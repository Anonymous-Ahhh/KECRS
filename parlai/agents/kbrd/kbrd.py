import copy
import os
import pickle as pkl
import re
from collections import defaultdict
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import nltk


from parlai.core.torch_agent import Output, TorchAgent, Batch
from parlai.core.utils import round_sigfigs

from .modules import KBRD



def _load_kg_embeddings(entity2entityId, dim, embedding_path):
    kg_embeddings = torch.zeros(len(entity2entityId), dim)
    with open(embedding_path, 'r') as f:
        for line in f.readlines():
            line = line.split('\t')
            entity = line[0]
            if entity not in entity2entityId:
                continue
            entityId = entity2entityId[entity]
            embedding = torch.Tensor(list(map(float, line[1:])))
            kg_embeddings[entityId] = embedding
    return kg_embeddings

def _load_text_embeddings(entity2entityId, dim, abstract_path):
    entities = []
    texts = []
    sent_tok = nltk.data.load('tokenizers/punkt/english.pickle')
    word_tok = nltk.tokenize.treebank.TreebankWordTokenizer()
    def nltk_tokenize(text):
        return [token for sent in sent_tok.tokenize(text)
                for token in word_tok.tokenize(sent)]

    with open(abstract_path, 'r') as f:
        for line in f.readlines():
            try:
                entity = line[:line.index('>')+1]
                if entity not in entity2entityId:
                    continue
                line = line[line.index('> "')+2:len(line)-line[::-1].index('@')-1]
                entities.append(entity)
                texts.append(line.replace('\\', ''))
            except Exception:
                pass
    vec_dim = 64
    try:
        model = Doc2Vec.load('doc2vec')
    except Exception:
        corpus = [nltk_tokenize(text) for text in texts]
        corpus = [
            TaggedDocument(words, ['d{}'.format(idx)])
            for idx, words in enumerate(corpus)
        ]
        model = Doc2Vec(corpus, vector_size=vec_dim, min_count=5, workers=28)
        model.save('doc2vec')

    full_text_embeddings = torch.zeros(len(entity2entityId), vec_dim)
    for i, entity in enumerate(entities):
        full_text_embeddings[entity2entityId[entity]] = torch.from_numpy(model.docvecs[i])

    return full_text_embeddings

class KbrdAgent(TorchAgent):
    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        super(KbrdAgent, cls).add_cmdline_args(argparser)
        agent = argparser.add_argument_group("Arguments")
        agent.add_argument("-ne", "--n-entity", type=int)
        agent.add_argument("-nr", "--n-relation", type=int)
        agent.add_argument("-dim", "--dim", type=int, default=128)
        # agent.add_argument("-hop", "--n-hop", type=int, default=2)
        agent.add_argument("-kgew", "--kge-weight", type=float, default=1)
        agent.add_argument("-l2w", "--l2-weight", type=float, default=2.5e-6)
        agent.add_argument("-nmem", "--n-memory", type=int, default=32)
        agent.add_argument(
            "-ium", "--item-update-mode", type=str, default="plus_transform"
        )
        agent.add_argument("-uah", "--using-all-hops", type=bool, default=True)
        agent.add_argument(
            "-lr", "--learningrate", type=float, default=3e-3, help="learning rate"
        )
        agent.add_argument("-nb", "--num-bases", type=int, default=8)
        KbrdAgent.dictionary_class().add_cmdline_args(argparser)
        return agent

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        init_model, is_finetune = self._get_init_model(opt, shared)

        self.id = "KbrdAgent"
        self.n_memory = opt["n_memory"]

        if not shared:
            # set up model from scratch

            self.kg = pkl.load(
                open(os.path.join(opt["datapath"], "redial", "subkg.pkl"), "rb")
            )
            self.movie_ids = pkl.load(
                open(os.path.join(opt["datapath"], "redial", "movie_ids.pkl"), "rb")
            )
            entity2entityId = pkl.load(
                open(os.path.join(opt["datapath"], "redial", "entity2entityId.pkl"), "rb")
            )
            entity_kg_emb = None
            abstract_path = 'dbpedia/short_abstracts_en.ttl'
            entity_text_emb = None
            opt["n_entity"] = len(entity2entityId)
            opt["n_relation"] = 214
            print("n_entity", opt["n_entity"])
            # encoder captures the input text
            self.model = KBRD(
                n_entity=opt["n_entity"],
                n_relation=opt["n_relation"],
                dim=opt["dim"],
                n_hop=2,
                kge_weight=opt["kge_weight"],
                l2_weight=opt["l2_weight"],
                n_memory=opt["n_memory"],
                item_update_mode=opt["item_update_mode"],
                using_all_hops=opt["using_all_hops"],
                kg=self.kg,
                entity_kg_emb=entity_kg_emb,
                entity_text_emb=entity_text_emb,
                num_bases=opt["num_bases"]
            )
            if init_model is not None:
                # load model parameters if available
                print("[ Loading existing model params from {} ]" "".format(init_model))
                states = self.load(init_model)
                if "number_training_updates" in states:
                    self._number_training_updates = states["number_training_updates"]

            if self.use_cuda:
                self.model.cuda()
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                opt["learningrate"],
            )

        elif "kbrd" in shared:
            # copy initialized data from shared table
            self.model = shared["kbrd"]
            self.kg = shared["kg"]
            self.movie_ids = shared["movie_ids"]
            self.optimizer = shared["optimizer"]

        self.metrics = defaultdict(float)
        self.counts = defaultdict(int)

    def report(self):
        """
        Report loss and perplexity from model's perspective.

        Note that this includes predicting __END__ and __UNK__ tokens and may
        differ from a truly independent measurement.
        """
        base = super().report()
        m = {}
        m["num_tokens"] = self.counts["num_tokens"]
        m["num_batches"] = self.counts["num_batches"]
        m["loss"] = self.metrics["loss"] / m["num_batches"]
        m["base_loss"] = self.metrics["base_loss"] / m["num_batches"]
        m["acc"] = self.metrics["acc"] / m["num_tokens"]
        m["auc"] = self.metrics["auc"] / m["num_tokens"]
        # Top-k recommendation Recall
        for x in sorted(self.metrics):
            if x.startswith("recall") and self.counts[x] > 200:
                m[x] = self.metrics[x] / self.counts[x]
                m["num_tokens_" + x] = self.counts[x]
            if x.startswith("precision") and self.counts[x] > 0:
                m[x] = self.metrics[x] / self.counts[x]
                m["num_tokens_" + x] = self.counts[x]
            if x.startswith("ndcg") and self.counts[x] > 0:
                m[x] = self.metrics[x] / self.counts[x]
                m["num_tokens_" + x] = self.counts[x]
            if x.startswith("same") and self.counts[x] > 0:
                m[x] = self.metrics[x] / self.counts[x]
                m["num_tokens_" + x] = self.counts[x]
        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            base[k] = round_sigfigs(v, 4)
        return base

    def reset_metrics(self):
        for key in self.metrics:
            self.metrics[key] = 0.0
        for key in self.counts:
            self.counts[key] = 0

    def share(self):
        """Share internal states."""
        shared = super().share()
        shared["kbrd"] = self.model
        shared["kg"] = self.kg
        shared["movie_ids"] = self.movie_ids
        shared["optimizer"] = self.optimizer
        return shared

    def vectorize(self, obs, history, **kwargs):
        if "text" not in obs:
            return obs

        if "labels" in obs:
            label_type = "labels"
        elif "eval_labels" in obs:
            label_type = "eval_labels"
        else:
            label_type = None
        if label_type is None:
            return obs

        # mentioned movies
        input_match = list(map(int, obs['label_candidates'][1].split()))
        labels_match = list(map(int, obs['label_candidates'][2].split()))
        entities_match = list(map(int, obs['label_candidates'][3].split()))
        liked_movie = list(map(int, obs['label_candidates'][5].split()))
        disliked_movie = list(map(int, obs['label_candidates'][6].split()))

        if labels_match == []:
            del obs["text"], obs[label_type]
            return obs

        input_vec = torch.zeros(self.model.n_entity)
        labels_vec = torch.zeros(self.model.n_entity, dtype=torch.long)
        input_vec[input_match] = 1
        input_vec[entities_match] = 1
        labels_vec[labels_match] = 1

        obs["text_vec"] = input_vec
        obs[label_type + "_vec"] = labels_vec

        # turn no.
        obs["turn"] = len(input_match)
        obs["liked_movie"] = liked_movie
        obs["disliked_movie"] = disliked_movie

        return obs

    def batchify(self, obs_batch, sort=False):
        """Override so that we can add memories to the Batch object."""
        batch = super().batchify(obs_batch, sort)
        valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if
                     self.is_valid(ex)]
        if len(valid_obs) == 0:
            return Batch()
        valid_inds, exs = zip(*valid_obs)
        # MOVIE ENTITIES
        turn = None
        if any('turn' in ex for ex in exs):
            turn = [ex.get('turn', []) for ex in exs]
        liked_movie = None
        if any('liked_movie' in ex for ex in exs):
            liked_movie = [ex.get('liked_movie', []) for ex in exs]
        disliked_movie = None
        if any('disliked_movie' in ex for ex in exs):
            disliked_movie = [ex.get('disliked_movie', []) for ex in exs]
        batch.turn = turn
        batch.liked_movie = liked_movie
        batch.disliked_movie = disliked_movie
        return batch

    def train_step(self, batch):
        self.model.train()
        bs = (batch.label_vec == 1).sum().item()
        labels = torch.zeros(bs, dtype=torch.long)

        # create subgraph for propagation
        seed_sets = []
        for i, (b, movieIdx) in enumerate(batch.label_vec.nonzero().tolist()):
            # seed set (i.e. mentioned movies + entitites)
            seed_set = batch.text_vec[b].nonzero().view(-1).tolist()
            labels[i] = movieIdx
            seed_sets.append(seed_set)

        if self.use_cuda:
            labels = labels.cuda()

        return_dict = self.model(seed_sets, labels)

        loss = return_dict["loss"]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.metrics["base_loss"] += return_dict["base_loss"].item()
        self.metrics["loss"] += loss.item()

        self.counts["num_tokens"] += bs
        self.counts["num_batches"] += 1
        self._number_training_updates += 1

    def eval_step(self, batch):
        if batch.text_vec is None:
            return

        self.model.eval()
        bs = (batch.label_vec == 1).sum().item()
        labels = torch.zeros(bs, dtype=torch.long)

        # create subgraph for propagation
        seed_sets = []
        turns = []
        positive_set_list = []
        liked_movie = []
        disliked_movie = []
        for i, (b, movieIdx) in enumerate(batch.label_vec.nonzero().tolist()):
            # seed set (i.e. mentioned movies + entitites)
            seed_set = batch.text_vec[b].nonzero().view(-1).tolist()
            liked_movie.append(batch.liked_movie[b])
            disliked_movie.append(batch.disliked_movie[b])
            labels[i] = movieIdx
            seed_sets.append(seed_set)
            turns.append(batch.turn[b])
            positive_set = batch.label_vec[b].nonzero().squeeze(dim=1).tolist()
            positive_set = [self.movie_ids.index(pos_item) for pos_item in positive_set]
            positive_set_list.append(set(positive_set))

        if self.use_cuda:
            labels = labels.cuda()

        return_dict = self.model(seed_sets, labels)

        loss = return_dict["loss"]

        self.metrics["base_loss"] += return_dict["base_loss"].item()
        self.metrics["loss"] += loss.item()
        self.counts["num_tokens"] += bs
        self.counts["num_batches"] += 1

        outputs = return_dict["scores"].cpu()
        outputs = outputs[:, torch.LongTensor(self.movie_ids)]
        pre_score, pred_idx = torch.topk(outputs, k=100, dim=1)
        target_idx_list = []
        counter = 0
        for b in range(bs):
            temp_seed_set = []
            for seed in seed_sets[b]:
                try:
                    # temp_seed_set.append(seed)
                    temp_seed_set.append(self.movie_ids.index(seed))
                except ValueError:
                    pass
            target_idx = self.movie_ids.index(labels[b].item())
            target_idx_list.append(target_idx)
            self.metrics["recall@1"] += int(target_idx in pred_idx[b][:1].tolist())
            self.metrics["recall@5"] += int(target_idx in pred_idx[b][:5].tolist())
            # if self.similarity_matrix[target_idx][int(pred_idx[b][:1])] > 0.5:
            #     self.metrics["recall@1@same"] += 1
            self.metrics["recall@10"] += int(target_idx in pred_idx[b][:10].tolist())
            self.metrics["recall@50"] += int(target_idx in pred_idx[b][:50].tolist())
            self.metrics[f"recall@1@turn{turns[b]}"] += int(target_idx in pred_idx[b][:1].tolist())
            self.metrics[f"recall@10@turn{turns[b]}"] += int(target_idx in pred_idx[b][:10].tolist())
            self.metrics[f"recall@50@turn{turns[b]}"] += int(target_idx in pred_idx[b][:50].tolist())
            self.metrics["same1"] += int(target_idx in temp_seed_set)
            # 测试集中推荐之前提到过电影的比例
            self.metrics["same2"] += int(pred_idx[b][:1].tolist()[0] in temp_seed_set)  # top-1预测之前提到过电影的比例
            self.metrics["same3"] += int(len(set(pred_idx[b][:5].tolist()) & set(temp_seed_set)) > 0)
            self.metrics["same4"] += int(len(set(pred_idx[b][:10].tolist()) & set(temp_seed_set)) > 0)
            # 测试集中推荐之前喜欢电影的比例
            # self.metrics["same5"] += int(pred_idx[b][:1].tolist()[0] in [self.movie_ids.index(d_m) for d_m in liked_movie[b]] and pred_idx[b][:1].tolist()[0] not in temp_seed_set)
            # self.metrics["same6"] += int(len((set(pred_idx[b][:5].tolist()) & set(temp_seed_set)) & set([self.movie_ids.index(d_m) for d_m in liked_movie[b]])))
            # self.metrics["same7"] += int(len((set(pred_idx[b][:10].tolist()) & set(temp_seed_set)) & set([self.movie_ids.index(d_m) for d_m in liked_movie[b]])))
            self.metrics["same5"] += int(target_idx in pred_idx[b][:1].tolist() and target_idx in set([self.movie_ids.index(d_m) for d_m in liked_movie[b]]))
            self.metrics["same6"] += int(target_idx in pred_idx[b][:1].tolist() and target_idx in set([self.movie_ids.index(d_m) for d_m in disliked_movie[b]]))
            # 测试集中推荐之前不喜欢电影的比例
            self.metrics["same8"] += int(pred_idx[b][:1].tolist()[0] in [self.movie_ids.index(d_m) for d_m in disliked_movie[b]] and pred_idx[b][:1].tolist()[0] not in temp_seed_set)
            self.metrics["same9"] += int(len((set(pred_idx[b][:5].tolist()) & set(temp_seed_set)) & set([self.movie_ids.index(d_m) for d_m in disliked_movie[b]])))
            self.metrics["same10"] += int(len((set(pred_idx[b][:10].tolist()) & set(temp_seed_set)) & set([self.movie_ids.index(d_m) for d_m in disliked_movie[b]])))

            self.counts[f"recall@1@turn{turns[b]}"] += 1
            self.counts[f"recall@10@turn{turns[b]}"] += 1
            self.counts[f"recall@50@turn{turns[b]}"] += 1
            self.counts[f"recall@1"] += 1
            self.counts[f"recall@5"] += 1
            # self.counts[f"recall@1@same"] += 1
            self.counts[f"recall@10"] += 1
            self.counts[f"recall@50"] += 1
            self.counts[f"same1"] += 1
            self.counts[f"same2"] += 1
            self.counts[f"same3"] += 1
            self.counts[f"same4"] += 1
            self.counts[f"same5"] += 1
            self.counts[f"same6"] += 1
            self.counts[f"same7"] += 1
            self.counts[f"same8"] += 1
            self.counts[f"same9"] += 1
            self.counts[f"same10"] += 1
        # print("recall@1", self.metrics["recall@1"], self.counts[f"recall@1"])
        # print("[target_idx_list]", target_idx_list)
        # return Output(list(map(str, pred_idx[:, 0].tolist())))
            # NDCG
            if target_idx in pred_idx[b][:5].tolist():
                self.metrics["ndcg@5"] += 1 / math.log(pred_idx[b][:10].tolist().index(target_idx) + 2, 2)
            self.counts[f"ndcg@5"] += 1
            if target_idx in pred_idx[b][:10].tolist():
                self.metrics["ndcg@10"] += 1 / math.log(pred_idx[b][:10].tolist().index(target_idx) + 2, 2)
            self.counts[f"ndcg@10"] += 1
            if target_idx in pred_idx[b][:20].tolist():
                self.metrics["ndcg@20"] += 1 / math.log(pred_idx[b][:20].tolist().index(target_idx) + 2, 2)
            self.counts[f"ndcg@20"] += 1
            if target_idx in pred_idx[b][:50].tolist():
                self.metrics["ndcg@50"] += 1 / math.log(pred_idx[b][:50].tolist().index(target_idx) + 2, 2)
            self.counts[f"ndcg@50"] += 1
            if counter == 0:
                target_id_set = positive_set_list[b]
                counter = len(target_id_set) - 1
                self.metrics["precision@1"] += len(target_id_set & set(pred_idx[b][:1].tolist()))
                self.metrics["precision@3"] += len(target_id_set & set(pred_idx[b][:3].tolist())) / 3
                self.metrics["precision@5"] += len(target_id_set & set(pred_idx[b][:5].tolist())) / 5
                self.metrics["precision@10"] += len(target_id_set & set(pred_idx[b][:10].tolist())) / 10
                self.metrics["precision@50"] += len(target_id_set & set(pred_idx[b][:50].tolist())) / 50
                self.counts[f"precision@1"] += 1
                self.counts[f"precision@3"] += 1
                self.counts[f"precision@5"] += 1
                self.counts[f"precision@10"] += 1
                self.counts[f"precision@50"] += 1
            else:
                counter -= 1

