# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.agents import Agent
from parlai.core.utils import warn_once
from parlai.core.utils import padded_3d, padded_tensor, round_sigfigs
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.core.torch_generator_agent import TorchGeneratorAgent
from parlai.core.torch_agent import TorchAgent, Batch, Output

from .modules import TransformerMemNetModel
from .modules import TransformerGeneratorModel

import torch
import torch.nn.functional as F
import re
import math
import os
import pickle as pkl


warn_once(
    "Public release transformer models are currently in beta. The name of "
    "command line options may change or disappear before a stable release. We "
    "welcome your feedback. Please file feedback as issues at "
    "https://github.com/facebookresearch/ParlAI/issues/new"
)


def add_common_cmdline_args(argparser):
    argparser.add_argument('-esz', '--embedding-size', type=int, default=300,
                           help='Size of all embedding layers')
    argparser.add_argument('-nl', '--n-layers', type=int, default=2)
    argparser.add_argument('-hid', '--ffn-size', type=int, default=300,
                           help='Hidden size of the FFN layers')
    argparser.add_argument('--dropout', type=float, default=0.0,
                           help='Dropout used in Vaswani 2017.')
    argparser.add_argument('--attention-dropout', type=float, default=0.0,
                           help='Dropout used after attention softmax.')
    argparser.add_argument('--relu-dropout', type=float, default=0.0,
                           help='Dropout used after ReLU. From tensor2tensor.')
    argparser.add_argument('--n-heads', type=int, default=2,
                           help='Number of multihead attention heads')
    argparser.add_argument('--learn-positional-embeddings', type='bool', default=False)
    argparser.add_argument('--embeddings-scale', type='bool', default=True)
    argparser.add_argument('--n-positions', type=int, default=None, hidden=True,
                           help='Number of positional embeddings to learn. Defaults '
                                'to truncate or 1024 if not provided.')


class Transformer(Agent):
    """
    Placeholder class, which just throws an error telling the user to specify
    whether they want the ranker or the generator.
    """
    def __init__(self, opt, shared=None):
        raise RuntimeError(
            "`--model transformer` is not a valid choice. Please select either "
            "`--model transformer/ranker` or `--model transformer/generator"
        )


class TransformerRankerAgent(TorchRankerAgent):
    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        super(TransformerRankerAgent, cls).add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Transformer Arguments')
        add_common_cmdline_args(agent)
        # memory and knowledge arguments
        agent.add_argument('--use-memories', type='bool', default=False,
                           help='use memories: must implement the function '
                                '`_vectorize_memories` to use this')
        agent.add_argument('--wrap-memory-encoder', type='bool',
                           default=False,
                           help='wrap memory encoder with MLP')
        agent.add_argument('--memory-attention', type=str, default='sqrt',
                           choices=['cosine', 'dot', 'sqrt'],
                           help='similarity for basic attention mechanism'
                                'when using transformer to encode memories')
        # model specific arguments
        agent.add_argument('--normalize-sent-emb', type='bool', default=False)
        agent.add_argument('--share-encoders', type='bool', default=True)
        agent.add_argument('--learn-embeddings', type='bool', default=True,
                           help='learn embeddings')
        agent.add_argument('--data-parallel', type='bool', default=False,
                           help='use model in data parallel, requires '
                                'multiple gpus')
        argparser.set_defaults(
            learningrate=0.0001,
            optimizer='adamax',
            truncate=1024,
        )
        cls.dictionary_class().add_cmdline_args(argparser)

        return agent

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.data_parallel = opt.get('data_parallel') and self.use_cuda
        if self.data_parallel:
            from parlai.core.distributed_utils import is_distributed
            if is_distributed():
                raise ValueError(
                    'Cannot combine --data-parallel and distributed mode'
                )
            self.model = torch.nn.DataParallel(self.model)

    def _score(self, output, cands):
        if cands.dim() == 2:
            return torch.matmul(output, cands.t())
        elif cands.dim() == 3:
            return torch.bmm(output.unsqueeze(1),
                             cands.transpose(1, 2)).squeeze(1)
        else:
            raise RuntimeError('Unexpected candidate dimensions {}'
                               ''.format(cands.dim()))

    def build_model(self, states=None):
        self.model = TransformerMemNetModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                self.model.embeddings.weight, self.opt['embedding_type']
            )
        return self.model

    def batchify(self, obs_batch, sort=False):
        """Override so that we can add memories to the Batch object."""
        batch = super().batchify(obs_batch, sort)
        if self.opt['use_memories']:
            valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if
                         self.is_valid(ex)]
            valid_inds, exs = zip(*valid_obs)
            mems = None
            if any('memory_vecs' in ex for ex in exs):
                mems = [ex.get('memory_vecs', None) for ex in exs]
            batch.memory_vecs = mems
        return batch

    def _vectorize_memories(self, obs):
        # TODO: move this to Torch Ranker Agent
        raise NotImplementedError(
            'Abstract class: user must implement this function to use memories'
        )

    def vectorize(self, *args, **kwargs):
        kwargs['add_start'] = False
        kwargs['add_end'] = False
        obs = super().vectorize(*args, **kwargs)
        if self.opt['use_memories']:
            obs = self._vectorize_memories(obs)
        return obs

    def encode_candidates(self, padded_cands):
        _, cands = self.model(
            xs=None,
            mems=None,
            cands=padded_cands,
        )

        return cands

    def score_candidates(self, batch, cand_vecs, cand_encs=None):
        # convoluted check that not all memories are empty
        if (self.opt['use_memories'] and batch.memory_vecs is not None and
                sum(len(m) for m in batch.memory_vecs)):
            mems = padded_3d(batch.memory_vecs, use_cuda=self.use_cuda,
                             pad_idx=self.NULL_IDX)
        else:
            mems = None

        if cand_encs is not None:
            # we pre-encoded the candidates, do not re-encode here
            cand_vecs = None

        context_h, cands_h = self.model(
            xs=batch.text_vec,
            mems=mems,
            cands=cand_vecs,
        )

        if cand_encs is not None:
            cands_h = cand_encs

        scores = self._score(context_h, cands_h)

        return scores


class TransformerRecGeneratorAgent(TorchGeneratorAgent):
    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('Transformer Arguments')
        add_common_cmdline_args(agent)
        agent.add_argument("-ne", "--n-entity", type=int)
        agent.add_argument("-nr", "--n-relation", type=int)
        agent.add_argument("-dim", "--dim", type=int, default=128)
        agent.add_argument("-hop", "--n-hop", type=int, default=2)
        agent.add_argument("-kgew", "--kge-weight", type=float, default=1)
        agent.add_argument("-l2w", "--l2-weight", type=float, default=2.5e-6)
        agent.add_argument("-nmem", "--n-memory", type=int, default=32)
        agent.add_argument(
            "-ium", "--item-update-mode", type=str, default="plus_transform"
        )
        agent.add_argument("-uah", "--using-all-hops", type=bool, default=True)
        cls.dictionary_class().add_cmdline_args(argparser)

        super(TransformerRecGeneratorAgent, cls).add_cmdline_args(argparser)
        return agent

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.valid_output = []
        self.valid_input = []
        self.valid_ground_truth = []
        self.kg = pkl.load(
            open(os.path.join(opt["datapath"], "redial", "subkg.pkl"), "rb")
        )
        self.n_entity = opt['n_entity']
        self.movie_ids = pkl.load(
            open(os.path.join(opt["datapath"], "redial", "movie_ids.pkl"), "rb")
        )
        entity2entityId = pkl.load(
            open(os.path.join(opt["datapath"], "redial", "entity2entityId.pkl"), "rb")
        )
        self.entityId2entity = {entity2entityId[x]: x for x in entity2entityId}
        id2entity = pkl.load(
            open(os.path.join(opt["datapath"], "redial", "id2entity.pkl"), "rb")
        )
        self.entity2id = {id2entity[x]: x for x in id2entity}
        with open(os.path.join(opt["datapath"], "redial", "movies_with_mentions.csv"), "r") as f:
            self.mid2movie = {}
            f.readline()
            for line in f:
                pos = line.index(",")
                key = int(line[:pos])
                val = line[pos + 1:-line[::-1].index(",") - 1]
                self.mid2movie[key] = val

    def build_model(self, states=None):
        self.model = TransformerGeneratorModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                self.model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        if self.use_cuda:
            self.model.cuda()
        return self.model

    def distinct_metrics(self):
        outs = [line.strip().split(" ") for line in self.valid_output]

        # outputs is a list which contains several sentences, each sentence contains several words
        unigram_count = 0
        bigram_count = 0
        trigram_count = 0
        quadragram_count = 0
        quintagram_count = 0
        unigram_set = set()
        bigram_set = set()
        trigram_set = set()
        quadragram_set = set()
        quintagram_set = set()
        for sen in outs:
            for word in sen:
                unigram_count += 1
                unigram_set.add(word)
            for start in range(len(sen) - 1):
                bg = str(sen[start]) + ' ' + str(sen[start+1])
                bigram_count += 1
                bigram_set.add(bg)
            for start in range(len(sen)-2):
                trg = str(sen[start]) + ' ' + str(sen[start+1]) + ' ' + str(sen[start+2])
                trigram_count += 1
                trigram_set.add(trg)
            for start in range(len(sen)-3):
                quadg = str(sen[start]) + ' ' + str(sen[start+1]) + \
                        ' ' + str(sen[start+2]) + ' ' + str(sen[start+3])
                quadragram_count += 1
                quadragram_set.add(quadg)
            for start in range(len(sen)-4):
                quing = str(sen[start]) + ' ' + str(sen[start+1]) + ' ' + \
                        str(sen[start+2]) + ' ' + str(sen[start+3]) + ' ' + str(sen[start+4])
                quintagram_count += 1
                quintagram_set.add(quing)
        dist1 = len(unigram_set) / len(outs)  # unigram_count
        dist2 = len(bigram_set) / len(outs)  # bigram_count
        dist3 = len(trigram_set)/len(outs)  # trigram_count
        dist4 = len(quadragram_set)/len(outs)  # quadragram_count
        dist5 = len(quintagram_set)/len(outs)  # quintagram_count
        return dist1, dist2, dist3, dist4, dist5

    def report(self):
        """
        Report loss and perplexity from model's perspective.

        Note that this includes predicting __END__ and __UNK__ tokens and may
        differ from a truly independent measurement.
        """
        base = super().report()
        m = {}
        num_tok = self.metrics['num_tokens']
        if num_tok > 0:
            m['loss'] = self.metrics['loss']
            if self.metrics['correct_tokens'] > 0:
                m['token_acc'] = self.metrics['correct_tokens'] / num_tok
            m['nll_loss'] = self.metrics['nll_loss'] / num_tok
            try:
                m['ppl'] = math.exp(m['nll_loss'])
            except OverflowError:
                m['ppl'] = float('inf')
        if self.metrics['num_pre'] > 0:
            m['accuracy'] = self.metrics["accuracy"] / self.metrics["num_pre"]
        if not self.is_training:
            m['dist1'], m['dist2'], m['dist3'], m['dist4'], m['dist5'] = self.distinct_metrics()
            with open("./test_output_kbrd.txt", "w", encoding="utf-8") as f:
                for output in self.valid_output:
                    f.write(output + "\n")
            with open("./test_input_kbrd.txt", "w", encoding="utf-8") as f:
                for output in self.valid_input:
                    f.write(output+"\n")
            with open("./test_ground_kbrd.txt", "w", encoding="utf-8") as f:
                for output in self.valid_ground_truth:
                    f.write(output+"\n")
            self.valid_output = []
        if self.metrics["recall@count"] > 0:
            m["recall"] = self.metrics["recall"] / self.metrics["recall@count"]
        if self.metrics['total_skipped_batches'] > 0:
            m['total_skipped_batches'] = self.metrics['total_skipped_batches']
        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            base[k] = round_sigfigs(v, 4)
        return base

    def vectorize(self, obs, history, **kwargs):
        kwargs['add_start'] = False
        kwargs['add_end'] = False
        obs = super().vectorize(obs, history, **kwargs)

        if "text" not in obs:
            return obs
        # match movies and entities
        input_match = list(map(int, obs['label_candidates'][1].split()))
        entities_match = list(map(int, obs['label_candidates'][3].split()))

        obs["movies"] = input_match + entities_match

        return obs

    def batchify(self, obs_batch, sort=False):
        """Override so that we can add memories to the Batch object."""
        batch = super().batchify(obs_batch, sort)
        valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if
                     self.is_valid(ex)]
        valid_inds, exs = zip(*valid_obs)
        # MOVIE ENTITIES
        movies = None
        if any('movies' in ex for ex in exs):
            movies = [ex.get('movies', []) for ex in exs]
        # label movie
        batch.movies = movies
        return batch

    def train_step(self, batch):
        """Train on a single batch of examples."""
        batchsize = batch.text_vec.size(0)
        # helps with memory usage
        self._init_cuda_buffer(batchsize, self.truncate or 256)
        self.model.train()
        self.zero_grad()

        if getattr(batch, 'movies', None):
            assert hasattr(self.model, 'kbrd')
            self.model.user_representation, _ = self.model.kbrd.user_representation(batch.movies)
            self.model.user_representation = self.model.user_representation.detach()
        try:
            loss = self.compute_loss(batch)
            self.metrics['loss'] += loss.item()
            self.backward(loss)
            self.update_params()
        except RuntimeError as e:
            # catch out of memory exceptions during fwd/bck (skip batch)
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch. '
                      'if this happens frequently, decrease batchsize or '
                      'truncate the inputs to the model.')
                self.metrics['total_skipped_batches'] += 1
                # gradients are synced on backward, now this model is going to be
                # out of sync! catch up with the other workers
                self._init_cuda_buffer(8, 8, True)
            else:
                raise e

    def eval_step(self, batch):
        """Evaluate a single batch of examples."""
        if batch.text_vec is None:
            return
        bsz = batch.text_vec.size(0)
        self.model.eval()
        cand_scores = None
        if getattr(batch, 'movies', None):
            assert hasattr(self.model, 'kbrd')
            self.model.user_representation, self.model.nodes_features = self.model.kbrd.user_representation(
                batch.movies)
            self.model.user_representation, self.model.nodes_features = self.model.user_representation.detach(), self.model.nodes_features.detach()

        if batch.label_vec is not None:
            # calculate loss on targets with teacher forcing
            loss = self.compute_loss(batch)  # noqa: F841  we need the side effects
            self.metrics['loss'] += loss.item()

        preds = None
        if self.skip_generation:
            warn_once(
                "--skip-generation does not produce accurate metrics beyond ppl",
                RuntimeWarning
            )
        elif self.beam_size == 1:
            # greedy decode
            _, preds, *_ = self.model(*self._model_input(batch), bsz=bsz)
        elif self.beam_size > 1:
            out = self.beam_search(
                self.model,
                batch,
                self.beam_size,
                start=self.START_IDX,
                end=self.END_IDX,
                pad=self.NULL_IDX,
                min_length=self.beam_min_length,
                min_n_best=self.beam_min_n_best,
                block_ngram=self.beam_block_ngram
            )
            beam_preds_scores, _, beams = out
            preds, scores = zip(*beam_preds_scores)

            if self.beam_dot_log is True:
                self._write_beam_dots(batch.text_vec, beams)

        cand_choices = None
        # TODO: abstract out the scoring here
        if self.rank_candidates:
            # compute roughly ppl to rank candidates
            cand_choices = []
            encoder_states = self.model.encoder(*self._model_input(batch))
            for i in range(bsz):
                num_cands = len(batch.candidate_vecs[i])
                enc = self.model.reorder_encoder_states(encoder_states, [i] * num_cands)
                cands, _ = padded_tensor(
                    batch.candidate_vecs[i], self.NULL_IDX, self.use_cuda
                )
                scores, _ = self.model.decode_forced(enc, cands)
                cand_losses = F.cross_entropy(
                    scores.view(num_cands * cands.size(1), -1),
                    cands.view(-1),
                    reduction='none',
                ).view(num_cands, cands.size(1))
                # now cand_losses is cands x seqlen size, but we still need to
                # check padding and such
                mask = (cands != self.NULL_IDX).float()
                cand_scores = (cand_losses * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
                _, ordering = cand_scores.sort()
                cand_choices.append([batch.candidates[i][o] for o in ordering])

        text = [self._v2t(p) for p in preds] if preds is not None else None
        # Replace __unk__ with recommendations when generating responses
        if text is not None and bsz == 1:
            for j, t in enumerate(text):
                scores = F.linear(self.model.user_representation, self.model.nodes_features,
                                  self.model.kbrd.output.bias)
                outputs = scores.cpu()
                outputs = outputs[0, torch.LongTensor(self.movie_ids)]
                rec_movies = list(map(lambda x: str(self.movie_ids[x]), outputs.argsort(descending=True).tolist()))
                movie_idx = 0
                while "__unk__" in t:
                    pos = t.index("__unk__")
                    while int(rec_movies[movie_idx]) in batch.movies[0]:
                        movie_idx += 1
                    entity = self.entityId2entity[int(rec_movies[movie_idx])]
                    if entity in self.entity2id:
                        mid = self.entity2id[entity]
                    else:
                        mid = entity
                    t = t[:pos] + "\"" + self.mid2movie[mid] + "\"" + t[pos + 7:]
                    movie_idx += 1
                text[j] = t

        text = [self._v2t(p) for p in preds] if preds is not None else None
        # self.valid_output.extend(preds.detach().cpu().tolist())
        self.valid_output.extend(text)
        input_text = [self._v2t(p).replace("__null__", "") for p in batch.text_vec] if batch.text_vec is not None else None
        label_text = [self._v2t(p).replace("__null__", "") for p in batch.label_vec] if batch.label_vec is not None else None
        self.valid_input.extend(input_text)
        self.valid_ground_truth.extend(label_text)

        return Output(text, cand_choices)
