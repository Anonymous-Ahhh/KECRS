import pickle as pkl
import json
import re
import time
from collections import defaultdict


def build_similarity_matrix_dbpedia():
    """
    Build simple similarity matrix for movie using entity id of dbpedia
    """
    id2entity = pkl.load(open("../../../data/redial/id2entity.pkl", "rb"))
    entity2entity_id_dbpedia = pkl.load(open("../../../data/redial/entity2entityId.pkl", "rb"))
    entity2entity_id_tmdb = pkl.load(open("../../../data/crs/entity2entity_id2.pkl", "rb"))
    # movie_kg = pkl.load(open("../../../data/crs/movie_kg.pkl", "rb"))
    dbpedia_entity_id2id = {}
    dbpedia_entity2tmdb_entity_id = {}
    for idx in id2entity:
        entity_dbpedia = id2entity[idx]
        if entity_dbpedia:
            entity_id_dbpedia = entity2entity_id_dbpedia[entity_dbpedia]
        else:
            entity_id_dbpedia = entity2entity_id_dbpedia[idx]
        if entity_id_dbpedia not in dbpedia_entity_id2id:
            dbpedia_entity_id2id[entity_id_dbpedia] = idx
        else:
            print(idx, dbpedia_entity_id2id[entity_id_dbpedia], entity_dbpedia)
        entity_id_tmdb = entity2entity_id_tmdb[str(idx)]
        dbpedia_entity2tmdb_entity_id[entity_id_dbpedia] = entity_id_tmdb
    pkl.dump(dbpedia_entity2tmdb_entity_id, open("../../../data/redial/dbpedia_entity2tmdb_entity_id2.pkl", "wb"))


def build_text_dict():
    text_dict = defaultdict(set)
    train_data = "../../../data/crs/train_data.jsonl"
    test_data = "../../../data/crs/test_data.jsonl"
    valid_data = "../../../data/crs/valid_data.jsonl"
    entity_index = pkl.load(open("../../../data/crs/entity2entity_id4.pkl", "rb"))
    entity_set = set()
    entity_list = []
    entity_index["comedies"] = entity_index["comedy"]
    entity_index["Shakespeare"] = entity_index["William Shakespeare"]
    for entity in entity_index:
        try:
            int(entity)
        except ValueError:
            entity_set.add(entity)
            entity_list.append(entity)
    length = [len(entity) for entity in entity_list]
    for entity in entity_list:
        if len(entity) > 30:
            print(entity)
    print(min(length), max(length))
    instances = []
    with open(train_data) as json_file:
        for line in json_file.readlines():
            instances.append(json.loads(line))
    with open(test_data) as json_file:
        for line in json_file.readlines():
            instances.append(json.loads(line))
    with open(valid_data) as json_file:
        for line in json_file.readlines():
            instances.append(json.loads(line))
    total_text = 0
    has_entity_text = 0
    instances_counter = 0
    start_time = time.time()
    for instance in instances:
        instances_counter += 1
        messages = instance["messages"]
        for message in messages:
            text = message["text"]
            total_text += 1
            if text == "" or len(text) < 3:
                continue
            for entity in entity_list:
                if len(entity) > len(text):
                    continue
                if entity.lower() in text.lower():
                    # print(1)
                    pass
                else:
                    continue
                _entity = '\\b' + entity + '\\b'
                match = re.findall(_entity.lower(), text.lower())
                if match:
                    text_dict[text].add(entity_index[entity])
                    has_entity_text += 1
        if instances_counter % 100 == 0:
            time_elapsed = time.time() - start_time
            start_time = time.time()
            print("Time for 100 instance / ", instances_counter, time_elapsed, "Total instance", len(instances),
                  "Estimated time (min)", len(instances) / 60 / 100 * time_elapsed, "Has entity text", has_entity_text)

    pkl.dump(text_dict, open("../../../data/crs/text_dict_tmdb4.pkl", "wb"))


def build_abstract_text_dict():
    abstracts = pkl.load(open("../../../data/crs/entity_overview4.pkl", "rb"))
    abstract_text_dict = defaultdict(set)
    entity_index = pkl.load(open("../../../data/crs/entity2entity_id4.pkl", "rb"))
    entity_set = set()
    entity_list = []
    entity_index["comedies"] = entity_index["comedy"]
    entity_index["Shakespeare"] = entity_index["William Shakespeare"]
    for entity in entity_index:
        try:
            int(entity)
        except ValueError:
            entity_set.add(entity)
            entity_list.append(entity)
    length = []
    start_time = time.time()
    for abstract in abstracts:
        if abstract == "":
            continue
        for entity in entity_list:
            if len(entity) > len(abstract):
                continue
            if entity.lower() not in abstract.lower():
                continue
            _entity = '\\b' + entity + '\\b'
            match = re.findall(_entity.lower(), abstract.lower())
            if match:
                abstract_text_dict[abstract].add(entity_index[entity])
        length.append(len(abstract_text_dict[abstract]))
    pkl.dump(abstract_text_dict, open("../../../data/crs/abstract_text_dict.pkl", "wb"))
    print(sum(length) / len(length), time.time() - start_time)


def movie2idx(text, entity2id):
    patten = re.compile("@\d+")
    movie_entity_idx = []
    movie_id_list = re.findall(patten, text)
    for movieId in movie_id_list:
        movie_entity_idx.append(entity2id[movieId[1:]])
        # movie_entity_idx.append(movieId[1:])
    return text, movie_entity_idx


def analyze_kg_text():
    train_data = "../../../data/crs/train_data.jsonl"
    valid_data = "../../../data/crs/valid_data.jsonl"
    test_data = "../../../data/crs/test_data.jsonl"
    text_dict = pkl.load(open("../../../data/crs/text_dict_tmdb4.pkl", "rb"))
    entity2id = pkl.load(open("../../../data/crs/entity2entity_id4.pkl", "rb"))
    instances = []
    with open(train_data) as json_file:
        for line in json_file.readlines():
            instances.append(json.loads(line))
    with open(valid_data) as json_file:
        for line in json_file.readlines():
            instances.append(json.loads(line))
    # with open(test_data) as json_file:
    #     for line in json_file.readlines():
    #         instances.append(json.loads(line))
    previous_target_pair = []
    previous_all_target_pair = []
    total_movie_mentioned = 0
    total_movie_mentioned_filtered = 0
    self_loop_counter = 0
    self_loop_counter_liked = 0
    self_loop_counter_disliked = 0
    self_loop_counter_unknown = 0
    # define iterator over all queries
    for instance in instances:
        initiator_id = instance["initiatorWorkerId"]
        respondent_id = instance["respondentWorkerId"]
        respondent_ques = instance["initiatorQuestions"]
        liked_movie = []
        for movie_id in respondent_ques:
            if respondent_ques[movie_id]["liked"] == 1:
                liked_movie.append(entity2id[movie_id])
                # liked_movie.append(movie_id)
        messages = instance["messages"]
        message_idx = 0
        mentioned_entities = []

        previously_mentioned_movies_list = []
        while message_idx < len(messages):
            source_text = []
            target_text = []
            while (
                    message_idx < len(messages)
                    and messages[message_idx]["senderWorkerId"] == initiator_id
            ):
                source_text.append(messages[message_idx]["text"])
                message_idx += 1
            while (
                    message_idx < len(messages)
                    and messages[message_idx]["senderWorkerId"] == respondent_id
            ):
                target_text.append(messages[message_idx]["text"])
                message_idx += 1
            source_text = [text for text in source_text if text != ""]
            target_text = [text for text in target_text if text != ""]
            if source_text != [] or target_text != []:
                for src in source_text:
                    mentioned_entities += text_dict[src]
                target_mentioned_entities = []
                for tgt in target_text:
                    target_mentioned_entities += text_dict[tgt]
                source_text = '\n'.join(source_text)
                target_text = '\n'.join(target_text)
                source_text, source_movie_list = movie2idx(source_text, entity2id)
                target_text, target_movie_list = movie2idx(target_text, entity2id)
                total_movie_mentioned += len(target_movie_list)
                for target_movie in target_movie_list:
                    if target_movie in source_movie_list + previously_mentioned_movies_list:
                        target_movie_list.remove(target_movie)
                        self_loop_counter += 1
                        if target_movie in liked_movie:
                            self_loop_counter_liked += 1
                        else:
                            self_loop_counter_unknown += 1
                for target_movie in target_movie_list:
                    if target_movie not in liked_movie:
                        target_movie_list.remove(target_movie)
                        self_loop_counter_disliked += 1
                if not [] == target_movie_list:
                    previous_target_pair.append(
                        (previously_mentioned_movies_list + source_movie_list + mentioned_entities,
                         list(set(target_movie_list)))
                    )
                previously_mentioned_movies_list += source_movie_list + target_movie_list
                mentioned_entities += target_mentioned_entities
    print(total_movie_mentioned, total_movie_mentioned_filtered, self_loop_counter)
    print(self_loop_counter_liked, self_loop_counter_disliked, self_loop_counter_unknown)
    analyze_kg_text_2(previous_target_pair)
    # pkl.dump(previous_target_pair, open("../../../data/crs/previous_target_pair_train_with_dialogue.pkl", "wb"))
    # pkl.dump(previous_target_pair, open("../../../data/crs/previous_target_pair_test_with_dialogue.pkl", "wb"))


def analyze_kg_text_2(previous_target_pair):
    # previous_target_pair = pkl.load(open("../../../data/crs/previous_target_pair_train_with_dialogue.pkl", "rb"))
    # previous_target_pair = pkl.load(open("../../../data/crs/previous_target_pair_test_with_dialogue.pkl", "rb"))
    # previous_target_pair_test = pkl.load(open("../../../data/crs/previous_target_pair_valid_test.pkl", "rb"))
    kg = pkl.load(open("../../../data/crs/movie_kg4.pkl", "rb"))
    c_t_pair = set()
    c_t_pair_dict = defaultdict(int)
    # c_t_pair_test = set()
    # c_t_pair_test_dict = defaultdict(int)
    for context_list, target_list in previous_target_pair:
        for target in target_list:
            for context in context_list:
                c_t_pair.add((context, target))
                c_t_pair_dict[(context, target)] += 1
    # for context_list, target_list in previous_target_pair_test:
    #     for target in target_list:
    #         for context in context_list:
    #             c_t_pair_test.add((context, target))
    #             c_t_pair_test_dict[(context, target)] += 1
    # # c_t_pair_dict_order = sorted(c_t_pair_dict.items(), key=lambda x: x[1], reverse=True)
    # c_t_pair_dict_order = sorted(c_t_pair_dict.items(), key=lambda x: x[1])
    # c_t_pair_test_dict_order = sorted(c_t_pair_test_dict.items(), key=lambda x: x[1], reverse=True)
    # print(len(c_t_pair), len(c_t_pair_test))
    # for key in c_t_pair_dict:
    #     if c_t_pair_dict[key] < 2:
    #         c_t_pair.remove(key)
    # for key in c_t_pair_test_dict:
    #     if c_t_pair_test_dict[key] < 2:
    #         c_t_pair_test.remove(key)
    # overlap = c_t_pair & c_t_pair_test
    # print(len(c_t_pair), len(c_t_pair_test), len(overlap))
    # print(1)

    def find_neighbour(head_list):
        one_hop_neighbour = set()
        for head in head_list:
            for relation, tail in list(set(kg[head])):
                if relation == 16:  # or relation == 0:
                    continue
                one_hop_neighbour.add(tail)
        # return None, one_hop_neighbour
        two_hop_neighbour = set()
        for head in list(one_hop_neighbour):
            kg_info = list(set(kg[head]))
            for relation, tail in kg_info:
                two_hop_neighbour.add(tail)
        # return one_hop_neighbour, two_hop_neighbour
        three_hop_neighbour = set()
        for head in list(two_hop_neighbour):
            kg_info = list(set(kg[head]))
            for relation, tail in kg_info:
                three_hop_neighbour.add(tail)
        four_hop_neighbour = set()
        for head in list(three_hop_neighbour):
            kg_info = list(set(kg[head]))
            for relation, tail in kg_info:
                four_hop_neighbour.add(tail)
        return one_hop_neighbour, four_hop_neighbour

    two_hop_find_num = 0
    total_pair = 0
    average_neighbour = 0
    average_neighbour_find = 0
    counter = 0
    for context_list, target_list in previous_target_pair:
        for target in target_list:
            total_pair += 1
            one_hop, neighbour = find_neighbour(context_list)
            average_neighbour += len(neighbour)
            if target in neighbour:
                two_hop_find_num += 1
                average_neighbour_find += len(neighbour)
            else:
                # print(context_list, target_list)
                target_neighbour = kg[target]
                if [] == target_neighbour:
                    counter += 1
    print("Total pair in training", total_pair)
    print("Find target in context neighbour number", two_hop_find_num)
    print("Average of neighbour of context", average_neighbour / total_pair)
    print("Average of neighbour of context of found", average_neighbour_find / two_hop_find_num)
    print(counter)
    print(len(previous_target_pair))


def get_path():
    kg = pkl.load(open("../../../data/crs/movie_kg4.pkl", "rb"))
    entity2id = pkl.load(open("../../../data/crs/entity2entity_id4.pkl", "rb"))
    id2entity = {}
    for e in entity2id:
        id2entity[entity2id[e]] = e
    source_e = 5688
    target_e = 5573
    source_one_hop_set = set([t for r, t in kg[source_e]])
    target_one_hop_set = set([t for r, t in kg[target_e]])
    overlap_set = source_one_hop_set & target_one_hop_set
    for r, t in list(set(kg[source_e])):
        if t in overlap_set:
            for r_, t_ in list(set(kg[target_e])):
                if t_ == t:
                    print(id2entity[source_e], '*', r, '*', id2entity[t], '*', r_, '*', id2entity[t_], id2entity[target_e])


def average_entity_dbpedia():
    # train_data = "../../../data/redial/train_data.jsonl"
    # valid_data = "../../../data/redial/valid_data.jsonl"
    # test_data = "../../../data/redial/test_data.jsonl"
    # text_dict = pkl.load(open("../../../data/redial/text_dict.pkl", "rb"))
    train_data = "../../../data/crs/train_data.jsonl"
    valid_data = "../../../data/crs/valid_data.jsonl"
    test_data = "../../../data/crs/test_data.jsonl"
    text_dict = pkl.load(open("../../../data/crs/text_dict_tmdb4.pkl", "rb"))
    instances = []
    with open(train_data) as json_file:
        for line in json_file.readlines():
            instances.append(json.loads(line))
    with open(valid_data) as json_file:
        for line in json_file.readlines():
            instances.append(json.loads(line))
    with open(test_data) as json_file:
        for line in json_file.readlines():
            instances.append(json.loads(line))
    source_entity_counter = 0
    target_entity_counter = 0
    source_sentence_counter = 0
    target_sentence_counter = 0
    for instance in instances:
        initiator_id = instance["initiatorWorkerId"]
        respondent_id = instance["respondentWorkerId"]
        messages = instance["messages"]
        message_idx = 0
        while message_idx < len(messages):
            source_text = []
            target_text = []
            while (
                    message_idx < len(messages)
                    and messages[message_idx]["senderWorkerId"] == initiator_id
            ):
                source_text.append(messages[message_idx]["text"])
                message_idx += 1
            while (
                    message_idx < len(messages)
                    and messages[message_idx]["senderWorkerId"] == respondent_id
            ):
                target_text.append(messages[message_idx]["text"])
                message_idx += 1
            source_text = [text for text in source_text if text != ""]
            target_text = [text for text in target_text if text != ""]
            if source_text != [] or target_text != []:
                source_sentence_counter += 1
                target_sentence_counter += 1
                mentioned_entities = []
                for src in source_text:
                    mentioned_entities += text_dict[src]
                target_mentioned_entities = []
                for tgt in target_text:
                    target_mentioned_entities += text_dict[tgt]
                source_entity_counter += len(mentioned_entities)
                target_entity_counter += len(target_mentioned_entities)
    print(source_entity_counter, target_entity_counter, source_sentence_counter, target_sentence_counter)
    print(source_entity_counter / source_sentence_counter,  target_entity_counter / target_sentence_counter)


if __name__ == '__main__':
    # build_similarity_matrix_dbpedia()
    # build_text_dict()
    # analyze_kg_text()
    # analyze_kg_text_2()
    # get_path()
    # build_abstract_text_dict()
    average_entity_dbpedia()