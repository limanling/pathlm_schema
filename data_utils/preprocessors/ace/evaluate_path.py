import ujson as json
from collections import defaultdict
from path_discover import edge_format, edge_instance_format, edge_text_format
from visualize_paths import COLOR_ENTITY, COLOR_EVENT, draw_graph, generate_networkx_from_paths, add_node_event, add_node_entity, add_edge
import networkx as nx
import os
import copy
import numpy as np
import math


event_type_pair_format = '%s__%s'


def is_event_instance(event_id):
    suffix = event_id.split('-')[-1]
    if suffix.startswith('EV'):
        return True
    else:
        return False


def get_event_doc(event_id):
    doc_id = event_id[:event_id.rfind('-')]
    return doc_id


def edge_reversed(edge):
    edge_type = edge.split(' ')[1]
    if edge_type.endswith('-1'):
        return True
    else:
        return False


def load_edge_instances_from_docs(edge_info_json, unique=True):
    edge_info = json.load(open(edge_info_json))
    # edge_instances = defaultdict(lambda: defaultdict(set))

    edge_event_map = dict()  # edge -> event
    for event_type_pair in edge_info:
        for edge in edge_info[event_type_pair]:
            # APW_ENG_20030311.0775-EV2 Destination APW_ENG_20030311.0775-E6
            node_instance_1, edge_type, node_instance_2 = edge.split(' ')
            if is_event_instance(node_instance_1):
                edge_event_map[edge] = node_instance_1
            if is_event_instance(node_instance_2):
                edge_event_map[edge] = node_instance_2

    return edge_info, edge_event_map


def load_path_instances_link(path_link_json):
    # print(path_link_json)
    paths_link_info = json.load(open(path_link_json))

    path_link_dict = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))
    path_link_dict_inst = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))
    for event_type_pair in paths_link_info:
        for path_info in paths_link_info[event_type_pair]:
            start_evt, end_evt = event_type_pair.split('__')
            if not allow_same_start_end:
                if start_evt == end_evt:
                    continue
            # path_textid = path_info['id']
            path = path_info['path']
            # count = path_info['count']
            # cooccur = path_info['cooccur']
            instances = path_info['instance']
            # instances_text = path_info['instance_text']

            # use path content as key:
            path_textid = ' '.join(path)

            for instance in instances:
                start_inst = instance[0]
                end_inst = instance[-1]
                start_doc = start_inst[:start_inst.rfind('-')]
                end_doc = end_inst[:end_inst.rfind('-')]
                # print(start_doc, end_doc, instance)
                path_link_dict[path_textid][start_doc][end_doc] += 1
                path_link_dict_inst[path_textid][start_inst][end_inst] += 1

    return path_link_dict, path_link_dict_inst


def load_path_instances(path_info_json, allow_same_start_end=False,
                        frequency_weight=False, load_edge=False, unique=True,
                        append_weight=False, path_weights=None):
    '''
    load node_instances
    :param path_info_json:
    :return:
    '''
    paths_info = json.load(open(path_info_json))

    path_dict = defaultdict(lambda: defaultdict())
    edge_instances = defaultdict(lambda : defaultdict(set))
    if frequency_weight:
        # if append_weight:
        #     path_weights = path_weights_test
        # else:
        if path_weights is None:
            path_weights = defaultdict(lambda : defaultdict())

    for event_type_pair in paths_info:
        for path_info in paths_info[event_type_pair]:
            start_evt, end_evt = event_type_pair.split('__')
            if not allow_same_start_end:
                if start_evt == end_evt:
                    continue
            # path_textid = path_info['id']
            path = path_info['path']
            count = path_info['count']
            # cooccur = path_info['cooccur']
            instances = path_info['instance']
            instances_text = path_info['instance_text']
            # event_type_pair = event_type_pair_format % (start_evt, end_evt)

            # use path content as key:
            path_textid = ' '.join(path)

            if append_weight:
                if path_textid in path_weights[event_type_pair]:
                    # path_dict[path_textid]['count'] += count
                    if frequency_weight:
                        path_weights[event_type_pair][path_textid] += count
                continue

            path_dict[path_textid]['path'] = ' '.join(path)
            path_dict[path_textid]['start'] = start_evt
            path_dict[path_textid]['end'] = end_evt
            path_dict[path_textid]['event_type_pair'] = event_type_pair
            # path_dict[path_textid]['count'] = count
            path_dict[path_textid]['edges'] = list()
            path_dict[path_textid]['nodes'] = set()

            if frequency_weight:
                path_weights[event_type_pair][path_textid] = count
                # if len(path) != 7:
                #     path_weights[event_type_pair][path_textid] = 0

            for element_idx, element in enumerate(path):
                if element_idx < 2:
                    continue
                if element_idx % 2 == 1:
                    continue
                # if path[element_idx-1].endswith('-1'):
                #     continue
                edge = edge_format % (path[element_idx-2], path[element_idx-1], path[element_idx])
                path_dict[path_textid]['edges'].append(edge)
                path_dict[path_textid]['nodes'].add(path[element_idx - 2])
                path_dict[path_textid]['nodes'].add(path[element_idx])

                if load_edge:
                    for instance in instances:
                        edge_instance = edge_instance_format % (
                            instance[int(element_idx/2 - 1)],
                            instance[int(element_idx/2) - 1],
                            instance[int(element_idx/2)]
                        )
                        # print(edge_instance)
                        if unique:
                            edge_instances[event_type_pair][edge].add(edge_instance)
                        else:
                            for _ in len(edge_instances[event_type_pair][edge]):
                                edge_instances[event_type_pair][edge].add(edge_instance)

                    # for instance_text in instances_text:
                    #     edge_text = instance_text[element_idx-1]
                    #     # print(edge_text)edge_instance
                    #     # print(edge_text)
                    #     edge_instances[event_type_pair][edge].add(edge_text)

    if load_edge:
        if frequency_weight:
            return edge_instances, path_dict, path_weights
        else:
            return edge_instances, path_dict
    else:
        if frequency_weight:
            return path_dict, path_weights
        else:
            return path_dict


def get_edge_weight(path_dict, path_weights):
    edge_weights = defaultdict(lambda : defaultdict(float))
    for path_textid in path_dict:
        event_type_pair = event_type_pair_format % (path_dict[path_textid]['start'], path_dict[path_textid]['end'])
        if path_textid not in path_weights[event_type_pair]:
            print('aaaaaaaaaa?????', path_textid)
            continue
        for edge in path_dict[path_textid]['edges']:
            edge_weights[event_type_pair][edge] += path_weights[event_type_pair][path_textid]
    # print(edge_weights)
    return edge_weights


def load_path_weight(path_weight_json, path_info_tsv, path_dict):
    path_weights = defaultdict(lambda : defaultdict()) # eventtype -> pathid -> path_weight

    index2pathID = dict()
    pathID2path = dict()
    for line in open(path_info_tsv):
        line = line.rstrip('\n')
        tabs = line.split('\t')
        try:
            index2pathID[tabs[0]] = int(tabs[1])
            pathID2path[tabs[1]] = tabs[3]
        except:
            pass

    path_weight_data = json.load(open(path_weight_json))
    for path_idx in path_weight_data:
        path_textid = pathID2path[path_idx]
        if path_textid not in path_dict:
            continue
        event_type_pair = path_dict[path_textid]['event_type_pair']
        path_weights[event_type_pair][pathID2path[path_idx]] = path_weight_data[path_idx]

    # print(path_weights)

    return path_weights


def average_reverse_weight(path_weights):
    # path_weights_new = defaultdict(lambda: defaultdict())  # eventtype -> pathid -> path_weight
    for event_type_pair in path_weights:
        start_type, end_type = event_type_pair.split('__')
        event_type_pair_reversed = '%s__%s' % (end_type, start_type)
        for path_textid in path_weights[event_type_pair]:
            path_textid_reversed = reverse_path(path_textid)
            # print(event_type_pair, path_textid)
            # print(event_type_pair_reversed, path_textid_reversed)
            # print('-----')
            if event_type_pair_reversed in path_weights and \
                    path_textid_reversed in path_weights[event_type_pair_reversed]:
                weight_reversed = path_weights[event_type_pair_reversed][path_textid_reversed]
                path_weights[event_type_pair][path_textid] += weight_reversed
                # path_weights[event_type_pair][path_textid] /= 2.0
    return path_weights


def load_path_weight_clmnsp(path_weight_json, path_info_tsv, path_dict,
                            path_nsp_weight_json=None, path_info_weight_tsv=None,
                            lm_weight_coefficient=1):
    path_nsp_weights = defaultdict(lambda: defaultdict(lambda : defaultdict()))  # eventtype -> pathid1 -> pathid2 -> path_weight
    if path_nsp_weight_json:
        index2nsppathIDs = dict()
        for line in open(path_info_weight_tsv):
            line = line.rstrip('\n')
            tabs = line.split('\t')
            # index   pathID  event_group     sentence1       sentence2       gold_label
            try:
                example_id = tabs[0]
                path_textid_1 = tabs[3]
                path_textid_2 = tabs[4]
                index2nsppathIDs[example_id] = (path_textid_1, path_textid_2)
            except:
                pass

        path_nsp_weight_data = json.load(open(path_nsp_weight_json))
        for path_idx in path_nsp_weight_data:
            path_textid_1, path_textid_2 = index2nsppathIDs[path_idx]
            event_type_pair_1 = path_dict[path_textid_1]['event_type_pair']
            # event_type_pair_2 = path_dict[path_textid_2]['event_type_pair']
            path_nsp_weights[event_type_pair_1][path_textid_1][path_textid_2] = path_nsp_weight_data[path_idx][1] # classfier has 2 output

    # load path lm losses
    path_weights = defaultdict(lambda: defaultdict())  # eventtype -> pathid -> path_weight

    index2pathID = dict()
    pathID2path = dict()
    for line in open(path_info_tsv):
        line = line.rstrip('\n')
        tabs = line.split('\t')
        try:
            index2pathID[tabs[0]] = int(tabs[1])
            pathID2path[tabs[1]] = tabs[3]
        except:
            pass

    path_weight_data = json.load(open(path_weight_json))
    for path_idx in path_weight_data:
        path_textid = pathID2path[path_idx]
        if path_textid not in path_dict:
            continue
        event_type_pair = path_dict[path_textid]['event_type_pair']
        path_weights[event_type_pair][pathID2path[path_idx]] = path_weight_data[path_idx] * lm_weight_coefficient


    # aggregate nsp path scores
    if path_nsp_weight_json:
        for event_type_pair in path_weights:
            for path_textid_1 in path_weights[event_type_pair]:
                # max_nsp_score = 0
                sum_nsp_score = 0
                for path_textid_2 in path_weights[event_type_pair]:
                    if path_textid_1 != path_textid_2:
                        try:
                            nsp_score = path_nsp_weights[event_type_pair][path_textid_1][path_textid_2]
                            # if nsp_score > max_nsp_score:
                            #     max_nsp_score = nsp_score
                            sum_nsp_score += nsp_score
                        except:
                            pass
                avg_nsp_score = sum_nsp_score / len(path_weights[event_type_pair])
                # print(path_weights[event_type_pair][path_textid_1], avg_nsp_score) #max_nsp_score)
                path_weights[event_type_pair][path_textid_1] -= avg_nsp_score #max_nsp_score

    return path_weights


def load_path_weight_clmnsp_old(path_weight_json, path_info_tsv, path_dict):
    path_weights = defaultdict(lambda : defaultdict()) # eventtype -> pathid -> path_weight

    # index2pathID = dict()
    index2path = dict()
    for line in open(path_info_tsv):
        line = line.rstrip('\n')
        tabs = line.split('\t')
        try:
            example_id = tabs[0]
            # example_path = tabs[3]
            # index2pathID[example_id] = int(tabs[1])
            index2path[int('%s0' % example_id)] = tabs[3]  # add sufix, 0: text_a; 1: text_b, 2: text_a_neg, 3: text_b_neg
            index2path[int('%s1' % example_id)] = tabs[4]
        except:
            pass

    path_weight_data = json.load(open(path_weight_json))
    for path_idx in path_weight_data:
        if not path_idx.endswith('0') and not path_idx.endswith('1'):
            continue
        path_textid = index2path[int(path_idx)]
        event_type_pair = path_dict[path_textid]['event_type_pair']
        path_weights[event_type_pair][path_textid] = path_weight_data[path_idx]

    return path_weights


def load_path_weight_mlm(path_weight_json, path_info_tsv, path_dict):
    path_weights = defaultdict(lambda : defaultdict()) # eventtype -> pathid -> path_weight

    # index2pathID = dict()
    index2path = dict()
    for line in open(path_info_tsv):
        line = line.rstrip('\n')
        tabs = line.split('\t')
        try:
            example_id = tabs[0]
            # example_path = tabs[3]
            # index2pathID[example_id] = int(tabs[1])
            index2path[int('%s0' % example_id)] = tabs[3]  # add sufix, 0: text_a; 1: text_b, 2: text_a_neg, 3: text_b_neg
            index2path[int('%s1' % example_id)] = tabs[4]
        except:
            pass

    path_weight_data = json.load(open(path_weight_json))
    for path_idx in path_weight_data:
        if not path_idx.endswith('0') and not path_idx.endswith('1'):
            continue
        path_textid = index2path[int(path_idx)]
        event_type_pair = path_dict[path_textid]['event_type_pair']
        path_weights[event_type_pair][path_textid] = path_weight_data[path_idx][1]  # classfier has 2 output



    return path_weights


# def graph_topK_edge(edge_weights, topk=10, max_better=False):
#     edge_weights_sorted = sorted(edge_weights.items(), key=lambda x:x[1], reverse=max_better)
#     schema_edges = [edge for edge, edge_weight in edge_weights_sorted[:topk]]
#     return schema_edges


def graph_topk_path(path_weights, path_dict, topk=10, max_better=False, filter_small_graph=False):
    if filter_small_graph:
        if len(path_weights) <= topk:
            return None, None

    path_weights_sorted = sorted(path_weights.items(), key=lambda x: x[1], reverse=max_better)
    schema_edges = defaultdict(float)
    schema_paths_topk = defaultdict(float)
    for path_textid, path_weight in path_weights_sorted[:topk]:
        # schema_edges.extend(path_dict[path_textid]['edges'])
        schema_paths_topk[path_textid] = path_weight
        for edge in path_dict[path_textid]['edges']:
            schema_edges[edge] += path_weight
    return schema_edges, schema_paths_topk


def graph_toppercent_path(path_weights, path_dict, toppercent=0.1, max_better=False):
    path_num = len(path_weights)
    # print(path_weights)
    topk = round(path_num * toppercent)
    # print('topk', topk, path_num, toppercent)

    path_weights_sorted = sorted(path_weights.items(), key=lambda x: x[1], reverse=max_better)
    schema_edges = defaultdict(float)
    schema_paths_topk = defaultdict(float)
    for path_textid, path_weight in path_weights_sorted[:topk]:
        # schema_edges.extend(path_dict[path_textid]['edges'])
        schema_paths_topk[path_textid] = path_weight
        for edge in path_dict[path_textid]['edges']:
            schema_edges[edge] += path_weight
    return schema_edges, schema_paths_topk


# correct solution:
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # print(x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


def softmax_path(path_weights, max_better, do_softmax=True):
    if max_better:
        path_weights = {path_id: 100 - path_w for path_id, path_w in path_weights.items()}
    if do_softmax:
        # path_weight_sum = sum(list(path_weights.values()))
        # path_weights = {path_id: float(path_w)/path_weight_sum for path_id, path_w in path_weights.items()}
        path_weights_softmax = softmax(list(path_weights.values()))
    else:
        path_weights_softmax = list(path_weights.values())
    path_weights = {path_id: path_weights_softmax[idx] for idx, path_id in enumerate(path_weights.keys())}
    return path_weights

def graph_topprob_path(path_weights, path_dict, topprob=0.5, max_better=False, do_softmax=True):
    path_weights = softmax_path(path_weights, max_better, do_softmax)
    path_weights_sorted = sorted(path_weights.items(), key=lambda x: x[1], reverse=True)
    prob_added = 0
    schema_edges = defaultdict(float)
    schema_paths_topk = defaultdict(float)
    for path_textid, path_weight in path_weights_sorted:
        if prob_added < topprob:
            schema_paths_topk[path_textid] = path_weight
            for edge in path_dict[path_textid]['edges']:
                schema_edges[edge] += path_weight
            prob_added += path_weight
    return schema_edges, schema_paths_topk





def instance_coverage(edge_weights_topk, edge_instances):
    recalled = set()
    # all = set()

    for edge in edge_weights_topk:
        # print('-------------', edge)
        # recalled += len(edge_instances[edge])
        if edge not in edge_instances:
            continue
        recalled.update(list(edge_instances[edge].keys()))
        # for inst in edge_instances[edge]:
        #     print(edge_instances[edge][inst])
        # for instance_path in edge_instances[edge]:
        #     print(instance_path)
        # print(edge_instances[edge])
        # print(recalled)
    # for edge in edge_weights:
    #     # all += len(edge_instances[edge])
    #     all.update(list(edge_instances[edge].keys()))

    return recalled


def instance_coverage_2gram(edge_instances, edge_neighbors_schema_graph):
    # print(len(edge_neighbors), edge_neighbors)
    all_2grams_schema_graph = 0
    matched_2grams_schame_graph = 0

    # matched_2_grams = dict()
    matched_2_grams = list()
    for edge_1 in edge_neighbors_schema_graph:
        for edge_2 in edge_neighbors_schema_graph[edge_1]:
            # print('2gram query', edge_1, edge_2)
            all_2grams_schema_graph += 1
            # if '%s | %s' % (edge_1, edge_2) in matched_2_grams or '%s | %s' % (edge_2, edge_1) in matched_2_grams:
            #     continue
            edge_neighbors_2gram = defaultdict(set)
            edge_neighbors_2gram[edge_1].add(edge_2)
            mathed_result = graph_matching([edge_1, edge_2], edge_instances,
                                           edge_event_map,
                                           edge_neighbors=edge_neighbors_2gram)
            # print('2gram mathed_result', mathed_result)
            # matched_2_grams['%s | %s' % (edge_1, edge_2)] = len(mathed_result)
            matched_2_grams.extend(mathed_result)

            if len(mathed_result) > 0:
                matched_2grams_schame_graph += 1

    # sum(list(matched_2_grams.values()))  # using dict

    return matched_2_grams, matched_2grams_schame_graph, all_2grams_schema_graph


def all_instances_gram(gram_n, edge_neighbors_instance_graph, edge_event_map):
    matched_edges_all = list() #defaultdict(list) #

    for edge_1_inst in edge_neighbors_instance_graph:
        finished_events = set()
        if edge_1_inst in edge_event_map:
            finished_events.add(edge_event_map[edge_1_inst])
        # print('edge_neighbors_instance_graph', len(edge_neighbors_instance_graph))
        matched_edges = traverse_instance_graph(edge_1_inst, edge_neighbors_instance_graph,
                                                [edge_1_inst],
                                                finished_events,
                                                edge_event_map,
                                                gram_n)
        matched_edges_all.extend(matched_edges)
        # print('edge_neighbors_instance_graph-1', len(edge_neighbors_instance_graph))
    return matched_edges_all


def instance_coverage_3gram(edge_instances, edge_neighbors_schema_graph):
    all_3grams_schema_graph = 0
    matched_3grams_schame_graph = 0

    if len(edge_neighbors_schema_graph) == 1:
        return [], 0, 0
    else:
        # matched_3_grams = dict()
        matched_3_grams = set()
        for edge_1 in edge_neighbors_schema_graph.keys():
            for edge_2 in edge_neighbors_schema_graph[edge_1]:
                if edge_2 not in edge_neighbors_schema_graph:
                    continue
                for edge_3 in edge_neighbors_schema_graph[edge_2]:
                    if edge_3 == edge_1:
                        continue
                    # print('- 3gram query', edge_1, edge_2, edge_3)
                    all_3grams_schema_graph += 1
                    # if '%s | %s | %s' % (edge_1, edge_2, edge_3) in matched_3_grams or \
                    #         '%s | %s | %s' % (edge_1, edge_3, edge_2) in matched_3_grams or \
                    #         '%s | %s | %s' % (edge_2, edge_3, edge_1) in matched_3_grams or \
                    #         '%s | %s | %s' % (edge_2, edge_1, edge_3) in matched_3_grams or \
                    #         '%s | %s | %s' % (edge_3, edge_1, edge_2) in matched_3_grams or \
                    #         '%s | %s | %s' % (edge_3, edge_2, edge_1) in matched_3_grams:
                    #     continue
                    edge_neighbors_3gram = defaultdict(set)
                    edge_neighbors_3gram[edge_1].add(edge_2)
                    edge_neighbors_3gram[edge_2].add(edge_3)
                    matched_graphs = graph_matching([edge_1, edge_2, edge_3], edge_instances,
                                                    edge_event_map,
                                                    edge_neighbors=edge_neighbors_3gram)
                    # matched_3_grams['%s | %s | %s' % (edge_1, edge_2, edge_3)] = len(matched_graphs)
                    # matched_3_grams.extend(matched_graphs)
                    for matched_graph in matched_graphs:
                        # print(' '.join(matched_graph))
                        # assert len(matched_graph) == 3
                        matched_3_grams.add(','.join(matched_graph))

                    if len(matched_graphs) > 0:
                        matched_3grams_schame_graph += 1
                        # print('3gram matched_3_grams', len(matched_3_grams), len(matched_graphs), matched_graphs)
    # print('matched_3grams_schame_graph', matched_3grams_schame_graph)
    # print('matched_3_grams_num', len(matched_3_grams))

    # sum(list(matched_3_grams.values()))  # using dict

    return matched_3_grams, matched_3grams_schame_graph, all_3grams_schema_graph


def traverse_instance_graph(edge_1_inst, edge_neighbors, matched_edges, finished_events,
                            edge_event_map, max_len):
    all_matched_edges = list()
    if edge_1_inst in edge_neighbors:
        for edge_2_inst in edge_neighbors[edge_1_inst]:
            matched_edges_ = copy.deepcopy(matched_edges)
            finished_events_ = copy.deepcopy(finished_events)
            # print('edge_1__edge2', max_len, len(matched_edges_), edge_1_inst, edge_2_inst)
            if edge_2_inst in matched_edges_:
                continue

            matched_edges_.append(edge_2_inst)
            if edge_2_inst in edge_event_map:
                finished_events_.add(edge_event_map[edge_2_inst])
            # print('matched_edges_', len(matched_edges_))

            if len(finished_events_) > 2:
                # if graph has more than 2 events, this is not a schema that we want
                pass
            elif len(matched_edges_) == max_len:
                all_matched_edges.append(matched_edges_)
            else:
                all_matched_edges.extend(traverse_instance_graph(edge_2_inst, edge_neighbors,
                                                                 matched_edges_,
                                                                 finished_events_,
                                                                 edge_event_map, max_len))

    return all_matched_edges


def traverse_graph(edge_1, edge_1_inst, edge_1_reversed, edge_neighbors, edge_instances, finsihed_edges,
                   finsihed_events, edge_event_map, matched_edges, max_len):
    all_matched_edges = list()
    for edge_2 in edge_neighbors[edge_1]:
        # print('edge_1__edge2', edge_1, edge_2)
        if edge_2 in finsihed_edges: # type level, not instance lvel
            continue
        if edge_2 not in edge_instances:
            continue
        edge_2_reversed = edge_reversed(edge_2)
        edge_2_instances = edge_instances[edge_2]
        for edge_2_inst in edge_2_instances:
            matched_edges_ = copy.deepcopy(matched_edges)
            finsihed_edges_ = copy.deepcopy(finsihed_edges)
            finsihed_events_ = copy.deepcopy(finsihed_events)
            # print('    ', edge_1_inst, edge_2_inst)  #, edge_instances[edge_1][edge_1_inst], edge_instances[edge_2][edge_2_inst])
            if edge_1_reversed:
                edge1_end = edge_1_inst.split(' ')[0]
            else:
                edge1_end = edge_1_inst.split(' ')[-1]
            if edge_2_reversed:
                edge2_start = edge_2_inst.split(' ')[-1]
            else:
                edge2_start = edge_2_inst.split(' ')[0]
            if edge1_end == edge2_start:
                matched_edges_.append(edge_2_inst)
                finsihed_edges_.add(edge_2)
                if edge_2_inst in edge_event_map:
                    finsihed_events_.add(edge_event_map[edge_2_inst])

                # print(len(matched_edges_))
                # print('matched_edges_part', len(finsihed_edges_), max_len, matched_edges_)

                if len(finsihed_events_) > 2:
                    # if graph has more than 2 events, this is not a schema that we want
                    pass
                elif len(finsihed_edges_) == max_len:
                    all_matched_edges.append(matched_edges_)
                    # print('all_matched_edges-maxlen-append', all_matched_edges)
                else:
                    all_matched_edges.extend(traverse_graph(edge_2, edge_2_inst, edge_2_reversed,
                                                            edge_neighbors, edge_instances,
                                                            finsihed_edges_,
                                                            finsihed_events_,
                                                            edge_event_map,
                                                            matched_edges_, max_len))
                    # print('all_matched_edges-extend', all_matched_edges)

    # print('all_matched_edges-final', all_matched_edges)

    return all_matched_edges


def path_textid2edges(path_textid):
    edges = list()

    path = path_textid.split(' ')
    for element_idx, element in enumerate(path):
        if element_idx < 2:
            continue
        if element_idx % 2 == 1:
            continue
        # if path[element_idx-1].endswith('-1'):
        #     continue
        edge = edge_format % (path[element_idx - 2], path[element_idx - 1], path[element_idx])
        edges.append(edge)

    edge_neighbors = defaultdict(set)
    for idx, edge in enumerate(edges):
        if idx == 0:
            continue
        edge_neighbors[edges[idx-1]].add(edge)

    return edges, edge_neighbors


def schema_path_matching(schema_paths_topk):
    matched_edges_schemapaths = defaultdict(list)
    for path_textid in schema_paths_topk:
        edges_path, edge_neighbors_path = path_textid2edges(path_textid)
        matched_edges_path = graph_matching(edges_path, edge_instances,
                       edge_event_map,
                       edge_neighbors=edge_neighbors_path)
        matched_edges_schemapaths[path_textid].extend(matched_edges_path)
    return matched_edges_schemapaths


def graph_matching(schema_edges, edge_instances, edge_event_map, edge_neighbors=None):

    if edge_neighbors is None:
        edge_neighbors = get_schema_graph_neighbor_dict(schema_edges)

    # print('schema_edges', schema_edges)
    if len(edge_neighbors) == 0:
        return []

    # get the instance-level neighbors
    matched_edges_all = set()
    for idx in range(len(edge_neighbors)):
        edge_1 = list(edge_neighbors.keys())[idx]
        # print('edge_1', edge_1)
        if edge_1 in edge_instances:
            edge_1_instances = edge_instances[edge_1]
            edge_1_reversed = edge_reversed(edge_1)
            break
        idx += 1
    if edge_1 not in edge_instances:
        return []

    for edge_1_inst in edge_1_instances:
        finsihed_edges = set()
        finsihed_edges.add(edge_1)
        finsihed_events = set()
        if edge_1_inst in edge_event_map:
            finsihed_events.add(edge_event_map[edge_1_inst])
        # print('edge_1_inst', edge_1_inst)  #, edge_1_instances[edge_1_inst])
        matched_edges_list = traverse_graph(edge_1, edge_1_inst, edge_1_reversed,
                                            edge_neighbors, edge_instances,
                                            finsihed_edges, finsihed_events,
                                            edge_event_map,
                                            [edge_1_inst], len(schema_edges))
        # print('matched_edges_list', matched_edges_list)

        # matched_edges_all.extend(matched_edges_list)
        for matched_edges_each in matched_edges_list:
            # assert len(matched_edges) == len(schema_edges)
            matched_edges_all.add(','.join(matched_edges_each))

    # # print(matched_edges_all)
    # matched_schema_graph_count = len(matched_edges_all)
    #
    # # count all subgraphs
    #
    # instance_graph_all = matched_schema_graph_count

    return matched_edges_all #matched_schema_graph_count, instance_graph_all


def get_schema_graph_neighbor_dict(schema_edges):
    edge_neighbors = defaultdict(set)
    for edge_1 in schema_edges:
        for edge_2 in schema_edges:
            if edge_1 != edge_2:
                if edge_1.split(' ')[-1] == edge_2.split(' ')[0]:
                    edge_neighbors[edge_1].add(edge_2)
    return edge_neighbors


def get_instance_graph_neighbor_dict(edge_instances, bidirection=True):
    edge_neighbors = defaultdict(set)
    for edge_1 in edge_instances:
        for edge_instance_1 in edge_instances[edge_1]:
            for edge_2 in edge_instances:
                if edge_1 != edge_2:
                    for edge_instance_2 in edge_instances[edge_2]:
                        if edge_instance_1 != edge_instance_2:
                            if edge_instance_1.split(' ')[-1] == edge_instance_2.split(' ')[0]:
                                edge_neighbors[edge_instance_1].add(edge_instance_2)
                            if bidirection:
                                if edge_instance_1.split(' ')[0] == edge_instance_2.split(' ')[0]:
                                    edge_neighbors[edge_instance_1].add(edge_instance_2)
                                elif edge_instance_1.split(' ')[-1] == edge_instance_2.split(' ')[-1]:
                                    edge_neighbors[edge_instance_1].add(edge_instance_2)
    return edge_neighbors


def generate_networkx_from_edges(edges, directed=True, need_split_path=False, weighted=True, unify_direction=True,
                                 distinguish_start_end=True, weight_coefficient=1):
    if directed:
        G = nx.DiGraph(overlap='false', splines='true')
    else:
        G = nx.Graph(overlap='false', splines='true')
    for edge in edges:
        # if need_split_path:
        node_1, edge_type, node_2 = edge.split(' ')
        # else:
        #     node_1, edge_type, node_2 = edge

        if ':' in node_1:
            # add events
            add_node_event(G, node_1)
        else:
            # add entities
            add_node_entity(G, node_1)

        if ':' in node_2:
            # add events
            add_node_event(G, node_2)
        else:
            # add entities
            add_node_entity(G, node_2)

        if weighted:
            edge_weight = edges[edge]
            if unify_direction:
                edge_reverse = edge_format % (node_2, edge_type.replace('-1', ''), node_1)
                # print(edge_reverse)
                if edge_reverse in edges:
                    edge_weight += edges[edge_reverse]
            edge_weight = edge_weight * weight_coefficient
            edge_type = '%s_%.1f' % (edge_type, edge_weight)
        if unify_direction:
            if '-1' in edge_type:
                edge_type = edge_type.replace('-1', '')
                add_edge(G, node_2, node_1, edge_type)
            else:
                add_edge(G, node_1, node_2, edge_type)
        else:
            add_edge(G, node_1, node_2, edge_type)

    return G


def reverse_path(path_textid):
    # print(path_textid.split(' '))
    path_elements = path_textid.split(' ')
    path_elements.reverse()
    for element_idx, element in enumerate(path_elements):
        if element_idx == 0 or element_idx == (len(path_elements) - 1):
            continue
        if element_idx % 2 == 1:
            # edge
            if '-1' in element:
                # remove -1
                path_elements[element_idx] = element.replace('-1', '')
            else:
                # add -1
                path_elements[element_idx] = element + '-1'
    return ' '.join(path_elements)


def write_paths_schema(schema_paths_topk, writer_paths, save_1hop=True):
    # writer_paths.write('\n'.join(list(schema_paths_topk.keys())).replace('-1', ''))
    # writer_paths.write('\n')
    path_saved = set()
    for path in schema_paths_topk:
        if not save_1hop:
            if len(path.split(' ')) == 5:
                continue
        path_reverse = reverse_path(path)
        # print('reverse', path, path_reverse)
        if path_reverse not in path_saved:
            writer_paths.write('%s\n' % path.replace('-1', ''))


def write_paths_all(path_weights, output_file_paths_all, save_1hop=True):
    with open(output_file_paths_all, 'w') as writer:
        path_weights_global = dict()
        for event_type_pair in path_weights:
            for path_textid in path_weights[event_type_pair]:
                # print(path_textid)
                if not save_1hop:
                    if len(path_textid.split(' ')) == 5:
                        continue
                path_textid_reverse = reverse_path(path_textid)
                if path_textid_reverse in path_weights_global:
                    continue
                path_weights_global[path_textid] = path_weights[event_type_pair][path_textid]
        for path_textid in path_weights_global:
            writer.write('%s\t%.9f\n' % (path_textid.replace('-1', ''), path_weights_global[path_textid]))


def get_lm_prob_2gram(path_weights_train):
    prob_2gram = defaultdict(lambda: defaultdict(float))  # word | previous_word
    for event_type_pair in path_weights_train:
        for path_textid in path_weights_train[event_type_pair]:
            path_content = path_textid.split(' ')
            for element_idx, element in enumerate(path_content):
                if element_idx == 0:
                    element_previous = 'SOS'
                else:
                    element_previous = path_content[element_idx - 1]
                prob_2gram[element_previous][element] += path_weights_train[event_type_pair][path_textid]

    # Let's transform the counts to probabilities
    for w1 in prob_2gram:
        total_count = float(sum(prob_2gram[w1].values()))
        for w2 in prob_2gram[w1]:
            prob_2gram[w1][w2] /= total_count

    return prob_2gram


def get_lm_prob_3gram(path_weights_train):
    prob_3gram = defaultdict(lambda: defaultdict(lambda : defaultdict(float)))  # word | previous_word
    for event_type_pair in path_weights_train:
        for path_textid in path_weights_train[event_type_pair]:
            path_content = path_textid.split(' ')
            for element_idx, element in enumerate(path_content):
                if element_idx == 0:
                    element_previous = 'SOS'
                    element_previous_previous = 'SOS'
                elif element_idx == 1:
                    element_previous = path_content[0]
                    element_previous_previous = 'SOS'
                else:
                    element_previous = path_content[element_idx - 1]
                    element_previous_previous = path_content[element_idx - 2]
                prob_3gram[element_previous_previous][element_previous][element] += path_weights_train[event_type_pair][path_textid]

    # Let's transform the counts to probabilities
    for w1 in prob_3gram:
        for w2 in prob_3gram[w1]:
            total_count = float(sum(prob_3gram[w1][w2].values()))
            for w3 in prob_3gram[w1][w2]:
                prob_3gram[w1][w2][w3] /= total_count

    return prob_3gram


def generate_lm_2gram(path_weights, path_weights_train, save_path_2gram=None,
                      path_logprob=False):
    prob_2gram = get_lm_prob_2gram(path_weights_train)

    path_weights_2gram = defaultdict(lambda: defaultdict(float))
    for event_type_pair in path_weights:
        for path_textid in path_weights[event_type_pair]:
            if path_logprob:
                path_weight = 0
            else:
                path_weight = 1
            path_content = path_textid.split(' ')
            for element_idx, element in enumerate(path_content):
                if element_idx == 0:
                    element_previous = 'SOS'
                else:
                    element_previous = path_content[element_idx - 1]

                if path_logprob:
                    if prob_2gram[element_previous][element] > 0:
                        path_weight = path_weight + math.log(prob_2gram[element_previous][element])
                    else:
                        path_weight = path_weight + math.log(1e-45)
                else:
                    path_weight = path_weight * prob_2gram[element_previous][element]
            path_weights_2gram[event_type_pair][path_textid] = path_weight

    if save_path_2gram:
        json.dump(path_weights_2gram, open(save_path_2gram, 'w'), indent=2)

    return path_weights_2gram


def generate_lm_3gram(path_weights, path_weights_train, save_path_3gram=None,
                      path_logprob=False):
    prob_3gram = get_lm_prob_3gram(path_weights_train)

    path_weights_3gram = defaultdict(lambda: defaultdict(float))
    for event_type_pair in path_weights:
        for path_textid in path_weights[event_type_pair]:
            if path_logprob:
                path_weight = 0
            else:
                path_weight = 1
            path_content = path_textid.split(' ')
            for element_idx, element in enumerate(path_content):
                if element_idx == 0:
                    element_previous = 'SOS'
                    element_previous_previous = 'SOS'
                elif element_idx == 1:
                    element_previous = path_content[0]
                    element_previous_previous = 'SOS'
                else:
                    element_previous = path_content[element_idx - 1]
                    element_previous_previous = path_content[element_idx - 2]
                if path_logprob:
                    if prob_3gram[element_previous_previous][element_previous][element] > 0:
                        path_weight = path_weight + math.log(prob_3gram[element_previous_previous][element_previous][element])
                    else:
                        path_weight = path_weight + math.log(1e-45)
                else:
                    path_weight = path_weight * prob_3gram[element_previous_previous][element_previous][element]
            path_weights_3gram[event_type_pair][path_textid] = path_weight

    if save_path_3gram:
        json.dump(path_weights_3gram, open(save_path_3gram, 'w'), indent=2)

    return path_weights_3gram


def perplexity(path_weight):
    weight_sum = 0
    path_num = 0
    for event_type_pair in path_weight:
        weight_sum += sum(list(path_weight[event_type_pair].values()))
        path_num += len(path_weight[event_type_pair])
    weight_avg = weight_sum / path_num
    print('weight_avg', weight_avg)
    perplexity = math.exp(-weight_avg)
    return perplexity


def get_lm_prob_1gram(path_weights_train):
    prob_1gram = defaultdict(float)
    for event_type_pair in path_weights_train:
        for path_textid in path_weights_train[event_type_pair]:
            for element in path_textid.split(' '):
                prob_1gram[element] += path_weights_train[event_type_pair][path_textid]

    # Let's transform the counts to probabilities
    total_count = float(sum(prob_1gram.values()))
    for w1 in prob_1gram:
        prob_1gram[w1] /= total_count

    return prob_1gram


def generate_lm_1gram(path_weights, path_weights_train, save_path_1gram=None,
                      path_logprob=False):
    prob_1gram = get_lm_prob_1gram(path_weights_train)

    path_weights_1gram = defaultdict(lambda : defaultdict(float))
    for event_type_pair in path_weights:
        for path_textid in path_weights[event_type_pair]:
            if path_logprob:
                path_weight = 0
            else:
                path_weight = 1
            for element in path_textid.split(' '):
                if path_logprob:
                    if prob_1gram[element] > 0:
                        path_weight = path_weight + math.log(prob_1gram[element])
                    else:
                        path_weight = path_weight + math.log(1e-45)
                else:
                    path_weight = path_weight * prob_1gram[element]
            # path_weight = path_weight * 100
            path_weights_1gram[event_type_pair][path_textid] = path_weight

    if save_path_1gram:
        json.dump(path_weights_1gram, open(save_path_1gram, 'w'), indent=2)

    return path_weights_1gram


def load_lm_ngram(save_path_ngram):
    data = json.load(open(save_path_ngram))
    return data


def coherence_doc(matched_schema_graphs, edge_event_map, subgraph_doc):
    for matched_schema_graph in matched_schema_graphs:
        # print('matched size', matched_schema_graph)
        docs = set()
        events = set()
        # find event and docid
        edge_instances = matched_schema_graph.split(',')
        for edge_instance in edge_instances:
            if edge_instance in edge_event_map:
                event = edge_event_map[edge_instance]
                events.add(event)

                doc_id = get_event_doc(event)
                docs.add(doc_id)

        # print('docs', docs)
        # print('events', events)
        assert len(events) <= 2
        docs = list(docs)

        # save event
        if len(docs) == 1:
            subgraph_doc[docs[0]][docs[0]] += 1
        elif len(docs) == 2:
            subgraph_doc[docs[0]][docs[1]] += 1  # for each doc, matched[doc1][doc2] = #num_subgraph
            # subgraph_doc[docs[1]][docs[0]] += 1

    return subgraph_doc


def coherence_doc_path_saved(schema_paths_topk, subgraph_doc, path_link_dict,
                             path_weights, max_better, do_softmax=True):
    path_weights = softmax_path(path_weights, max_better, do_softmax)

    for path_textid in schema_paths_topk:
        if path_textid not in path_link_dict:
            continue
        for doc_1 in path_link_dict[path_textid]:
            for doc_2 in path_link_dict[path_textid][doc_1]:
                subgraph_doc[doc_1][doc_2] += path_link_dict[path_textid][doc_1][doc_2] \
                                              * path_weights[path_textid]

    return subgraph_doc, path_weights


def coherence_doc_graph_saved(schema_paths_topk, subgraph_doc, path_link_dict_inst,
                             path_weights, max_better, do_softmax=True):
    # path_weights = softmax_path(path_weights, max_better, do_softmax)
    matched_subgraph = defaultdict(lambda: defaultdict(list))
    for path_textid in schema_paths_topk:
        if path_textid not in path_link_dict:
            continue
        for inst_1 in path_link_dict_inst[path_textid]:
            for inst_2 in path_link_dict_inst[path_textid][inst_1]:
                matched_subgraph[inst_1][inst_2].append(path_textid)

    graph_size = len(schema_paths_topk)
    for inst_1 in matched_subgraph:
        for inst_2 in matched_subgraph[inst_1]:
            matched_path_num = len(set(matched_subgraph[inst_1][inst_2]))
            if matched_path_num == graph_size:
                doc_1 = get_event_doc(inst_1)
                doc_2 = get_event_doc(inst_2)
                subgraph_doc[doc_1][doc_2] += 1

    return subgraph_doc


def coherence_doc_path(matched_schema_paths, edge_event_map, subgraph_doc, path_weights,
                       max_better, do_softmax=True):
    path_weights = softmax_path(path_weights, max_better, do_softmax)
    for path_textid in matched_schema_paths:
        for schema_path in matched_schema_paths[path_textid]:
            # print('matched path', schema_path)
            docs = set()
            events = set()
            # find event and docid
            edge_instances = schema_path.split(',')
            for edge_instance in edge_instances:
                if edge_instance in edge_event_map:
                    event = edge_event_map[edge_instance]
                    events.add(event)

                    doc_id = get_event_doc(event)
                    docs.add(doc_id)

            # print('docs', docs)
            # print('events', events)
            assert len(events) <= 2
            docs = list(docs)

            # save event
            if len(docs) == 1:
                subgraph_doc[docs[0]][docs[0]] += path_weights[path_textid]
            elif len(docs) == 2:
                subgraph_doc[docs[0]][docs[1]] += path_weights[path_textid]  # for each doc, matched[doc1][doc2] = #num_subgraph
                # subgraph_doc[docs[1]][docs[0]] += 1

    return subgraph_doc, path_weights


def coherence_normalized_cut(subgraph_doc):
    if len(subgraph_doc) == 0:
        return 0

    # print('subgraph_doc', subgraph_doc)
    coherence_all = 0
    doc_across_all = 0
    doc_within_all = 0
    for doc_1 in subgraph_doc:
        doc_within = 0
        doc_across = 0
        for doc_2 in subgraph_doc[doc_1]:
            if doc_1 == doc_2:
                doc_within += subgraph_doc[doc_1][doc_2]
            else:
                doc_across += subgraph_doc[doc_1][doc_2]

        # coherence = (doc_across + 1) / float(doc_within + 1)
        coherence = float(doc_within + 1) / float(doc_across + 1)
        coherence_all += coherence
        doc_across_all += doc_across
        doc_within_all += doc_within

    coherence_average = coherence_all #/ len(subgraph_doc)

    return coherence_average, doc_across_all, doc_within_all


# def coherence_path(path_dict, path_weights):
#     # for every event, find the paths in doc and out-of-doc
#     for path_textid in path_dict:


# def save_instance_type_mapping(matched_n_grams_all):
#     # ['APW_ENG_20030322.0119-EV17 Agent Turkey', 'APW_ENG_20030311.0775-E20 PART-WHOLE Turkey', 'APW_ENG_20030322.0119-EV16 Agent Turkey']
#     for matched_n_gram in matched_n_grams_all:
#         print(matched_n_gram)
#     return None, None


def draw_graph_dynamic_html(event_type_pair_save, output_html_file, topk):
    head = open('head').read()
    tail = open('tail').read()
    with open(output_html_file, 'w') as f:
        f.write(head)
        f.write('<h3>')
        f.write(event_type_pair_save)
        f.write('</h3> <br>')
        # f.write('<p>frequency-baseline      vs       PathLM</p>')
        f.write('<p> Graph1      vs       Graph2 </p>')
        f.write('<iframe src="../test_graph_freq_top'+str(topk)+'_nolink_same/' + event_type_pair_save +'.pdf" class="figure-img rounded" style="height:700px; width: 49%; max-width: 640px; padding: 1px; background-color: #fff;">')
        f.write('</iframe>')
        f.write(
            '<iframe src="../test_graph_xlent_top'+str(topk)+'_nolink_same/' + event_type_pair_save +'.pdf" class="figure-img rounded" style="height:700px; width: 49%; max-width: 640px; padding: 1px; background-color: #fff;">')
        f.write('</iframe>')
        f.write('<br>')
        f.write('<p> Graph3        vs       Graph4</p>')
        f.write(
            '<iframe src="../test_graph_1gram_top'+str(topk)+'_nolink_same/' + event_type_pair_save +'.pdf" ' +
            'class="figure-img rounded" style="height:700px; width: 49%; ' +
            'max-width: 640px; padding: 1px; background-color: #fff;">')
        f.write('</iframe>')
        f.write(
            '<iframe src="../test_graph_2gram_top'+str(topk)+'_nolink_same/' + event_type_pair_save +'.pdf" class="figure-img rounded" style="height:700px; width: 49%; max-width: 640px; padding: 1px; background-color: #fff;">')
        f.write('</iframe>')
        f.write(tail)


if __name__=='__main__':

    threshold = False  # probability
    # topk = 3
    if threshold:
        top_prob = 0.1
    else:
        top_percent = 0.2
    # method = 'freq'
    # method = 'freq_train'
    # method = 'freq_train_only'
    # method = 'xlnet'
    # method = 'mlm'
    method = '1gram'
    # method = '2gram'
    # method = '3gram'
    # method = 'xlnet_nsp'
    # method = 'xlnet_nspcotrain'
    link = False
    evaluate_1gram = False
    evaluate_2gram = False
    evaluate_3gram = False
    evaluate_path = True
    evaluate_graph = False
    reverse_average = True

    remove_doc = False
    # schema_set = 'train'
    schema_set = 'test'
    # test_set = 'dev'
    # test_set = 'train'
    test_set = 'test'
    # append_train_freq = True
    filter_small_graph = False
    unique = True
    save_1hop = True
    visualization = False
    ace_dir = 'ace.system'
    allow_same_start_end = True
    append_train_freq = False
    path_perplexity = None
    path_logprob = True

    xlent_score_path = 'xlnet_large_ep20_lr1e-4_bs16/test_loss_step_1222.json'
    # xlent_score_path = 'xlnet_large_system_ep20_lr1e-4_bs16/test_loss_step_1504.json'
    # xlent_score_path = 'xlnet_large_system_ep5_lr1e-4_bs16/test_loss_step_940.json'
    # xlent_score_path = 'xlnet_large_system_ep100_lr1e-3_bs16/%s_loss_step_16920.json' % schema_set
    # xlent_score_path = 'xlnet_large_ep100_lr1e-3_bs16/%s_loss_step_48880.json' % schema_set
    # 'xlnet_ep20_lr1e-3_bs8/eval_loss_step_19536.json'
    # xlent_score_path = 'xlnet_ep20_lr1e-3_bs8/test_loss_step_19536.json'
    # xlent_score_path = 'xlnet_ep20_lr1e-3_bs8/test_loss_step_14652.json'
    # xlent_score_path = 'xlnet_ep20_lr1e-3_bs8/train_loss_step_12210.json'
    # xlent_score_path = 'xlnet_ep20_lr1e-3_bs8/test_loss_step_17094.json'
    # 'xlnet_ep50_lr1e-5_bs8/eval_loss_step_61050.json'
    # xlent_score_path = 'xlnet_ep20_lr1e-3_bs8/test_loss_step_24420.json'
    # xlent_score_path = 'xlnet_ep20_lr1e-3_bs8/eval_loss_step_21978.json'
    # xlent_score_path = 'xlnet_ep20_lr1e-3_bs8/eval_loss_step_2442.json'
    # xlent_score_path = 'xlnet_ep20_lr1e-3_bs8/eval_loss_step_9768.json'
    # '0324_negbalance_littleparam_little_ep20_lr1e-5_bs8/eval_loss_step_11960.json'

    # xlent_score_path_lm = 'xlnetnsp_new_w1_w50_ep10_lr1e-3_bs16/test_loss_lm_step_27126.json'
    # xlent_score_path_nsp = 'xlnetnsp_new_w1_w50_ep10_lr1e-3_bs16/test_scores_nsp_step_27126.json'
    # xlent_score_path_lm = 'xlnetnsp_new_w50_w1_ep10_lr1e-3_bs16/test_loss_lm_step_40689.json'
    # xlent_score_path_nsp = 'xlnetnsp_new_w50_w1_ep10_lr1e-3_bs16/test_scores_nsp_step_40689.json'
    # xlnet_score_path_lm = 'xlnetnsp_new_w50_w1_ep50_lr1e-3_bs16/test_loss_lm_step_67815.json'
    # xlent_score_path_nsp = 'xlnetnsp_new_w50_w1_ep50_lr1e-3_bs16/test_scores_nsp_step_67815.json'
    # xlnet_score_path_lm = 'xlnetnsp_new_w50_w1_ep50_lr1e-3_bs16/test_loss_lm_step_90420.json'
    # xlent_score_path_nsp = 'xlnetnsp_new_w50_w1_ep50_lr1e-3_bs16/test_scores_nsp_step_90420.json'
    # xlnet_score_path_lm = 'xlnetnsp_large_w50_w1_ep50_lr1e-3_bs16/%s_loss_lm_step_67815.json' % schema_set
    # xlent_score_path_nsp = 'xlnetnsp_large_w50_w1_ep50_lr1e-3_bs16/%s_scores_nsp_step_67815.json' % schema_set
    # xlnet_score_path_lm = 'xlnetnsp_large_system_w50_w1_ep5_lr1e-4_bs16/%s_loss_lm_step_1417.json' % schema_set
    # xlent_score_path_nsp = 'xlnetnsp_large_system_w50_w1_ep5_lr1e-4_bs16/%s_scores_nsp_step_1417.json' % schema_set
    # xlnet_score_path_lm = 'xlnetnsp_large_w50_w1_ep5_lr1e-4_bs16/%s_loss_lm_step_2260.json' % schema_set
    # xlent_score_path_nsp = 'xlnetnsp_large_w50_w1_ep5_lr1e-4_bs16/%s_scores_nsp_step_2260.json' % schema_set
    xlnet_score_path_lm = 'xlnetnsp_large_w50_w1_ep2_lr1e-4_bs16/%s_loss_lm_step_3616.json' % schema_set
    xlent_score_path_nsp = 'xlnetnsp_large_w50_w1_ep2_lr1e-4_bs16/%s_scores_nsp_step_3616.json' % schema_set

    output_dir = '../../../vis/%s/%s_graph_%s_%s_%s_%s' % (ace_dir, test_set, method,
                                                           'topprob%.2f' % top_prob if threshold else 'toppercent%.2f' % top_percent,
                                                            'link' if link else 'nolink',
                                                            'same' if allow_same_start_end else 'nosame')
    output_file_edges = os.path.join(output_dir, '0graph_%s_%s.txt' % (method,
                                                                             'topprob%.2f' % top_prob if threshold else 'toppercent%.2f' % top_percent))
    output_file_paths = os.path.join(output_dir, '0paths_each_type_%s_%s.txt' % (method,
                                                                                      'topprob%.2f' % top_prob if threshold else 'toppercent%.2f' % top_percent))
    output_file_paths_all = os.path.join(output_dir, '0paths_all_%s_%s.txt' % (method,
                                                                                    'topprob%.2f' % top_prob if threshold else 'toppercent%.2f' % top_percent))

    # path_info_json = '../../../data/%s/%s.paths.direct.json' % (ace_dir, schema_set)
    path_info_json = '../../../data/%s/%s.paths%s%s.json' % (
        ace_dir,
        schema_set,
        '.direct',
        # '.link' if link else '',
        '.removed' if remove_doc else '',
    )
    # path_info_train_json = '../../../data/%s/train.paths.direct.json' % ace_dir
    path_info_train_json = '../../../data/%s/train.paths%s%s.json' % (
        ace_dir,
        '.direct',
        # '.link' if link else '',
        '.removed' if remove_doc else '',
    )
    edge_info_json = '../../../data/%s/%s.edges%s%s%s.json' % (
        ace_dir,
        test_set,
        '.direct',
        '.link' if link else '',
        '.removed' if remove_doc else '',
    )
    path_link_json = '../../../data/%s/%s.paths%s%s%s.json' % (
        ace_dir,
        test_set,
        '.direct',
        '.link',
        '.removed' if remove_doc else '',
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    writer_edges = open(output_file_edges, 'w')
    writer_paths = open(output_file_paths, 'w')
    edge_instances, edge_event_map = load_edge_instances_from_docs(edge_info_json, unique=unique)

    if method == 'mlm':
        # mlm, using pathclassifier
        frequency_weight = False
        max_better = True
        do_softmax = True
        path_info_tsv = '../../../data/%s/spnsp/%s.tsv' % (ace_dir, schema_set)
        path_weight_json = '../../../path_ft/%s/' \
                            'spnsp_1_50_50_ep100_lr1e-3_bs8/eval_scores_sp_step_271260.json' \
                           % ace_dir
                           # 'spnsp_nolossweight_ep50_lr1e-2_bs8/eval_scores_sp_step_step_45210.bin.json'
        path_dict = load_path_instances(path_info_json, frequency_weight=frequency_weight,
                                        load_edge=False, unique=unique, allow_same_start_end=allow_same_start_end)
        path_weights = load_path_weight_mlm(path_weight_json, path_info_tsv, path_dict)

    elif method == 'xlnet':
        # xlent, using loss to rank
        frequency_weight = False
        max_better = False
        do_softmax = True
        weight_coefficient = 1
        path_info_tsv = '../../../data/%s/lm/%s.tsv' % (ace_dir, schema_set)
        path_weight_json = '../../../path_ft/%s/%s' % (ace_dir, xlent_score_path)
        path_dict = load_path_instances(path_info_json, frequency_weight=frequency_weight,
                                        load_edge=False, unique=unique, allow_same_start_end=allow_same_start_end)
        path_weights = load_path_weight(path_weight_json, path_info_tsv, path_dict)

    elif method == 'xlnet_nsp':
        frequency_weight = False
        max_better = False
        do_softmax = True
        weight_coefficient = 0.1
        path_info_tsv = '../../../data/%s/nsp/%s_lm.tsv' % (ace_dir, schema_set)
        path_nsp_info_tsv = '../../../data/%s/nsp/%s_nsp.tsv' % (ace_dir, schema_set)
        path_weight_json = '../../../path_ft/%s/%s' % (ace_dir, xlnet_score_path_lm)
        path_nsp_weight_json = '../../../path_ft/%s/%s' % (ace_dir, xlent_score_path_nsp)
        path_dict = load_path_instances(path_info_json, frequency_weight=frequency_weight,
                                        load_edge=False, unique=unique, allow_same_start_end=allow_same_start_end)
        path_weights = load_path_weight_clmnsp(path_weight_json, path_info_tsv, path_dict,
                            path_nsp_weight_json=path_nsp_weight_json, path_info_weight_tsv=path_nsp_info_tsv,
                            lm_weight_coefficient=weight_coefficient)

    elif method == 'xlnet_nspcotrain':
        frequency_weight = False
        max_better = False
        do_softmax = False
        weight_coefficient = 10
        path_info_tsv = '../../../data/%s/nsp/%s_lm.tsv' % (ace_dir, schema_set)
        path_weight_json = '../../../path_ft/%s/%s' % (ace_dir, xlnet_score_path_lm)
        path_dict = load_path_instances(path_info_json, frequency_weight=frequency_weight,
                                        load_edge=False, unique=unique, allow_same_start_end=allow_same_start_end)
        path_weights = load_path_weight_clmnsp(path_weight_json, path_info_tsv, path_dict,
                                               path_nsp_weight_json=None,
                                               path_info_weight_tsv=None)


    elif method.startswith('freq'):
        # if method == 'freq':
        #     append_train_freq = False
        #     train_freq = False
        # elif method == 'freq_train':
        #     append_train_freq = True
        #     train_freq = False
        # elif method == 'freq_train_only':
        #     append_train_freq = False
        #     train_freq = True
        frequency_weight = True
        max_better = True
        do_softmax = False
        weight_coefficient = 1
        path_weight_json = 'freq'
        path_dict, path_weights = load_path_instances(path_info_json, frequency_weight=frequency_weight,
                                                        load_edge=False, unique=unique,
                                                        allow_same_start_end=allow_same_start_end)
        # if append_train_freq:
        if method == 'freq_train':
            _, path_weights = load_path_instances(path_info_train_json, allow_same_start_end=allow_same_start_end,
                            frequency_weight=True, load_edge=False, unique=unique,
                            append_weight=True, path_weights=path_weights)
        elif method == 'freq_train_only':
            _, path_weights = load_path_instances(path_info_train_json, allow_same_start_end=allow_same_start_end,
                                                  frequency_weight=True, load_edge=False, unique=unique,
                                                  append_weight=True, path_weights=None)

    elif method == '1gram':
        frequency_weight = True
        max_better = True
        do_softmax = False
        weight_coefficient = 1
        path_weight_json = '../../../data/%s/lm_1gram_path_weights.json' % ace_dir
        _, path_weights_train = load_path_instances(path_info_train_json, frequency_weight=frequency_weight,
                                                      load_edge=False, unique=unique,
                                                      allow_same_start_end=allow_same_start_end)
        path_dict, path_weights_raw = load_path_instances(path_info_json,
                                                      frequency_weight=frequency_weight,
                                                      load_edge=False, unique=unique,
                                                      allow_same_start_end=allow_same_start_end)
        path_weights = generate_lm_1gram(path_weights_raw, path_weights_train, save_path_1gram=path_weight_json,
                                         path_logprob=path_logprob)

        # path_dict = load_path_instances(path_info_json,
        #                                   frequency_weight=False,
        #                                   load_edge=False, unique=unique,
        #                                   allow_same_start_end=allow_same_start_end)
        # path_weights = load_lm_ngram(path_weight_json)
        path_perplexity = perplexity(path_weights)

    elif method == '2gram':
        frequency_weight = True
        max_better = True
        do_softmax = False
        weight_coefficient = 1
        path_weight_json = '../../../data/%s/lm_2gram_path_weights.json' % ace_dir
        path_dict_train, path_weights_train = load_path_instances(path_info_train_json, frequency_weight=frequency_weight,
                                                      load_edge=False, unique=unique,
                                                      allow_same_start_end=allow_same_start_end)
        path_dict, path_weights = load_path_instances(path_info_json,
                                                      frequency_weight=frequency_weight,
                                                      load_edge=False, unique=unique,
                                                      allow_same_start_end=allow_same_start_end)
        path_weights = generate_lm_2gram(path_weights, path_weights_train, save_path_2gram=path_weight_json,
                                         path_logprob=path_logprob)

        # path_dict = load_path_instances(path_info_json,
        #                                   frequency_weight=False,
        #                                   load_edge=False, unique=unique,
        #                                   allow_same_start_end=allow_same_start_end)
        # path_weights = load_lm_ngram(path_weight_json)
        path_perplexity = perplexity(path_weights)

    elif method == '3gram':
        frequency_weight = True
        max_better = True
        do_softmax = False
        weight_coefficient = 1
        path_weight_json = '../../../data/%s/lm_3gram_path_weights.json' % ace_dir
        path_dict_train, path_weights_train = load_path_instances(path_info_train_json,
                                                                  frequency_weight=frequency_weight,
                                                                  load_edge=False, unique=unique,
                                                                  allow_same_start_end=allow_same_start_end)
        path_dict, path_weights = load_path_instances(path_info_json,
                                                      frequency_weight=frequency_weight,
                                                      load_edge=False, unique=unique,
                                                      allow_same_start_end=allow_same_start_end)
        path_weights = generate_lm_3gram(path_weights, path_weights_train, save_path_3gram=path_weight_json,
                                         path_logprob=path_logprob)
        path_perplexity = perplexity(path_weights)


    if reverse_average:
        path_weights = average_reverse_weight(path_weights)



    coverage_eachtype = set()
    allinstance_eachtype = set()
    matched_schema_graph_count_eachtype = set()
    instance_graph_all_eachtype = set()
    matched_2_grams_eachtype = set()
    matched_3_grams_eachtype = set()
    matched_2_grams_all_eachtype = set()
    matched_3_grams_all_eachtype = set()
    num_coverage_eachtype = list()
    num_allinstance_eachtype = list()
    num_fullinstance_eachtype = list()
    num_matched_schema_graph_count_eachtype = list()
    num_instance_graph_all_eachtype = list()
    num_matched_2_grams_eachtype = list()
    num_matched_3_grams_eachtype = list()
    num_matched_2_grams_all_eachtype = list()
    num_matched_3_grams_all_eachtype = list()
    num_matched_2_grams_full_eachtype = list()
    num_matched_3_grams_full_eachtype = list()
    num_matched_2_grams_schema_eachtype = list()
    num_matched_3_grams_schema_eachtype = list()
    num_matched_2_grams_all_schema_eachtype = list()
    num_matched_3_grams_all_schema_eachtype = list()
    subgraph_doc_all = defaultdict(lambda: defaultdict(int))

    if evaluate_1gram or evaluate_2gram or evaluate_3gram:
        edge_weights = get_edge_weight(path_dict, path_weights)
        all_instance_edge_combine = sum(len(edge_instances[edge]) for edge in edge_instances)
    if evaluate_2gram or evaluate_3gram:
        edge_neighbors_instance_graph = get_instance_graph_neighbor_dict(edge_instances, bidirection=True)
    if evaluate_2gram:
        matched_2_grams_all = all_instances_gram(2, edge_neighbors_instance_graph, edge_event_map)
        matched_2_grams_all_eachtype.update([' '.join(seq) for seq in matched_2_grams_all])
        # instance2type_2gram, type2instance_2gram = save_instance_type_mapping(matched_2_grams_all)
    if evaluate_3gram:
        matched_3_grams_all = all_instances_gram(3, edge_neighbors_instance_graph, edge_event_map)
        matched_3_grams_all_eachtype.update([' '.join(seq) for seq in matched_3_grams_all])
        # instance2type_3gram, type2instance_3gram = save_instance_type_mapping(matched_3_grams_all)
    if evaluate_path or evaluate_graph:
        path_link_dict, path_link_dict_inst = load_path_instances_link(path_link_json)
    # print(path_link_dict)

    for event_type_pair in path_weights:
        print('%s\t%d' % (event_type_pair, len(path_weights[event_type_pair])))
        continue
        # if event_type_pair != 'Movement:Transport__Movement:Transport':
        #     continue
        # print('==================', event_type_pair)
        # print(event_type_pair)
        # if load_edge_from_path:
        #     edge_instances_typepair = edge_instances[event_type_pair]
        # else:
        edge_instances_typepair = edge_instances

        if not threshold:
            # schema_edges, schema_paths_topk = graph_topk_path(path_weights[event_type_pair], path_dict,
            #                                                   topk=topk, max_better=max_better,
            #                                                   filter_small_graph=filter_small_graph)
            schema_edges, schema_paths_topk = graph_toppercent_path(path_weights[event_type_pair], path_dict,
                                                                    toppercent=top_percent,
                                                                    max_better=max_better)

        else:
            schema_edges, schema_paths_topk = graph_topprob_path(path_weights[event_type_pair], path_dict,
                                                                 topprob=top_prob, max_better=max_better)
        if schema_edges is None:
            continue
        # print('schema_edges', schema_edges)


        # visualization
        if visualization:
            G = generate_networkx_from_edges(schema_edges, directed=True, need_split_path=True, weighted=True,
                                             distinguish_start_end=False, weight_coefficient=weight_coefficient)
            event_type_pair_name = event_type_pair.replace(':','')
            draw_graph(G, os.path.join(output_dir, '%s.pdf' % event_type_pair_name))
            print('<li><a href="./html/%s.html" target="_blank">graph schema between %s </a></li>'
                  % (event_type_pair_name, event_type_pair_name))
            # print('http://blender.cs.illinois.edu/software/schema/top%d/html/%s.html' % (topk, event_type_pair_name))
            os.makedirs(os.path.join(output_dir, 'html'), exist_ok=True)
            output_html_file = os.path.join(output_dir, 'html', '%s.html' % event_type_pair_name)
            draw_graph_dynamic_html(event_type_pair_name, output_html_file, 'topprob%.2f' % top_prob if threshold else 'toppercent%.2f' % top_percent)


        # instance coverage
        if evaluate_1gram:
            matched_edge_list = instance_coverage(schema_edges, edge_instances_typepair)
            coverage_eachtype.update(matched_edge_list)
            full_list = instance_coverage(edge_weights[event_type_pair], edge_instances_typepair)
            num_coverage_eachtype.append(len(matched_edge_list))
            num_fullinstance_eachtype.append(len(full_list))

        if evaluate_graph:
            # # subgraph matching
            # # matched_schema_graphs = graph_matching(schema_edges, edge_instances_typepair, edge_event_map)
            # # matched_schema_graph_count = len(matched_schema_graphs)
            # # num_matched_schema_graph_count_eachtype.append(matched_schema_graph_count)
            # matched_edges_schemapaths_dict = schema_path_matching(schema_paths_topk)
            # # matched_schema_graph_count = len(matched_edges_schemapaths_dict.values())
            # matched_schema_graph_count = 0
            # # for path_textid in matched_edges_schemapaths_dict:
            # #     matched_schema_graph_count += path_weights[event_type_pair][path_textid] * len(matched_edges_schemapaths_dict[path_textid])
            # # num_matched_schema_graph_count_eachtype.append(matched_schema_graph_count)
            #
            # # coherence
            # # subgraph_doc_all = coherence_doc(matched_edges_schemapaths, edge_event_map, subgraph_doc_all)
            # subgraph_doc_all = coherence_doc_path(matched_edges_schemapaths_dict, edge_event_map, subgraph_doc_all,
            #                                       path_weights[event_type_pair], max_better, do_softmax)
            subgraph_doc_all = coherence_doc_graph_saved(schema_paths_topk, subgraph_doc_all, path_link_dict_inst,
                             path_weights[event_type_pair], max_better, do_softmax=do_softmax)

        if evaluate_path:
            subgraph_doc_all, path_weights_softmax = coherence_doc_path_saved(schema_paths_topk, subgraph_doc_all, path_link_dict,
                             path_weights[event_type_pair], max_better, do_softmax=do_softmax)
            # path_weights[event_type_pair] = path_weights_softmax

        if evaluate_2gram:
            # 2gram
            edge_neighbors_schema_graph = get_schema_graph_neighbor_dict(schema_edges)
            edge_neighbors_schema_graph_full = get_schema_graph_neighbor_dict(edge_weights[event_type_pair])
            matched_2_grams, matched_2grams_schame, all_2grams_schema = instance_coverage_2gram(edge_instances_typepair,
                                                                                                edge_neighbors_schema_graph)
            matched_2_grams_eachtype.update([' '.join(seq) for seq in matched_2_grams])
            num_matched_2_grams_eachtype.append(len(matched_2_grams))
            num_matched_2_grams_schema_eachtype.append(matched_2grams_schame)
            num_matched_2_grams_all_schema_eachtype.append(all_2grams_schema)
            matched_2_grams_full, matched_2grams_schame_full, all_2grams_schema_full = instance_coverage_2gram(edge_instances_typepair,
                                                                                                               edge_neighbors_schema_graph_full)
            num_matched_2_grams_full_eachtype.append(len(matched_2_grams_full))
            # instance all
            num_matched_2_grams_all_eachtype.append(len(matched_2_grams_all))

        # 3gram
        if evaluate_3gram:
            matched_3_grams, matched_3grams_schame, all_3grams_schema = instance_coverage_3gram(edge_instances_typepair,
                                                                                                edge_neighbors_schema_graph)

            matched_3_grams_eachtype.update([' '.join(seq) for seq in matched_3_grams])
            num_matched_3_grams_eachtype.append(len(matched_3_grams))
            num_matched_3_grams_schema_eachtype.append(matched_3grams_schame)
            num_matched_3_grams_all_schema_eachtype.append(all_3grams_schema)
            matched_3_grams_full, matched_3grams_schame_full, all_3grams_schema_full = instance_coverage_3gram(
                edge_instances_typepair,
                edge_neighbors_schema_graph_full)
            num_matched_3_grams_full_eachtype.append(len(matched_3_grams_full))
            # print('schema\'s 2gram/3gram:', all_2grams_schema, all_3grams_schema)
            # instance all
            num_matched_3_grams_all_eachtype.append(len(matched_3_grams_all))

        # write_schema_edges
        writer_edges.write('%s\n' % ','.join(list(schema_edges.keys())).replace('-1', ''))
        write_paths_schema(schema_paths_topk, writer_paths, save_1hop=save_1hop)

    write_paths_all(path_weights, output_file_paths_all, save_1hop=save_1hop)

    writer_edges.flush()
    writer_edges.close()
    writer_paths.flush()
    writer_paths.close()

    print(path_info_train_json)
    print(path_weight_json if path_weight_json else 'freq')
    print('topk', 'topprob%.2f' % top_prob if threshold else 'toppercent%.2f' % top_percent)
    print('method', method)
    print('ace_dir', ace_dir)
    print('training included in weighting', append_train_freq)
    print('filter_small_graph', filter_small_graph)
    print('allow same?', 'with_same' if allow_same_start_end else 'no_same')
    print('link?', 'link' if link else 'nolink')
    print('reverse_average?', reverse_average)
    print('remove_doc?', remove_doc)
    print('path_perplexity', path_perplexity)

    # edge instance
    if evaluate_1gram:
        intersection_instance_edge = sum(num_coverage_eachtype)
        all_instance_edge = all_instance_edge_combine * len(num_coverage_eachtype)
        recall_edge = intersection_instance_edge / all_instance_edge
        f1_edge = 2 * recall_edge * 1 / (recall_edge + 1)
        intersection_instance_edge_combine = len(coverage_eachtype)
        recall_edge_combine = intersection_instance_edge_combine / all_instance_edge_combine
        f1_edge_combine = 2 * recall_edge_combine * 1 / (recall_edge_combine + 1)
        full_instance_edge = sum(num_fullinstance_eachtype)
        recall_edge_full = intersection_instance_edge / full_instance_edge
        f1_edge_full = 2 * recall_edge_full * 1 / (recall_edge_full + 1)
        print('edge',  intersection_instance_edge)
        print('edge_recall', recall_edge, recall_edge_combine, recall_edge_full)
        print('edge_f1', f1_edge, f1_edge_combine, f1_edge_full)

    # 2grams
    if evaluate_2gram:
        # print(num_matched_2_grams_eachtype)
        # print(num_matched_2_grams_all_eachtype)
        # print(num_matched_2_grams_schema_eachtype)
        # print(num_matched_2_grams_all_schema_eachtype)
        intersection_instance_2gram = sum(num_matched_2_grams_eachtype)
        all_instance_2gram = sum(num_matched_2_grams_all_eachtype)
        intersection_schema_2gram = sum(num_matched_2_grams_schema_eachtype)
        all_schema_2gram = sum(num_matched_2_grams_all_schema_eachtype)
        recall_2gram = intersection_instance_2gram / all_instance_2gram
        precision_2gram = intersection_schema_2gram / all_schema_2gram
        f1_2gram = 2 * recall_2gram * precision_2gram / (recall_2gram + precision_2gram)
        intersection_instance_2gram_combine = len(matched_2_grams_eachtype)
        all_instance_2gram_combine = len(matched_2_grams_all_eachtype)
        recall_2gram_combine = intersection_instance_2gram_combine / all_instance_2gram_combine
        f1_2gram_combine = 2 * recall_2gram_combine * precision_2gram / (recall_2gram_combine + precision_2gram)
        all_instance_2gram_full = sum(num_matched_2_grams_full_eachtype)
        recall_2gram_full = intersection_instance_2gram / all_instance_2gram_full
        f1_2gram_full = 2 * recall_2gram_full * precision_2gram / (recall_2gram_full + precision_2gram)
        print('2-grams', intersection_instance_2gram)
        print('2-grams-precision', precision_2gram)
        print('2-grams-recall', recall_2gram, recall_2gram_combine, recall_2gram_full)
        print('2-grams-f1', f1_2gram, f1_2gram_combine, f1_2gram_full)

    # 3grams
    if evaluate_3gram:
        # print(num_matched_3_grams_eachtype)
        # print(num_matched_3_grams_all_eachtype)
        # print(num_matched_3_grams_schema_eachtype)
        # print(num_matched_3_grams_all_schema_eachtype)
        intersection_instance_3gram = sum(num_matched_3_grams_eachtype)
        all_instance_3gram = sum(num_matched_3_grams_all_eachtype)
        intersection_schema_3gram = sum(num_matched_3_grams_schema_eachtype)
        all_schema_3gram = sum(num_matched_3_grams_all_schema_eachtype)
        recall_3gram = intersection_instance_3gram / all_instance_3gram
        precision_3gram = intersection_schema_3gram / all_schema_3gram
        f1_3gram = 2 * recall_3gram * precision_3gram / (recall_3gram + precision_3gram)
        intersection_instance_3gram_combine = len(matched_3_grams_eachtype)
        all_instance_3gram_combine = len(matched_3_grams_all_eachtype)
        recall_3gram_combine = intersection_instance_3gram_combine / all_instance_3gram_combine
        f1_3gram_combine = 2 * recall_3gram_combine * precision_3gram / (recall_3gram_combine + precision_3gram)
        all_instance_3gram_full = sum(num_matched_3_grams_full_eachtype)
        recall_3gram_full = intersection_instance_3gram / all_instance_3gram_full
        f1_3gram_full = 2 * recall_3gram_full * precision_3gram / (recall_3gram_full + precision_3gram)
        print('3-grams', intersection_instance_3gram)
        print('3-grams-precision', precision_3gram)
        print('3-grams-recall', recall_3gram, recall_3gram_combine, recall_3gram_full)
        print('3-grams-f1', f1_3gram, f1_3gram_combine, f1_3gram_full)

    # subgraph
    # intersection_instance_subgraph = sum(num_matched_schema_graph_count_eachtype)
    # print('subgraph', intersection_instance_subgraph)
    coherence_graphs, doc_across_all, doc_within_all = coherence_normalized_cut(subgraph_doc_all)
    print('coherence_graphs', coherence_graphs)
    print('doc_across_all', doc_across_all)
    print('doc_within_all', doc_within_all)
    print('doc_across_all / doc_within_all', doc_across_all / doc_within_all)
    print('doc_within_all / doc_across_all', doc_within_all / doc_across_all)
    print('doc_within_all / doc_across_all+doc_within_all', doc_within_all / (doc_within_all + doc_across_all))
    json.dump(subgraph_doc_all, open(os.path.join(output_dir, 'subgraph_doc_all.json'), 'w'), indent=2)


