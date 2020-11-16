import json
import itertools
import random
import copy

from collections import defaultdict, Counter

mention_type_values = {'NAM': 0, 'NOM': 1, 'PRO': 2, 'UNK': 3}
edge_format = '%s %s %s' # h_type, r_type, t_type
edge_instance_format = '%s %s %s' # h_id, r_type, t_id
edge_text_format = '%s | %s | %s | %s' # h_text, t_text, sent_id, sent_text

def undirect_path(path):
    '''
    forward and backward are the same
    :param path:
    :return:
    '''
    nodes, types, texts = path
    types_fwd = ','.join(types)
    types_bwd = ','.join(reversed(types))
    if types_fwd < types_bwd:
        return path
    else:
        return (
            list(reversed(nodes)),
            list(reversed(types)),
            list(reversed(texts))
        )


def load_entity_linking(entitylinking_tab):
    entity_ace2wiki = dict()
    for line in open(entitylinking_tab):
        tabs = line.split('\t')
        if len(tabs) > 1:
            entity_id_ace = tabs[1]
            entity_id_wiki = tabs[4]
            mention_type = tabs[6]
            if mention_type == 'PRO' or mention_type == 'NOM':
                entity_ace2wiki[entity_id_ace] = entity_id_ace
                continue
            if entity_id_wiki.startswith('NIL'):
                entity_ace2wiki[entity_id_ace] = entity_id_ace
                continue
            # if 'CNN_IP_20030405.1600.00-3-E40' in entity_id_ace:
            #     print(line)
            entity_id_ace = entity_id_ace[:entity_id_ace.rfind('-')]
            entity_ace2wiki[entity_id_ace] = entity_id_wiki

    # print(entity_ace2wiki)
    return entity_ace2wiki


def load_entity_relation_event(insts, edge_info,
                               entity_props, event_props, entity_text, event_text,
                               entity_next_entities, entity_next_events, event_next_entities,
                               linking=False, entity_ace2wiki=None,
                               use_entity_sub_type=False,
                               use_relation_sub_type=False,
                               direct=True,
                                entity_num = 0,
                                relation_num = 0,
                                event_num = 0,
                                role_num = 0,
                               ):
    # Collect entities, relations, events
    for inst in insts:
        inst_id = '%s: %s' % (inst['sent_id'], inst['sentence'])  # inst['sent_id']
        inst_entities = inst['entity_mentions']
        inst_relations = inst['relation_mentions']
        inst_events = inst['event_mentions']

        entity_num += len(inst_entities)
        relation_num += len(inst_relations)
        event_num += len(inst_events)


        for entity in inst_entities:
            entity_mention_id = entity['id']
            entity_id = entity_mention_id[:entity_mention_id.rfind('-')]
            if linking:
                if entity_id in entity_ace2wiki:
                    entity_id = entity_ace2wiki[entity_id]

            # Map coreferential entity mentions to one entity
            # mention_entity_mapping[entity_mention_id] = entity_id
            # Add entity text
            entity_text[entity_id].append((entity['text'],
                                           entity['mention_type']))
            # Add entity properties
            if entity_id not in entity_props:
                entity_props[entity_id] = (
                    entity['entity_type'],
                    entity['entity_subtype'],
                )
        for event in inst_events:
            event_mention_id = event['id']
            event_id = event_mention_id[:event_mention_id.rfind('-')]
            # Map coreferential event mentions to one event
            # mention_event_mapping[event_mention_id] = event_id
            # Add event text
            event_text[event_id].append((event['trigger']['text']))
            # Add event property
            if event_id not in event_props:
                event_props[event_id] = event['event_type']
            event_type = event_props[event_id]
            # Add next entities
            role_num += len(event['arguments'])
            for arg in event['arguments']:
                entity_id = arg['entity_id']
                entity_id = entity_id[:entity_id.rfind('-')]
                if linking:
                    if entity_id in entity_ace2wiki:
                        entity_id = entity_ace2wiki[entity_id]

                # save edge info
                entity_type, entity_sub_type = entity_props[entity_id]
                if use_entity_sub_type:
                    entity_type = '{}:{}'.format(entity_type, entity_sub_type)
                edge_info[edge_format % (event_type, arg['role'], entity_type)][
                    edge_instance_format % (event_id, arg['role'], entity_id)].append(
                    edge_text_format % (event['trigger']['text'], arg['text'], inst['sent_id'], inst['sentence']))
                # inverse to <t, r, h>, but the instance remains  the original order of <h,r,t> !!
                edge_info[edge_format % (entity_type, '%s-1' % arg['role'], event_type)][
                    edge_instance_format % (event_id, '%s' % arg['role'], entity_id)].append(
                    edge_text_format % (event['trigger']['text'], arg['text'], inst['sent_id'], inst['sentence']))

                if not any([entity_id == eid
                            for eid, _, _ in event_next_entities[event_id]]):
                    event_next_entities[event_id].append((entity_id, arg['role'], inst_id))
                if not any([event_id == eid
                            for eid, _, _ in entity_next_events[entity_id]]):
                    if direct:
                        entity_next_events[entity_id].append((event_id, '%s-1' % arg['role'], inst_id))
                    else:
                        entity_next_events[entity_id].append((event_id, arg['role'], inst_id))

        for relation in inst_relations:
            relation_type = relation['relation_type']
            relation_subtype = relation['relation_subtype']
            arg1 = relation['arguments'][0]
            arg2 = relation['arguments'][1]
            arg1_entity_id = arg1['entity_id']
            arg1_entity_id = arg1_entity_id[:arg1_entity_id.rfind('-')]
            if linking:
                if arg1_entity_id in entity_ace2wiki:
                    arg1_entity_id = entity_ace2wiki[arg1_entity_id]
            arg2_entity_id = arg2['entity_id']
            arg2_entity_id = arg2_entity_id[:arg2_entity_id.rfind('-')]
            if linking:
                if arg2_entity_id in entity_ace2wiki:
                    arg2_entity_id = entity_ace2wiki[arg2_entity_id]

            # save edge info
            arg1_entity_type, arg1_entity_sub_type = entity_props[arg1_entity_id]
            arg2_entity_type, arg2_entity_sub_type = entity_props[arg2_entity_id]
            if use_entity_sub_type:
                arg1_entity_type = '{}:{}'.format(arg1_entity_type, arg1_entity_sub_type)
                arg2_entity_type = '{}:{}'.format(arg2_entity_type, arg2_entity_sub_type)
            if use_relation_sub_type:
                relation_type_edge_save = '{}:{}'.format(relation_type, relation_subtype)
                relation_type_edge_save_reverse = '{}-1:{}-1'.format(relation_type, relation_subtype)
            else:
                relation_type_edge_save = relation_type
                relation_type_edge_save_reverse = '%s-1' % relation_type
            edge_info[edge_format % (arg1_entity_type, relation_type_edge_save, arg2_entity_type)][
                edge_instance_format % (arg1_entity_id,
                                        relation_type_edge_save,
                                        arg2_entity_id)].append(
                edge_text_format % (arg1['text'], arg2['text'], inst['sent_id'], inst['sentence']))
            # inverse to <t, r, h>, but the instance remains  the original order of <h,r,t>
            edge_info[edge_format % (arg2_entity_type, relation_type_edge_save_reverse, arg1_entity_type)][
                edge_instance_format % (arg1_entity_id,
                                        relation_type_edge_save,
                                        arg2_entity_id)].append(
                edge_text_format % (arg1['text'], arg2['text'], inst['sent_id'], inst['sentence']))

            # Add next entities
            if not any([arg2_entity_id == eid
                        for eid, _, _, _ in entity_next_entities[arg1_entity_id]]):
                entity_next_entities[arg1_entity_id].append((arg2_entity_id,
                                                             relation_type,
                                                             relation_subtype,
                                                             inst_id))
            if not any([arg1_entity_id == eid
                        for eid, _, _, _ in entity_next_entities[arg2_entity_id]]):
                if direct:
                    entity_next_entities[arg2_entity_id].append((arg1_entity_id,
                                                                 '%s-1' % relation_type,
                                                                 '%s-1' % relation_subtype,
                                                                 inst_id))
                else:
                    entity_next_entities[arg2_entity_id].append((arg1_entity_id,
                                                                 relation_type,
                                                                 relation_subtype,
                                                                 inst_id))

    return entity_props, event_props, entity_text, event_text, \
           entity_next_entities, entity_next_events, event_next_entities, \
            entity_num, relation_num, event_num, role_num


def entity_text_canonical(entity_text, event_text):
    # Select text for each entity/event
    entity_text_ = {}
    for entity_id, text_list in entity_text.items():
        c = Counter()
        c.update([text for text, mention_type in text_list])
        text_list.sort(key=lambda x: (
            # NAM > NOM > PRO
            mention_type_values[x[1]],
            # prefer the most frequent one
            -c[x[0]],
            # prefer the longest one
            -len(x[0])
        ))
        entity_text_[entity_id] = text_list[0][0]
    entity_text = entity_text_

    event_text_ = {}
    for event_id, text_list in event_text.items():
        c = Counter()
        c.update(text_list)
        text_list.sort(key=lambda x: (
            # prefer the most frequent one
            -c[x],
            # prefer the shortest one
            len(x[0])
        ))
        event_text_[event_id] = text_list[0]
    event_text = event_text_

    return entity_text, event_text


def find_paths(entity_props, event_props, entity_text, event_text,
               entity_next_entities, entity_next_events, event_next_entities,
               schema_id_mapping, path_id_mapping, max_path_len,
               endpoint_path_count, path_instance_mapping, path_instance_text_mapping,
               path_cooccur_freq_count, path_freq_count,
               assign_node_id=False,
               save_instance=True
               ):
    # Find event to event paths
    paths = []
    for start_event_id, next_entities in event_next_entities.items():
        start_event_type = event_props[start_event_id]
        cur_paths = []

        # Initial paths
        for entity_id, role, sent_id in next_entities:
            # print(sent_id)
            entity_type, entity_sub_type = entity_props[entity_id]
            if use_entity_sub_type:
                entity_type = '{}:{}'.format(entity_type, entity_sub_type)
            cur_paths.append((
                # node ids
                [start_event_id, entity_id],
                # node types
                [start_event_type, role, entity_type],
                # node texts
                [event_text[start_event_id], sent_id, entity_text[entity_id]]
            ))

        while cur_paths:
            cur_paths_ = []
            for path in cur_paths:
                nodes, types, texts = path
                if len(nodes) < max_path_len - 1:
                    # Add next entity
                    for entity_id, rel_type, rel_sub_type, sent_id in entity_next_entities[nodes[-1]]:
                        if entity_id not in nodes:
                            if use_relation_sub_type:
                                rel_type = '{}:{}'.format(rel_type, rel_sub_type)
                            entity_type, entity_sub_type = entity_props[entity_id]
                            if use_entity_sub_type:
                                entity_type = '{}:{}'.format(entity_type, entity_sub_type)
                            cur_paths_.append((
                                nodes + [entity_id],
                                types + [rel_type, entity_type],
                                texts + [sent_id, entity_text[entity_id]]
                            ))
                # Add next event
                for event_id, role, sent_id in entity_next_events[nodes[-1]]:
                    if event_id > start_event_id:
                        paths.append((
                            nodes + [event_id],
                            types + [role, event_props[event_id]],
                            texts + [sent_id, event_text[event_id]]
                        ))
            cur_paths = cur_paths_

    # Group paths of same same start and end
    endpoint_paths = defaultdict(lambda: defaultdict(list))  # defaultdict(list)
    for path in paths:
        nodes, types_, texts_ = path
        endpoints = (nodes[0], nodes[-1])
        endpoints_type = '%s__%s' % (types_[0], types_[-1])
        # if types_[0] == 'Business:End-Org' and types_[-1] == 'Personnel:End-Position':
        # if types_[0] == 'Justice:Convict' and types_[-1] == 'Justice:Trial-Hearing':
        # if endpoints_type == 'Conflict:Attack__Movement:Transport':
        endpoint_paths[endpoints_type][endpoints].append(path)  # endpoint_paths[endpoints].append(path)

    # assign schema_id to entities, entity id will start from 0 for each endpoints_type
    # endpoint_paths_schema = defaultdict(list)
    for endpoints_type in endpoint_paths:
        for endpoints, path_list in endpoint_paths[endpoints_type].items():
            # path_id_set = set()
            # unique_path_list = []
            for path in path_list:
                if not direct:
                    path = undirect_path(path)
                nodes, types, texts = path

                # assign schema_id to entities, the entity_id must be generated inside the same schema event_types
                # path_node_edge = list()
                # path_node_edge.append(nodes[0]) #append(types[0])  #for event: add type instead of instance
                for node_idx, node in enumerate(nodes[1:-1]):
                    path_element_id = 2 * (node_idx + 1)

                    if assign_node_id:
                        # node key = the shortest path to event
                        node_key_left = types[:path_element_id + 1]
                        print(node_key_left)
                        node_key_left = ','.join(node_key_left)
                        node_key_left_len = len(node_key_left)
                        node_key_right = types[path_element_id:]
                        print(node_key_right)
                        node_key_right = ','.join(node_key_right)
                        node_key_right_len = len(node_key_right)
                        if node_key_left in schema_id_mapping[endpoints_type]:
                            schema_id_nodekey, schema_key_len_nodekey = schema_id_mapping[endpoints_type][node_key_left]
                            if node_key_right in schema_id_mapping[endpoints_type]:
                                if node_key_left_len > node_key_right_len:
                                    schema_id_nodekey, schema_key_len_nodekey = schema_id_mapping[endpoints_type][
                                        node_key_right]
                            if node in schema_id_mapping[endpoints_type]:
                                # both node and nodekey already has schema id
                                schema_id_node, schema_key_len_node = schema_id_mapping[endpoints_type][node]
                                if schema_key_len_nodekey < schema_key_len_node:
                                    # if conflict, if new path is shorter, use the shorter one
                                    schema_id_mapping[endpoints_type][node] = (
                                        schema_id_nodekey, schema_key_len_nodekey)
                            else:
                                # only node_key_left in schema_id_mapping, propogate to node, save schema id for node
                                schema_id_mapping[endpoints_type][node] = (schema_id_nodekey, schema_key_len_nodekey)
                        elif node_key_right in schema_id_mapping[endpoints_type]:
                            schema_id_nodekey, schema_key_len_nodekey = schema_id_mapping[endpoints_type][
                                node_key_right]
                            if node in schema_id_mapping[endpoints_type]:
                                # both node and nodekey already has schema id
                                schema_id_node, schema_key_len_node = schema_id_mapping[endpoints_type][node]
                                if schema_key_len_nodekey < schema_key_len_node:
                                    # if conflict, if new path is shorter, use the shorter one
                                    schema_id_mapping[endpoints_type][node] = (
                                        schema_id_nodekey, schema_key_len_nodekey)
                            else:
                                # only node_key_left in schema_id_mapping, propogate to node, save schema id for node
                                schema_id_mapping[endpoints_type][node] = (schema_id_nodekey, schema_key_len_nodekey)
                        elif node in schema_id_mapping[endpoints_type]:
                            # only node in schema_id_mapping, check whether new path is shorter, if so, create new key
                            pass
                            # schema_id_node, schema_key_len_node = schema_id_mapping[endpoints_type][node]
                            # if node_key_left_len < schema_key_len_node:
                            #     schema_id = '%s_%d' % (types[path_element_id], len(schema_id_mapping[endpoints_type]))
                            #     # save schema id for node and node_key_left
                            #     schema_id_mapping[endpoints_type][node_key_left] = (schema_id, node_key_left_len)
                            #     schema_id_mapping[endpoints_type][node] = (schema_id, node_key_left_len)
                        else:
                            # both node and nodekey not in schema
                            schema_id = '%s_%d' % (types[path_element_id], len(schema_id_mapping[endpoints_type]))
                            if node_key_left_len <= node_key_right_len:
                                # save schema id for node and node_key
                                schema_id_mapping[endpoints_type][node_key_left] = (schema_id, node_key_left_len)
                                schema_id_mapping[endpoints_type][node] = (schema_id, node_key_left_len)
                            else:
                                # save schema id for node and node_key
                                schema_id_mapping[endpoints_type][node_key_right] = (schema_id, node_key_right_len)
                                schema_id_mapping[endpoints_type][node] = (schema_id, node_key_right_len)

                #     path_node_edge.append(types[path_element_id - 1])
                #     path_node_edge.append(node)
                # path_node_edge.append(types[-2])
                # path_node_edge.append(nodes[-1])  #append(types[-1])  #for event: add type instead of instance
                # endpoint_paths_schema[endpoints_type].append(path_node_edge)

    # assign path id
    doc_path_id_set = set()
    for endpoints_type in endpoint_paths:
        endpoint_path_count[endpoints_type] += len(endpoint_paths[endpoints_type])
        for endpoints, path_list in endpoint_paths[endpoints_type].items():
            # path_id_set = set()
            # unique_path_list = []
            path_id_list = []  # ??? where to put??
            for path in path_list:
                # print(path)
                if not direct:
                    path = undirect_path(path)
                nodes, types, texts = path

                types_schema = list()
                for element_idx, element_type in enumerate(types):
                    if element_idx % 2 == 0:
                        # node
                        node_idx = int(element_idx / 2)
                        element = nodes[node_idx]
                        if node_idx == 0 or node_idx == len(types) - 1:
                            # # event
                            # if element in event_props:
                            # event_type = event_props[element]
                            # types_schema.append(event_type)
                            types_schema.append(element_type)
                        else:
                            # entity
                            if assign_node_id:
                                if element in schema_id_mapping[endpoints_type]:
                                    schema_id, key_len = schema_id_mapping[endpoints_type][element]
                                    types_schema.append(schema_id)
                            else:
                                types_schema.append(element_type)
                    else:
                        # edge
                        types_schema.append(element_type)

                    # Get path id

                path_key = ','.join(types_schema)
                if path_key in path_id_mapping:
                    path_id = path_id_mapping[path_key]
                else:
                    path_id = len(path_id_mapping)
                    path_id_mapping[path_key] = path_id

                path_id_list.append(path_id)
                # path_text[path_id].add(', '.join(texts))

                # add instance mapping:
                if save_instance:
                    path_instance_mapping[path_id].append(nodes)
                    path_instance_text_mapping[path_id].append(texts)

            # doc_path_id_set.update(path_id_set)
            doc_path_id_set.update(path_id_list)

            # for path_id_1, path_id_2 in itertools.combinations(path_id_set, 2):
            for path_id_1, path_id_2 in itertools.combinations(path_id_list, 2):
                # Count co-occurrence in a doc
                path_cooccur_freq_count[path_id_1][path_id_2] += 1
                path_cooccur_freq_count[path_id_2][path_id_1] += 1

            # for path_id, path in unique_path_list:
            for path_id in path_id_list:
                # Add path frequency
                path_freq_count[path_id] += 1

    # path_doc_freq_count.update(doc_path_id_set)

    return schema_id_mapping, path_id_mapping, max_path_len, \
            endpoint_path_count, path_instance_mapping, path_instance_text_mapping, \
            path_cooccur_freq_count, path_freq_count


def traverse_event_event(input_files,
                         entitylinking_tab=None,
                         linking=False,
                         output_file=None,
                         output_file_nodeid_mapping=None,
                         output_file_edge_info=None,
                         max_path_len=6,
                         use_entity_sub_type=False,
                         use_relation_sub_type=False,
                         direct=True,
                         assign_node_id=False,
                         save_instance=True,
                         remove_docs=None):

    if entitylinking_tab is not None:
        entity_ace2wiki = load_entity_linking(entitylinking_tab)

    # Load data
    data = defaultdict(list)
    for input_file in input_files:
        with open(input_file, 'r') as r:
            for line in r:
                inst = json.loads(line)

                docid = inst['doc_id']

                if remove_docs is not None and docid in remove_docs:
                    continue

                if inst['entity_mentions']: # and len(inst['event_mentions']) > 1:
                    data[docid].append(inst)
    print('#doc: {}'.format(len(data)))
    print('#sent: {}'.format(sum([len(v) for _, v in data.items()])))


    path_id_mapping = {}
    if save_instance:
        path_instance_mapping = defaultdict(list)
        path_instance_text_mapping = defaultdict(list)

    path_freq_count = Counter()
    # path_doc_freq_count = Counter()
    path_cooccur_freq_count = defaultdict(Counter)
    # path_text = defaultdict(set)
    endpoint_path_count = Counter()
    edge_info = defaultdict(lambda : defaultdict(list)) # edge -> edge_instance_unique -> edge_instance_text
    # doc_events = defaultdict(list)

    if assign_node_id:
        schema_id_mapping = defaultdict(lambda : defaultdict())
    else:
        schema_id_mapping = None

    entity_num = 0
    relation_num = 0
    event_num = 0
    role_num = 0

    if linking:
        entity_props = {}
        event_props = {}
        entity_text = defaultdict(list)
        event_text = defaultdict(list)
        entity_next_entities = defaultdict(list)
        entity_next_events = defaultdict(list)
        event_next_entities = defaultdict(list)

        for doc_id, insts in data.items():
            entity_props, event_props, entity_text, event_text, \
            entity_next_entities, entity_next_events, event_next_entities, \
            entity_num, relation_num, event_num, role_num = load_entity_relation_event(
                insts, edge_info,
                entity_props, event_props, entity_text, event_text,
                entity_next_entities, entity_next_events, event_next_entities,
                linking=linking, entity_ace2wiki=entity_ace2wiki,
                use_entity_sub_type=use_entity_sub_type,
                use_relation_sub_type=use_relation_sub_type,
                direct=direct,
                entity_num=entity_num, relation_num=relation_num,
                event_num=event_num, role_num=role_num,
            )

        entity_text, event_text = entity_text_canonical(entity_text, event_text)

        schema_id_mapping, path_id_mapping, max_path_len, \
        endpoint_path_count, path_instance_mapping, path_instance_text_mapping, \
        path_cooccur_freq_count, path_freq_count = find_paths(entity_props, event_props, entity_text, event_text,
                                                              entity_next_entities, entity_next_events,
                                                              event_next_entities,
                                                              schema_id_mapping, path_id_mapping, max_path_len,
                                                              endpoint_path_count, path_instance_mapping,
                                                              path_instance_text_mapping,
                                                              path_cooccur_freq_count, path_freq_count,
                                                              assign_node_id=assign_node_id,
                                                              save_instance=save_instance
                                                              )
    else:
        for doc_id, insts in data.items():
            entity_props = {}
            event_props = {}
            entity_text = defaultdict(list)
            event_text = defaultdict(list)
            entity_next_entities = defaultdict(list)
            entity_next_events = defaultdict(list)
            event_next_entities = defaultdict(list)

            entity_props, event_props, entity_text, event_text, \
            entity_next_entities, entity_next_events, event_next_entities, \
            entity_num, relation_num, event_num, role_num = load_entity_relation_event(
                insts, edge_info,
                entity_props, event_props, entity_text, event_text,
                entity_next_entities, entity_next_events, event_next_entities,
                linking=linking, entity_ace2wiki=entity_ace2wiki,
                use_entity_sub_type=use_entity_sub_type,
                use_relation_sub_type=use_relation_sub_type,
                direct=direct,
                entity_num=entity_num, relation_num=relation_num,
                event_num=event_num, role_num=role_num,
            )

            entity_text, event_text = entity_text_canonical(entity_text, event_text)

            schema_id_mapping, path_id_mapping, max_path_len, \
            endpoint_path_count, path_instance_mapping, path_instance_text_mapping, \
            path_cooccur_freq_count, path_freq_count = find_paths(entity_props, event_props, entity_text, event_text,
                                                                  entity_next_entities, entity_next_events,
                                                                  event_next_entities,
                                                                  schema_id_mapping, path_id_mapping, max_path_len,
                                                                  endpoint_path_count, path_instance_mapping,
                                                                  path_instance_text_mapping,
                                                                  path_cooccur_freq_count, path_freq_count,
                                                                  assign_node_id=assign_node_id,
                                                                  save_instance=save_instance
                                                                  )


    # Output paths
    id_path_mapping = {i: p for p, i in path_id_mapping.items()}
    path_info_by_endpoints = defaultdict(list)
    for path_id, freq in path_freq_count.most_common():
        path_key = id_path_mapping[path_id]
        # doc_freq = path_doc_freq_count[path_id]
        # print('Path: {}'.format(path_key))
        # print('Freq/Doc freq: {}/{}'.format(freq, doc_freq))
        # for co_path_id, co_freq in path_cooccur_freq_count[path_id].most_common(100):
        #     co_path_key = id_path_mapping[co_path_id]
        #     prob = co_freq / freq
        #     print('{:.3f} => {}'.format(prob, co_path_key))
        # print('-' * 80)

        # text_list = list(path_text[path_id])
        # random.shuffle(text_list)

        path_types = path_key.split(',')
        endpoints_type = '%s__%s' % (path_types[0], path_types[-1])
        # print(endpoints_type, endpoint_path_count[endpoints_type])
        path_info = {
            'id': path_id,
            'path': path_types,
            'probability': float(freq) / endpoint_path_count[endpoints_type],
            'count': freq,
            # 'doc_count': doc_freq,
            # 'cooccur': [(i, f / freq, f, id_path_mapping[i]) for i, f in path_cooccur_freq_count[path_id].most_common(5)],
            'cooccur': [{'id': i, 'prob': f / freq, 'freq': f} for i, f in path_cooccur_freq_count[path_id].most_common()],
            # 'example': text_list[:20],
            'instance': path_instance_mapping[path_id],
            'instance_text': path_instance_text_mapping[path_id],
        }
        path_info_by_endpoints[endpoints_type].append(path_info)
    json.dump(path_info_by_endpoints, open(output_file, 'w'), indent=2)

    with open(output_file.replace('.json', '.txt'), 'w') as w:
        for endpoint in path_info_by_endpoints:
            for path_instance in path_info_by_endpoints[endpoint]:
                w.write('%s\n' % ','.join(path_instance['path']))

    if assign_node_id:
        json.dump(schema_id_mapping, open(output_file_nodeid_mapping, 'w'), indent=2)

    # save edge_info
    json.dump(edge_info, open(output_file_edge_info, 'w'), indent=2)
    print('save edge info to ', output_file_edge_info)

    print('entity_num, relation_num, event_num, role_num', entity_num, relation_num, event_num, role_num)


if __name__ == '__main__':
    use_entity_sub_type = False
    use_relation_sub_type = use_entity_sub_type
    direct = True
    save_instance = True
    assign_node_id = False
    linking = False
    remove_doc = False

    if remove_doc:
        lines = open('removed_doclist.txt', 'r').readlines()
        remove_docs = set([line.rstrip('\n') for line in lines])
    else:
        remove_docs = None

    for filename in ['train', 'dev', 'test']:
        print(filename)
        traverse_event_event(
            [
                # '/shared/nas/data/m1/manling2/pathlm/pathlm/data/ace/train.oneie.json',
                # '../../../data/ace/%s.oneie.json' % filename,
                '/shared/nas/data/m1/yinglin8/projects/oneie/data/ere/processed/english/eng.merge.%s.json' % filename,
             ],
            entitylinking_tab='/shared/nas/data/m1/manling2/evt-rep/data/ace/%s_link.tab' % filename,
            linking=linking,
            output_file='../../../data/ace%s/%s.paths%s%s%s.json' % (
                '.fine' if use_entity_sub_type else '',
                filename,
                '.direct' if direct else '',
                '.link' if linking else '',
                '.removed' if remove_doc else '',
            ),
            # output_file_nodeid_mapping='../../../data/ace/paths.schemaid.idmapping.json',
            output_file_edge_info = '../../../data/ace%s/%s.edges%s%s%s.json' % (
                '.fine' if use_entity_sub_type else '',
                filename,
                '.direct' if direct else '',
                '.link' if linking else '',
                '.removed' if remove_doc else '',
            ),
            use_entity_sub_type=use_entity_sub_type,
            use_relation_sub_type=use_relation_sub_type,
            direct=direct,
            assign_node_id=assign_node_id,
            save_instance=save_instance,
            remove_docs=remove_docs
        )


