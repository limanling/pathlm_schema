import json

test='test'

input_file = 'oneie_result/result.%s.baseline.json' % test

with open(input_file, 'r') as r:
    for line in r:
        inst = json.loads(line)

        sent_id_system = inst['sent_id']
        doc_id = sent_id_system[:sent_id_system.rfind('-')]

        if inst['pred']:  # and len(inst['event_mentions']) > 1:
            inst_entities = inst['pred']['entities']
            inst_relations = inst['pred']['relations']
            inst_events = inst['pred']['triggers']
            inst_roles = inst['pred']['roles']

            has_attack = False
            has_not_attack = False
            if len(inst_relations) > 1 and len(inst_events) > 1 and len(inst_roles) > 1:
                for event_idx, event in enumerate(inst_events):
                    start_idx, end_idx, event_type = event
                    if 'Attack' in event_type:
                        has_attack = True
                    if 'Transport' in event_type:
                        has_not_attack = True

            if has_attack and has_not_attack:
                print(sent_id_system, ' '.join(inst['tokens']))

xlent_set = set()
input_file_xlnet = 'oneie_result/result.%s.xlnet.json' % test
print('-------xlnet---------')
with open(input_file_xlnet, 'r') as r:
    for line in r:
        inst = json.loads(line)

        sent_id_system = inst['sent_id']
        doc_id = sent_id_system[:sent_id_system.rfind('-')]

        if inst['pred']:  # and len(inst['event_mentions']) > 1:
            inst_entities = inst['pred']['entities']
            inst_relations = inst['pred']['relations']
            inst_events = inst['pred']['triggers']
            inst_roles = inst['pred']['roles']

            has_attack = False
            has_not_attack = False
            if len(inst_relations) > 1 and len(inst_events) > 1 and len(inst_roles) > 1:
                for event_idx, event in enumerate(inst_events):
                    start_idx, end_idx, event_type = event
                    if 'Attack' in event_type:
                        has_attack = True
                    if 'Transport' in event_type:
                        has_not_attack = True

            if has_attack and has_not_attack:
                print(sent_id_system, inst['tokens'])
                xlent_set.add(sent_id_system)

print('==========')
ace_gt = '../../../data/ace/%s.oneie.json' % test
with open(ace_gt, 'r') as r:
    for line in r:
        inst = json.loads(line)
        # print(line)

        sent_id_system = inst['sent_id']
        doc_id = sent_id_system[:sent_id_system.rfind('-')]

        has_attack = False
        has_not_attack = False
        if len(inst['relation_mentions']) > 1 and len(inst['event_mentions']) > 1 :

            inst_entities = inst['event_mentions']
            for event in inst_entities:
                event_type = event['event_type']
                if 'Attack' in event_type:
                    has_attack = True
                if 'Transport' in event_type:
                    has_not_attack = True

        if has_attack and has_not_attack:
            print(sent_id_system, ' '.join(inst['tokens']))
            # entity_mention_id = entity['id']
            # entity_mention_start = entity['start']
            # entity_mention_end = entity['end']
            # entity_mention_id_formapping = '%s:%s_%s' % (sent_id, entity_mention_start, entity_mention_end)
            # entitymention_mapping2ace[entity_mention_id_formapping] = entity_mention_id

# with open(input_file, 'r') as r:
#     for line in r:
#         inst = json.loads(line)
#
#         sent_id_system = inst['sent_id']
#         sent_tokens = inst['tokens']
#         sent_tokens_content = tokens2content(sent_tokens)
#         doc_id_system = sent_id_system[:sent_id_system.rfind('-')]
#
#         senttokens2sentid_system[sent_tokens_content] = sent_id_system
#         sentid2senttokens_system[sent_id_system] = sent_tokens_content
#         # print(sent_id_system)
#         # sent_id_ace = senttokens2sentid_ace[sent_tokens_content]
#         sent_id_ace = None
#         for sent_id_ace in senttokens2sentid_ace[sent_tokens_content]:
#             doc_id_ace = sent_id_ace[:sent_id_ace.rfind('-')]
#             if doc_id_system == doc_id_ace:
#                 break
#         # print(sent_id_system, sent_tokens_content)
#         # assert doc_id_system == doc_id_ace
#         # if sent_id_ace:
#         #     print('---', sent_id_ace)
#
#         inst_id = '%s: %s' % (sent_id_system, sent_tokens_content)  # inst['sent_id']
#         inst_entities = inst['graph']['entities']
#         inst_relations = inst['graph']['relations']
#         inst_events = inst['graph']['triggers']
#         inst_roles = inst['graph']['roles']