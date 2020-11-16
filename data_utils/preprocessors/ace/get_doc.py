from collections import defaultdict
import json

input_files = ['../../../data/ace/test.oneie.json']

doc_set = set()
# Load data
data = defaultdict(list)
for input_file in input_files:
    with open(input_file, 'r') as r:
        for line in r:
            inst = json.loads(line)
            doc_set.add(inst['doc_id'])
            # if inst['entity_mentions']:  # and len(inst['event_mentions']) > 1:
            #     data[inst['doc_id']].append(inst)
# print('#doc: {}'.format(len(data)))
# print('#sent: {}'.format(sum([len(v) for _, v in data.items()])))

for doc_id in doc_set:
    print(doc_id)