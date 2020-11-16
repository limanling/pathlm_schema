import pandas as pd
from collections import defaultdict
import ujson as json
# import json

path_file = '../../../data/ace/id.paths.all.json'
path_file_new = '../../../data/ace/id.paths.all.cooccur.json'

print(open(path_file).read())
paths_df = pd.read_json(path_file, lines=True)

path_dict = paths_df.to_dict('records')

for row in path_dict:
    start_evt = row["path"][0]
    end_evt = row["path"][-1]
    path = row['path']
    path_id = row['id']
    count = row['count']
    cooccur = row['cooccur']

    cooccur_new = dict()
    for path_cooccur in cooccur:
        path_cooccur_id = path_cooccur[0]
        probability = path_cooccur[1]
        cooccur_count = round(probability * count)  # math.ceil(probability * count)
        cooccur_new[path_cooccur_id] = cooccur_count

    row['cooccur'] = cooccur_new

json.dump(path_dict, open(path_file_new, 'w'), indent=2)