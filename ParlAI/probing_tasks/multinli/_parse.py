"""
Script to process MultiNLI dataset into ParlAI format.

Note that this script should be manually run by the user and not through ParlAI
to process the data, and hence _parse.py instead of parse.py.
"""
from pathlib import Path
import pickle
import json


MULTINLI_PREMISE_KEY = 'sentence1'
MULTINLI_HYPO_KEY = 'sentence2'
MULTINLI_ANSWER_KEY = 'gold_label'


# Load data
project_dir = Path(__file__).resolve().parent.parent.parent.parent
data_dir = Path(project_dir, 'data', 'probing', 'multinli')
train_path = data_dir.joinpath('multinli_1.0_train.jsonl')
dev_path = data_dir.joinpath('multinli_1.0_dev_matched.jsonl')
test_path = data_dir.joinpath('multinli_1.0_dev_mismatched.jsonl')

train = [json.loads(l) for l in open(train_path)]
dev = [json.loads(l) for l in open(dev_path)]
test = [json.loads(l) for l in open(test_path)]
data = train + dev + test

# Save files
multinli_path = data_dir.joinpath('multinli.txt')
label_path = data_dir.joinpath('labels.txt')
info_path = data_dir.joinpath('info.pkl')

multinli_file = open(multinli_path, 'w', encoding="utf-8")
label_file = open(label_path, 'w')
info_file = open(info_path, 'wb')

# Process data
for line in data:
    premise = line[MULTINLI_PREMISE_KEY]
    hypo = line[MULTINLI_HYPO_KEY]
    label = line[MULTINLI_ANSWER_KEY]

    label_file.write(label + '\n')
    multinli_file.write('text:' + premise + '\n' + hypo + '\tlabels: \tepisode_done:True\n')


# Save data info
info = {'n_train': len(train),
        'n_dev': len(dev),
        'n_test': len(test)}

pickle.dump(info, info_file)
