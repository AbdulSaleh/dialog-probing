# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to create dsct8 dataset in ParlAI format.

Note that this script should be manually run by the user and not through ParlAI
to process the data, and hence _parse.py instead of parse.py.
"""

import json
from pathlib import Path
import pickle
import os
import random

random.seed(0)
project_dir = Path(__file__).resolve().parent.parent.parent.parent
data_dir = Path(project_dir, 'data', 'probing', 'dstc8')


def concat_json_files(folderpath):
    combined_list = []
    for dirpath, _dirs, files in os.walk(folderpath):
        for filename in files:
            if 'dialogues_' in filename:
                json_list = json.load(open(os.path.join(dirpath, filename)))
                combined_list += json_list
    return combined_list


train_path = data_dir.joinpath('train')
test_path = data_dir.joinpath('test')
dev_path = data_dir.joinpath('dev')

train_data = concat_json_files(train_path)
test_data = concat_json_files(test_path)
dev_data = concat_json_files(dev_path)  # dev data is ignored for now

# Save files
dialog_path = data_dir.joinpath('dstc8.txt')
label_path = data_dir.joinpath('labels.txt')
info_path = data_dir.joinpath('info.pkl')

dialog_file = open(dialog_path, 'w')
label_file = open(label_path, 'w')
info_file = open(info_path, 'wb')


# Process data
def process_split(data):
    count = 0
    for example in data:
        dialog = example['turns']
        turns = [turn['utterance'] for turn in dialog]
        possible_turns = []
        for i in range(0, min(float('inf'), len(dialog)), 2):
            intents = [dialog[i]['frames'][0]['state']['active_intent']]
            if len(set(intents)) == 1:
                possible_turns.append(i)

        if len(possible_turns) == 0:
            continue

        chosen_turn = random.choice(possible_turns)
        line = ('text:' + '\n'.join(turns[:chosen_turn + 1])
                + '\tlabels:' + '\tepisode_done:True\n')
        label = dialog[chosen_turn]['frames'][0]['state']['active_intent']

        dialog_file.write(line)
        label_file.write(label)
        count += 1
    return count


train_data_len = process_split(train_data)
dev_data_len = process_split(dev_data)
test_data_len = process_split(test_data)

# Save data info
info = {'n_train': train_data_len,
        'n_dev': dev_data_len,
        'n_test': test_data_len}

pickle.dump(info, info_file)
