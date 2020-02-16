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

number_turns = 3 # This can be 1, 3, or 5

project_dir = Path(__file__).resolve().parent.parent.parent
data_dir = Path(project_dir, 'data', 'probing', 'dsct8_multi')

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
dev_data = concat_json_files(dev_path) # dev data is ignored for now

# Save files
dialog_path = data_dir.joinpath('dsct8_multi.txt')
label_path = data_dir.joinpath('labels.txt')
info_path = data_dir.joinpath('info.pkl')

dialog_file = open(dialog_path, 'w')
label_file = open(label_path, 'w')
info_file = open(info_path, 'wb')

# Process data
def process_split(data):
    count = 0
    for example in data:
        dialog = example['turns'][:number_turns]
        # text = '\n'.join([turn['text'] for turn in dialog])
        active_intent = dialog[-1]['frames'][0]['state']['active_intent']
        if active_intent:
          utterances = list(map(lambda x: x['utterance'], dialog))
          utterances_joined = '\n'.join(utterances)
          label_file.write(active_intent + '\n')
          dialog_file.write(
              f"text:{utterances_joined}\tlabels: \tepisode_done:True\n")
          count += 1
    return count


train_data_len = process_split(train_data)
test_data_len = process_split(test_data)

# Save data info
info = {'n_train': train_data_len,
        'n_test': test_data_len}

pickle.dump(info, info_file)
