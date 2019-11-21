# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This preprocessing script was used to create ParlAI's version of the
data. It was run in a script called parse.py inside ijcnlp_dailydialog/ after
uncompressing the original directory and all subdirectories.
"""

import json
from pathlib import Path
import random

project_dir = Path(__file__).resolve().parent.parent.parent.parent
data_dir = Path(project_dir, 'data', 'probing', 'shuffle_dailydialog')
train_path = data_dir.joinpath('train.json')
test_path = data_dir.joinpath('test.json')
valid_path = data_dir.joinpath('valid.json')

train = list(map(json.loads, open(train_path, 'r').readlines()))
test = list(map(json.loads, open(test_path, 'r').readlines()))
valid = list(map(json.loads, open(valid_path, 'r').readlines()))
data = train + test + valid

# Save files
question_path = data_dir.joinpath('shuffle_dailydialog.txt')
label_path = data_dir.joinpath('labels.txt')
info_path = data_dir.joinpath('info.pkl')

question_file = open(question_path, 'w')
label_file = open(label_path, 'w')
info_file = open(info_path, 'wb')

# Process data
for i, episode in enumerate(data):
  dialogue = episode['dialogue']
  # TODO consider making this exactly half of dataset
  shuffle = i > (len(data) / 2)
  if shuffle:
    random.Random(0).shuffle(dialogue)
    label_file.write('True' + '\n')
  else:
    label_file.write('False' + '\n')
  for i in range(0, len(dialogue), 2):
    utterance_1 = dialogue[i]['text']
    if i == len(dialogue) - 1:
      utterance_2 = ''
    else:
      utterance_2 = dialogue[i + 1]['text']
    episode_done = i >= len(dialogue) - 2
    question_file.write(
        f'text:{utterance_1}\tlabels:{utterance_2}\tepisode_done:{episode_done}\n')
