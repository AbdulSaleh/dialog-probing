"""
Script to process WNLI dataset into ParlAI format.

Note that this script should be manually run by the user and not through ParlAI
to process the data, and hence _parse.py instead of parse.py.
"""
import pickle
from pathlib import Path
import json
import re
# Load data
project_dir = Path(__file__).resolve().parent
train_path = project_dir.joinpath('train.jsonl')
val_path = project_dir.joinpath('val.jsonl')
# These don't have labels so they're not used here (I think they're
# intended for evaluating on the benchmark site)
test_path = project_dir.joinpath('test.jsonl')

train = open(train_path, 'r').readlines()
test = open(test_path, 'r').readlines()
validation = open(val_path, 'r').readlines()
data = train + validation

project_dir = Path(__file__).resolve().parent.parent.parent.parent
data_dir = Path(project_dir, 'data', 'probing', 'winograd')

# Save files
question_path = data_dir.joinpath('winograd.txt')
label_path = data_dir.joinpath('labels.txt')
info_path = data_dir.joinpath('info.pkl')

question_file = open(question_path, 'w')
label_file = open(label_path, 'w')
info_file = open(info_path, 'wb')

# Process data
for line in data:
  example = json.loads(line)
  input_text = example['text']
  input_lower = input_text.lower()
  word_1 = example['target']['span1_text']
  word_2 = example['target']['span2_text']
  label = example['label']

  label_file.write(str(label) + '\n')
  # TODO what do labels mean (if anything) here?
  question_file.write('text:' + input_text + '\tlabels: \n')
  question_file.write(
      'text:'
  + f"Does {word_1} refer to {word_2}?"
      + '\tlabels: \tepisode_done:True\n')


# Save data info
info = {'n_train': len(train),
        'n_val': len(validation)}

pickle.dump(info, info_file)
