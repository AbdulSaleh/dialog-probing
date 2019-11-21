"""
Script to process WNLI dataset into ParlAI format.

Note that this script should be manually run by the user and not through ParlAI
to process the data, and hence _parse.py instead of parse.py.
"""
import pickle
from pathlib import Path
import csv
from os import mkdir


project_dir = Path(__file__).resolve().parent.parent.parent.parent
data_dir = Path(project_dir, 'data', 'probing', 'wnli')
mkdir(data_dir)

# Load data
train_path = data_dir.joinpath('train.tsv')
dev_path = data_dir.joinpath('dev.tsv')

train_data = csv.DictReader(open(train_path, 'r'), dialect='excel-tab')
dev_data = csv.DictReader(open(dev_path, 'r'), dialect='excel-tab')

# Save files
question_path = data_dir.joinpath('wnli.txt')
label_path = data_dir.joinpath('labels.txt')
info_path = data_dir.joinpath('info.pkl')

question_file = open(question_path, 'w')
label_file = open(label_path, 'w')
info_file = open(info_path, 'wb')


# Process data
def process_examples(data):
  example_count = 0
  for example in train_data:
    label_file.write(str(example['label']) + '\n')
    question_file.write(
        f"text:{example['sentence1']}\n{example['sentence2']}\tlabels:\tepisode_done:True\n")
    example_count += 1
  return example_count


train_data_len = process_examples(train_data)
dev_data_len = process_examples(dev_data)


# Save data info
info = {'n_train': train_data_len,
        'n_dev': dev_data_len}

pickle.dump(info, info_file)
