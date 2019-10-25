"""
Script to process TREC Question Classification dataset into ParlAI format.

Note that this script should be manually run by the user and not through ParlAI
to process the data, and hence _parse.py instead of parse.py.
"""
from pathlib import Path
import pickle

# Load data
project_dir = Path(__file__).resolve().parent.parent.parent.parent
data_dir = Path(project_dir, 'data', 'probing', 'trecquestion')
train_path = data_dir.joinpath('train_5500.label')
test_path = data_dir.joinpath('TREC_10.label')

train = open(train_path, 'r', encoding='ISO-8859-1').readlines()
test = open(test_path, 'r', encoding='ISO-8859-1').readlines()
data = train + test

# Save files
question_path = data_dir.joinpath('trecquestion.txt')
label_path = data_dir.joinpath('labels.txt')

question_file = open(question_path, 'w')
label_file = open(label_path, 'w')

# Process data
for line in data:
    label = line[:line.index(' ')].strip()
    question = line[line.index(' ') + 1:].rstrip()

    label_file.write(label + '\n')
    question_file.write('text:' + question + '\tlabels: \tepisode_done:True\n')
