"""
Script to process TREC Question Classification dataset into ParlAI format.

Note that this script should be manually run by the user and not through ParlAI
to process the data, and hence _parse.py instead of parse.py.
"""
from pathlib import Path


# Call from ParlAI directory
project_dir = Path(__file__).parent.parent.parent.parent
data_dir = Path(project_dir, 'data', 'probing', 'trecquestion')
data_path = data_dir.joinpath('train_5500.label')
question_path = data_dir.joinpath('trecquestion.txt')
label_path = data_dir.joinpath('labels.txt')


data = open(data_path, 'r', encoding='ISO-8859-1').readlines()
question_file = open(question_path, 'w')
label_file = open(label_path, 'w')


for line in data:
    label = line[:line.index(' ')].strip()
    question = line[line.index(' ') + 1:].rstrip()

    label_file.write(label + '\n')
    question_file.write('text:' + question + '\tlabels: \tepisode_done:True\n')
