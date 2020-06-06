"""
Note that this script should be manually run by the user and not through ParlAI
to process the data, and hence _parse.py instead of parse.py.
"""
from pathlib import Path
import os
import pickle
import json
import random
random.seed(0)

categories = ['/Greeting&Greeting', '/Offering&Response', '/Question&Answer', '/Others']

examples = []

project_dir = Path(__file__).resolve().parent.parent.parent.parent
data_dir = Path(project_dir, 'ParlAI', 'data', 'probing', 'scenariosa')

question_path = data_dir.joinpath('scenariosa_single.txt')
label_path = data_dir.joinpath('labels_single.txt')
info_path = data_dir.joinpath('info_single.pkl')

question_file = open(question_path, 'w')
label_file = open(label_path, 'w')
info_file = open(info_path, 'wb')

n_pos = 0
n_neg = 0
n_neu = 0

for cat in categories:
    directory = str(data_dir) + cat + '/'
    for filename in os.listdir(directory):
        with open(directory + filename, 'r') as f:
            example = f.readlines()
        utts = []
        sents = []
        for line in example:
            if line[0] in ['A', 'B']:
                stripline = line.rstrip()
                if stripline[-2] == '-':
                    utts.append(stripline[4:-3])
                    sents.append('neg')
                elif stripline[-1] == '0':
                    utts.append(stripline[4:-2])
                    sents.append('neu')
                else:
                    utts.append(stripline[4:-2])
                    sents.append('pos')
        if len(utts) == 0:
            continue

        ### this is because we're just dong the first line
        text = 'text:' + utts[0] + '\tlabels:\tepisode_done:True\n'
        if sents[0] == 'pos':
            examples.append(['pos', text])
            n_pos += 1
        elif sents[0] == 'neg':
            examples.append(['neg', text])
            n_neg += 1
        else:
            examples.append(['neu', text])
            n_neu += 1
        ###

random.shuffle(examples)

n_train = int(0.85 * len(examples))
n_test = len(examples) - n_train

for ex in examples:
    question_file.write(ex[1])
    label_file.write(ex[0] + '\n')

print('n_train:', n_train)
print('n_test', n_test)

print('n_pos:', n_pos)
print('n_neg:', n_neg)
print('n_neu:', n_neu)

# Save data info
info = {'n_train': n_train,
        'n_test': n_test}

pickle.dump(info, info_file)


