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

question_path = data_dir.joinpath('scenariosa_multiple.txt')
label_path = data_dir.joinpath('labels_multiple.txt')
info_path = data_dir.joinpath('info_multiple.pkl')

question_file = open(question_path, 'w')
label_file = open(label_path, 'w')
info_file = open(info_path, 'wb')

n_pos = 0
n_neg = 0

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
        non_neutral_turns = []
        for i in range(len(sents)):
            if sents[i] != 'neu':
                non_neutral_turns.append(i)
        if not non_neutral_turns:
            continue
        selected_turn = random.choice(non_neutral_turns)

        text = 'text:' + '\n'.join(utts[:selected_turn+1]) + '\tlabels:\tepisode_done:True\n'
        if sents[selected_turn] == 'pos':
            examples.append(['pos', text])
            n_pos += 1
        elif sents[selected_turn] == 'neg':
            examples.append(['neg', text])
            n_neg += 1
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

# Save data info
info = {'n_train': n_train,
        'n_test': n_test}

pickle.dump(info, info_file)


