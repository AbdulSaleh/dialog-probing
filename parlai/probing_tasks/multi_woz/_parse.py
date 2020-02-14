"""
Note that this script should be manually run by the user and not through ParlAI
to process the data, and hence _parse.py instead of parse.py.
"""
from pathlib import Path
from collections import Counter
import pickle
import json
import random
random.seed(0)

project_dir = Path(__file__).resolve().parent.parent.parent.parent
data_dir = Path(project_dir, 'data', 'probing', 'multi_woz')

question_path = data_dir.joinpath('multi_woz.txt')
label_path = data_dir.joinpath('labels.txt')
info_path = data_dir.joinpath('info.pkl')

question_file = open(question_path, 'w')
label_file = open(label_path, 'w')
info_file = open(info_path, 'wb')

with open(data_dir.joinpath('data.json'), encoding='latin-1') as json_file:
    dataset = json.load(json_file)
    test_codes = set(open(data_dir.joinpath('testListFile.txt')).read().splitlines())
    valid_codes = set(open(data_dir.joinpath('valListFile.txt')).read().splitlines())
    train_codes = set(dataset.keys()) - valid_codes - test_codes
    splits = {'train': train_codes, 'dev': valid_codes, 'test': test_codes}

    info = {'n_train': 0, 'n_dev': 0, 'n_test': 0}
    dialogs = []
    labels = []
    for split, codes in splits.items():
        for c in codes:
            example = dataset[c]
            try:
                turns = [turn['text'] for turn in example['log']]
                dialog_acts = [[act_type for act_type in turn['dialog_act']] for turn in example['log']]
            except KeyError:
                continue

            possible_turns = []
            for i in range(0, min(float('inf'), len(turns)), 2):
                if len(set(dialog_acts[i])) == 1:
                    possible_turns.append(i)

            if len(possible_turns) == 0:
                continue

            chosen_turn = random.choice(possible_turns)
            dialogs.append('text:' + '\n'.join(turns[:chosen_turn+1])
                           + '\tlabels:' + '\tepisode_done:True\n')
            labels.append(dialog_acts[chosen_turn][0])
            info['n_'+split] += 1


for dialog, label in zip(dialogs, labels):
    question_file.write(dialog)
    label_file.write(label + '\n')

pickle.dump(info, info_file)

# Summary statistics
train_stats = {label: count / info['n_train'] for label, count in Counter(labels[:info['n_train']]).items()}
valid_stats = {label: count / info['n_dev'] for label, count in Counter(labels[info['n_train']:info['n_train']+info['n_dev']]).items()}
test_stats = {label: count / info['n_test'] for label, count in Counter(labels[info['n_train']+info['n_dev']:]).items()}

print("For train data", train_stats)
print("For valid data", valid_stats)
print("For test data", test_stats)
