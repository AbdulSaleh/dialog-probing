"""
Note that this script should be manually run by the user and not through ParlAI
to process the data, and hence _parse.py instead of parse.py.
"""
from pathlib import Path
import pickle
import json
import random
random.seed(0)

labels = ['taxi', 'police', 'restaurant', 'hospital', 'hotel', 'attraction', 'train']
labels_dict = {'taxi': 0, 'police': 0, 'restaurant': 0, 'hospital': 0, 'hotel': 0, 'attraction': 0, 'train': 0}
examples_dict = {'taxi': [], 'police': [], 'restaurant': [], 'hospital': [], 'hotel': [], 'attraction': [], 'train': []}
master_train = []
master_test = []

project_dir = Path(__file__).resolve().parent.parent.parent.parent
data_dir = Path(project_dir, 'data', 'probing', 'multi_woz')

question_path = data_dir.joinpath('multi_woz.txt')
label_path = data_dir.joinpath('labels.txt')
info_path = data_dir.joinpath('info.pkl')

question_file = open(question_path, 'w')
label_file = open(label_path, 'w')
info_file = open(info_path, 'wb')

with open(str(data_dir.joinpath('multi_woz_data.json')), encoding='latin-1') as json_file:
    dataset = json.load(json_file)
    for ep in dataset:
        example = dataset[ep]
        example_subjs = []
        for subj_type in labels:
            if example['goal'][subj_type] != {}:
                example_subjs.append(subj_type)
        if len(example_subjs) > 1:
            continue
        label = example_subjs[0]

        example_list = [label]

        all_turns = [turn['text'] for turn in example['log']]
        for i in range(1, len(all_turns)):
            example_list.append('text:' + '\n'.join(all_turns[:i]) + '\tlabels:' + '\tepisode_done:True\n')
            # question_file.write('text:' + '\n'.join(all_turns[:i]) + '\tlabels:' + '\tepisode_done:True\n')
            # label_file.write(label + '\n')
            labels_dict[label] += 1
        examples_dict[label].append(example_list)

n_train = 0
n_test = 0

for lab in labels:
    random.shuffle(examples_dict[lab])
    lab_n_examples = len(examples_dict[lab])
    lab_n_train = int(0.85 * lab_n_examples)
    master_train += examples_dict[lab][:lab_n_train]
    master_test += examples_dict[lab][lab_n_train:]

random.shuffle(master_test)
random.shuffle(master_train)

for dialog in master_train:
    label = dialog[0]
    for i in range(1, len(dialog)):
        question_file.write(dialog[i])
        label_file.write(label + '\n')
        n_train += 1

for dialog in master_test:
    label = dialog[0]
    for i in range(1, len(dialog)):
        question_file.write(dialog[i])
        label_file.write(label + '\n')
        n_test += 1


print('n_train:', n_train)
print('n_test', n_test)

print(labels_dict)

# Save data info
info = {'n_train': n_train,
        'n_test': n_test}

pickle.dump(info, info_file)


