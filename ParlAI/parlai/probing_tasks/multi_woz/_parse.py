"""
Script to process TREC Question Classification dataset into ParlAI format.

Note that this script should be manually run by the user and not through ParlAI
to process the data, and hence _parse.py instead of parse.py.
"""
from pathlib import Path
import pickle
import json

labels = ['taxi', 'police', 'restaurant', 'hospital', 'hotel', 'attraction', 'train']

project_dir = Path(__file__).resolve().parent.parent.parent.parent
data_dir = Path(project_dir, 'data', 'probing', 'multi_woz')

question_path = data_dir.joinpath('multi_woz.txt')
label_path = data_dir.joinpath('labels.txt')
info_path = data_dir.joinpath('info.pkl')

question_file = open(question_path, 'w')
label_file = open(label_path, 'w')
info_file = open(info_path, 'wb')

example_count = 0
turn_count = 0

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
        customer_turns = []
        concierge_turns = []
        example_count += 1
        for i, turn in enumerate(example['log']):
            if i % 2 == 0:
                customer_turns.append(turn['text'])
                turn_count += 1
            else:
                concierge_turns.append(turn['text'])
        n_turns = len(customer_turns)
        for i in range(n_turns):
            question_file.write('text:' + customer_turns[i] + '\tlabels:' + concierge_turns[i])
            if i != n_turns-1:
                question_file.write('\n')
            else:
                question_file.write('\tepisode_done:True\n')
            label_file.write(label + '\n')

n_train = int(0.75 * example_count)

print('n_train:', n_train)
print('n_test', example_count - n_train)

# Save data info
info = {'n_train': n_train,
        'n_test': example_count - n_train}

pickle.dump(info, info_file)


