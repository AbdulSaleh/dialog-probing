"""
Note that this script should be manually run by the user and not through ParlAI
to process the data, and hence _parse.py instead of parse.py.
"""
from pathlib import Path
import pickle
import json

labels = ['taxi', 'police', 'restaurant', 'hospital', 'hotel', 'attraction', 'train']
labels_dict = {'taxi': 0, 'police': 0, 'restaurant': 0, 'hospital': 0, 'hotel': 0, 'attraction': 0, 'train': 0}

project_dir = Path(__file__).resolve().parent.parent.parent.parent
data_dir = Path(project_dir, 'data', 'probing', 'multi_woz')

question_path = data_dir.joinpath('multi_woz.txt')
label_path = data_dir.joinpath('labels.txt')
info_path = data_dir.joinpath('info.pkl')

question_file = open(question_path, 'w')
label_file = open(label_path, 'w')
info_file = open(info_path, 'wb')

example_count = 0

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

        all_turns = [turn['text'] for turn in example['log']]
        for i in range(1, len(all_turns)):
            question_file.write('text:' + '\n'.join(all_turns[:i]) + '\tlabels:' + '\tepisode_done:True\n')
            label_file.write(label + '\n')
            example_count += 1
            labels_dict[label] += 1

n_train = int(0.75 * example_count)

print('n_train:', n_train)
print('n_test', example_count - n_train)

print(labels_dict)

# Save data info
info = {'n_train': n_train,
        'n_test': example_count - n_train}

pickle.dump(info, info_file)


