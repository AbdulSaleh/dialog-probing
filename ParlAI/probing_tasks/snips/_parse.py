"""
Script to process Snips dataset into ParlAI format.

Note that this script should be manually run by the user and not through ParlAI
to process the data, and hence _parse.py instead of parse.py.
"""
from pathlib import Path
import pickle
import json

labels = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic',
          'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent']

test_dict_count = {'AddToPlaylist': 0, 'BookRestaurant': 0, 'GetWeather': 0, 'PlayMusic': 0,
                   'RateBook': 0, 'SearchCreativeWork': 0, 'SearchScreeningEvent': 0}

# Load data
project_dir = Path(__file__).resolve().parent.parent.parent.parent
data_dir = Path(project_dir, 'data', 'probing', 'snips')

question_path = data_dir.joinpath('snips.txt')
label_path = data_dir.joinpath('labels.txt')
info_path = data_dir.joinpath('info.pkl')

question_file = open(question_path, 'w', encoding='utf-8')
label_file = open(label_path, 'w')
info_file = open(info_path, 'wb')

n_train = 0
n_test = 0

# Process train data
for label in labels:
    f_name = data_dir.joinpath(label, 'train_' + label + '_full.json')
    with open(f_name, encoding="utf-8", errors='ignore') as f:
        dataset = json.load(f)
        for example in dataset[label]:
            text = ''.join([t['text'] for t in example['data']])
            text = text.replace('\n', '').replace('\t', '')
            question_file.write(('text:' + text + '\tlabels: \tepisode_done:True\n'))
            label_file.write(label + '\n')
            n_train += 1

# Process test data
for label in labels:
    f_name = data_dir.joinpath(label, 'validate_' + label + '.json')
    with open(f_name, encoding="utf-8", errors='ignore') as f:
        dataset = json.load(f)
        for example in dataset[label]:
            text = (''.join([t['text'] for t in example['data']]))
            text = text.replace('\n', '').replace('\t', '')
            question_file.write('text:' + text + '\tlabels: \tepisode_done:True\n')
            label_file.write(label + '\n')
            test_dict_count[label] += 1
            n_test += 1

print('train', n_train)
print('test', n_test)

print(test_dict_count)

# Save data info
info = {'n_train': n_train,
        'n_test': n_test}

pickle.dump(info, info_file)
