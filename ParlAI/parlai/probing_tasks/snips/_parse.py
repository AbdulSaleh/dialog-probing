"""
Script to process TREC Question Classification dataset into ParlAI format.

Note that this script should be manually run by the user and not through ParlAI
to process the data, and hence _parse.py instead of parse.py.
"""
from pathlib import Path
import pickle
import json

labels = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic',
          'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent']

# Load data
project_dir = Path(__file__).resolve().parent.parent.parent.parent
data_dir = Path(project_dir, 'data', 'probing', 'snips')

question_path = data_dir.joinpath('snips.txt')
label_path = data_dir.joinpath('labels.txt')
info_path = data_dir.joinpath('info.pkl')

question_file = open(question_path, 'w')
label_file = open(label_path, 'w')
info_file = open(info_path, 'wb')

counts = [0, 0]

for i, label in enumerate(labels):

    for j, filename in enumerate(['/train_' + label + '_full.json', '/validate_' + label + '.json']):
        with open(str(data_dir.joinpath(label + filename)), encoding='latin-1') as json_file:
            dataset = json.load(json_file)
            for example in dataset[label]:
                example_text = ''
                for phrase in example['data']:
                    example_text += phrase['text']
                question_file.write('text:' + example_text + '\tlabels: \tepisode_done:True\n')
                label_file.write(label + '\n')
                counts[j] += 1

print('n_train:', counts[0])
print('n_test', counts[1])

# Save data info
info = {'n_train': counts[0],
        'n_test': counts[1]}

pickle.dump(info, info_file)

