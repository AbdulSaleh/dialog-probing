"""
Script to create shuffled utterance daily dialog dataset in ParlAI format.

Note that this script should be manually run by the user and not through ParlAI
to process the data, and hence _parse.py instead of parse.py.
"""

import pickle
import json
import random
from pathlib import Path
random.seed(1984)


project_dir = Path(__file__).resolve().parent.parent.parent.parent
data_dir = Path(project_dir, 'data', 'probing', 'topic_dailydialog')
test_path = data_dir.joinpath('test.json')

test = list(map(json.loads, open(test_path, 'r').readlines()))
data = test

# Save files
dialogs_path = data_dir.joinpath('dialogs.txt')
label_path = data_dir.joinpath('labels.txt')
info_path = data_dir.joinpath('info.pkl')

dialogs_file = open(dialogs_path, 'w')
label_file = open(label_path, 'w')
info_file = open(info_path, 'wb')


# Process data
for episode in data:
    dialog = episode['dialogue']
    text = '\n'.join([turn['text'] for turn in dialog])

    dialogs_file.write(
        f'text:{text}\tlabels: \tepisode_done:True\n'
    )

    topic = episode['topic']
    label_file.write(topic+'\n')


# Save data info
n_train = int(len(data) * 0.85)
info = {'n_train': n_train,
        'n_test': len(data) - n_train}

pickle.dump(info, info_file)
