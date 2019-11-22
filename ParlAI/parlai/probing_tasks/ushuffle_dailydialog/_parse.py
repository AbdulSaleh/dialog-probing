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
data_dir = Path(project_dir, 'data', 'probing', 'ushuffle_dailydialog')
test_path = data_dir.joinpath('test.json')

test = list(map(json.loads, open(test_path, 'r').readlines()))
data = test

# Save files
shuffle_path = data_dir.joinpath('shuffled.txt')
label_path = data_dir.joinpath('labels.txt')
info_path = data_dir.joinpath('info.pkl')

shuffle_file = open(shuffle_path, 'w')
label_file = open(label_path, 'w')
info_file = open(info_path, 'wb')

# Pick dialogs to shuffle
n = len(data)
indices = range(len(data))
shuffle = random.sample(indices, n//2)

# Process data
for i, episode in enumerate(data):
    dialog = episode['dialogue']

    # Might choose to limit dialog length
    conv_len = random.sample(range(2, len(dialog)+1), 1)[0]
    dialog = dialog[:conv_len]

    if i in shuffle:
        random.shuffle(dialog)
        label_file.write('True' + '\n')
    else:
        label_file.write('False' + '\n')

    text = '\n'.join([turn['text'] for turn in dialog])
    shuffle_file.write(
        f'text:{text}\tlabels: \tepisode_done:True\n'
    )

# Save data info
n_train = int(len(data) * 0.85)
info = {'n_train': n_train,
        'n_test': len(data) - n_train}

pickle.dump(info, info_file)
