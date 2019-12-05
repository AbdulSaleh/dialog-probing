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

from sklearn.model_selection import train_test_split


project_dir = Path(__file__).resolve().parent.parent.parent.parent
data_dir = Path(project_dir, 'data', 'probing', 'act_dailydialog')
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

# Get labels to determine stratified splits
dialogs = []
labels = []
for episode in data:
    dialog = episode['dialogue']
    dialogs.append(dialog)

    acts = []
    for turn in dialog:
        acts.append(turn['act'])

    labels.append(random.choice(acts))

X_train, X_test, y_train, y_test = train_test_split(
    dialogs, labels, test_size=0.15, random_state=1984, stratify=labels
)

data = X_train + X_test

# Process data
c = 0
for dialog in data:
    running = []
    for turn in dialog:
        c += 1
        running.append(turn['text'])
        text = '\n'.join(running)
        dialogs_file.write(
            f'text:{text}\tlabels: \tepisode_done:True\n'
        )

        act = turn['act']
        label_file.write(act+'\n')

# Save data info
n_train = int(0.85 * c)
info = {'n_train': n_train,
        'n_test': c - n_train}

pickle.dump(info, info_file)
