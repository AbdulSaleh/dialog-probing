"""
Note that this script should be manually run by the user and not through ParlAI
to process the data, and hence _parse.py instead of parse.py.
"""
from pathlib import Path
import pickle
import json

project_dir = Path(__file__).resolve().parent.parent.parent.parent
data_dir = Path(project_dir, 'data', 'probing', 'copa')

question_path = data_dir.joinpath('copa.txt')
label_path = data_dir.joinpath('labels.txt')
info_path = data_dir.joinpath('info.pkl')

question_file = open(question_path, 'w')
label_file = open(label_path, 'w')
info_file = open(info_path, 'wb')

with open(str(data_dir.joinpath('copa.json'))) as json_file:
    dataset = json.load(json_file)

    for example in dataset['item']:
        label = int(example['@most-plausible-alternative'])-1
        context = example['p']
        a1 = example['a1'][:-1]
        a2 = example['a2']
        question = context + '\nEither ' + a1 + ' or ' + a2 + ' Which one?'
        question_file.write('text:' + question + '\tlabels:' + '\tepisode_done:True\n')
        label_file.write(str(label) + '\n')

print('n_train:', 500)
print('n_test', 500)

# Save data info
info = {'n_train': 500,
        'n_test': 500}

pickle.dump(info, info_file)

