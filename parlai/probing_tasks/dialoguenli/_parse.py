"""
Script to process TREC Question Classification dataset into ParlAI format.

Note that this script should be manually run by the user and not through ParlAI
to process the data, and hence _parse.py instead of parse.py.
"""
from pathlib import Path
import pickle
import json
import random

random.seed(0)

project_dir = Path(__file__).resolve().parent.parent.parent.parent
data_dir = Path(project_dir, 'ParlAI', 'data', 'probing', 'dialoguenli')

question_path = data_dir.joinpath('dialoguenli.txt')
label_path = data_dir.joinpath('labels.txt')
info_path = data_dir.joinpath('info.pkl')

question_file = open(question_path, 'w')
label_file = open(label_path, 'w')
info_file = open(info_path, 'wb')

example_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
examples = []

word_map = {'am ': 'are ', 'was ': 'were ', 'i ': 'you ', 'me ': 'you ', 'my ': 'your ', 'mine ': 'yours '}

for i, dset_type in enumerate(['train', 'dev', 'test']):
    with open(str(data_dir.joinpath('dialogue_nli_' + dset_type + '.jsonl'))) as json_file:
        dataset = json.load(json_file)
    for example in dataset:
        s1 = example['sentence1']
        s2 = example['sentence2']
        for k, v in word_map.items():
            s2 = s2.replace(k, v)
        s = s1 + '\n' + s2
        text = 'text:' + s + '\tlabels:\tepisode_done:True\n'
        label = example['label']
        examples.append([label, text])
        example_counts[label] += 1

random.shuffle(examples)

n_train = int(0.85 * len(examples))
n_test = len(examples) - n_train

for ex in examples:
    question_file.write(ex[1])
    label_file.write(ex[0] + '\n')

print('n_train:', n_train)
print('n_test', n_test)

print('n_pos:', example_counts['positive'])
print('n_neg:', example_counts['negative'])
print('n_neu:', example_counts['neutral'])

# Save data info
info = {'n_train': n_train,
        'n_test': n_test}

pickle.dump(info, info_file)

