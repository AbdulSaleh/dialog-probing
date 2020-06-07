"""
Script to process DialogueNLI dataset into ParlAI format.

Note that this script should be manually run by the user and not through ParlAI
to process the data, and hence _parse.py instead of parse.py.
"""
from pathlib import Path
import pickle
import json
import random

random.seed(0)

project_dir = Path(__file__).resolve().parent.parent.parent.parent
data_dir = Path(project_dir, 'data', 'probing', 'dialoguenli')

text_path = data_dir.joinpath('dialoguenli.txt')
label_path = data_dir.joinpath('labels.txt')
info_path = data_dir.joinpath('info.pkl')

text_file = open(text_path, 'w')
label_file = open(label_path, 'w')
info_file = open(info_path, 'wb')

example_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
examples = []

word_map = {'am': 'are', 'was': 'were', 'i': 'you', 'me': 'you', 'my': 'your', 'mine': 'yours'}

info = {'n_train': 0, 'n_dev': 0, 'n_test': 0}
for i, split in enumerate(['train', 'dev', 'test']):
    data_file = open(str(data_dir.joinpath('dialogue_nli_' + split + '.jsonl')))
    data = json.load(data_file)

    for example in data:
        s1 = example['sentence1']
        s2 = example['sentence2']
        s2_words = s2.split()
        s2 = " ".join([w if w not in word_map else word_map[w] for w in s2_words])
        s = s1 + '\n' + s2
        text = 'text:' + s + '\tlabels:\tepisode_done:True\n'
        label = example['label']
        examples.append([label, text])
        example_counts[label] += 1

        info['n_' + split] += 1

for ex in examples:
    text_file.write(ex[1])
    label_file.write(ex[0] + '\n')

print('n_pos:', example_counts['positive'])
print('n_neg:', example_counts['negative'])
print('n_neu:', example_counts['neutral'])

print(info)
pickle.dump(info, info_file)

