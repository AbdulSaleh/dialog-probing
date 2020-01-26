"""
Script to process TREC Question Classification dataset into ParlAI format.

Note that this script should be manually run by the user and not through ParlAI
to process the data, and hence _parse.py instead of parse.py.
"""
from pathlib import Path
import pickle
import json

project_dir = Path(__file__).resolve().parent.parent.parent.parent
data_dir = Path(project_dir, 'data', 'probing', 'squad2')

question_path = data_dir.joinpath('squad2.txt')
label_path = data_dir.joinpath('labels.txt')
info_path = data_dir.joinpath('info.pkl')

question_file = open(question_path, 'w')
label_file = open(label_path, 'w')
info_file = open(info_path, 'wb')

example_count = 0
question_count = 0
set_example_counts = [0, 0]

for i, dset_type in enumerate(['train', 'dev']):
    with open(str(data_dir.joinpath(dset_type + '-v2.0.json'))) as json_file:
        dataset = json.load(json_file)
    for article in dataset['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                question_file.write('text:' + context + '\n' + question + '\tlabels:\tepisode_done:True\n')
                if qa['is_impossible']:
                    label_file.write('1' + '\n')
                else:
                    label_file.write('0' + '\n')
                question_count += 1
        set_example_counts[i] += 1
        example_count += 1

print('n_train_paragraphs:', set_example_counts[0])
print('n_test_paragraphs:', set_example_counts[1])

print('n_train_questions:', example_count)
print('n_test_questions:', question_count)

# Save data info
info = {'n_train': set_example_counts[0],
        'n_test': set_example_counts[1]}

pickle.dump(info, info_file)


