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

calendar_labels = ['event', 'time', 'date', 'party', 'room', 'agenda']
weather_labels = ['location', 'temperature', 'weather_attribute']
poi_labels = ['poi', 'traffic-info', 'poi_type', 'address', 'distance']
labels = calendar_labels + weather_labels + poi_labels + ['none']

project_dir = Path(__file__).resolve().parent.parent.parent.parent
data_dir = Path(project_dir, 'data', 'probing', 'sdd')

question_path = data_dir.joinpath('sdd.txt')
label_path = data_dir.joinpath('labels.txt')
info_path = data_dir.joinpath('info.pkl')

question_file = open(question_path, 'w')
label_file = open(label_path, 'w')
info_file = open(info_path, 'wb')

set_example_counts = [0, 0]

for i, dset_type in enumerate(['train', 'dev', 'test']):
    with open(str(data_dir.joinpath('kvret_' + dset_type + '_public.json'))) as json_file:
        dataset = json.load(json_file)
    for example in dataset:
        all_turns = []
        label_turns = []
        for j, turn in enumerate(example['dialogue']):
            if j % 2 == 0:
                if turn['turn'] != 'driver':
                    label_turns.append(None)  # hacky but this is to fix the dataset being stupid
                    break
                all_turns.append(turn['data']['utterance'])
            else:
                if turn['turn'] != 'assistant':
                    label_turns.append(None)  # hacky but this is to fix the dataset being stupid
                    break
                all_turns.append(turn['data']['utterance'])
                requested_here = []
                for label_type in turn['data']['requested']:
                    if turn['data']['requested'][label_type] == True:
                        requested_here.append(label_type)
                        if requested_here[-1] == 'date':  # merge date and time to reduce conflicts
                            requested_here[-1] = 'time'
                        elif requested_here[-1] in ['poi_type', 'address']:  # merge poi and poit type to reduce conflicts
                            requested_here[-1] = 'poi'
                        elif requested_here[-1] == 'distance':  # merge poi and poit type to reduce conflicts
                            requested_here[-1] = 'traffic'
                        elif requested_here[-1] in ['party', 'room', 'agenda']:
                            requested_here[-1] = 'event'
                if len(requested_here) == 0:
                    label_turns.append('none')
                else:
                    label_turns.append(random.choice(requested_here))
                    label_turns.append(random.choice(requested_here))  # do this two times
        if any([at == '' for at in all_turns]):  # this datset is full of trash
            continue
        if any([lt == None for lt in label_turns]):  # this datset is full of trash
            continue
        if len(all_turns) != len(label_turns):  # fuck this dataset
            continue
        for j in range(1, len(all_turns)):
            question_file.write('text:' + '\n'.join(all_turns[:j]) + '\tlabels:' + '\tepisode_done:True\n')
            label_file.write(label_turns[j-1] + '\n')
            if i == 0 or i == 1:
                set_example_counts[0] += 1
            else:
                set_example_counts[1] += 1

        # user_turns = []
        # assistant_turns = []
        # label_turns = []
        # for j, turn in enumerate(example['dialogue']):
        #     if j % 2 == 0:
        #         if turn['turn'] != 'driver':
        #             label_turns.append(None)  # hacky but this is to fix the dataset being stupid
        #             break
        #         user_turns.append(turn['data']['utterance'])
        #     else:
        #         if turn['turn'] != 'assistant':
        #             label_turns.append(None)  # hacky but this is to fix the dataset being stupid
        #             break
        #         assistant_turns.append(turn['data']['utterance'])
        #         requested_here = []
        #         for label_type in turn['data']['requested']:
        #             if turn['data']['requested'][label_type] == True:
        #                 requested_here.append(label_type)
        #                 if requested_here[-1] == 'date':  # merge date and time to reduce conflicts
        #                     requested_here[-1] = 'time'
        #                 elif requested_here[-1] in ['poi_type', 'address']:  # merge poi and poit type to reduce conflicts
        #                     requested_here[-1] = 'poi'
        #                 elif requested_here[-1] == 'distance':  # merge poi and poit type to reduce conflicts
        #                     requested_here[-1] = 'traffic'
        #                 elif requested_here[-1] in ['party', 'room', 'agenda']:
        #                     requested_here[-1] = 'event'
        #         if len(requested_here) == 0:
        #             label_turns.append('none')
        #         else:
        #             label_turns.append(random.choice(requested_here))
        # if any([ut == '' for ut in user_turns]):  # this datset is full of trash
        #     continue
        # if any([at == '' for at in assistant_turns]):  # omfg
        #     continue
        # if len(user_turns) != len(assistant_turns) or len(user_turns) != len(label_turns):  # fuck this dataset
        #     continue
        # n_turns = len(user_turns)
        # example_count += 1
        # turn_count += n_turns
        # if i == 0:
        #     set_example_counts[0] += 1
        # else:
        #     set_example_counts[1] += 1
        # for k in range(n_turns):
        #     question_file.write('text:' + user_turns[k] + '\tlabels:' + assistant_turns[k])
        #     if k != n_turns - 1:
        #         question_file.write('\n')
        #     else:
        #         question_file.write('\tepisode_done:True\n')
        #     label_file.write(label_turns[k] + '\n')

print('n_train:', set_example_counts[0])
print('n_test', set_example_counts[1])

# Save data info
info = {'n_train': set_example_counts[0],
        'n_test': set_example_counts[1]}

pickle.dump(info, info_file)

