"""
Note that this script should be manually run by the user and not through ParlAI
to process the data, and hence _parse.py instead of parse.py.
"""
from pathlib import Path
import pickle
import json
import random
random.seed(0)

labels = ['Attraction-Inform', 'Restaurant-Request', 'general-bye', 'general-thank', 'Restaurant-Inform',
          'Hotel-Request', 'Police-Request', 'Police-Inform', 'Hospital-Request', 'Train-Request', 'general-greet',
          'Taxi-Inform', 'Taxi-Request', 'Hotel-Inform', 'Hospital-Inform', 'Train-Inform', 'Attraction-Request']

labels_dict = {label: 0 for label in labels}
examples_dict = {label: [] for label in labels}
master_train = []
master_test = []

project_dir = Path(__file__).resolve().parent.parent.parent.parent
data_dir = Path(project_dir, 'ParlAI', 'data', 'probing', 'multi_woz2')

question_path = data_dir.joinpath('multi_woz2.txt')
label_path = data_dir.joinpath('labels.txt')
info_path = data_dir.joinpath('info.pkl')

question_file = open(question_path, 'w')
label_file = open(label_path, 'w')
info_file = open(info_path, 'wb')

with open(str(data_dir.joinpath('multi_woz2_data.json')), encoding='latin-1') as json_file:
    dataset = json.load(json_file)
    for ep in dataset:
        example = dataset[ep]

        try:
            all_turns = [turn['text'] for turn in example['log']]
            all_dialog_acts = [[act_type for act_type in turn['dialog_act']] for turn in example['log']]
            user_dialog_acts = [all_dialog_acts[i] for i in range(0, len(all_dialog_acts), 2)]
        except KeyError:
            continue

        possible_turns = []
        for i in range(len(user_dialog_acts)):
            if len(user_dialog_acts[i]) == 1:
                possible_turns.append(i)

        if len(possible_turns) == 0:
            continue

        chosen_turn = random.choice(possible_turns)
        chosen_text = 'text:' + '\n'.join(all_turns[:chosen_turn*2+1]) + '\tlabels:' + '\tepisode_done:True\n'
        dialog_act = user_dialog_acts[chosen_turn][0]

        labels_dict[dialog_act] += 1
        examples_dict[dialog_act].append([dialog_act, chosen_text])

n_train = 0
n_test = 0

for lab in labels:
    random.shuffle(examples_dict[lab])
    lab_n_examples = len(examples_dict[lab])
    lab_n_train = int(0.85 * lab_n_examples)
    master_train += examples_dict[lab][:lab_n_train]
    master_test += examples_dict[lab][lab_n_train:]

random.shuffle(master_test)
random.shuffle(master_train)

for dialog in master_train:
    label = dialog[0]
    for i in range(1, len(dialog)):
        question_file.write(dialog[i])
        label_file.write(label + '\n')
        n_train += 1

for dialog in master_test:
    label = dialog[0]
    for i in range(1, len(dialog)):
        question_file.write(dialog[i])
        label_file.write(label + '\n')
        n_test += 1


print('n_train:', n_train)
print('n_test', n_test)

print(labels_dict)

# Save data info
info = {'n_train': n_train,
        'n_test': n_test}

pickle.dump(info, info_file)


