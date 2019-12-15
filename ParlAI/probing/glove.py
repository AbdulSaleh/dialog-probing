"""Generate bag of vectors representation for a given probing task."""
import pickle
import json
import argparse
from pathlib import Path
import csv
from itertools import chain
from probing.utils import load_glove, encode_glove


def setup_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--tasks', type=str, nargs='+',
                        required=True,
                        help='Usage: -t trecquestion or -t trecquestion wnli multinli'
                        '\nOnly compatible with names in probing_tasks')

    return vars(parser.parse_args())


def process_task(task_name, save_dir, glove):
    task_dir = save_dir.joinpath(task_name)

    if not task_dir.exists():
        print('*' * 10, '\n', '*' * 10)
        print(f'Creating dir to save {task_name} probing outputs at {task_dir}')
        print('*' * 10, '\n', '*' * 10)
        task_dir.mkdir(parents=True)

    # Create save file
    save_path = task_dir.joinpath(task_name + '.pkl')
    save_file = open(save_path, 'wb')

    # Load and process data depending on task
    print(f'Loading {task_name} data!')
    if task_name == 'trecquestion':
        data_dir = Path(project_dir, 'data', 'probing', 'trecquestion')
        train_path = data_dir.joinpath('train_5500.label')
        test_path = data_dir.joinpath('TREC_10.label')

        train = open(train_path, 'r', encoding='ISO-8859-1').readlines()
        test = open(test_path, 'r', encoding='ISO-8859-1').readlines()
        data = train + test

        questions = [line[line.index(' ') + 1:].rstrip() for line in data]
        embeddings = encode_glove(questions, glove, dict=dict)

    elif task_name == 'wnli':
        data_dir = Path(project_dir, 'data', 'probing', 'wnli')
        train_path = data_dir.joinpath('train.tsv')
        dev_path = data_dir.joinpath('dev.tsv')

        train_data = csv.DictReader(open(train_path, 'r'), dialect='excel-tab')
        dev_data = csv.DictReader(open(dev_path, 'r'), dialect='excel-tab')
        data = chain(train_data, dev_data)

        examples = [example['sentence1'] + ' ' + example['sentence2'] for example in data]
        embeddings = encode_glove(examples, glove, dict=dict)

    elif task_name == 'multinli':
        MULTINLI_PREMISE_KEY = 'sentence1'
        MULTINLI_HYPO_KEY = 'sentence2'

        data_dir = Path(project_dir, 'data', 'probing', 'multinli')
        train_path = data_dir.joinpath('multinli_1.0_train.jsonl')
        dev_path = data_dir.joinpath('multinli_1.0_dev_matched.jsonl')
        test_path = data_dir.joinpath('multinli_1.0_dev_mismatched.jsonl')

        train = [json.loads(l) for l in open(train_path)]
        dev = [json.loads(l) for l in open(dev_path)]
        test = [json.loads(l) for l in open(test_path)]
        data = train + dev + test

        examples = []
        for line in data:
            premise = line[MULTINLI_PREMISE_KEY]
            hypo = line[MULTINLI_HYPO_KEY]
            examples.append(premise + ' ' + hypo)
        embeddings = encode_glove(examples, glove, dict=dict)

    elif task_name == 'snips':
        labels = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic',
                  'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent']

        data_dir = Path(project_dir, 'data', 'probing', 'snips')
        examples = []
        # Process train
        for label in labels:
            f_name = data_dir.joinpath(label, 'train_' + label + '_full.json')
            with open(f_name, encoding='latin-1') as f:
                dataset = json.load(f)
                for example in dataset[label]:
                    text = ''.join([t['text'] for t in example['data']])
                    examples.append(text)
        # Process test
        for label in labels:
            f_name = data_dir.joinpath(label, 'validate_' + label + '.json')
            with open(f_name, encoding='latin-1') as f:
                dataset = json.load(f)
                for example in dataset[label]:
                    text = ''.join([t['text'] for t in example['data']])
                    examples.append(text)
        embeddings = encode_glove(examples, glove, dict=dict)

    elif task_name == 'ushuffle_dailydialog':
        # raise NotImplemented('Bug needs to be fixed')
        # This task is an exception as we load the shuffled and processed
        # probing data in ParlAI format instead of the raw data.
        data_dir = Path(project_dir, 'data', 'probing', 'ushuffle_dailydialog')
        data = open(data_dir.joinpath('shuffled.txt'))

        dialogs = []
        for line in data:
            line = line.rstrip('\n')
            turn = line.split('\t')[0]
            if turn.startswith('text:'):
                # start a new episode
                episode = []
                turn = turn[len('text:'):]
            episode.append(turn)

            if 'episode_done:True' in line:
                dialogs.append(' '.join(episode))

        embeddings = encode_glove(dialogs, glove, dict=dict)

    elif task_name == 'act_dailydialog':
        data_dir = Path(project_dir, 'data', 'probing', 'act_dailydialog')
        data = open(data_dir.joinpath('dialogs.txt'))

        dialogs = []
        for line in data:
            line = line.rstrip('\n')
            turn = line.split('\t')[0]
            if turn.startswith('text:'):
                # start a new episode
                episode = []
                turn = turn[len('text:'):]
            episode.append(turn)

            if 'episode_done:True' in line:
                dialogs.append(' '.join(episode))

        embeddings = encode_glove(dialogs, glove, dict=dict)

    elif task_name == 'sentiment_dailydialog':
        data_dir = Path(project_dir, 'data', 'probing', 'sentiment_dailydialog')
        data = open(data_dir.joinpath('dialogs.txt'))

        dialogs = []
        for line in data:
            line = line.rstrip('\n')
            turn = line.split('\t')[0]
            if turn.startswith('text:'):
                # start a new episode
                episode = []
                turn = turn[len('text:'):]
            episode.append(turn)

            if 'episode_done:True' in line:
                dialogs.append(' '.join(episode))

        embeddings = encode_glove(dialogs, glove, dict=dict)

    elif task_name == 'topic_dailydialog':
        data_dir = Path(project_dir, 'data', 'probing', 'topic_dailydialog')
        data = open(data_dir.joinpath('dialogs.txt'))

        dialogs = []
        for line in data:
            line = line.rstrip('\n')
            turn = line.split('\t')[0]
            if turn.startswith('text:'):
                # start a new episode
                episode = []
                turn = turn[len('text:'):]
            episode.append(turn)

            if 'episode_done:True' in line:
                dialogs.append(' '.join(episode))

        embeddings = encode_glove(dialogs, glove, dict=dict)

    elif task_name == 'multi_woz':
        data_dir = Path(project_dir, 'data', 'probing', 'multi_woz')
        data = open(data_dir.joinpath('multi_woz.txt'))

        examples = []
        for line in data:
            line = line.rstrip('\n')
            turn = line.split('\t')[0]
            if turn.startswith('text:'):
                # start a new episode
                episode = []
                turn = turn[len('text:'):]
            episode.append(turn)

            if 'episode_done:True' in line:
                examples.append(' '.join(episode))

    else:
        raise NotImplementedError(f'Probing task: {task_name} not supported')

    pickle.dump(embeddings, save_file)
    print(f'Done embedding {task_name} data with GloVe')


if __name__ == "__main__":
    opt = setup_args()

    project_dir = Path(__file__).resolve().parent.parent

    # Load GloVe
    glove_path = project_dir.joinpath('data', 'models', 'glove_vectors', 'glove.840B.300d.txt')
    glove = load_glove(glove_path)

    # Create save dir for embeddings
    save_dir = project_dir.joinpath('trained', 'GloVe', 'probing')
    if not save_dir.exists():
        print('*' * 10, '\n', '*' * 10)
        print(f'Creating dir to save GloVe bag of vectors embeddings at {save_dir}')
        print('*' * 10, '\n', '*' * 10)
        save_dir.mkdir(parents=True)

    # Check for dict
    dict_exsists = list(save_dir.glob('*.dict'))
    if dict_exsists:
        dict_path = list(save_dir.glob('*.dict'))[0]
        lines = open(dict_path).readlines()
        dict = set(line.split('\t')[0] for line in lines)
        print('#' * 10, '\n', '#' * 10)
        print(f'Found dict at {dict_path}')
        print('#' * 10, '\n', '#' * 10)
    else:
        dict = None
        print('#' * 10, '\n', '#' * 10)
        print('No dict found!! Using entire GloVe vocab.')
        print('#' * 10, '\n', '#' * 10)

    task_names = opt['tasks']
    for task_name in task_names:
        process_task(task_name, save_dir, glove)
