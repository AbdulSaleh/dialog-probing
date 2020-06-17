"""Generate bag of vectors representation for a given probing task."""
import os
import argparse
import pickle
import json
import csv
import zipfile
from pathlib import Path
from itertools import chain
import urllib.request
from importlib import import_module
import numpy as np
from probing.utils import load_glove, encode_glove


def setup_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--tasks', type=str, nargs='+',
                        required=True,
                        help='Usage: -t trecquestion or -t trecquestion wnli multinli'
                        '\nOnly compatible with names in probing_tasks')

    parser.add_argument('--dict-path', type=str,
                        help='Include path to a model dict to restrict the vocabulary '
                             'size used by GloVe for comparibility.')
    return vars(parser.parse_args())


def process_task(task_name, save_dir, glove, dict):
    task_dir = save_dir.joinpath(task_name)

    if not task_dir.exists():
        print('*' * 10, '\n', '*' * 10)
        print(f'Creating dir to save {task_name} probing outputs at {task_dir}')
        print('*' * 10, '\n', '*' * 10)
        task_dir.mkdir(parents=True)

    # Create save file
    save_path = task_dir.joinpath(task_name + '.pkl')
    save_file = open(save_path, 'wb')

    # Check if task data exists, if not then build task
    data_dir = Path(project_dir, 'data', 'probing', task_name)
    if not data_dir.exists():
        build = import_module('.'.join(['probing', 'tasks', task_name, 'build']))
        build.build({'datapath': Path(__file__).parent.parent.joinpath('data')})


    # Load and process data depending on task
    print(f'Loading {task_name} data!')
    if task_name == 'trecquestion':
        data_dir = Path(project_dir, 'data', 'probing', 'trecquestion', 'trecquestion_orig')
        train_path = data_dir.joinpath('train.txt')
        test_path = data_dir.joinpath('test.txt')

        train = open(train_path, 'r', encoding='ISO-8859-1').readlines()
        test = open(test_path, 'r', encoding='ISO-8859-1').readlines()
        data = train + test

        questions = [line[line.index(' ') + 1:].rstrip() for line in data]
        embeddings = encode_glove(questions, glove, dict=dict)

    elif task_name == 'wnli':
        data_dir = Path(project_dir, 'data', 'probing', 'wnli', 'wnli_orig')
        train_path = data_dir.joinpath('train.tsv')
        dev_path = data_dir.joinpath('dev.tsv')

        train_data = csv.DictReader(open(train_path, 'r'), dialect='excel-tab')
        dev_data = csv.DictReader(open(dev_path, 'r'), dialect='excel-tab')
        data = list(chain(train_data, dev_data))

        sent1 = [example['sentence1'] for example in data]
        sent2 = [example['sentence2'] for example in data]
        sent1 = encode_glove(sent1, glove, dict=dict)
        sent2 = encode_glove(sent2, glove, dict=dict)
        embeddings = np.hstack((sent1, sent2))

    elif task_name == 'snips':
        labels = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic',
                  'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent']

        data_dir = Path(project_dir, 'data', 'probing', 'snips', 'snips_orig')
        examples = []
        # Process train
        for label in labels:
            f_name = data_dir.joinpath(label, 'train.json')
            with open(f_name, encoding='latin-1') as f:
                dataset = json.load(f)
                for example in dataset[label]:
                    text = ''.join([t['text'] for t in example['data']])
                    examples.append(text)
        # Process test
        for label in labels:
            f_name = data_dir.joinpath(label, 'test.json')
            with open(f_name, encoding='latin-1') as f:
                dataset = json.load(f)
                for example in dataset[label]:
                    text = ''.join([t['text'] for t in example['data']])
                    examples.append(text)
        embeddings = encode_glove(examples, glove, dict=dict)

    elif task_name == 'dailydialog_topic':
        data_dir = Path(project_dir, 'data', 'probing', 'dailydialog_topic')
        data = open(data_dir.joinpath('dialogs.txt'))

        history = []
        current = []
        for line in data:
            line = line.rstrip('\n')
            turn = line.split('\t')[0]
            if turn.startswith('text:'):
                # start a new episode
                episode = []
                turn = turn[len('text:'):]
            episode.append(turn)

            if 'episode_done:True' in line:
                history.append(' '.join(episode))
                current.append(turn)

        history = encode_glove(history, glove, dict=dict)
        current = encode_glove(current, glove, dict=dict)
        embeddings = np.hstack((history, current))

    elif task_name == 'multiwoz':
        data_dir = Path(project_dir, 'data', 'probing', 'multiwoz')
        data = open(data_dir.joinpath('multiwoz.txt'))

        history = []
        current = []
        for line in data:
            line = line.rstrip('\n')
            turn = line.split('\t')[0]
            if turn.startswith('text:'):
                # start a new episode
                episode = []
                turn = turn[len('text:'):]
            episode.append(turn)

            if 'episode_done:True' in line:
                history.append(' '.join(episode))
                current.append(turn)

        history = encode_glove(history, glove, dict=dict)
        current = encode_glove(current, glove, dict=dict)
        embeddings = np.hstack((history, current))

    elif task_name == 'dialoguenli':
        data_dir = Path(project_dir, 'data', 'probing', 'dialoguenli')
        data = open(data_dir.joinpath('dialoguenli.txt'))

        history = []
        current = []
        for line in data:
            line = line.rstrip('\n')
            turn = line.split('\t')[0]
            if turn.startswith('text:'):
                # start a new episode
                episode = []
                turn = turn[len('text:'):]
            episode.append(turn)

            if 'episode_done:True' in line:
                history.append(' '.join(episode))
                current.append(turn)

        history = encode_glove(history, glove, dict=dict)
        current = encode_glove(current, glove, dict=dict)
        embeddings = np.hstack((history, current))

    elif task_name == 'sgd':
        data_dir = Path(project_dir, 'data', 'probing', 'sgd')
        data = open(data_dir.joinpath('sgd.txt'))

        history = []
        current = []
        for line in data:
            line = line.rstrip('\n')
            turn = line.split('\t')[0]
            if turn.startswith('text:'):
                # start a new episode
                episode = []
                turn = turn[len('text:'):]
            episode.append(turn)

            if 'episode_done:True' in line:
                history.append(' '.join(episode))
                current.append(turn)

        history = encode_glove(history, glove, dict=dict)
        current = encode_glove(current, glove, dict=dict)
        embeddings = np.hstack((history, current))

    elif task_name == 'scenariosa':
        data_dir = Path(project_dir, 'data', 'probing', 'scenariosa')
        data = open(data_dir.joinpath('scenariosa.txt'))

        history = []
        current = []
        for line in data:
            line = line.rstrip('\n')
            turn = line.split('\t')[0]
            if turn.startswith('text:'):
                # start a new episode
                episode = []
                turn = turn[len('text:'):]
            episode.append(turn)

            if 'episode_done:True' in line:
                history.append(' '.join(episode))
                current.append(turn)

        history = encode_glove(history, glove, dict=dict)
        current = encode_glove(current, glove, dict=dict)
        embeddings = np.hstack((history, current))

    else:
        raise NotImplementedError(f'Probing task: {task_name} not supported')

    pickle.dump(embeddings, save_file)
    print(f'Done embedding {task_name} data with GloVe')


if __name__ == "__main__":
    opt = setup_args()

    project_dir = Path(__file__).resolve().parent.parent
    glove_dir = project_dir.joinpath('data', 'models', 'glove_vectors')

    # Load GloVe
    glove_path = glove_dir.joinpath('glove.840B.300d.txt')
    if not glove_path.exists():
        try:
            glove_dir.mkdir(parents=True)
        except FileExistsError:
            pass
        print('Downloading GloVe embeddings! This might take a few minutes...')
        zip_path = glove_dir.joinpath('glove.840B.300d.zip')
        url = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
        urllib.request.urlretrieve(url, zip_path)
        print('Done downloading GloVe!')

        print('Unzipping GloVe embeddings! Just a few more minutes...')
        with zipfile.ZipFile(zip_path) as f:
            f.extractall(path=glove_dir)
        print('Done unzipping Glove!')
        os.remove(zip_path)

    glove = load_glove(glove_path)

    # Create save dir for embeddings
    save_dir = project_dir.joinpath('trained', 'GloVe', 'probing')
    if not save_dir.exists():
        print('*' * 10, '\n', '*' * 10)
        print(f'Creating dir to save GloVe bag of vectors embeddings at {save_dir}')
        print('*' * 10, '\n', '*' * 10)
        save_dir.mkdir(parents=True)

    # Check for dict
    dict_path = Path(opt['dict_path'])
    if dict_path.exists():
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
        process_task(task_name, save_dir, glove, dict)
