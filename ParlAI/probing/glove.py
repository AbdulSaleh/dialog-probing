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

    parser.add_argument('-t', '--task', type=str, required=True,
                        help='Usage: -t trecquestion\nOnly compatible with names in probing_tasks')

    return vars(parser.parse_args())


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

    task_name = opt['task']
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
        embeddings = encode_glove(questions, glove)
    elif task_name == 'wnli':
        data_dir = Path(project_dir, 'data', 'probing', 'wnli')
        train_path = data_dir.joinpath('train.tsv')
        dev_path = data_dir.joinpath('dev.tsv')

        train_data = csv.DictReader(open(train_path, 'r'), dialect='excel-tab')
        dev_data = csv.DictReader(open(dev_path, 'r'), dialect='excel-tab')
        data = chain(train_data, dev_data)

        examples = [example['sentence1'] + ' ' + example['sentence2'] for example in data]
        embeddings = encode_glove(examples, glove)

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
        embeddings = encode_glove(examples, glove)

    elif task_name == 'snips':
        labels = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic',
                  'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent']

        data_dir = Path(project_dir, 'data', 'probing', 'snips')
        question_path = data_dir.joinpath('snips.txt')
        label_path = data_dir.joinpath('labels.txt')
        info_path = data_dir.joinpath('info.pkl')

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
        embeddings = encode_glove(examples, glove)

    elif task_name == 'ushuffle_dailydialog':
        # This task is an exception as we load the shuffled and processed
        # probing data in ParlAI format instead of the raw data.
        data_dir = Path(project_dir, 'data', 'probing', 'ushuffle_dailydialog')
        data = open(data_dir.joinpath('shuffle.txt'))

        dialogs = []
        dialog = ''
        for turn in data:
            turn = turn.split('\t')
            text = turn[0][len('text:'):] + ' '
            dialog += text

            if len(turn) == 3:
                # implies dialog is over
                dialogs.append(dialog)
                dialog = ''

        examples = dialogs
        embeddings = encode_glove(examples, glove)

    else:
        raise NotImplementedError(f'Probing task: {task_name} not supported')

    pickle.dump(embeddings, save_file)
    print(f'Done embedding {task_name} data with GloVe')
