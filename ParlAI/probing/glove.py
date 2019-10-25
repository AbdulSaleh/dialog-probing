"""Generate bag of vectors representation for a given [robing task
"""
import pickle
import argparse
from pathlib import Path

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
    else:
        raise NotImplementedError(f'Probing task: {task_name} not supported')

    pickle.dump(embeddings, save_file)
    print(f'Done embedding {task_name} data with GloVe')





