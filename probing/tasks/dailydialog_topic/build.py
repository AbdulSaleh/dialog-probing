import os
from pathlib import Path
import pickle
import json
import random
import parlai.core.build_data as build_data
from sklearn.model_selection import train_test_split


def create_probing_format(orig_dpath):
    random.seed(1984)
    data_dir = orig_dpath.parent

    test_path = orig_dpath.joinpath('test.json')
    test = list(map(json.loads, open(test_path, 'r').readlines()))
    data = test

    # Save files
    dialogs_path = data_dir.joinpath('dialogs.txt')
    label_path = data_dir.joinpath('labels.txt')
    info_path = data_dir.joinpath('info.pkl')

    dialogs_file = open(dialogs_path, 'w')
    label_file = open(label_path, 'w')
    info_file = open(info_path, 'wb')

    labels = [ep['topic'] for ep in data]
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.15, random_state=1984, stratify=labels
    )

    data = X_train + X_test

    # Process data
    for episode in data:
        dialog = episode['dialogue']
        text = '\n'.join([turn['text'] for turn in dialog])

        dialogs_file.write(
            f'text:{text}\tepisode_done:True\n'
        )

        topic = episode['topic']
        label_file.write(topic + '\n')

    # Save data info
    n_train = int(len(data) * 0.85)
    info = {'n_train': n_train,
            'n_test': len(data) - n_train}

    pickle.dump(info, info_file)


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt['datapath'], 'probing', 'dailydialog_topic')

    # check if data had been previously built
    if not build_data.built(dpath):
        print('[building data: ' + dpath + ']')

        build_data.make_dir(dpath)

        fname = 'dailydialog.zip'
        url = 'https://www.dropbox.com/s/eplorb4w3mhgihp/dailydialog.zip?dl=1'
        build_data.download(url, dpath, fname)
        build_data.unzip(dpath, fname)

        orig_dpath = os.path.join(dpath, 'dailydialog_orig')
        os.rename(os.path.join(dpath, 'dailydialog'), orig_dpath)

        create_probing_format(Path(orig_dpath))

        # mark the data as built
        build_data.mark_done(dpath)


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent.parent.parent.parent
    datapath = project_dir.joinpath('data')
    opt = {'datapath': str(datapath)}
    build(opt)