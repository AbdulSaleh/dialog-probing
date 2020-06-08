# Download and build the data if it does not exist.

import os
import json
import pickle
from pathlib import Path
import parlai.core.build_data as build_data


def download_data(dpath):
    labels = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic',
              'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent']

    for label in labels:
        build_data.make_dir(os.path.join(dpath, label))

        train_url = 'https://raw.githubusercontent.com/snipsco/nlu-benchmark/'\
                    'master/2017-06-custom-intent-engines/{}/'\
                    'train_{}_full.json'.format(label, label)
        test_url = 'https://raw.githubusercontent.com/snipsco/nlu-benchmark/'\
                   'master/2017-06-custom-intent-engines/{}/'\
                   'validate_{}.json'.format(label, label)

        build_data.download(train_url, os.path.join(dpath, label), 'train.json')
        build_data.download(test_url, os.path.join(dpath, label), 'test.json')


def create_probing_format(orig_dpath):
    labels = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic',
              'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent']

    data_dir = orig_dpath.parent

    question_path = data_dir.joinpath('snips.txt')
    label_path = data_dir.joinpath('labels.txt')
    info_path = data_dir.joinpath('info.pkl')

    question_file = open(question_path, 'w', encoding='utf-8')
    label_file = open(label_path, 'w')
    info_file = open(info_path, 'wb')

    n_train = 0
    n_test = 0

    # Process train data
    for label in labels:
        f_name = orig_dpath.joinpath(label, 'train.json')
        with open(f_name, encoding="utf-8", errors='ignore') as f:
            dataset = json.load(f)
            for example in dataset[label]:
                text = ''.join([t['text'] for t in example['data']])
                text = text.replace('\n', '').replace('\t', '')
                question_file.write(('text:' + text + '\tepisode_done:True\n'))
                label_file.write(label + '\n')
                n_train += 1

    # Process test data
    for label in labels:
        f_name = orig_dpath.joinpath(label, 'test.json')
        with open(f_name, encoding="utf-8", errors='ignore') as f:
            dataset = json.load(f)
            for example in dataset[label]:
                text = (''.join([t['text'] for t in example['data']]))
                text = text.replace('\n', '').replace('\t', '')
                question_file.write('text:' + text + '\tepisode_done:True\n')
                label_file.write(label + '\n')
                n_test += 1

    # Save data info
    info = {'n_train': n_train,
            'n_test': n_test}

    pickle.dump(info, info_file)


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt['datapath'], 'probing', 'snips')

    # check if data had been previously built
    if not build_data.built(dpath):
        print('[building data: ' + dpath + ']')

        build_data.make_dir(dpath)
        orig_dpath = os.path.join(dpath, 'snips_orig')

        download_data(orig_dpath)
        create_probing_format(Path(orig_dpath))

        # mark the data as built
        build_data.mark_done(dpath)


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent.parent.parent.parent
    datapath = project_dir.joinpath('data')
    opt = {'datapath': str(datapath)}
    build(opt)