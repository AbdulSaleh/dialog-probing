import os
import pickle
import json
from pathlib import Path
import random
from collections import Counter
import parlai.core.build_data as build_data


def create_probing_format(orig_dpath):
    random.seed(0)
    data_dir = orig_dpath.parent

    text_path = data_dir.joinpath('multiwoz.txt')
    label_path = data_dir.joinpath('labels.txt')
    info_path = data_dir.joinpath('info.pkl')

    text_file = open(text_path, 'w')
    label_file = open(label_path, 'w')
    info_file = open(info_path, 'wb')

    with open(orig_dpath.joinpath('data.json'), encoding='latin-1') as json_file:
        dataset = json.load(json_file)
        test_codes = set(open(orig_dpath.joinpath('testListFile.txt')).read().splitlines())
        valid_codes = set(open(orig_dpath.joinpath('valListFile.txt')).read().splitlines())
        train_codes = set(dataset.keys()) - valid_codes - test_codes
        splits = {'train': train_codes, 'dev': valid_codes, 'test': test_codes}

        info = {'n_train': 0, 'n_dev': 0, 'n_test': 0}
        dialogs = []
        labels = []
        for split, codes in splits.items():
            for c in codes:
                example = dataset[c]
                try:
                    turns = [turn['text'] for turn in example['log']]
                    dialog_acts = [[act_type for act_type in turn['dialog_act']] for turn in example['log']]
                except KeyError:
                    continue

                possible_turns = []
                for i in range(0, min(float('inf'), len(turns)), 2):
                    if len(set(dialog_acts[i])) == 1:
                        possible_turns.append(i)

                if len(possible_turns) == 0:
                    continue

                chosen_turn = random.choice(possible_turns)
                dialogs.append('text:' + '\n'.join(turns[:chosen_turn + 1]) +
                               '\tepisode_done:True\n')
                labels.append(dialog_acts[chosen_turn][0])
                info['n_' + split] += 1

    for dialog, label in zip(dialogs, labels):
        text_file.write(dialog)
        label_file.write(label + '\n')

    pickle.dump(info, info_file)


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt['datapath'], 'probing', 'multiwoz')

    # check if data had been previously built
    if not build_data.built(dpath):
        print('[building data: ' + dpath + ']')

        build_data.make_dir(dpath)

        # Download the data
        fname = 'multiwoz_orig.zip'
        url = 'https://github.com/budzianowski/multiwoz/raw/master/data/MultiWOZ_2.1.zip'
        build_data.download(url, dpath, fname)
        build_data.unzip(dpath, fname)

        orig_dpath = os.path.join(dpath, 'multiwoz_orig')
        os.rename(os.path.join(dpath, 'MultiWOZ_2.1'), orig_dpath)

        create_probing_format(Path(orig_dpath))

        # mark the data as built
        build_data.mark_done(dpath)


if __name__ == '__main__':
    opt = {'datapath': 'C:/Users/Abdul/Workspace/media_lab/dialog-probing/data/'}
    build(opt)