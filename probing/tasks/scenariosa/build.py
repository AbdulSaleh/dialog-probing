import os
import pickle
import json
import random
import rarfile
from pathlib import Path
import parlai.core.build_data as build_data


def create_probing_format(orig_dpath):
    random.seed(0)
    data_dir = orig_dpath.parent

    text_path = data_dir.joinpath('scenariosa.txt')
    label_path = data_dir.joinpath('labels.txt')
    info_path = data_dir.joinpath('info.pkl')

    text_file = open(text_path, 'w', encoding='utf=8')
    label_file = open(label_path, 'w')
    info_file = open(info_path, 'wb')

    n_pos = 0
    n_neg = 0
    n_neu = 0

    dialogs_dir = orig_dpath.joinpath('InteractiveSentimentDataset')
    examples = []
    for filename in os.listdir(dialogs_dir):
        with open(dialogs_dir.joinpath(filename), 'r', encoding='cp1252') as f:
            example = f.readlines()
        utts = []
        sents = []
        for line in example:
            if line[0] in ['A', 'B']:
                stripline = line.rstrip()
                if stripline[-2] == '-':
                    utts.append(stripline[4:-3])
                    sents.append('neg')
                elif stripline[-1] == '0':
                    utts.append(stripline[4:-2])
                    sents.append('neu')
                else:
                    utts.append(stripline[4:-2])
                    sents.append('pos')
        if len(utts) == 0:
            continue

        selected_turn = random.choice(range(len(utts)))
        if sents[selected_turn] == 'neu':
            selected_turn = random.choice(range(len(utts)))

        text = 'text:' + '\n'.join(utts[:selected_turn + 1]) + '\tlabels:\tepisode_done:True\n'
        if sents[selected_turn] == 'pos':
            examples.append(['pos' + '\n', text])
            n_pos += 1
        elif sents[selected_turn] == 'neg':
            examples.append(['neg' + '\n', text])
            n_neg += 1
        elif sents[selected_turn] == 'neu':
            examples.append(['neu' + '\n', text])
            n_neu += 1

    random.shuffle(examples)
    n_train = int(0.85 * len(examples))
    n_test = len(examples) - n_train

    for ex in examples:
        text_file.write(ex[1])
        label_file.write(ex[0])

    # Save data info
    info = {'n_train': n_train,
            'n_test': n_test}

    pickle.dump(info, info_file)


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt['datapath'], 'probing', 'scenariosa')

    # check if data had been previously built
    if not build_data.built(dpath):
        print('[building data: ' + dpath + ']')

        build_data.make_dir(dpath)

        fname = 'scenriosa_orig.zip'
        url = 'https://www.dropbox.com/sh/hz7lvr2hniwg9uq/AACB2aNOqarKNUHQGj6AN0Uva?dl=1'
        build_data.download(url, dpath, fname)
        build_data.unzip(dpath, fname)

        orig_dpath = os.path.join(dpath, 'scenariosa_orig')
        os.rename(os.path.join(dpath, 'ScenarioSA'), orig_dpath)

        create_probing_format(Path(orig_dpath))

        # mark the data as built
        build_data.mark_done(dpath)

if __name__ == '__main__':
    opt = {'datapath': 'C:/Users/Abdul/Workspace/media_lab/dialog-probing/data/'}
    build(opt)