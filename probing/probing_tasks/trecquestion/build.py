import os
import pickle
from pathlib import Path
import parlai.core.build_data as build_data


def create_probing_format(orig_dpath):
    train_path = orig_dpath.joinpath('train.txt')
    test_path = orig_dpath.joinpath('test.txt')

    train = open(train_path, 'r', encoding='ISO-8859-1').readlines()
    test = open(test_path, 'r', encoding='ISO-8859-1').readlines()
    data = train + test

    # Save files
    question_path = orig_dpath.parent.joinpath('trecquestion.txt')
    label_path = orig_dpath.parent.joinpath('labels.txt')
    info_path = orig_dpath.parent.joinpath('info.pkl')

    question_file = open(question_path, 'w')
    label_file = open(label_path, 'w')
    info_file = open(info_path, 'wb')

    # Process data
    for line in data:
        label = line[:line.index(' ')].strip().split(':')[0]
        question = line[line.index(' ') + 1:].rstrip()

        label_file.write(label + '\n')
        question_file.write('text:' + question + ' \tepisode_done:True\n')

    # Save data info
    info = {'n_train': len(train),
            'n_test': len(test)}

    pickle.dump(info, info_file)


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt['datapath'], 'probing', 'trecquestion')

    # check if data had been previously built
    if not build_data.built(dpath):
        print('[building data: ' + dpath + ']')

        build_data.make_dir(dpath)

        train_fname = 'train.txt'
        train_url = 'https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label'
        test_fname = 'test.txt'
        test_url = 'https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label'
        orig_dpath = os.path.join(dpath, 'trecquestion_orig')
        build_data.make_dir(orig_dpath)
        build_data.download(train_url, orig_dpath, train_fname)
        build_data.download(test_url, orig_dpath, test_fname)

        create_probing_format(Path(orig_dpath))

        # mark the data as built
        build_data.mark_done(dpath)


