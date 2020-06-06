# Download and build the data if it does not exist.

import os
import csv
import pickle
import zipfile
from pathlib import Path
import parlai.core.build_data as build_data


def create_probing_format(orig_dpath):
    # Load raw data
    train_path = orig_dpath.joinpath('train.tsv')
    dev_path = orig_dpath.joinpath('dev.tsv')

    train_data = csv.DictReader(open(train_path, 'r'), dialect='excel-tab')
    dev_data = csv.DictReader(open(dev_path, 'r'), dialect='excel-tab')

    # Save files
    question_path = orig_dpath.parent.joinpath('wnli.txt')
    label_path = orig_dpath.parent.joinpath('labels.txt')
    info_path = orig_dpath.parent.joinpath('info.pkl')

    question_file = open(question_path, 'w')
    label_file = open(label_path, 'w')
    info_file = open(info_path, 'wb')

    # Process data
    def process_split(data):
        count = 0
        for example in data:
            label_file.write(str(example['label']) + '\n')
            question_file.write(
                f"text:{example['sentence1']}\n{example['sentence2']}\tepisode_done:True\n")
            count += 1
        return count

    train_data_len = process_split(train_data)
    dev_data_len = process_split(dev_data)

    # Save data info
    info = {'n_train': train_data_len,
            'n_test': dev_data_len}

    pickle.dump(info, info_file)


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt['datapath'], 'probing', 'wnli')

    # check if data had been previously built
    if not build_data.built(dpath):
        print('[building data: ' + dpath + ']')

        build_data.make_dir(dpath)

        # Download data
        fname = 'wnli_orig.zip'
        url = 'https://firebasestorage.googleapis.com/' \
              'v0/b/mtl-sentence-representations.appspot.com/' \
              'o/data%2FWNLI.zip?alt=media&token=068ad0a0-ded7-4bd7-99a5-5e00222e0faf'
        build_data.download(url, dpath, fname)

        build_data.unzip(dpath, fname)

        orig_dpath = os.path.join(dpath, 'wnli_orig')
        os.rename(os.path.join(dpath, 'WNLI'), orig_dpath)

        # Process the data
        create_probing_format(Path(orig_dpath))

        # mark the data as built
        build_data.mark_done(dpath)
