# Download and build the data if it does not exist.

import os
import pickle
import json
import random
from pathlib import Path
import parlai.core.build_data as build_data


def create_probing_format(orig_dpath):
    random.seed(0)
    data_dir = orig_dpath.parent

    def concat_json_files(folderpath):
        combined_list = []
        for dirpath, _dirs, files in os.walk(folderpath):
            for filename in files:
                if 'dialogues_' in filename:
                    json_list = json.load(open(os.path.join(dirpath, filename)))
                    combined_list += json_list
        return combined_list

    train_path = orig_dpath.joinpath('train')
    test_path = orig_dpath.joinpath('dev')

    train_data = concat_json_files(train_path)
    test_data = concat_json_files(test_path)

    # Save files
    dialog_path = data_dir.joinpath('sgd.txt')
    label_path = data_dir.joinpath('labels.txt')
    info_path = data_dir.joinpath('info.pkl')

    dialog_file = open(dialog_path, 'w')
    label_file = open(label_path, 'w')
    info_file = open(info_path, 'wb')

    # Process data
    def process_split(data):
        count = 0
        for example in data:
            dialog = example['turns']
            turns = [turn['utterance'] for turn in dialog]

            chosen_turn = random.choice(range(0, len(dialog), 2))
            line = ('text:' + '\n'.join(turns[:chosen_turn + 1])
                    + '\tepisode_done:True\n')
            label = dialog[chosen_turn]['frames'][0]['state']['active_intent']

            dialog_file.write(line)
            label_file.write(label + '\n')
            count += 1
        return count

    train_data_len = process_split(train_data)
    test_data_len = process_split(test_data)

    # Save data info
    info = {'n_train': train_data_len,
            'n_test': test_data_len}

    pickle.dump(info, info_file)


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt['datapath'], 'probing', 'sgd')

    # check if data had been previously built
    if not build_data.built(dpath):
        print('[building data: ' + dpath + ']')

        build_data.make_dir(dpath)

        # Download the data.
        fname = 'sgd_orig.zip'
        url = 'https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/archive/master.zip'
        build_data.download(url, dpath, fname)
        build_data.unzip(dpath, fname)

        orig_dpath = os.path.join(dpath, 'sgd_orig')
        os.rename(os.path.join(dpath, 'dstc8-schema-guided-dialogue-master'), orig_dpath)

        create_probing_format(Path(orig_dpath))

        # mark the data as built
        build_data.mark_done(dpath)


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent.parent.parent.parent
    datapath = project_dir.joinpath('data')
    opt = {'datapath': str(datapath)}
    build(opt)