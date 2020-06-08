import os
from pathlib import Path
import pickle
import json
import random
import parlai.core.build_data as build_data


def create_probing_format(orig_dpath):
    random.seed(0)
    data_dir = orig_dpath.parent

    text_path = data_dir.joinpath('dialoguenli.txt')
    label_path = data_dir.joinpath('labels.txt')
    info_path = data_dir.joinpath('info.pkl')

    text_file = open(text_path, 'w')
    label_file = open(label_path, 'w')
    info_file = open(info_path, 'wb')

    example_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
    examples = []

    word_map = {'am': 'are', 'was': 'were', 'i': 'you', 'me': 'you', 'my': 'your', 'mine': 'yours'}

    info = {'n_train': 0, 'n_dev': 0, 'n_test': 0}
    for i, split in enumerate(['train', 'dev', 'test']):
        data_file = open(str(orig_dpath.joinpath('dnli', 'dialogue_nli_' + split + '.jsonl')))
        data = json.load(data_file)

        for example in data:
            s1 = example['sentence1']
            s2 = example['sentence2']
            s2_words = s2.split()
            s2 = " ".join([w if w not in word_map else word_map[w] for w in s2_words])
            s = s1 + '\n' + s2
            text = 'text:' + s + '\tepisode_done:True\n'
            label = example['label']
            examples.append([label, text])
            example_counts[label] += 1

            info['n_' + split] += 1

    for ex in examples:
        text_file.write(ex[1])
        label_file.write(ex[0] + '\n')

    pickle.dump(info, info_file)


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt['datapath'], 'probing', 'dialoguenli')

    # check if data had been previously built
    if not build_data.built(dpath):
        print('[building data: ' + dpath + ']')

        build_data.make_dir(dpath)

        fname = 'dnli.zip'
        url = 'https://www.dropbox.com/s/h65c5i8o7q9d2kk/dnli.zip?dl=1'
        build_data.download(url, dpath, fname)
        build_data.unzip(dpath, fname)

        orig_dpath = os.path.join(dpath, 'dialoguenli_orig')
        os.rename(os.path.join(dpath, 'dnli'), orig_dpath)

        create_probing_format(Path(orig_dpath))

        # mark the data as built
        build_data.mark_done(dpath)


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent.parent.parent.parent
    datapath = project_dir.joinpath('data')
    opt = {'datapath': str(datapath)}
    build(opt)