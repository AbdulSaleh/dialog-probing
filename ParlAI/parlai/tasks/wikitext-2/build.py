"""Script to download and process wikitext-2 in ParlAI format."""

import os
from zipfile import ZipFile
from pathlib import Path

from tqdm import tqdm
from nltk import sent_tokenize

import parlai.core.build_data as build_data


SILENCE_TOKEN = '__SILENCE__'


def create_dialog(conv):
    conv_len = len(conv) if len(conv) % 2 == 0 else len(conv)-1
    if not conv_len: return
    _conv = conv[:conv_len]

    processed = ''
    pairs = list(zip(_conv[::2], _conv[1::2]))
    for p in pairs:
        line = f'text:{p[0]}\tlabels:{p[1]}\n'
        processed += line
    # undo new line
    processed = processed.strip('\n')
    processed += '\tepisode_done:True\n'
    return processed


def ParlAI_format(path):
    data_paths = path.glob('*')
    # Iterate over train, test, and valid data
    for file in data_paths:
        split = file.stem.split('.')[1]
        print(f'***\nProcessing {split} wikitext-2 split!\n***')

        output = open(path.joinpath(split + '.txt'), 'w')
        data = open(file, encoding='utf-8')
        for line in tqdm(data):
            # Clean weird symbols
            line = line.encode("ascii", errors="ignore").decode()
            sents = sent_tokenize(line)
            if len(sents) < 2:
                continue

            # Create one perspective.
            # For example, the conversation x1, y1, x2, y2, x3 becomes:
            # x1 y1
            # x2 y2
            output.write(create_dialog(sents))

            # Create other perspective.
            # __SILENCE__ x1
            # y1 x2
            # y2 x3
            output.write(create_dialog([SILENCE_TOKEN] + sents))


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt['datapath'], 'wikitext-2')

    # check if data had been previously built
    if not build_data.built(dpath):
        build_data.make_dir(dpath)

        # Download data
        url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
        fname = 'wikitext-2.zip'
        build_data.download(url, dpath, fname)

        # Extract data
        zip_path = Path(dpath).joinpath(fname)
        with ZipFile(zip_path) as zip:
            for info in zip.infolist():
                if info.filename[-1] == '/':
                    continue
                info.filename = os.path.basename(info.filename)
                zip.extract(info, dpath)
        os.remove(zip_path)

        # Convert to parlai format
        ParlAI_format(Path(dpath))
        build_data.mark_done(dpath)
