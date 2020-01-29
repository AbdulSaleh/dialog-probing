#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os
import json
from .nlpaugmentation import augment_dataset


def build(opt):
    dpath = os.path.join(opt['datapath'], 'dailydialog_augmented')
    version = 'None'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        remote_fname = 'dailydialog.tar.gz'
        local_fname = 'dailydialog_augmented.tar.gz'
        url = 'http://parl.ai/downloads/dailydialog/' + remote_fname
        build_data.download(url, dpath, local_fname)
        build_data.untar(dpath, local_fname)

        fpath = os.path.join(dpath, 'train.json')
        with open(fpath, mode='r+') as f:
            data = []
            print('augmenting dailydialog')
            for line in f:
                dialog = list(map(lambda obj_dialog: obj_dialog['text'], json.loads(line)['dialogue']))
                data.append(dialog)
            augmented_data = augment_dataset(data)
            def package_dialog(dialog):
                packaged_utterances = list(map(lambda utterance: {'emotion': "", 'act': "", 'text': utterance}, dialog))
                return {"fold": "train", "topic": "", "dialogue": packaged_utterances}
            augmented_data = list(map(package_dialog, augmented_data))
            f.seek(0)
            json.dump(augmented_data, f)
            
        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
