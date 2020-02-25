#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os
import shutil


def build(opt):
  # get path to data directory
  dpath = os.path.join(opt['datapath'], 'probing', 'dstc8')

  # check if data had been previously built
  if not build_data.built(dpath):
    print('[building data: ' + dpath + ']')

    # mark the data as built
    build_data.mark_done(dpath)

# def build(opt):
#   dpath = os.path.join(opt['datapath'], 'probing', 'dsct8')
#   version = 'None'
#
#   if not build_data.built(dpath, version_string=version):
#     print('[building data: ' + dpath + ']')
#     if build_data.built(dpath):
#       # An older version exists, so remove these outdated files.
#       build_data.remove_dir(dpath)
#     build_data.make_dir(dpath)
#
#     # Download the data.
#     fname = 'dsct8_multi.zip'
#     url = 'https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/archive/master.zip'
#     build_data.download(url, dpath, fname)
#     build_data.untar(dpath, fname)
#     unzipped_dir = os.path.join(dpath, 'dstc8-schema-guided-dialogue-master')
#     for filepath in os.scandir(unzipped_dir):
#       if filepath.is_dir():
#         shutil.move(filepath.path, os.path.join(dpath, filepath.name))
#     shutil.rmtree(unzipped_dir)
#
#     # Mark the data as built.
#     build_data.mark_done(dpath, version_string=version)
#
#
# if __name__ == "__main__":
#   build({'datapath': 'parlai/data'})
