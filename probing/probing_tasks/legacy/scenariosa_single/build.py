import os
import parlai.core.build_data as build_data


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt['datapath'], 'probing', 'scenariosa')

    # check if data had been previously built
    if not build_data.built(dpath):
        print('[building data: ' + dpath + ']')

        # mark the data as built
        build_data.mark_done(dpath)

# def build(opt):
#     dpath = os.path.join(opt['datapath'], 'probing', 'snips')
#
#     if not build_data.built(dpath, version_string=version):
#
#         print('[building data: ' + dpath + ']')
#
#         if build_data.built(dpath):
#             # An older version exists, so remove these outdated files.
#             build_data.remove_dir(dpath)
#         build_data.make_dir(dpath)
#
#         # Download the data.
#         fname = 'benchmark_data.json'
#         url = 'https://raw.githubusercontent.com/snipsco/nlu-benchmark/master/2016-12-built-in-intents/' + fname
#         build_data.download(url, dpath, fname)
#
#         # Mark the data as built.
#         build_data.mark_done(dpath, version_string=version)
