import os
import parlai.core.build_data as build_data


def build(opt):
  # get path to data directory
  dpath = os.path.join(opt['datapath'], 'probing', 'winograd')

  # check if data had been previously built
  if not build_data.built(dpath):
    print('[building data: ' + dpath + ']')

    # mark the data as built
    build_data.mark_done(dpath)
