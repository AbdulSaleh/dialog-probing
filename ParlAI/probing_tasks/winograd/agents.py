#!/usr/bin/env python3

from parlai.core.teachers import ParlAIDialogTeacher
from .build import build

import copy
import os


def _path(opt):
  # Build the data if it doesn't exist.
  build(opt)
  return os.path.join(opt['datapath'], 'probing', 'winograd', 'winograd.txt')


class DefaultTeacher(ParlAIDialogTeacher):
  def __init__(self, opt, shared=None):
    opt = copy.deepcopy(opt)
    opt['parlaidialogteacher_datafile'] = _path(opt)
    super().__init__(opt, shared)
