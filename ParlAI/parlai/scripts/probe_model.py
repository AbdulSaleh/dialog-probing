#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Script for probing model on task specified.


Examples
--------

.. code-block:: shell

  python probe_model.py -t "trecquestion" -mf "transformer"

"""

from parlai.core.params import ParlaiParser, print_announcements
from parlai.core.agents import create_agent
from parlai.core.logs import TensorboardLogger
from parlai.core.metrics import aggregate_task_reports
from parlai.core.worlds import create_task
from parlai.core.utils import TimeLogger

import random



def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Probe a model')
    parser.add_pytorch_datateacher_args()
    # Get command line arguments
    parser.add_argument('-probe', type=bool, default=True)
    parser.add_argument('-ne', '--num-examples', type=int, default=-1)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=2)

    TensorboardLogger.add_cmdline_args(parser)
    # TODO: datatype='probe'
    # Maybe keep datatype valid because it has nice properties. We want inference mode.
    # Need a flag to return representations though...
    parser.set_defaults(datatype='valid')
    return parser


def _probe_single_world(opt, agent, task):
    print(
        '[ Probing task {} using datatype {}. ] '.format(
            task, opt.get('datatype', 'N/A')
        )
    )
    task_opt = opt.copy()  # copy opt since we're editing the task
    task_opt['task'] = task
    world = create_task(task_opt, agent)  # create worlds for tasks

    # set up logging
    # log_every_n_secs = opt.get('log_every_n_secs', -1)
    # if log_every_n_secs <= 0:
    #     log_every_n_secs = float('inf')
    # log_time = TimeLogger()

    # # max number of examples to evaluate
    # max_cnt = opt['num_examples'] if opt['num_examples'] > 0 else float('inf')
    cnt = 0

    while not world.epoch_done():
        cnt += opt.get('batchsize', 1)
        world.parley()
        if opt['display_examples']:
            # display examples
            print(world.display() + '\n~~')
        # if log_time.time() > log_every_n_secs:
        #     report = world.report()
        #     text, report = log_time.log(report['exs'], world.num_examples(), report)
        #     print(text)

    report = world.report()
    world.reset()
    return report







