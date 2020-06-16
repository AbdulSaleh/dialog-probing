#!/usr/bin/env python3

"""Script iterates through the tasks specified and
generates vector representations to be used for probing the given model.

Examples
--------

.. code-block:: shell

  python probe_model.py -mf trained/dailydialog/scratch_seq2seq/seq2seq -t probing.tasks.trecquestion --probe encoder_state --batchsize 512
"""

import sys
import warnings
import pickle
from pathlib import Path
from parlai.core.params import ParlaiParser, print_announcements
from parlai.core.agents import create_agent
from parlai.core.logs import TensorboardLogger
from parlai.core.metrics import aggregate_task_reports
from parlai.core.worlds import create_task
from parlai.core.utils import TimeLogger

import random


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Evaluate a model')
    parser.add_pytorch_datateacher_args()
    # Get command line arguments

    # Probing command line arguments
    parser.add_argument('--probe', type=str, default=None,
                        choices=['word_embeddings', 'encoder_state', 'combined'],
                        help="Specify the type of representations to generate for probing. "
                             "See 'Probing Neural Dialog for Conversational Understanding' for more details.")

    parser.add_argument('-t', '--tasks', type=str, nargs='+',
                        required=True,
                        help='Usage: -t trecquestion or -t trecquestion wnli multiwoz'
                             '\nOnly compatible with names in probing/tasks')
    # Other command line arguments
    parser.add_argument('-ne', '--num-examples', type=int, default=-1)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=2)
    parser.add_argument(
        '-micro',
        '--aggregate-micro',
        type='bool',
        default=False,
        help='If multitasking, average metrics over the '
        'number of examples. If false, averages over the '
        'number of tasks.',
    )
    parser.add_argument(
        '-mcs',
        '--metrics',
        type=str,
        default='default',
        help='list of metrics to show/compute, e.g. all, default,'
        'or give a list split by , like '
        'ppl,f1,accuracy,hits@1,rouge,bleu'
        'the rouge metrics will be computed as rouge-1, rouge-2 and rouge-l',
    )
    TensorboardLogger.add_cmdline_args(parser)
    parser.set_defaults(datatype='valid')
    return parser


def _probe_single_world(opt, agent, task):
    print(
        '[ Evaluating task {} using datatype {}. ] '.format(
            task, opt.get('datatype', 'N/A')
        )
    )
    task_opt = opt.copy()  # copy opt since we're editing the task
    task_opt['task'] = task
    world = create_task(task_opt, agent)  # create worlds for tasks

    # set up logging
    log_every_n_secs = opt.get('log_every_n_secs', -1)
    if log_every_n_secs <= 0:
        log_every_n_secs = float('inf')
    log_time = TimeLogger()

    # max number of examples to evaluate
    max_cnt = opt['num_examples'] if opt['num_examples'] > 0 else float('inf')
    cnt = 0

    while not world.epoch_done() and cnt < max_cnt:
        cnt += opt.get('batchsize', 1)
        world.parley()
        if opt['display_examples']:
            # display examples
            print(world.display() + '\n~~')
        if log_time.time() > log_every_n_secs:
            report = world.report()
            _report = {'exs': cnt}
            text, report = log_time.log(cnt, world.num_examples(), _report)
            print(text)

    # Create save folder for probing outputs
    task_name = world.opt['task'].split('.')[-2]
    model_dir = Path(world.opt['model_file']).parent
    probing_dir = model_dir.joinpath('probing')
    probing_module_dir = probing_dir.joinpath(world.opt['probe'])
    task_dir = probing_module_dir.joinpath(task_name)
    save_path = task_dir.joinpath(task_name + '.pkl')
    if not probing_dir.exists():
        print("*" * 10, "\n", "*" * 10)
        print(f"Creating dir to save probing outputs at {probing_dir}")
        print("*" * 10, "\n", "*" * 10)
        probing_dir.mkdir()

    if not probing_module_dir.exists():
        print("*" * 10, "\n", "*" * 10)
        print(f"Creating dir to save {world.opt['probe']} probing outputs at {probing_module_dir}")
        print("*" * 10, "\n", "*" * 10)
        probing_module_dir.mkdir()

    if not task_dir.exists():
        print("*" * 10, "\n", "*" * 10)
        print(f"Creating dir to save {task_name} probing outputs at {task_dir}")
        print("*" * 10, "\n", "*" * 10)
        task_dir.mkdir()

    if save_path.exists():
        warnings.warn(f"\nVector representations for probing already exists at {save_path}!!\n"
                      "They will be overwritten.", RuntimeWarning)


    print("*" * 10, "\n", "*" * 10)
    print(f"Creating pickle file to save {task_name} probing outputs at {save_path}")
    print("*" * 10, "\n", "*" * 10)
    # Save probing outputs
    try:
        pickle.dump(world.agents[1].probing_outputs, open(save_path, 'wb'))
    except:
        pickle.dump(world.agents[1].probing_outputs, open(save_path, 'wb'), protocol=4)

    report = world.report()
    world.reset()
    return report


def probe_model(opt, print_parser=None):
    """Evaluates a model.

    :param opt: tells the evaluation function how to run
    :param bool print_parser: if provided, prints the options that are set within the
        model after loading the model
    :return: the final result of calling report()
    """
    random.seed(42)

    # load model and possibly print opt
    agent = create_agent(opt, requireModelExists=True)
    if print_parser:
        # show args after loading model
        print_parser.opt = agent.opt
        print_parser.print_args()

    tasks = opt['tasks']
    reports = []
    for task in tasks:
        task_report = _probe_single_world(opt, agent, task)
        reports.append(task_report)

    report = aggregate_task_reports(
        reports, tasks, micro=opt.get('aggregate_micro', True)
    )

    print('[ Finished probing tasks ]'.format(tasks))


if __name__ == '__main__':
    parser = setup_args()
    probe_model(parser.parse_args(print_args=False))
