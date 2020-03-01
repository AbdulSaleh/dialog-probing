"""Script for summarizing and presenting probing results
"""

import os
import json
from pathlib import Path
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--latex', action='store_true')
args = parser.parse_args()


project_dir = Path(__file__).resolve().parent.parent
datasets = ['dailydialog', 'wikitext-103']
modules = ['encoder_embeddings', 'encoder_state', 'encoder_embeddings_state']
models = ['scratch_seq2seq', 'scratch_seq2seq_within',
          'scratch_seq2seq_att', 'scratch_seq2seq_att_within',
          'scratch_transformer', 'scratch_transformer_within',
          'large_seq2seq', 'large_seq2seq_att', 'large_transformer',
          'finetuned_seq2seq', 'finetuned_seq2seq_within',
          'finetuned_seq2seq_att', 'finetuned_seq2seq_att_within',
          'finetuned_transformer', 'finetuned_transformer_within']

tasks = ['trecquestion', 'dialoguenli', 'multi_woz', 'dstc8', 'snips',
         'wnli', 'scenariosa', 'topic_dailydialog', 'ushuffle_dailydialog']
tasks_dict = {'trecquestion': 'TREC',
              'dialoguenli': 'DNLI',
              'multi_woz': 'MWOZ',
              'dstc8': 'DSTC8',
              'snips': 'SNIPS',
              'wnli': 'WNLI',
              'scenariosa': 'SSA',
              'topic_dailydialog': 'Topic DD',
              'ushuffle_dailydialog': 'Shuffle DD'}
full_results = {}
# tasks = set()
for dataset in datasets:
    model_dirs = []
    for m in project_dir.joinpath('trained', dataset).glob("*"):
        if m.is_dir():
            model_dirs.append(m)

    for model_dir in model_dirs:
        model = model_dir.stem
        if model not in models:
            continue

        full_results[model] = {}

        module_dirs = []
        for m in model_dir.joinpath('probing').glob("*"):
            if m.is_dir():
                module_dirs.append(m)
        for module_dir in module_dirs:
            module = module_dir.stem
            if module not in modules:
                continue

            full_results[model][module] = {}

            task_dirs = []
            for m in module_dir.glob("*"):
                if m.is_dir():
                    task_dirs.append(m)

            for task_dir in task_dirs:
                task = task_dir.stem
                results = json.load(open(task_dir.joinpath('results.json')))
                full_results[model][module][task] = results['mean']
                # tasks.add(task)

# tasks = list(tasks)
longest_model = max([len(m) for m in models])
header = "Model" + " " * (longest_model - len("Model")) + "\t"
for task in tasks:
    header += tasks_dict[task] + "\t"

for module in modules:
    print()
    print(10 * '*')
    print(10 * '*')
    print(module)
    print(10 * '*')
    print(10 * '*')
    print()
    print(header)
    for model in models:
        row = model + " " * (len(header.split('\t')[0]) - len(model)) + "\t"
        for task in tasks:
            acc = full_results[model][module][task]
            row = row + '{:0.1f} , '.format(acc*100)

        print(row)

print('#' * 3)
print('#' * 3)
print('#' * 3)

if args.latex:
    for module in modules:
        print()
        print(10 * '*')
        print(10 * '*')
        print(module)
        print(10 * '*')
        print(10 * '*')
        print()
        print(header)
        for model in models:
            row = model + " " * (len(header.split('\t')[0]) - len(model)) + "\t" + '&'
            for task in tasks:
                acc = full_results[model][module][task]
                row = row + '{:0.1f} & '.format(acc*100)
            print(row[:-2] + " \\\\ ")

