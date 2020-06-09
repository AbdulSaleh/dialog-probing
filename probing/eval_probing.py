"""Loads embeddings and evaluates their performance on a given probing task
"""
import json
import pickle
import argparse
from time import time
from pathlib import Path

import numpy as np
import torch.optim as optim
from skorch import NeuralNetClassifier, NeuralNetBinaryClassifier
from skorch.callbacks import Checkpoint
from skorch.helper import predefined_split
from skorch.dataset import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from probing.mlp import MLP


def setup_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--task', type=str, required=True,
                        help='Usage: -t trecquestion\nOnly compatible with names in probing_tasks')
    parser.add_argument('-p', '--probing-module', type=str,
                        choices=['word_embeddings', 'encoder_state', 'combined'])

    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Usage: -m GloVe or -m trained\dailydialg\seq2seq\n'
                             'Model directory of embeddings to be probed.')

    parser.add_argument('-r', '--runs', type=int, default=1,
                        help='Number of times to train MLP with new random inits each time.\n'
                             'Required for creating confidence intervals')
    parser.add_argument('-ep', '--max_epochs', type=int, default=100)
    parser.add_argument('-bs', '--batch_size', type=int, default=128)

    parser.add_argument('-hidden', '--hidden_layer_dim', type=int, default=128)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-drop', '--dropout', type=float, default=0.5)
    parser.add_argument('-l2', '--l2-weight', type=float, default=0)

    return vars(parser.parse_args())


if __name__ == '__main__':
    start = time()

    opt = setup_args()
    task_name = opt['task']
    model = opt['model']
    runs = opt['runs']

    project_dir = Path(__file__).resolve().parent.parent

    # Load embeddings
    if model == 'GloVe':
        probing_dir = project_dir.joinpath('trained', 'GloVe', 'probing', task_name)
    else:
        module = opt['probing_module']
        probing_dir = project_dir.joinpath('trained', model, 'probing', module, task_name)

    embeddings_path = probing_dir.joinpath(task_name + '.pkl')
    print(f'Loading embeddings from {embeddings_path}')
    X = pickle.load(open(embeddings_path, 'rb'))
    X = X.astype(np.float32)

    # Load labels
    labels_path = project_dir.joinpath('data', 'probing', task_name, 'labels.txt')
    print(f'Loading labels from {labels_path}')
    labels = open(labels_path).readlines()
    y = np.unique(labels, return_inverse=True)[1]
    y = y.astype(np.float32) if len(set(y)) == 2 else y.astype(np.int64)

    # Load info
    info_path = project_dir.joinpath('data', 'probing', task_name, 'info.pkl')
    info = pickle.load(open(info_path, 'rb'))
    n_train = info['n_train']

    # Split data
    X_train = X[:n_train]
    y_train = y[:n_train]

    if 'n_dev' in info:
        # Use pre defined dev split
        n_dev = info['n_dev']
        X_val = X[n_train: n_train+n_dev]
        y_val = y[n_train: n_train+n_dev]
        X_test = X[n_train+n_dev:]
        y_test = y[n_train+n_dev:]
    else:
        # Create custom stratified dev split from train
        X_test = X[n_train:]
        y_test = y[n_train:]
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, stratify=y_train.astype(np.int64),
            random_state=1984
        )

    # Number of features, and number of classes
    # Only one output neuron in binary case
    input_dim = len(X[0])
    output_dim = 1 if len(set(y)) == 2 else len(set(y))
    Net = NeuralNetBinaryClassifier if len(set(y)) == 2 else NeuralNetClassifier

    # Compile test acc for conf intervals
    test_acc_list = []
    for run in range(1, runs+1):
        print('')
        print(15 * '*')
        print(f'Starting Run number: {run}')
        print(15 * '*')
        print('')

        save_dir = probing_dir.joinpath('runs', str(run))

        # Init skorch classifier
        print('Initializing MLP!')
        net = Net(
            # Architecture
            module=MLP,
            module__input_dim=input_dim,
            module__output_dim=output_dim,
            module__hidden_dim=opt['hidden_layer_dim'],
            optimizer__weight_decay=opt['l2_weight'],
            module__dropout=opt['dropout'],
            device='cuda',
            # Training
            max_epochs=opt['max_epochs'],
            batch_size=opt['batch_size'],
            callbacks=[Checkpoint(dirname=save_dir,
                                  f_params='params.pt',
                                  f_optimizer=None,
                                  f_history=None,
                                  monitor='valid_loss_best')],
            # train_split is validation data
            train_split=predefined_split(Dataset(X_val, y_val)),
            # Optimizer
            optimizer=optim.Adam,
            lr=opt['learning_rate'],
            # Data
            iterator_train__shuffle=True,
            verbose=(runs == 1)
        )

        net.fit(X_train, y_train)

        # Reload best valid loss checkpoint
        net.load_params(save_dir.joinpath('params.pt'))

        # Evaluate
        preds = net.predict(X_train)
        train_acc = accuracy_score(y_train, preds)

        preds = net.predict(X_val)
        val_acc = accuracy_score(y_val, preds)

        preds = net.predict(X_test)
        test_acc = accuracy_score(y_test, preds)

        # Save results
        results = {'test_acc': test_acc,
                   'val_acc': val_acc,
                   'train_acc': train_acc,
                   'model': model,
                   'task': task_name,
                   'architecture': str(net),
                   'history': net.history}

        print(f'Test acc: {test_acc}',
              f'Valid acc: {val_acc}',
              f'Train acc: {train_acc}')

        results_path = save_dir.joinpath('training_results.json')
        print(f'Saving training results to {results_path}')
        json.dump(results, open(results_path, 'w'))

        # Keep track for confidence interval
        test_acc_list.append(test_acc)

    # Calculate confidence intervals
    results_path = probing_dir.joinpath('results.json')
    mean = np.mean(test_acc_list)
    stddev = np.std(test_acc_list)
    stderr = stddev / runs
    results = {'mean': mean,
               'lower': mean - 2 * stderr,
               'upper': mean + 2 * stderr,
               'stddev': stddev,
               'stderr': stderr}
    print(results)
    json.dump(results, open(results_path, 'w'))

    print("Time elapsed: {:0.1f} minutes".format((time() - start)/60))
