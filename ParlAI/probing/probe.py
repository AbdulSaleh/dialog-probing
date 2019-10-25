"""Loads embeddings and evaluates their performance on a given probing task
"""
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch.optim as optim
from skorch import NeuralNetClassifier
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

    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Usage: -m GloVe\nModel directory of embeddings to be probed.' 
                             'Assumes models saved to ParlAI\\trained')
    return vars(parser.parse_args())


if __name__ == '__main__':
    opt = setup_args()
    task_name = opt['task']
    model = opt['model']

    project_dir = Path(__file__).resolve().parent.parent

    # Load embeddings
    embeddings_path = project_dir.joinpath('trained', model, 'probing',
                                           task_name, task_name + '.pkl')
    print(f'Loading embeddings from {embeddings_path}')
    X = pickle.load(open(embeddings_path, 'rb'))
    X = X.astype(np.float32)

    # Load labels
    labels_path = project_dir.joinpath('data', 'probing', task_name, 'labels.txt')
    print(f'Loading labels from {labels_path}')
    labels = open(labels_path).readlines()
    y = np.unique(labels, return_inverse=True)[1]
    y = y.astype(np.int64)

    # Load info
    info_path = project_dir.joinpath('data', 'probing', task_name, 'info.pkl')
    info = pickle.load(open(info_path, 'rb'))
    n_train = info['n_train']

    # Split data
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=1984
    )

    # Number of features, and number of classes
    input_dim = len(X[0])
    output_dim = len(set(y))

    # Init skorch classifier
    print('Initializing MLP!')
    net = NeuralNetClassifier(
        # Architecture
        module=MLP,
        module__input_dim=input_dim,
        module__output_dim=output_dim,
        module__hidden_dim=128,
        module__dropout=0.5,
        device='cuda',
        # Training
        max_epochs=250,
        batch_size=128,
        #callbacks=[Checkpoint(monitor='valid_acc_best')],
        # train_split is validation data
        train_split=predefined_split(Dataset(X_val, y_val)),
        # Optimizer
        optimizer=optim.Adam,
        lr=0.001,
        # Data
        iterator_train__shuffle=True,
    )

    net.fit(X_train, y_train)

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

    results_path = project_dir.joinpath('trained', model, 'probing',
                                        task_name, 'training_results.pkl')
    print(f'Saving training results to {results_path}')
    pickle.dump(results, open(results_path, 'wb'))



