"""Loads embeddings and evaluates their performance on a given probing task
"""
from mlp import MLP
import numpy as np
import sklearn


def load_trec_labels():
  return np.random.rand(5000)  # TODO get actual labels


def load_trec_data():
  return np.random.rand(5000, 300)  # TODO get actual labels


def eval_trec():
  X = load_trec_data()
  labels = load_trec_labels()
  X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
      X, labels, test_size=0.2)
  model = MLP(300, len(labels.unique()))
  model.train(X_train, y_train, X_test, y_test)
