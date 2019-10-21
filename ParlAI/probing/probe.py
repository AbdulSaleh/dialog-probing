"""Loads embeddings and evaluates their performance on a given probing task
"""
from mlp import MLP
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import pickle


def load_trec_labels():
  return torch.empty(5452, dtype=torch.double).random_(
      2).unsqueeze(1)  # TODO get actual labels


def load_trec_data():
  data = pickle.load(
      open(
          '../trained/dailydialog/small_default_transformer/probing/trecquestion/trecquestion.pkl',
          'rb'))
  return torch.from_numpy(data)


def eval_trec():
  X = load_trec_data()
  labels = load_trec_labels()
  # X_train, X_test, y_train, y_test = torch.utils.data.random_split(
  #     X.numpy(), labels.numpy(), test_size=0.2)
  num_labels = len(np.unique(labels))
  print(f'num labels: {num_labels}')
  model = MLP(300, num_labels)
  print(f'data shape: {torch.cat((X, labels), dim=1).size()}')
  model.train(torch.cat((X, labels), dim=1))


if __name__ == "__main__":
  # TODO set cmd line args for different datasets and probing tasks
  eval_trec()
