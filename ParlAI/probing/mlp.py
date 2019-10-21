"""Implements MLP trained on embeddings for probing tasks
"""
import torch
from torch.utils.data import Dataset
from torch import nn
import torch.optim as optim


class MLP():
  def __init__(
          self,
          d_in,
          d_out,
          hidden_dim=100):
    self.model = torch.nn.Sequential(
        torch.nn.Linear(d_in, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, d_out),
    )

  def train(self, train_x, train_y, test_x, test_y, epochs=15, log_every_n=100):
    optimizer = optim.Adam(self.model.parameters())
    optimizer.zero_grad()
    self.loss_fn = nn.CrossEntropyLoss()
    traindata = TrecData(train_x, train_y)

    for epoch in epochs:
      optimizer.zero_grad()
      trainloader = torch.utils.data.DataLoader(
          traindata, batch_size=64, shuffle=True)
      loss = 0
      for step, (x, y) in enumerate(trainloader):
        y_pred = self.model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
      print(f'Train loss: {loss}')
      self.evaluate(test_x, test_y)

  def evaluate(self, test_x, test_y):
    y_pred = self.model(test_x)
    loss = self.loss_fn(y_pred, test_y)
    print(f'Test loss: {loss}')


class TrecData(Dataset):
  def __init__(self, x, y):
    super(TrecData, self).__init__()
    assert x.shape[0] == y.shape[0]
    self.x = x
    self.y = y

  def __len__(self):
    return self.y.shape[0]

  def __getitem__(self, index):
    return self.x[index], self.y[index]
