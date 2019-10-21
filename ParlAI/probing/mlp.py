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

  def train(self, data, epochs=1000, log_every_n=10, split_ratio=0.2):
    self.model.double()
    optimizer = optim.Adam(self.model.parameters())
    optimizer.zero_grad()
    self.loss_fn = nn.CrossEntropyLoss()
    num_examples = data.size()[0]
    data = data[torch.randperm(num_examples)]  # shuffle data
    # this is unused atm, TODO figure out how to reincorporate to avoid all the
    # gross last col indexing
    dataset = TrecData(data)
    split_index = int(data.size()[0] * split_ratio)
    test_data, train_data = data[:split_index, :], data[split_index:, :]
    print(f'Training on {num_examples - split_index}, Testing on {split_index}')
    for epoch in range(epochs):
      optimizer.zero_grad()
      trainloader = torch.utils.data.DataLoader(
          train_data, batch_size=64, shuffle=True)
      loss = 0
      for example in trainloader:
        x = example[:, :-1]
        y = example[:, -1].long()
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
      if epoch % 10 == 0:
        print(f'Epoch: {epoch}')
        print(f'Train loss: {loss}')
        self.evaluate(test_data)

  def evaluate(self, test_data):
    y_pred = self.model(test_data[:, :-1])
    loss = self.loss_fn(y_pred, test_data[:, -1].long())
    print(f'Test loss: {loss}')


class TrecData(Dataset):
  def __init__(self, data):
    super(TrecData, self).__init__()
    self.x = data[:, :-1]
    self.y = data[:, -1]
    assert self.x.shape[0] == self.y.shape[0]

  def __len__(self):
    return self.y.shape[0]

  def __getitem__(self, index):
    return self.x[index], self.y[index].long()
