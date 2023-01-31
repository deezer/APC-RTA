import torch
import torch.nn as nn
from .base import AggregatorBase
from src.rta.utils import get_device

class GRUNet(AggregatorBase):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(get_device())
        return hidden
    
    def aggregate(self, X, pad_mask):
      bs = X.shape[0]
      H = self.init_hidden(bs)
      l = X.shape[1]
      output = torch.zeros(X.shape).to(get_device())
      for i in range(1, l):
        o, H = self.forward(X[:, :i], H)
        output[:,i] = o
      return output
    
    def aggregate_single(self, X, pad_mask):
      # necessary function for common RTA interface
      bs = X.shape[0]
      H = self.init_hidden(bs)
      return self.forward(X, H)[0]