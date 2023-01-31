import torch
import torch.nn as nn
from src.rta.utils import get_device

class AggregatorBase(nn.Module):
  """Base class for aggregator functions."""
  def __init__(self):
    super(AggregatorBase, self).__init__()
    return

  def aggregate(self, X, pad_mask):
    # Performs cumulative aggregation for all sub-playlists in the playlist
    l = X.shape[1]
    divs = torch.arange(1, l+1).unsqueeze(1).to(get_device())
    output = torch.cumsum(X, dim=1) / divs
    return output

  def aggregate_single(self, X, pad_mask):
    # Necessary function for common RTA interface
    # Performs aggregation for the playlist
    return self.aggregate(X, pad_mask)[:,-1,...]