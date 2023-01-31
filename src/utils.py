import torch
import numpy as np
def padded_avg(X, pad):
  X_s = X.shape
  n = torch.sum(pad, dim=1).unsqueeze(1)
  while (len(pad.shape) < len(X_s)):
    pad = pad.unsqueeze(-1)
  return torch.sum(X * pad, dim=1) / n

def mean_FM(E):
  return torch.mean(torch.stack(E, dim=1), dim=1)

def get_device():
  if torch.cuda.is_available():
    dev = torch.device('cuda')
  else:
    dev = torch.device('cpu')
  return dev

def array_mapping(source, target):
  return  np.array([np.where(target== pid)[0][0] for pid in source])