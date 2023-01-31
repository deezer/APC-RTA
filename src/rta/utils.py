import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Any

def padded_avg(X, pad):
    # Computes the average without taking into account indices of padded tokens
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

class CustomTransformerDecoderLayer(nn.TransformerDecoderLayer):
    # A modification of the eponymous pytorch class that doesn't require to use a memory as input.
  def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt