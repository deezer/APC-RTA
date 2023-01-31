import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Any
from .base import AggregatorBase
from src.rta.utils import CustomTransformerDecoderLayer, get_device


def generate_square_subsequent_mask(seq_len):
    # generates a mask that only allows previous tokens to be seen
  mask = (torch.triu(torch.ones(seq_len, seq_len)) - torch.eye((seq_len))).bool()
  return mask

class DecoderModel(AggregatorBase):
    def __init__(self,
                 embd_size,
                 use_position=True,
                 max_len=11,
                 n_layers=1,
                 n_head=8,
                 drop_p=0.1):
        super(DecoderModel, self).__init__()
        self.use_position = use_position

        if self.use_position:
          self.max_len = max_len
          self.position_embedding = nn.Embedding(max_len, embd_size)
          torch.nn.init.normal_(self.position_embedding.weight)

        #norm = torch.nn.LayerNorm(embd_size)
        decoder_layer = CustomTransformerDecoderLayer(d_model=embd_size, nhead=n_head, dim_feedforward =4096, activation='gelu', dropout=drop_p)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
    
    def forward(self, x, pad_mask):
      ## NEW_FORWARD (different from the original model)
      seq_len = x.shape[1]
      bs = x.shape[0]
      mask = generate_square_subsequent_mask(seq_len).to(get_device())
      X = x.transpose(1,0)
      if self.use_position:
        if seq_len < self.max_len:
          pos_embeddings = self.position_embedding(torch.Tensor(range(self.max_len-seq_len, self.max_len)).long().to(get_device())).unsqueeze(1).repeat((1,bs,1))
          X = X + pos_embeddings
        else:
          pos = torch.ones(seq_len)
          pos[-(self.max_len-1):] = torch.Tensor(range(1, self.max_len))
          pos_embeddings = self.position_embedding(pos.long().to(get_device())).unsqueeze(1).repeat((1,bs,1))
          X = X + pos_embeddings
      output = self.transformer_decoder(X, None, tgt_mask=mask, tgt_key_padding_mask = pad_mask).transpose(1,0)
      return output

    def aggregate(self, X, pad_mask):
      # necessary function for common RTA interface
      return self.forward(X, pad_mask)