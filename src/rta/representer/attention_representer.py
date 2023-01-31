import torch
import torch.nn as nn
from .base_representer import BaseEmbeddingRepresenter
from src.rta.utils import CustomTransformerDecoderLayer

class AttentionFMRepresenter(BaseEmbeddingRepresenter):
    def __init__(self,
              data_manager,
              emb_dim,
              n_att_heads = 1,
              n_att_layers = 1,
              dropout_att=0.2):
      super().__init__(data_manager, emb_dim)
      decoder_layer = CustomTransformerDecoderLayer(d_model=emb_dim, nhead=n_att_heads, dim_feedforward=1024, activation='gelu', dropout=dropout_att, batch_first =True)
      self.attention = nn.TransformerDecoder(decoder_layer, num_layers=n_att_layers)
      self.linear = nn.Linear(5, 1)

    def forward(self, x):
        albs = self.song_album[x]
        arts = self.song_artist[x]
        pops = self.data_manager.get_pop_bucket(self.song_pop[x]).int()
        durs = self.data_manager.get_duration_bucket(self.song_dur[x]).int()

        X = self.embedding(x)
        X_albs = self.album_embedding(albs)
        X_arts = self.artist_embedding(arts)
        X_pops = self.pop_embedding(pops)
        X_durs = self.dur_embedding(durs)

        return self.attention_FM((X, X_albs, X_arts, X_pops, X_durs))

    def attention_FM(self, E):
      E_stack = torch.stack(E, dim=-2)
      origin_shape = E_stack.shape
      E_extended = E_stack.view((-1, origin_shape[-2], origin_shape[-1]))
      E_attention = self.attention (E_extended, None)
      E_agg = torch.transpose(self.linear(torch.transpose(E_attention, 2, 1)), 2, 1).view((-1, origin_shape[1], origin_shape[-1]))
      return E_agg