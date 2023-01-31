import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base import AggregatorBase
from src.rta.utils import get_device

class GatedCNN(nn.Module):
    '''
        In : (N, sentence_len)
        Out: (N, sentence_len, embd_size)
    '''
    def __init__(self,
                 embd_size,
                 n_layers,
                 kernel_size,
                 out_chs,
                 res_block_count,
                 k_pool=3,
                 drop_p=0.0):
        super(GatedCNN, self).__init__()
        kernel = (kernel_size, embd_size)
        self.res_block_count = res_block_count
        self.n_layers = n_layers
        self.k_pool = k_pool
        self.padding_0 = nn.ConstantPad2d((0, 0, kernel_size - 1, 0), 0) # not using future songs to predict current songs
        self.conv_0 = nn.Conv2d(1, out_chs, kernel)
        self.b_0 = nn.Parameter(torch.randn(1, out_chs, 1))
        self.conv_gate_0 = nn.Conv2d(1, out_chs, kernel)
        self.c_0 = nn.Parameter(torch.randn(1, out_chs, 1))
        self.batch_norm_0 = nn.BatchNorm1d(out_chs)
        self.relu_0 = nn.ReLU()
        self.drop_layer_0 = nn.Dropout(p=drop_p)

        self.paddings = nn.ModuleList([nn.ConstantPad1d((kernel[0] - 1, 0), 0) for _ in range(n_layers)])  # not using future songs to predict current songs
        self.bottle_conv = nn.ModuleList([nn.Conv1d(out_chs, out_chs, kernel[0]) for _ in range(n_layers)])
        self.b = nn.ParameterList([nn.Parameter(torch.randn(1, out_chs, 1)) for _ in range(n_layers)])
        self.bottle_conv_gate = nn.ModuleList([nn.Conv1d(out_chs, out_chs, kernel[0]) for _ in range(n_layers)])
        self.c = nn.ParameterList([nn.Parameter(torch.randn(1, out_chs, 1)) for _ in range(n_layers)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(out_chs) for _ in range(n_layers)])
        self.relus = nn.ModuleList([nn.ReLU() for _ in range(n_layers)])
        self.drop_layers = nn.ModuleList([nn.Dropout(p=drop_p) for _ in range(n_layers)])

        self.fc = nn.Linear(out_chs * self.k_pool, embd_size) # the regression model outputs an embedding
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_in')

    def forward(self, x):

        # CNN
        l = x.shape[1]
        bs = x.shape[0]
        x = x.unsqueeze(1) # (bs, Cin, seq_len, embd_size), insert Channnel-In dim
        x = self.padding_0(x)

        # Conv2d
        #    Input : (bs, Cin,  Hin,  Win )
        #    Output: (bs, Cout, Hout, Wout)
        A = self.conv_0(x).squeeze(3)      # (bs, Cout, seq_len)
        A += self.b_0.repeat(1, 1, l)

        B = self.conv_gate_0(x).squeeze(3) # (bs, Cout, seq_len)
        B += self.c_0.repeat(1, 1, l)
        h = A * torch.sigmoid(B)    # (bs, Cout, seq_len)
        h = self.batch_norm_0(h)
        #h = self.relu_0(h)
        h = self.drop_layer_0(h)
        res_input = h # TODO this is h1 not h0

        for i in range(self.n_layers):
            h = self.paddings[i](h)
            A = self.bottle_conv[i](h) # + self.b[i].repeat(1, 1, seq_len)
            A += self.b[i]
            B = self.bottle_conv_gate[i](h) # + self.c[i].repeat(1, 1, seq_len)
            B += self.c[i]
            h = A * torch.sigmoid(B) # (bs, Cout, seq_len)
            h = self.batch_norms[i](h)
            #h = self.relus[i](h)
            h = self.drop_layers[i](h)
            if i % self.res_block_count == 0: # size of each residual block
                h += res_input
                res_input = h
        h =  torch.topk(h, k =self.k_pool, dim=2)[0]
        h = h.view(bs, -1) # (bs, Cout*seq_len)
        h = self.fc(h)
        return h

    def aggregate(self, X, pad_mask):
        l = X.shape[1]
        output = torch.zeros(X.shape).to(get_device())
        for i in range(1, l):
            o = self.forward(X[:, :i])
            output[:, i] = o
        return output

    def aggregate_single(self, X, pad_mask):
        # necessary function for common RTA interface
        return self.forward(X)