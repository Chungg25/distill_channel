import torch
import torch.nn as nn
import math

from layers.decomp import DECOMP
from layers.network import Network

from layers.revin import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # Parameters
        seq_len = configs.seq_len   # lookback window L
        pred_len = configs.pred_len # prediction length (96, 192, 336, 720)
        c_in = configs.enc_in       # input channels
        d_model = configs.d_model    # dimension of model
        period_len = configs.period_len  # period length
        nhead = configs.n_head      # number of attention heads
        groups = configs.group_channel    # number of group channels

        # Patching
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        # Normalization
        self.revin = configs.revin
        self.revin_layer = RevIN(c_in,affine=True,subtract_last=False)

        # Moving Average
        self.ma_type = configs.ma_type
        alpha = configs.alpha       # smoothing factor for EMA (Exponential Moving Average)
        beta = configs.beta         # smoothing factor for DEMA (Double Exponential Moving Average)

        dropout = configs.dropout
        num_layers = configs.num_layers

        self.decomp = DECOMP(self.ma_type, alpha, beta, period_len)

        self.net = Network(seq_len, pred_len, patch_len, stride, padding_patch, dropout, d_model, nhead, num_layers, groups)

    def forward(self, x):
        # x: [Batch, Input, Channel]
        # Normalization
        if self.revin:
            x = self.revin_layer(x, 'norm')


        if self.ma_type == 'reg':   # If no decomposition, directly pass the input to the network
            x = self.net(x, x)
        if self.ma_type == 'sma':  
            resid_init, trend_init = self.decomp(x)
            x = self.net(resid_init, trend_init)
        if self.ma_type == 'dema':
            seasonal_init, trend_init = self.decomp(x)
            x = self.net(seasonal_init, trend_init)

        if self.revin:
            x = self.revin_layer(x, 'denorm')

        return x