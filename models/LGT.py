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
        d_model_channel = configs.d_model_channel    # dimension of model
        d_model_spectral = configs.d_model_spectral
        period_len = configs.period_len  # period length
        nhead = configs.n_head      # number of attention heads
        expand = configs.expand

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
        num_groups = configs.num_groups

        self.decomp = DECOMP(self.ma_type, alpha, beta, period_len)

        self.net = Network(seq_len, pred_len, patch_len, stride, padding_patch, dropout, d_model_channel, d_model_spectral, nhead, num_layers, expand, num_groups)

    def forward(self, x, current_epoch=0, max_epochs=1):
        # x: [Batch, Input, Channel]
        # Normalization
        if self.revin:
            x = self.revin_layer(x, 'norm')
        aux_losses = {}
        if self.ma_type == 'reg':   
            x, aux_losses = self.net(x, x, current_epoch, max_epochs)
        if self.ma_type == 'sma':  
            resid_init, trend_init = self.decomp(x)
            x, aux_losses = self.net(resid_init, trend_init, current_epoch, max_epochs)
        if self.ma_type == 'dema':
            seasonal_init, trend_init = self.decomp(x)
            x, aux_losses = self.net(seasonal_init, trend_init, current_epoch, max_epochs)
        if self.revin:
            x = self.revin_layer(x, 'denorm')
        return x, aux_losses