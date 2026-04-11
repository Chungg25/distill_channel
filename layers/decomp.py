import torch
from torch import nn

from layers.ema import EMA
from layers.dema import DEMA
from layers.sma import moving_avg

class DECOMP(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, ma_type, alpha, beta, window_size):
        super(DECOMP, self).__init__()
        if ma_type == 'ema':
            self.ma = EMA(alpha)
        elif ma_type == 'dema':
            self.ma = DEMA(alpha)
        elif ma_type == 'sma':
            self.ma = moving_avg()
        

    def forward(self, x):
        moving_average = self.ma(x)
        res = x - moving_average
        return res, moving_average