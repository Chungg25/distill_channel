import torch
import torch.nn as nn

from layers.decomp import DECOMP
from layers.network import Network
from layers.revin import RevIN

# canonical calendar used by the model, identical for every data set and sampling rate:
#   (hour of day, day of week, day of month, day of year), each in [-0.5, 0.5]
MARK_DIM = 4


def steps_per_hour(freq):
    """Sampling rate in steps per hour from a pandas-style frequency ('h', '15min', '10min', 't', 'd', ...)."""
    f = str(freq).strip().lower()
    num = ''.join(ch for ch in f if ch.isdigit())
    n = float(num) if num else 1.0
    unit = ''.join(ch for ch in f if ch.isalpha())
    if unit in ('min', 't'):
        return 60.0 / n
    if unit == 's':
        return 3600.0 / n
    if unit == 'h':
        return 1.0 / n
    if unit in ('d', 'b'):
        return 1.0 / (24.0 * n)
    if unit == 'w':
        return 1.0 / (168.0 * n)
    return 1.0


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len

        # per-channel RevIN calibration: the only channel-specific parameters (re-initialised on a
        # new domain by load_transfer)
        self.revin = configs.revin
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)

        self.ma_type = configs.ma_type
        self.decomp = DECOMP(self.ma_type, configs.alpha, configs.beta, configs.period_len)

        # calendar of the target timestamps (known in advance, never target values)
        self.raw_marks = getattr(configs, 'embed', 'timeF') != 'timeF'
        self.no_marks = str(getattr(configs, 'data', '')) == 'Solar'      # loader gives no timestamps

        self.net = Network(configs.seq_len, configs.pred_len, configs.patch_len, configs.stride,
                           configs.padding_patch, dropout=configs.dropout, d_model=configs.d_model,
                           nhead=configs.n_head, num_layers=configs.num_layers,
                           num_groups=configs.num_groups,
                           steps_per_day=24.0 * steps_per_hour(getattr(configs, 'freq', 'h')),
                           mark_dim=MARK_DIM)

    # ------------------------------------------------------------------------------------------
    def _future_marks(self, y_mark):
        """Loader calendar of the label + horizon timestamps -> canonical [B, H, 4], or None."""
        if y_mark is None or self.no_marks or y_mark.shape[1] < self.pred_len:
            return None
        m = y_mark[:, -self.pred_len:]
        if self.raw_marks:                                   # [month, day, weekday, hour(, minute)]
            lo = m.new_tensor([1., 1., 0., 0.])
            hi = m.new_tensor([12., 31., 6., 23.])
            r = (m[..., :4] - lo) / (hi - lo) - 0.5
            return r[..., [3, 2, 1, 0]]                      # hour, weekday, day, month (~ day of year)
        if m.shape[-1] >= MARK_DIM:                          # timeF: last 4 = hour, weekday, monthday, yearday
            return m[..., -MARK_DIM:]
        pad = m.new_zeros(*m.shape[:-1], MARK_DIM - m.shape[-1])   # daily data: no hour
        return torch.cat([pad, m], dim=-1)

    def _inputs(self, x):
        if self.ma_type == 'reg':
            return x, x
        return self.decomp(x)                                # sma: (resid, trend); dema: (seasonal, trend)

    def forward(self, x, current_epoch=0, max_epochs=1, x_mark=None, y_mark=None):
        # x: [B, L, C]; y_mark: [B, label_len + H, m] calendar of the target timestamps
        if self.revin:
            x = self.revin_layer(x, 'norm')
        x1, x2 = self._inputs(x)
        x, aux = self.net(x1, x2, self._future_marks(y_mark))
        if self.revin:
            x = self.revin_layer(x, 'denorm')
        return x, aux

    @torch.no_grad()
    def forward_views(self, x, x_mark=None, y_mark=None):
        """Per-view forecasts in the data space (analysis only). RevIN de-normalisation is affine per
        channel and the fusion weights sum to one, so the de-normalised fused forecast is the same convex
        combination of the de-normalised views (and the cross part maps linearly: only the scale applies)."""
        if self.revin:
            x = self.revin_layer(x, 'norm')
        x1, x2 = self._inputs(x)
        views, w, fused, cross = self.net.forward_views(x1, x2, self._future_marks(y_mark))
        if self.revin:
            views = torch.stack([self.revin_layer(views[:, v], 'denorm') for v in range(3)], dim=1)
            fused = self.revin_layer(fused, 'denorm')
            scale = self.revin_layer.stdev
            if self.revin_layer.affine:
                scale = scale / (self.revin_layer.affine_weight + self.revin_layer.eps ** 2)
            cross = cross * scale
        return {'views': views, 'weights': w, 'fused': fused, 'cross': cross}

    def load_transfer(self, state):
        """Load weights trained on another data set (any number of channels). Only the per-channel RevIN
        calibration may differ in shape; it keeps this model's identity initialisation. Returns the
        keys that were not transferred."""
        own = self.state_dict()
        skip = [k for k, v in state.items() if k.startswith('revin_layer.') and (k not in own or own[k].shape != v.shape)]
        missing, unexpected = self.load_state_dict({k: v for k, v in state.items() if k not in skip}, strict=False)
        bad = [k for k in missing if not k.startswith('revin_layer.')] + list(unexpected)
        if bad:
            raise RuntimeError(f'incompatible keys for transfer: {bad}')
        return skip