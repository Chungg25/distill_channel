"""
TriSpect: tri-view direct long-term forecaster, trained with the forecasting (MAE) loss only.

  temporal : how each series evolves. Patch encoder (local conv + GLU + transformer) and patch
             decoder of the production model, unchanged.
  spectral : what is periodic inside the window. Learned gain x per-window Wiener (SNR) gain and a
             learned phase on the spectrum, gated residual, coefficients on the shared horizon basis.
  channel  : what the OTHER channels say. Channels are softly assigned to G roles; channel c reads
             the role means computed WITHOUT itself (exact leave-one-out), mixed by attention among
             the roles. Its forecast is the sum of an own part (its own window) and a cross part
             (the other channels only):  y_C = y_own + y_cross.

Horizon basis (shared by spectral and channel): y(t) = sum_r a_r U_r(t),
  U(t) = MLP( calendar of the target time t ; physical lead time of t ),
so no parameter depends on the horizon H, and the same model is defined for any H and any sampling
rate (the lead time is measured in days, the calendar in hour / weekday / month-day / year-day).

Fusion: y = s(h, x) * sum_v w_v(h, x) f_v,  w = softmax_v( sum_b c_vb P_b(h) + alpha_v (Omega(x) - 1/2) )
  convex weights (Legendre basis over the horizon, Omega = spectral entropy of the input window),
  a shared amplitude s, subset view dropout -> every sub-ensemble is a valid forecast and the exact
  Shapley attribution of the views is defined.

No parameter depends on the number, order or identity of the channels (RevIN affine aside).
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


# ----------------------------------------------------------------------------------------------
# Horizon basis
# ----------------------------------------------------------------------------------------------
class HorizonBasis(nn.Module):
    """
    R forecast waveforms U(t) [H, R], generated from time instead of stored per horizon step.

      lead time  tau_j = j / steps_per_day (days), j = 1..H:
                 sin / cos (2 pi k tau)      k = 1..n_daily     (daily harmonics)
                 sin / cos (2 pi k tau / 7)  k = 1..n_weekly    (weekly harmonics)
                 exp(-tau / s)               s in decay_days    (relaxation from the current state)
      calendar   sin / cos (2 pi m) of the target timestamp's hour of day and day of week
                 (known in advance; zeros when the data set has no timestamps)
      U(t)     = MLP(features(t))

    The lead-time features depend on H and on the sampling rate only through the time grid (buffers,
    not parameters), so a trained model can be evaluated on any horizon and any sampling rate.
    """

    def __init__(self, pred_len, steps_per_day, rank, mark_dim=4, hidden=64,
                 n_daily=6, n_weekly=3, decay_days=(0.25, 1.0, 4.0, 16.0)):
        super().__init__()
        self.mark_dim = mark_dim
        self.rank = rank
        tau = torch.arange(1, pred_len + 1, dtype=torch.float64) / steps_per_day
        feats = []
        for k in range(1, n_daily + 1):
            feats += [torch.sin(2 * math.pi * k * tau), torch.cos(2 * math.pi * k * tau)]
        for k in range(1, n_weekly + 1):
            feats += [torch.sin(2 * math.pi * k * tau / 7), torch.cos(2 * math.pi * k * tau / 7)]
        feats += [torch.exp(-tau / s) for s in decay_days]
        self.register_buffer('lead', torch.stack(feats, -1).float(), persistent=False)   # [H, F]
        n_in = self.lead.shape[-1] + 2 * mark_dim
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, rank))

    def forward(self, y_mark=None):
        """y_mark: [B, H, mark_dim] calendar of the target timestamps, or None -> U [B or 1, H, R]."""
        H = self.lead.shape[0]
        if y_mark is None:
            cal = self.lead.new_zeros(1, H, 2 * self.mark_dim)
        else:
            ang = 2 * math.pi * y_mark
            cal = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        lead = self.lead[None].expand(cal.shape[0], -1, -1)
        return self.net(torch.cat([lead, cal], dim=-1))


# ----------------------------------------------------------------------------------------------
# Temporal view (production model, unchanged)
# ----------------------------------------------------------------------------------------------
class PatchChannelGLU(nn.Module):
    def __init__(self, patch_len, d_model):
        super().__init__()
        self.linear_a = nn.Linear(patch_len, d_model)
        self.linear_b = nn.Linear(patch_len, d_model)

    def forward(self, x):
        return self.linear_a(x) * torch.sigmoid(self.linear_b(x))


class LocalTemporal(nn.Module):
    def __init__(self, kernel_size, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, dilation=dilation,
                              padding=(kernel_size - 1) // 2 * dilation)

    def forward(self, x):
        # identical to self.conv(x), written as k shifted multiply-adds (a 1-in / 1-out
        # convolution is slow on GPU and CPU); parameters stay in self.conv
        k, d = self.conv.kernel_size[0], self.conv.dilation[0]
        p, T = self.conv.padding[0], x.shape[-1]
        w = self.conv.weight.view(-1)
        xp = F.pad(x, (p, p))
        out = self.conv.bias + w[0] * xp[..., 0:T]
        for j in range(1, k):
            out = out + w[j] * xp[..., j * d:j * d + T]
        return out


class TemporalView(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, d_model, nhead, num_layers, dropout):
        super().__init__()
        self.pred_len, self.patch_len, self.stride, self.padding_patch = pred_len, patch_len, stride, padding_patch
        patch_num = (seq_len - patch_len) // stride + 1
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1
        self.future_patch_num = math.ceil(pred_len / patch_len)
        F_p = self.future_patch_num
        self.patch_conv = LocalTemporal(kernel_size=3, dilation=1)
        self.patch_glu = PatchChannelGLU(patch_len, d_model)
        self.patch_embed = nn.Linear(d_model, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
                                       dropout=dropout, batch_first=True, activation='gelu', norm_first=True),
            num_layers=num_layers, enable_nested_tensor=False)
        self.repr_norm = nn.LayerNorm(d_model)
        self.patch_forecast = nn.Sequential(nn.Linear(patch_num, d_model), nn.GELU(), nn.Dropout(dropout),
                                            nn.Linear(d_model, F_p))
        self.context_generator = nn.Linear(d_model, F_p)
        self.forecast_gate = nn.Linear(patch_num, F_p)
        self.smoothing_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, padding_mode='replicate')
        self.base_linear = nn.Linear(d_model, patch_len)
        self.patch_decoder = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                                           nn.Linear(d_model, patch_len))

    def forward(self, x):
        """x: [B, C, L] -> [B, C, H]."""
        B, C, L = x.shape
        s = x.reshape(B * C, L)
        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        BC, P, Lp = s.shape
        s = s.reshape(BC * P, 1, Lp)
        s = (self.patch_conv(s) + s).reshape(BC, P, Lp)
        s = self.patch_embed(self.patch_glu(s))
        h = self.repr_norm(self.transformer_encoder(s))                   # [BC, P, d]
        context = self.context_generator(h[:, -1, :]).unsqueeze(1)        # [BC, 1, F_p]
        h_t = h.transpose(1, 2)                                            # [BC, d, P]
        f = (self.patch_forecast(h_t) + context) * torch.sigmoid(self.forecast_gate(h_t))
        f = (f + self.smoothing_conv(f)).transpose(1, 2)                   # [BC, F_p, d]
        y = self.base_linear(f) + self.patch_decoder(f)                   # [BC, F_p, patch_len]
        return y.reshape(B, C, -1)[..., :self.pred_len]


# ----------------------------------------------------------------------------------------------
# Spectral view (production filter, horizon basis head)
# ----------------------------------------------------------------------------------------------
class SpectralView(nn.Module):
    """
    X(w) -> g(w) X(w) e^{i phi_w}, for every window and channel:
      g(w) = sigmoid(a_w) * P(w) / (P(w) + N(w))     learned gain x Wiener gain
      P(w) = |X(w)|^2 of this window,  N(w) = softplus(rho_w) * T  a learned noise floor per frequency
    The Wiener factor keeps a frequency in proportion to its signal-to-noise ratio in THIS window
    (strong components pass, components at the level of the noise floor are attenuated), so the filter
    adapts to every input; N starts small, i.e. at the fixed learned gain of the original filter.
    Learned phase shift (scaling |X| and shifting angle(X), computed without abs / angle); gated
    residual, LayerNorm; coefficients on the horizon basis.
    """

    def __init__(self, seq_len, rank, dropout=0.1, width=16):
        super().__init__()
        n_freq = seq_len // 2 + 1
        self.amp_gate = nn.Parameter(torch.ones(n_freq))
        self.phase_shift = nn.Parameter(torch.zeros(n_freq))
        self.noise_rho = nn.Parameter(torch.full((n_freq,), -4.0))       # softplus(-4) = 0.018 -> N small
        self.norm = nn.LayerNorm(seq_len)
        self.res_gate = nn.Sequential(nn.Linear(seq_len, seq_len//2), nn.GELU(),
                                      nn.Linear(seq_len//2, seq_len), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)
        self.coef = nn.Sequential(nn.Linear(seq_len, width), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(width, rank))

    def forward(self, x):
        """x: [B, C, L] -> coefficients [B, C, R]; also stores |X|^2 for the fusion statistic."""
        T = x.shape[-1]
        X = torch.fft.rfft(x, dim=-1)
        xr, xi = X.real, X.imag
        power = xr ** 2 + xi ** 2
        self._last_power = power
        g = torch.sigmoid(self.amp_gate) * power / (power + F.softplus(self.noise_rho) * T)
        c, s = torch.cos(self.phase_shift), torch.sin(self.phase_shift)
        refined = torch.fft.irfft(torch.complex(g * (xr * c - xi * s), g * (xr * s + xi * c)), n=T, dim=-1)
        z = self.dropout(self.norm(x + self.res_gate(x) * refined))
        return self.coef(z)


# ----------------------------------------------------------------------------------------------
# Channel view: roles, exact leave-one-out, own + cross
# ----------------------------------------------------------------------------------------------
class ChannelView(nn.Module):
    """
      e_c       = W_e x_c                                           channel token
      q(g | c)  = softmax_g( <K e_c, P_g> / (sqrt(D) tau) )          soft role membership
      role tokens (attention WEIGHTS only, from all channels):
                  t_g = sum_c q_gc V e_c / sum_c q_gc ;  per layer l: M_l = attention among roles,
                  t <- t + mix_l(t), t <- t + FFN(t)
      values (exact leave-one-out):
                  z_g^(-c) = sum_{c' != c} q_gc' V e_c' / (sum_{c' != c} q_gc' + eps)
                  r_c      = sum_g q_gc [T z^(-c)]_g ,   T = prod_l ((1 - rho_l) I + rho_l M_l),
                  M_l = attention among roles averaged over heads
      forecast  : coef_c = Own(e_c) + W_x (gate(e_c) * W_o r_c)    -> y_C = y_own + y_cross

    Channel c's own window enters the cross part only through attention weights (q, M, gate); every
    value reaching it comes from the other channels. The cross part is linear in those values:
      y_cross,c = U W_x diag(gate_c) W_o sum_{c' != c} A[c, c'] V e_c'     (exact, see relations()).
    Cost O(B C G D) + O(B G^2 D): no tensor of size C x G x D; the C-sized maps are [B, C, G].
    """

    def __init__(self, seq_len, rank, d_model=32, num_groups=16, n_head=4, num_layers=2, dropout=0.1,
                 temperature=0.7, eps=1e-3):
        super().__init__()
        assert d_model % n_head == 0, 'd_model must be divisible by n_head'
        self.D, self.G, self.h = d_model, num_groups, n_head
        self.dh = d_model // n_head
        self.eps = eps
        self.embed = nn.Linear(seq_len, d_model)
        self.roles = nn.Parameter(torch.randn(num_groups, d_model))
        self.assign_k = nn.Linear(d_model, d_model, bias=False)
        self.log_tau = nn.Parameter(torch.tensor(math.log(temperature)))
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.layers = nn.ModuleList([nn.ModuleDict({
            'norm1': nn.LayerNorm(d_model),
            'q': nn.Linear(d_model, d_model, bias=False),
            'k': nn.Linear(d_model, d_model, bias=False),
            'norm2': nn.LayerNorm(d_model),
            'ffn': nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * d_model, d_model)),
        }) for _ in range(num_layers)])
        self.mix_logit = nn.Parameter(torch.zeros(num_layers))               # rho = 1/2 at init
        self.out = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.own = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * d_model, rank))
        self.cross = nn.Linear(d_model, rank)

    def _split(self, t):                                                   # [..., D] -> [..., h, dh]
        return t.reshape(*t.shape[:-1], self.h, self.dh)

    def _roles(self, e):
        """Membership q [B, C, G], role sums S [B, G, D] and masses Z [B, G], mixing T [B, G, G]."""
        logits = torch.einsum('bcd,gd->bcg', self.assign_k(e), self.roles) / (math.sqrt(self.D) * self.log_tau.exp())
        q = torch.softmax(logits, dim=-1)                                  # [B, C, G]
        v = self.value(e)                                                  # [B, C, D]
        S = torch.einsum('bcg,bcd->bgd', q, v)                             # [B, G, D]
        Z = q.sum(1)                                                       # [B, G]
        t = S / (Z[..., None] + self.eps)                                  # role tokens (all channels)
        B, G = Z.shape
        I = torch.eye(G, device=e.device, dtype=e.dtype)
        T = None
        rho = torch.sigmoid(self.mix_logit)                                # [layers]
        for l, layer in enumerate(self.layers):
            u = layer['norm1'](t)
            qm, km = self._split(layer['q'](u)), self._split(layer['k'](u))      # [B, G, h, dh]
            M = torch.softmax(torch.einsum('bghd,bkhd->bhgk', qm, km) / math.sqrt(self.dh), dim=-1).mean(1)
            T_l = (1 - rho[l]) * I + rho[l] * M                            # [B, G, G], rows sum to 1
            t = T_l @ t
            t = t + layer['ffn'](layer['norm2'](t))
            T = T_l if T is None else T_l @ T
        return q, v, S, Z, T

    def _cross(self, q, v, S, Z, T):
        """r [B, C, D] (before W_o): r_c = sum_k a_ck (S_k - q_kc v_c), a_ck = w_ck / (Z_k - q_kc + eps)."""
        w = q if T is None else q @ T                                     # [B, C, G]
        a = w / ((Z[:, None, :] - q).clamp_min(0) + self.eps)             # [B, C, G]
        b = (a * q).sum(-1, keepdim=True)                                  # [B, C, 1]
        return a @ S - b * v, a

    def forward(self, x):
        """x: [B, C, L] -> own and cross coefficients [B, C, R] each."""
        e = self.embed(x)
        q, v, S, Z, T = self._roles(e)
        r, _ = self._cross(q, v, S, Z, T)
        if not self.training:
            self._last_assign = q.detach()
        cross = self.cross(self.gate(e) * self.out(r))
        return self.own(e), cross

    @torch.no_grad()
    def relations(self, x):
        """
        Analysis. x: [B, C, L] normalised input. Returns
          assign [B, C, G]  role membership q(g | c)
          A      [B, C, C]  exact weight of channel c' in the cross context of c: r_c = sum_c' A[c, c'] V e_c';
                            A[c, c] = 0; each row sums to 1 minus the mass of roles that only c fills
        """
        e = self.embed(x)
        q, v, S, Z, T = self._roles(e)
        r, a = self._cross(q, v, S, Z, T)
        C = q.shape[1]
        A = (a @ q.transpose(1, 2)) * (1 - torch.eye(C, device=x.device, dtype=x.dtype))
        self._last_check = float((A @ v - r).abs().max()) if C > 1 else 0.0
        return q, A


# ----------------------------------------------------------------------------------------------
# Fusion helpers
# ----------------------------------------------------------------------------------------------
def legendre_basis(n_basis, H):
    """Legendre polynomials P_0..P_{n-1} on H points of [-1, 1] -> [n, H]."""
    t = torch.linspace(-1.0, 1.0, H, dtype=torch.float64)
    P = [torch.ones_like(t), t]
    for n in range(1, n_basis - 1):
        P.append(((2 * n + 1) * t * P[n] - n * P[n - 1]) / (n + 1))
    return torch.stack(P[:n_basis]).float()


def spectral_entropy(power, eps=1e-8):
    """Normalised spectral entropy (0 = one sinusoid, 1 = white noise) from |rfft|^2 [.., F] -> [..]."""
    P = power[..., 1:]
    p = P / (P.sum(-1, keepdim=True) + eps)
    return -(p * (p + eps).log()).sum(-1) / math.log(P.shape[-1])


class _ConvexFusion(torch.autograd.Function):
    """
    y = s_b[b, c] s_h[h] * sum_v F_v[b, c] E_v[h] f_v[b, c, h] / sum_v F_v[b, c] E_v[h]

    Forward and backward are written out so that nothing of size [B, C, H] is stored besides the three
    view forecasts (which exist anyway); the normaliser is recomputed in the backward pass.
    """

    @staticmethod
    def _parts(f, E, Fv):
        den = torch.einsum('vbc,vh->bch', Fv, E)
        num = sum(Fv[v][..., None] * E[v] * f[v] for v in range(len(f)))
        return num, den

    @staticmethod
    def forward(ctx, f0, f1, f2, E, Fv, s_h, s_b):
        f = (f0, f1, f2)
        num, den = _ConvexFusion._parts(f, E, Fv)
        ctx.save_for_backward(f0, f1, f2, E, Fv, s_h, s_b)
        return num / den * s_h * s_b[..., None]

    @staticmethod
    def backward(ctx, g):
        f0, f1, f2, E, Fv, s_h, s_b = ctx.saved_tensors
        f = (f0, f1, f2)
        num, den = _ConvexFusion._parts(f, E, Fv)
        mix = num / den                                                    # convex mixture [B, C, H]
        gk = g * s_h * s_b[..., None]                                      # dL/d(mix)
        gd = gk / den
        grad_f = [gd * (Fv[v][..., None] * E[v]) for v in range(3)]
        # d mix / d(F_v E_v) = (f_v - mix) / den
        grad_E, grad_F = [], []
        for v in range(3):
            t = gd * (f[v] - mix)
            grad_E.append(torch.einsum('bch,bc->h', t, Fv[v]))
            grad_F.append(torch.einsum('bch,h->bc', t, E[v]))
        gm = g * mix
        grad_sh = torch.einsum('bch,bc->h', gm, s_b)
        grad_sb = torch.einsum('bch,h->bc', gm, s_h)
        return grad_f[0], grad_f[1], grad_f[2], torch.stack(grad_E), torch.stack(grad_F), grad_sh, grad_sb


VIEW_NAMES = ('temporal', 'spectral', 'channel')
SUBSET_MASKS = torch.tensor([[(m >> v) & 1 for v in range(3)] for m in range(8)], dtype=torch.bool)


def subset_forecast(views, weights, mask):
    """Forecast of a sub-ensemble (weights of the kept views renormalised). views, weights [B, 3, H, C]."""
    m = mask.to(views.device).to(views.dtype)[None, :, None, None]
    w = weights * m
    w = w / w.sum(1, keepdim=True).clamp_min(1e-12)
    return (w * views).sum(1)


# ----------------------------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------------------------
class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, dropout=0.1, d_model=32,
                 nhead=4, num_layers=2, num_groups=16, steps_per_day=24.0, mark_dim=4,
                 view_dropout=0.2, fusion_basis=4):
        super().__init__()
        self.pred_len = pred_len
        rank = max(32, 2 * d_model)
        # calendar of the target time: hour of day and day of week only (patterns repeated hundreds of
        # times in any training split); day-of-month / day-of-year repeat at most a few times and let the
        # basis memorise the seasons of the training years
        self.basis = HorizonBasis(pred_len, steps_per_day, rank, mark_dim=2)
        self.temporal = TemporalView(seq_len, pred_len, patch_len, stride, padding_patch, d_model,
                                     nhead, num_layers, dropout)
        self.spectral = SpectralView(seq_len, rank, dropout)
        self.channel = ChannelView(seq_len, rank, d_model, num_groups, nhead, num_layers=2, dropout=dropout)

        self.view_dropout = view_dropout
        self.register_buffer('fusion_basis', legendre_basis(fusion_basis, pred_len), persistent=False)
        self.fusion_coef = nn.Parameter(torch.zeros(3, fusion_basis))     # uniform 1/3 at init
        self.fusion_alpha = nn.Parameter(torch.zeros(3))
        self.amp_coef = nn.Parameter(torch.zeros(fusion_basis))           # s = 1 at init
        self.amp_alpha = nn.Parameter(torch.zeros(()))
        subsets = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=torch.bool)
        self.register_buffer('view_subsets', subsets, persistent=False)

    # ------------------------------------------------------------------------------------------
    def _views(self, x1, x2, y_mark, parts=False):
        """x1, x2 [B, L, C] -> temporal, spectral, channel-own, channel-cross [B, C, H], Omega [B, C].
        Without parts the channel view is returned whole in place of channel-own and cross is None."""
        x1 = x1.permute(0, 2, 1)
        decomposed = x2 is not None and x2 is not x1 and x2.data_ptr() != x1.data_ptr()
        x2 = x2.permute(0, 2, 1) if x2 is not None else x1
        B, C = x1.shape[0], x1.shape[1]
        Ut = self.basis(None if y_mark is None else y_mark[..., :2]).transpose(1, 2).expand(B, -1, -1)   # [B, R, H]
        temporal = self.temporal(x1)
        own_c, cross_c = self.channel(x1)
        if parts:      # analysis: the cross-channel part separately
            spectral, own, cross = (torch.cat([self.spectral(x2), own_c, cross_c], 1) @ Ut).split(C, 1)
        else:          # one batched product with the basis: [B, 2C, R] @ [B, R, H]
            spectral, channel = (torch.cat([self.spectral(x2), own_c + cross_c], 1) @ Ut).split(C, 1)
            own, cross = channel, None
        if decomposed:
            total = x1 + x2
            X = torch.fft.rfft(total, dim=-1)
            power = X.real ** 2 + X.imag ** 2
        else:
            power = self.spectral._last_power
        return temporal, spectral, own, cross, spectral_entropy(power)

    def view_keep_mask(self, B, C, drop=False, device=None):
        keep = None
        if drop and self.view_dropout > 0:
            use = torch.rand(B, device=device) < self.view_dropout
            idx = torch.randint(0, 6, (B,), device=device)
            keep = torch.ones(B, 3, dtype=torch.bool, device=device)
            keep[use] = self.view_subsets[idx[use]]
        if C == 1:                                                         # no other channel
            no_ch = torch.tensor([True, True, False], device=device)
            keep = no_ch.expand(B, 3) if keep is None else keep & no_ch
            keep = torch.where(keep.any(1, keepdim=True), keep, no_ch.expand(B, 3))
        return keep

    def fusion_weights_for(self, B, C, omega, keep=None):
        """Convex weights [B, 3, C, H]."""
        logit = (self.fusion_coef @ self.fusion_basis)[None, :, None, :]
        logit = logit + self.fusion_alpha[None, :, None, None] * (omega[:, None, :, None] - 0.5)
        w = torch.softmax(logit, dim=1)
        if keep is not None:
            w = w * keep[:, :, None, None].to(w.dtype)
            w = w / w.sum(dim=1, keepdim=True)
        return w.expand(B, 3, C, self.pred_len)

    def amplitude(self, omega):
        """Shared amplitude s(h, x) > 0 [B, C, H]."""
        return torch.exp((self.amp_coef @ self.fusion_basis)[None, None, :] + self.amp_alpha * (omega[..., None] - 0.5))

    def _combine(self, views, omega, keep):
        """s * sum_v w_v f_v without forming [B, 3, C, H] weights. The fusion logit is a sum of a horizon
        term and a sample term, so exp(logit_v) = E_v[h] * F_v[b, c] (outer product):
          w_v = E_v F_v / sum_u E_u F_u,   s = exp(k . P(h)) * exp(kappa (Omega - 1/2)).
        Identical to fusion_weights_for() / amplitude() (used for the analysis read-out)."""
        hl = self.fusion_coef @ self.fusion_basis                          # [3, H]
        E = torch.exp(hl - hl.max(0, keepdim=True).values)                # [3, H]
        sl = self.fusion_alpha[:, None, None] * (omega[None] - 0.5)        # [3, B, C]
        Fv = torch.exp(sl - sl.max(0, keepdim=True).values)                # [3, B, C]
        if keep is not None:
            Fv = Fv * keep.t()[:, :, None].to(Fv.dtype)
        s_h = torch.exp(self.amp_coef @ self.fusion_basis)                 # [H]
        s_b = torch.exp(self.amp_alpha * (omega - 0.5))                    # [B, C]
        return _ConvexFusion.apply(*views, E, Fv, s_h, s_b)

    def forward(self, x1, x2=None, y_mark=None):
        """x1, x2: [B, L, C] normalised inputs; y_mark: [B, H, 4] calendar of the target times or None."""
        temporal, spectral, channel, _, omega = self._views(x1, x2, y_mark)
        B, C, _ = temporal.shape
        keep = self.view_keep_mask(B, C, drop=self.training, device=temporal.device)
        out = self._combine((temporal, spectral, channel), omega, keep)
        self._last_omega = omega.detach()
        return out.permute(0, 2, 1), {}

    @torch.no_grad()
    def forward_views(self, x1, x2=None, y_mark=None):
        """Per-view forecasts after the shared amplitude [B, 3, H, C], weights [B, 3, H, C], fused [B, H, C],
        and the cross-channel part of the channel view [B, H, C] (fused = sum_v w_v views_v exactly)."""
        temporal, spectral, own, cross, omega = self._views(x1, x2, y_mark, parts=True)
        B, C, _ = temporal.shape
        s = self.amplitude(omega)[:, None]
        views = torch.stack([temporal, spectral, own + cross], dim=1) * s
        w = self.fusion_weights_for(B, C, omega, self.view_keep_mask(B, C, device=temporal.device))
        fused = (w * views).sum(1)
        cross_s = cross * s[:, 0]
        return views.transpose(2, 3), w.transpose(2, 3), fused.transpose(1, 2), cross_s.transpose(1, 2)

    @torch.no_grad()
    def explain(self):
        """Interpretable quantities (run an eval forward first for the input-dependent ones)."""
        info = {'fusion_weights': torch.softmax(self.fusion_coef @ self.fusion_basis, 0).cpu(),   # [3, H] at Omega = 1/2
                'fusion_entropy_coef': self.fusion_alpha.detach().cpu(),
                'amplitude_horizon': torch.exp(self.amp_coef @ self.fusion_basis).cpu(),
                'amplitude_entropy_coef': float(self.amp_alpha),
                'spectral_gain': torch.sigmoid(self.spectral.amp_gate).detach().cpu(),
                'spectral_phase': self.spectral.phase_shift.detach().cpu(),
                'spectral_noise_floor': (F.softplus(self.spectral.noise_rho)
                                         * self.spectral.norm.normalized_shape[0]).detach().cpu(),
                'channel_temperature': float(self.channel.log_tau.exp()),
                'channel_role_mixing': torch.sigmoid(self.channel.mix_logit).detach().cpu()}
        if getattr(self, '_last_omega', None) is not None:
            info['input_spectral_entropy_mean'] = float(self._last_omega.mean())
        if hasattr(self.channel, '_last_assign'):
            p = self.channel._last_assign.mean(dim=(0, 1))
            info['channel_effective_roles'] = float(torch.exp(-(p * (p + 1e-12).log()).sum()))
            info['channel_role_membership'] = self.channel._last_assign.cpu()
        return info


# ----------------------------------------------------------------------------------------------
# Contribution diagnostics (evaluation only: uses targets, never used for training)
# ----------------------------------------------------------------------------------------------
class ViewDiagnostics:
    """
    Streaming diagnostics over a split.
    update(views [B,3,H,C], weights [B,3,H,C], y [B,H,C], x=None, cross=None)
      x     [B,L,C] raw input window -> baseline "mean of the window" for the exact Shapley game
      cross [B,H,C] cross-channel part of the channel view -> exact counterfactual "no other channel"
    result(): mae_view, mae_fused, jensen_gain, mean_weight, error_corr, and with x: shapley [3],
      shapley_share, unique [3], interaction [3,3], shapley_horizon [3,H]; with cross: mae_without_cross,
      cross_gain = MAE(without cross) - MAE(fused).
    """

    def __init__(self):
        self.n = 0
        self.s = {}

    def _add(self, k, v):
        v = v.detach().double().cpu() if torch.is_tensor(v) else float(v)
        self.s[k] = self.s.get(k, 0) + v

    @torch.no_grad()
    def update(self, views, weights, y, x=None, cross=None):
        fused = (weights * views).sum(1)
        err = views - y[:, None]
        self.n += y.numel()
        self._add('abs_view', err.abs().sum(dim=(0, 2, 3)))
        self._add('abs_fused', (fused - y).abs().sum())
        self._add('abs_weighted', (weights * err.abs()).sum())
        self._add('w', weights.sum(dim=(0, 2, 3)))
        e = err.transpose(0, 1).reshape(3, -1).double()
        self._add('e1', e.sum(-1))
        self._add('e2', e @ e.T)
        if cross is not None:
            self._add('abs_nocross', (fused - weights[:, 2] * cross - y).abs().sum())
        if x is not None:
            self._add('n_h', y.shape[0] * y.shape[2])
            base = x.mean(dim=1, keepdim=True)
            self._add('sub_h_0', (base - y).abs().sum(dim=(0, 2)))
            for m in range(1, 8):
                f = fused if m == 7 else subset_forecast(views, weights, SUBSET_MASKS[m])
                self._add(f'sub_h_{m}', (f - y).abs().sum(dim=(0, 2)))

    def _attribution(self):
        s = self.s
        mae_h = torch.stack([s[f'sub_h_{m}'] / s['n_h'] for m in range(8)])
        v_h = mae_h[0][None] - mae_h
        v_h[0] = 0.0
        fact = [math.factorial(k) for k in range(5)]

        def shap(v):
            out = []
            for i in range(3):
                acc = 0
                for m in range(8):
                    if not (m >> i) & 1:
                        k = bin(m).count('1')
                        acc = acc + fact[k] * fact[2 - k] / fact[3] * (v[m | (1 << i)] - v[m])
                out.append(acc)
            return torch.stack(out)

        def inter(v, i, j):
            k_ = 3 - i - j
            acc = 0
            for m in (0, 1 << k_):
                k = bin(m).count('1')
                acc = acc + fact[k] * fact[1 - k] / fact[2] * (v[m | (1 << i) | (1 << j)] - v[m | (1 << i)] - v[m | (1 << j)] + v[m])
            return acc

        v = v_h.mean(dim=1)
        phi = shap(v)
        im = torch.zeros(3, 3, dtype=v.dtype)
        for i in range(3):
            for j in range(3):
                if i != j:
                    im[i, j] = inter(v, i, j)
        return {'mae_baseline': float(mae_h[0].mean()), 'value': v.float(), 'shapley': phi.float(),
                'shapley_share': (phi / v[7].clamp_min(1e-12)).float(), 'shapley_horizon': shap(v_h).float(),
                'unique': torch.stack([v[7] - v[7 & ~(1 << i)] for i in range(3)]).float(),
                'interaction': im.float()}

    def result(self):
        n, s = self.n, self.s
        mean = s['e1'] / n
        cov = s['e2'] / n - torch.outer(mean, mean)
        sd = cov.diagonal().clamp_min(1e-24).sqrt()
        out = {'mae_view': (s['abs_view'] / n).float(), 'mae_fused': float(s['abs_fused'] / n),
               'jensen_gain': float((s['abs_weighted'] - s['abs_fused']) / n),
               'mean_weight': (s['w'] / n).float(), 'error_corr': (cov / torch.outer(sd, sd)).float()}
        if 'abs_nocross' in s:
            out['mae_without_cross'] = float(s['abs_nocross'] / n)
            out['cross_gain'] = out['mae_without_cross'] - out['mae_fused']
        if 'n_h' in s:
            out.update(self._attribution())
        return out