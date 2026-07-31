import torch
from torch import nn
import torch.nn.functional as F
import math

class GroupChannelBlock(nn.Module):
    def __init__(self, seq_len, pred_len,
                 d_model=64, num_groups=8,
                 nhead=4, num_layers=2,
                 dropout=0.1, expand=2, temperature=0.7):
        super().__init__()

        self.embed = nn.Linear(seq_len, d_model)

        # self.group_tokens = nn.Parameter(torch.randn(num_groups, d_model))
        self.group_seed = nn.Parameter(torch.randn(num_groups, d_model))
        
        # Sinh ra cả Scale (Gamma) và Shift (Beta)
        self.modulator = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model * 2) # x2 để chia làm scale và shift
        )

        self.temperature = temperature
        self.num_groups = num_groups

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(
                    d_model, nhead, batch_first=True, dropout=dropout
                ),
                "norm1": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model)
                ),
                "norm2": nn.LayerNorm(d_model),
                "attn_gate": nn.Linear(d_model, d_model),
            })
            for _ in range(num_layers)
        ])

        self.channel_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        self.head = nn.Sequential(
            nn.Linear(d_model, pred_len * expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_len * expand, pred_len)
        )

    def forward(self, x):
        # x: [B, C, T]
        B, C, T = x.shape

        x_embed = self.embed(x)  # [B, C, D]

        context = x_embed.mean(dim=1)  # [B, d_model]
        
        # Sinh ra modulation [B, 1, d_model * 2]
        mod = self.modulator(context).unsqueeze(1) 
        scale, shift = mod.chunk(2, dim=-1) # [B, 1, d_model] mỗi phần tử

        # FiLM: Scale & Shift
        group_tokens = self.group_seed.unsqueeze(0) * scale + shift

        sim = torch.einsum('bcd,bgd->bcg', x_embed, group_tokens) / math.sqrt(self.embed.out_features)

        assign  = torch.softmax(sim / self.temperature, dim=-1)

        self._last_assign = assign

        group_feat = torch.einsum('bcg,bcd->bgd', assign, x_embed)
        group_feat = group_feat / (assign.sum(dim=1).unsqueeze(-1) + 1e-6)

        for layer in self.layers:
            residual = group_feat
            normed1 = layer["norm1"](group_feat)
            attn_out, _ = layer["attn"](normed1, normed1, normed1)
            attn_gate = torch.sigmoid(layer["attn_gate"](residual))
            group_feat = residual + attn_out * attn_gate

            residual = group_feat
            normed2 = layer["norm2"](group_feat)
            ffn_out = layer["ffn"](normed2)
            group_feat = residual + ffn_out

        self._last_group_feat = group_feat 

        out = torch.einsum('bcg,bgd->bcd', assign, group_feat)

        gate = self.channel_gate(x_embed)
        out = out * gate + x_embed  

        self._last_latent = out  

        out = self.head(out)

        return out

    def get_aux_losses(self):
        """
        Compute the three MoE auxiliary losses.

        Returns:
            dict with keys 'diversity', 'balance', 'sharpness'
        """
        losses = {}

        group_feat = self._last_group_feat  # [B, G, D]
        G_norm = F.normalize(group_feat.mean(dim=0), dim=-1)  # [G, D]
        gram = G_norm @ G_norm.T  # [G, G]
        mask = ~torch.eye(self.num_groups, dtype=torch.bool, device=gram.device)
        losses['diversity'] = gram[mask].pow(2).mean()

        assign = self._last_assign  # [B, C, G]
        p_bar = assign.mean(dim=1).mean(dim=0)  # [G]
        losses['anti_collapse'] = -torch.log(p_bar + 1e-6).mean()

        eps = 1e-8
        log_assign = (assign + eps).log()
        entropy = -(assign * log_assign).sum(dim=-1)
        losses['sharpness'] = entropy.mean()

        return losses

class DecorrelationLoss(nn.Module):
    def forward(self, *branch_outputs):
        loss = 0.0
        n = len(branch_outputs)
        for i in range(n):
            for j in range(i + 1, n):
                # [B, C, D] or [B, C, T] → [B, C]
                yi = branch_outputs[i].mean(dim=-1)
                yj = branch_outputs[j].mean(dim=-1)
                cos_sim = F.cosine_similarity(yi, yj, dim=-1)  # [B]
                loss = loss + (cos_sim ** 2).mean()
        return loss

class AdaptiveAuxLossWeighter(nn.Module):
    """
    Inverse-scaling adaptive weighter:
    - When main_loss is HIGH (early training) → reduce aux weights (let model learn forecasting first)
    - When main_loss is LOW (model converged) → increase aux weights (refine structure)
    - Sharpness has warmup: off for first 5 epochs, then linear ramp
    - Cosine decay towards end of training
    """

    def __init__(self, lambda_decor=0.001, lambda_div=0.001,
                 lambda_collapse=0.005, lambda_sharp=0.005, ema_decay=0.95):
        super().__init__()
        self.lambda_base = {
            'decorrelation': lambda_decor,
            'diversity': lambda_div,
            'anti_collapse': lambda_collapse,
            'sharpness': lambda_sharp,
        }
        self.ema_decay = ema_decay
        self.register_buffer('ema_main_loss', torch.tensor(1.0))

    @torch.no_grad()
    def compute_weights(self, aux_losses, current_epoch=0, max_epochs=1, main_loss=None):
        weights = {}

        if main_loss is not None:
            self.ema_main_loss.copy_(
                self.ema_decay * self.ema_main_loss + (1 - self.ema_decay) * main_loss
            )

        inv_scale = 1.0 / (1.0 + self.ema_main_loss.item())

        if max_epochs > 1:
            decay = 0.5 * (1 + math.cos(math.pi * current_epoch / max_epochs))
        else:
            decay = 1.0

        for key in self.lambda_base:
            if key not in aux_losses:
                continue

            if key == 'sharpness':
                warmup_epochs = 5
                if current_epoch < warmup_epochs:
                    weights[key] = 0.0
                    continue
                ramp = min(1.0, (current_epoch - warmup_epochs) / warmup_epochs)
                weights[key] = self.lambda_base[key] * ramp * inv_scale * decay
            else:
                weights[key] = self.lambda_base[key] * inv_scale * decay

        return weights



class PatchChannelGLU(nn.Module):
    def __init__(self, patch_len, d_model):
        super().__init__()
        self.linear_a = nn.Linear(patch_len, d_model)
        self.linear_b = nn.Linear(patch_len, d_model)

    def forward(self, x):
        a = self.linear_a(x)
        b = torch.sigmoid(self.linear_b(x))
        return a * b


class LocalTemporal(nn.Module):
    def __init__(self, kernel_size, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) // 2 * dilation,
        )

    def forward(self, x):
        return self.conv(x)


class SpectralTimeBlock(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        expand,
        dropout=0.1
    ):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len

        self.F = seq_len // 2 + 1

        self.amp_gate = nn.Parameter(torch.ones(self.F))
        self.phase_shift = nn.Parameter(torch.zeros(self.F))

        self.norm = nn.LayerNorm(seq_len)

        bottleneck_dim = max(16, seq_len // 8)
        self.res_gate = nn.Sequential(
            nn.Linear(seq_len, seq_len//2),
            nn.GELU(),                           
            nn.Linear(seq_len//2, seq_len),
            nn.Sigmoid()                          
        )

        self.dropout = nn.Dropout(dropout)

        self.proj = nn.Sequential(
            nn.Linear(seq_len, 16 * 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16 * 1, pred_len)
        )

    def forward(self, x):
        """
        x: [B,C,T]
        """
        B, C, T = x.shape

        x_freq = torch.fft.rfft(x, dim=-1)  # [B, C, F]

        amp = torch.abs(x_freq)       # [B, C, F]
        phase = torch.angle(x_freq)   # [B, C, F]

        amp = amp * torch.sigmoid(self.amp_gate)   # [F] broadcasts to [B, C, F]

        phase = phase + self.phase_shift  # [F] broadcasts to [B, C, F]

        x_freq_refined = torch.complex(
            amp * torch.cos(phase),
            amp * torch.sin(phase)
        )

        x_time_refined = torch.fft.irfft(x_freq_refined, n=T, dim=-1)

        gate = self.res_gate(x)  # [B, C, T]
        x_time = x + gate * x_time_refined

        x_time = self.norm(x_time)
        self._last_latent = x_time  # [B, C, T] for decorrelation loss
        x_time = self.dropout(x_time)

        out = self.proj(x_time)

        return out
        
class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch,
                 dropout=0.1, d_model=64, nhead=4, num_layers=2, expand=2, num_groups=128):
        super().__init__()

        self.pred_len = pred_len    
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = (seq_len - patch_len) // stride + 1
        self.future_patch_num = math.ceil(pred_len / patch_len)
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1
        self.patch_num = patch_num
        self.decor_loss = DecorrelationLoss()
        self.aux_weighter = AdaptiveAuxLossWeighter()
        self.seasonal_channel = GroupChannelBlock(
            seq_len, pred_len, d_model=d_model, num_groups=num_groups,  
            nhead=nhead, dropout=dropout, expand=expand
        )
        self.patch_conv = LocalTemporal(kernel_size=3, dilation=1)
        self.patch_glu = PatchChannelGLU(patch_len, d_model)
        self.patch_embed = nn.Linear(d_model, d_model)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
                dropout=dropout, batch_first=True, activation='gelu', 
                norm_first=True 
            ),
            num_layers=num_layers
        )
        self.repr_norm = nn.LayerNorm(d_model)

        self.patch_forecast = nn.Sequential(
            nn.Linear(self.patch_num, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.future_patch_num)
        )

        # self.patch_forecast = nn.Sequential(
        #     nn.Linear(self.patch_num, self.future_patch_num)
        # )
        
        self.context_generator = nn.Linear(d_model, self.future_patch_num)
        
        self.forecast_gate = nn.Linear(self.patch_num, self.future_patch_num)
        
        self.smoothing_conv = nn.Conv1d(
            in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1, padding_mode='replicate'
        )
        
        self.base_linear = nn.Linear(d_model, patch_len)
        self.patch_decoder = nn.Sequential(
            nn.Linear(d_model, d_model*1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model*1, patch_len)
        )

        # self.patch_decoder = nn.Sequential(
        #     nn.Linear(d_model, patch_len)
        # )
        
        self.spectral = SpectralTimeBlock(seq_len, pred_len, expand)

        self.fusion_weights = nn.Parameter(torch.ones(3)) 
  

    def forward(self, x1, x2, current_epoch=0, max_epochs=1):
        x1 = x1.permute(0, 2, 1)  
        x2 = x2.permute(0, 2, 1)
        B, C, I = x1.shape

        channel = self.seasonal_channel(x1)  

        s_flat = x1.reshape(B * C, I)
        if self.padding_patch == 'end':
            s_flat = self.padding_patch_layer(s_flat)

        s_patch = s_flat.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        BC, P, L = s_patch.shape

        s_patch = s_patch.reshape(BC * P, 1, L)
        residual = s_patch
        s_patch = self.patch_conv(s_patch)
        s_patch = s_patch + residual
        s_patch = s_patch.reshape(BC, P, L)

        s_patch = self.patch_glu(s_patch)
        s_patch = self.patch_embed(s_patch)

        s_patch = self.transformer_encoder(s_patch)
        x = self.repr_norm(s_patch) # [BC, P, d_model]
        self._temporal_latent = x.mean(dim=1).view(B, C, -1)  # [B, C, d_model]
        
        last_state = x[:, -1, :] 
        context = self.context_generator(last_state).unsqueeze(1) # [BC, 1, Future_P]

        x_t = x.transpose(1, 2) # [BC, d_model, P]
        base_forecast = self.patch_forecast(x_t) # [BC, d_model, Future_P]

        forecast = base_forecast + context 
        gate = torch.sigmoid(self.forecast_gate(x_t)) # Mở cổng
        future_features = forecast * gate # [BC, d_model, Future_P]

        x_conv = self.smoothing_conv(future_features)
        future_features = future_features + x_conv 
        
        future_features = future_features.transpose(1, 2) # [BC, Future_P, d_model]
        x_decoded = self.base_linear(future_features) + self.patch_decoder(future_features)
        
        x = x_decoded.reshape(B * C, self.future_patch_num * self.patch_len)
        temporal = x.view(B, C, -1)

        spectral = self.spectral(x2).view(B, C, self.pred_len)

        w = F.softplus(self.fusion_weights)
        x = w[0] * temporal + w[1] * spectral + w[2] * channel
        x = x.permute(0,2,1)

        aux_losses = {}
        if self.training:
            channel_latent = self.seasonal_channel._last_latent  # [B, C, D]
            temporal_latent = self._temporal_latent               # [B, C, D]
            spectral_latent = self.spectral._last_latent          # [B, C, T]
            aux_losses['decorrelation'] = self.decor_loss(
                channel_latent, temporal_latent, spectral_latent
            )
            aux_losses.update(self.seasonal_channel.get_aux_losses())
            
        return x, aux_losses
    
