import torch
from torch import nn
import torch.nn.functional as F

class AdaptiveFusion(nn.Module):
    def __init__(self, pred_len):
        super().__init__()
        self.fc = nn.Linear(pred_len*2, pred_len)

    def forward(self, s, t):
           alpha = torch.sigmoid(self.fc(torch.cat([s, t], dim=-1)))
           return alpha * s + (1 - alpha) * t  # [B*C, pred_len]

# ========================
# Patch GLU
# ========================
class PatchChannelGLU(nn.Module):
    def __init__(self, patch_len, d_model):
        super().__init__()
        self.linear_a = nn.Linear(patch_len, d_model)
        self.linear_b = nn.Linear(patch_len, d_model)

    def forward(self, x):
        a = self.linear_a(x)
        b = torch.sigmoid(self.linear_b(x))
        return a * b


# ========================
# Local Temporal Conv
# ========================
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

class LiteGroupTransformerChannel(nn.Module):
    def __init__(self, seq_len, pred_len,
                 d_model=64, num_groups=8, nhead=4, dropout=0.1):
        super().__init__()


        self.embed = nn.Linear(seq_len, d_model)


        # 🔥 learnable group centers (vẫn giữ để so sánh, nhưng sẽ dùng MLP cho assignment)
        self.group_tokens = nn.Parameter(torch.randn(num_groups, d_model))

        # 🔥 Non-linear assignment MLP
        self.assign_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_groups)
        )

        # 🔥 transformer trên group (NHẸ vì G nhỏ)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )

        self.group_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2   # 🔥 chỉ 1 layer là đủ
        )

        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, pred_len * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_len * 2, pred_len)
        )

    def forward(self, x):
        # x: [B, C, T]
        B, C, T = x.shape

        # ======================
        # 1. EMBED
        # ======================
        x_embed = self.embed(x)  # [B, C, D]

        # ======================
        # 2. GROUP ASSIGN (dùng MLP non-linearity)
        # ======================
        # [B, C, D] -> [B, C, num_groups]
        assign_score = self.assign_mlp(x_embed)
        assign = torch.softmax(assign_score, dim=-1)

        # ======================
        # 3. BUILD GROUP
        # ======================
        group_feat = torch.einsum('bcg,bcd->bgd', assign, x_embed)
        group_feat = group_feat / (assign.sum(dim=1).unsqueeze(-1) + 1e-6)

        # ======================
        # 🔥 4. TRANSFORMER ON GROUP
        # ======================
        residual = group_feat
        group_feat = self.group_transformer(group_feat)
        group_feat = self.norm(group_feat + residual)

        # ======================
        # 5. BACK TO CHANNEL
        # ======================
        out = torch.einsum('bcg,bgd->bcd', assign, group_feat)

        # ======================
        # 6. RESIDUAL
        # ======================
        out = self.norm(out + x_embed)

        # ======================
        # 7. HEAD
        # ======================
        out = self.head(out)

        return out

import torch
import torch.nn as nn

class SpectralTrendRefineBlock(nn.Module):
    def __init__(self, seq_len, pred_len, dropout=0.1):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.F = seq_len // 2 + 1

        # 🔥 chỉ giữ spectral transform (KHÔNG filter nữa)
        self.weight_real = nn.Parameter(torch.ones(self.F))
        self.weight_imag = nn.Parameter(torch.ones(self.F))

        # 🔥 residual scaling để tránh phá trend gốc
        self.beta = nn.Parameter(torch.tensor(0.1))

        self.norm = nn.LayerNorm(seq_len)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [B, C, T] (đã là TREND từ decomposition)

        # ======================
        # 1. FFT
        # ======================
        x_freq = torch.fft.rfft(x, dim=-1)

        # ======================
        # 2. SPECTRAL TRANSFORM (NO FILTER)
        # ======================
        real = x_freq.real * self.weight_real
        imag = x_freq.imag * self.weight_imag
        x_freq_refined = torch.complex(real, imag)

        # ======================
        # 3. IFFT
        # ======================
        x_time_refined = torch.fft.irfft(
            x_freq_refined, n=self.seq_len, dim=-1
        )

        # ======================
        # 4. RESIDUAL (QUAN TRỌNG)
        # ======================
        # chỉ học correction thay vì thay thế hoàn toàn
        x_time = x + self.beta * x_time_refined

        # ======================
        # 5. NORMALIZE + DROPOUT
        # ======================
        x_time = self.norm(x_time)
        x_time = self.dropout(x_time)

        # ======================
        # 6. FORECAST
        # ======================
        out = self.proj(x_time)

        return out

# ========================
# 🔥 MAIN NETWORK
# ========================
class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch,
                 dropout=0.1, d_model=64, nhead=4, num_layers=2, groups=8):
        super().__init__()

        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch

        patch_num = (seq_len - patch_len) // stride + 1
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        self.patch_num = patch_num
        self.alpha = nn.Parameter(torch.full((1, 862, 1), -1.0))

        # 🔥 Channel path (seasonal)
        self.seasonal_channel = LiteGroupTransformerChannel(
            seq_len,
            pred_len,
            d_model=d_model,
            num_groups=groups,   # 🔥 quan trọng
            nhead=nhead,
            dropout=dropout
        )

        # 🔥 Temporal path (GIỮ NGUYÊN)
        self.patch_conv = LocalTemporal(kernel_size=3, dilation=1)
        self.patch_glu = PatchChannelGLU(patch_len, d_model)
        self.patch_embed = nn.Linear(d_model, d_model)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            ),
            num_layers=num_layers
        )

        # self.temporal_pool = nn.Sequential(
        #     nn.Linear(d_model, d_model),
        #     nn.GELU(),
        #     nn.Linear(d_model, 1)
        # )

        # self.out_linear = nn.Linear(d_model, pred_len)

        # Seasonal head (temporal path)
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear_seasonal = nn.Linear(self.patch_num * d_model, pred_len*2)
        self.gelu_seasonal = nn.GELU()
        self.dropout_seasonal = nn.Dropout(dropout)
        self.linear_seasonal2 = nn.Linear(pred_len*2, pred_len)

        self.adaptive_fusion = AdaptiveFusion(pred_len)

        self.trend = SpectralTrendRefineBlock(seq_len, pred_len)

        # (Optional) learnable fusion
        # self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, s, t):
        # s, t: [B, seq_len, C]

        s = s.permute(0, 2, 1)  # [B, C, T]
        t = t.permute(0, 2, 1)

        B, C, I = s.shape

        # ======================
        # 🔥 1. CHANNEL PATH
        # ======================
        s_channel = self.seasonal_channel(s)   # [B, C, pred_len]

        # ======================
        # 🔥 2. TEMPORAL PATH (NGUYÊN BẢN)
        # ======================
        s_flat = s.reshape(B * C, I)

        if self.padding_patch == 'end':
            s_flat = self.padding_patch_layer(s_flat)

        s_patch = s_flat.unfold(
            dimension=-1,
            size=self.patch_len,
            step=self.stride
        )

        BC, P, L = s_patch.shape

        s_patch = s_patch.reshape(BC * P, 1, L)
        residual = s_patch
        s_patch = self.patch_conv(s_patch)
        s_patch = s_patch + residual
        s_patch = s_patch.reshape(BC, P, L)

        s_patch = self.patch_glu(s_patch)
        s_patch = F.gelu(s_patch)
        s_patch = self.patch_embed(s_patch)

        s_patch_residual = s_patch
        s_patch = self.transformer_encoder(s_patch)
        s_patch = s_patch + s_patch_residual

        s_patch = self.flatten(s_patch)

        s_temporal = self.linear_seasonal(s_patch)
        s_temporal = self.gelu_seasonal(s_temporal)
        s_temporal = self.dropout_seasonal(s_temporal)
        s_temporal = self.linear_seasonal2(s_temporal).view(B, C, self.pred_len)


        # attn = self.temporal_pool(s_patch)          # [BC, P, 1]
        # attn = torch.softmax(attn, dim=1)           # normalize theo patch

        # # weighted sum
        # pooled = (s_patch * attn).sum(dim=1)       # [BC, D]

        # pooled = pooled + s_patch.mean(dim=1)

        # # project
        # s_temporal = self.out_linear(pooled).view(B, C, self.pred_len)

        s = self.adaptive_fusion(s_channel, s_temporal)
        s = s.view(B, C, self.pred_len)


        # ======================
        # 🔥 4. TREND (KHÔNG ĐỤNG)
        # ======================
        t = t.reshape(B * C, I)

        t = self.trend(t).view(B, C, self.pred_len)

        # ======================
        # 🔥 5. FINAL
        # ======================
        x = s + t
        x = x.permute(0, 2, 1)

        return x