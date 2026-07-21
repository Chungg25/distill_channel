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

        self.group_tokens = nn.Parameter(torch.randn(num_groups, d_model))

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
                "norm2": nn.LayerNorm(d_model)
            })
            for _ in range(num_layers)
        ])

        self.channel_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, pred_len*expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_len*expand, pred_len)
        )

    def forward(self, x):
        # x: [B, C, T]
        B, C, T = x.shape

        x_embed = self.embed(x)  # [B, C, D]

        context = x_embed.mean(dim=1, keepdim=True)  # [B, 1, D]

        group_tokens = self.group_tokens.unsqueeze(0) + context  # [B, G, D]

        sim = torch.einsum('bcd,bgd->bcg', x_embed, group_tokens)

        assign  = torch.softmax(sim / self.temperature, dim=-1)

        self._last_assign = assign

        group_feat = torch.einsum('bcg,bcd->bgd', assign, x_embed)
        group_feat = group_feat / (assign.sum(dim=1).unsqueeze(-1) + 1e-6)

        for layer in self.layers:
            residual = group_feat
            attn_out, _ = layer["attn"](group_feat, group_feat, group_feat)
            group_feat = layer["norm1"](residual + attn_out)

            residual = group_feat
            ffn_out = layer["ffn"](group_feat)
            group_feat = layer["norm2"](residual + ffn_out)

        out = torch.einsum('bcg,bgd->bcd', assign, group_feat)

        gate = self.channel_gate(x_embed)
        out = out * gate + x_embed  

        out = self.head(out)

        return out

    def get_aux_losses(self):
        """
        Compute the three MoE auxiliary losses.

        Returns:
            dict with keys 'diversity', 'balance', 'sharpness'
        """
        losses = {}

        # --- Diversity Loss: L_div = ||G̃ G̃ᵀ − I||²_F ---
        # Forces group tokens to be orthogonal (non-redundant experts)
        G_norm = F.normalize(self.group_tokens, dim=-1)  # [G, D]
        gram = G_norm @ G_norm.T                          # [G, G]
        I = torch.eye(self.num_groups, device=gram.device)
        losses['diversity'] = (gram - I).pow(2).mean()
        self._last_gram = gram 
        # --- Balance Loss: L_bal = Σ_g (p̄_g − 1/G)² ---
        # Forces channels to be distributed evenly across groups
        assign = self._last_assign               # [B, C, G]
        p_bar = assign.mean(dim=1).mean(dim=0)   # [G]
        losses['balance'] = ((p_bar - 1.0 / self.num_groups) ** 2).sum()
        self._last_p_bar = p_bar 

        # --- Sharpness Loss (Entropy): L_sharp = −(1/C)ΣΣ A_{c,g} log A_{c,g} ---
        # Minimizing entropy forces decisive (one-hot-like) assignments
        eps = 1e-8
        log_assign = (assign + eps).log()
        entropy = -(assign * log_assign).sum(dim=-1)
        losses['sharpness'] = entropy.mean()
        self._last_entropy = entropy 
        losses['_num_groups'] = self.num_groups 

        return losses

class DecorrelationLoss(nn.Module):
    """
    L_decor = Σ_{i≠j} ( cos_sim(Y_i, Y_j) )²

    Penalizes cosine similarity between branch outputs to enforce
    orthogonal (complementary) feature learning.
    """

    def forward(self, *branch_outputs):
        loss = 0.0
        n = len(branch_outputs)
        for i in range(n):
            for j in range(i + 1, n):
                yi = branch_outputs[i].reshape(branch_outputs[i].shape[0], -1)
                yj = branch_outputs[j].reshape(branch_outputs[j].shape[0], -1)
                cos_sim = F.cosine_similarity(yi, yj, dim=-1)  # [B]
                loss = loss + (cos_sim ** 2).mean()
        return loss

class AdaptiveAuxLossWeighter(nn.Module):
    """
    lambda_i = lambda_base_i * tanh(signal_i / tau)
    """

    def __init__(self, lambda_decor=0.005, lambda_div=0.01,
                 lambda_bal=0.05, lambda_sharp=0.05, tau_init=0.3, ema_decay=0.9):
        super().__init__() 
        self.lambda_base = {
            'decorrelation': lambda_decor,
            'diversity': lambda_div,
            'balance': lambda_bal,
            'sharpness': lambda_sharp,
        }
        self.ema_decay = ema_decay
        self.register_buffer('tau_decor', torch.tensor(tau_init))
        self.register_buffer('tau_div', torch.tensor(tau_init))
        self.register_buffer('tau_bal', torch.tensor(tau_init))
        self.register_buffer('tau_sharp', torch.tensor(tau_init))
        self.tau_dict = {
            'decorrelation': 'tau_decor', 'diversity': 'tau_div',
            'balance': 'tau_bal', 'sharpness': 'tau_sharp',
        }


    @torch.no_grad()
    def compute_weights(self, aux_losses, current_epoch=0, max_epochs=1):
        weights = {}
        if max_epochs > 1:
            decay_factor = 0.5 * (1 + math.cos(math.pi * current_epoch / max_epochs))
        else:
            decay_factor = 1.0
        # Lấy chính xác số groups từ cấu trúc mạng (default 128 đề phòng lỗi)
        G = aux_losses.get('_num_groups', 128)
        max_entropy = math.log(G)
        for key, loss_tensor in aux_losses.items():
            # Bỏ qua các key dùng làm metadata (bắt đầu bằng '_')
            if key.startswith('_'):
                continue
            val = loss_tensor.detach()
            if val.dim() > 0: val = val.mean()
            # Tái sử dụng giá trị Loss làm Signal
            if key == 'decorrelation':
                signal = torch.sqrt(val)  
            elif key == 'diversity':
                signal = torch.sqrt(val)  
            elif key == 'balance':
                signal = torch.sqrt(val)  
            elif key == 'sharpness':
                # Tính chuẩn xác tuyệt đối không cần ước lượng
                signal = (val / max_entropy).clamp(0, 1)
            else:
                signal = val
            # Cập nhật EMA Tau
            tau_name = self.tau_dict[key]
            current_tau = getattr(self, tau_name)
            if self.training:
                new_tau = self.ema_decay * current_tau + (1 - self.ema_decay) * signal
                current_tau.copy_(new_tau.clamp(min=1e-4))
            
            # Tính weight
            current_lambda = self.lambda_base[key] * decay_factor
            weights[key] = current_lambda * torch.tanh(signal / current_tau)
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

        self.kernel = nn.Parameter(
            torch.eye(2)
            .unsqueeze(0)
            .repeat(self.F, 1, 1)
        )
        self.beta = nn.Parameter(
            torch.tensor(0.1)
        )

        self.norm = nn.LayerNorm(seq_len)

        self.dropout = nn.Dropout(dropout)

        self.proj = nn.Sequential(
            nn.Linear(seq_len, pred_len*expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_len*expand, pred_len)
        )

    def forward(self, x):
        """
        x: [B,C,T]
        """

        B, C, T = x.shape

        x_freq = torch.fft.rfft(
            x,
            dim=-1
        )
        # [B,C,F]

        Re = x_freq.real
        Im = x_freq.imag

        real_new = (
            self.kernel[:, 0, 0] * Re +
            self.kernel[:, 0, 1] * Im
        )

        imag_new = (
            self.kernel[:, 1, 0] * Re +
            self.kernel[:, 1, 1] * Im
        )

        x_freq_refined = torch.complex(
            real_new,
            imag_new
        )

        x_time_refined = torch.fft.irfft(
            x_freq_refined,
            n=T,
            dim=-1
        )


        x_time = x + self.beta * x_time_refined

        x_time = self.norm(x_time)
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
        self.future_patch_num = math.ceil(
            pred_len / patch_len
        )
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        self.patch_num = patch_num
        self.alpha = nn.Parameter(torch.ones(1, 7, 1))

        self.decor_loss = DecorrelationLoss()
        self.aux_weighter = AdaptiveAuxLossWeighter()

        self.seasonal_channel = GroupChannelBlock(
            seq_len,
            pred_len,
            d_model=d_model,
            num_groups=num_groups,  
            nhead=nhead,
            dropout=dropout,
            expand=expand
        )

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

        self.patch_forecast = nn.Sequential(
            nn.Linear(self.patch_num, self.future_patch_num),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.future_patch_num, self.future_patch_num)
        )

        self.patch_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, patch_len)
        )
        
        self.patch_importance = nn.Linear(d_model,1)
        self.spectral = SpectralTimeBlock(seq_len, pred_len, expand)


    def forward(self, x1, x2, current_epoch=0, max_epochs=1):
        # s, t: [B, seq_len, C]

        x1 = x1.permute(0, 2, 1)  # [B, C, T]
        x2 = x2.permute(0, 2, 1)

        B, C, I = x1.shape

        channel = self.seasonal_channel(x1)   # [B, C, pred_len]

        s_flat = x1.reshape(B * C, I)

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

        importance = torch.softmax(
            self.patch_importance(s_patch),
            dim=1
        )

        s_patch = s_patch * importance

        x = s_patch

        x = x.transpose(1, 2)

        x = self.patch_forecast(x)

        x = x.transpose(1, 2)

        x = self.patch_decoder(x)

        x = x.reshape(
            B * C,
            self.future_patch_num * self.patch_len
        )

        temporal = x.view(
            B,
            C,
            self.pred_len
        )

        alpha = torch.sigmoid(self.alpha)
        s = alpha * channel + (1 - alpha) * temporal

        # s = channel + temporal

        spectral = self.spectral(x2).view(B, C, self.pred_len)
        
        x = s + spectral
        x = x.permute(0, 2, 1)

        aux_losses = {}
        if self.training:
            aux_losses['decorrelation'] = self.decor_loss(
                channel, temporal, spectral
            )
            aux_losses.update(self.seasonal_channel.get_aux_losses())

            adaptive_w = self.aux_weighter.compute_weights(
                aux_losses,
                current_epoch=current_epoch,
                max_epochs=max_epochs
            )
            aux_losses['_adaptive_weights'] = adaptive_w
            

        return x, aux_losses
    
