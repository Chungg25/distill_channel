import torch
from torch import nn

class PatchChannelGLU(nn.Module):
        def __init__(self, patch_len, d_model):
            super().__init__()
            self.linear_a = nn.Linear(patch_len, d_model)
            self.linear_b = nn.Linear(patch_len, d_model)
        def forward(self, x):  # x: [Batch*Channel, Patch_num, Patch_len]
            a = self.linear_a(x)
            b = torch.sigmoid(self.linear_b(x))
            return a * b
        

class PatchChannelGLUMix(nn.Module):
    def __init__(self, patch_len, in_channels, d_model):
        super().__init__()
        self.linear_a = nn.Linear(in_channels * patch_len, d_model)
        self.linear_b = nn.Linear(in_channels * patch_len, d_model)
    def forward(self, x):  # x: [Batch, Patch_num, Channel, Patch_len]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [Batch, Patch_num, Channel*Patch_len]
        a = self.linear_a(x)
        b = torch.sigmoid(self.linear_b(x))
        return a * b  # [Batch, Patch_num, d_model]
        
class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, droup_out = 0, d_model=64, nhead=4, num_layers=2):
        super(Network, self).__init__()
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = d_model
        self.drop_out = droup_out
        self.patch_num = (seq_len - patch_len)//stride + 1

        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1

        # Patch Channel GLU + Patch Embedding
        self.patch_channel_glu = PatchChannelGLU(patch_len, d_model)
        
        self.patch_embed = nn.Linear(d_model, d_model)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=self.drop_out)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Flatten Head
        self.flatten = nn.Flatten(start_dim=-2)
        self.fc_out = nn.Sequential(
            nn.Linear(self.patch_num * d_model, pred_len * 2),
            nn.GELU(),
            nn.Dropout(self.drop_out),
            nn.Linear(pred_len * 2, pred_len),
            nn.Dropout(self.drop_out)
        )


        # Linear Stream (trend)
        self.fc_trend2 = nn.Linear(seq_len, pred_len * 2)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len)
        self.fc_trend3 = nn.Linear(pred_len, pred_len)
        # Streams Concatination
        self.fc_concat = nn.Sequential(
            nn.Linear(pred_len * 2, pred_len),
            nn.Dropout(self.drop_out)
        )

    def forward(self, s, t):
        # s: [Batch, Input, Channel] (seasonality)
        # t: [Batch, Input, Channel] (trend)
        s = s.permute(0,2,1) # [Batch, Channel, Input]
        t = t.permute(0,2,1)
        B, C, I = s.shape
        s = torch.reshape(s, (B*C, I)) # [Batch*Channel, Input]
        t = torch.reshape(t, (B*C, I))
        # Patching
        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride) # [Batch*Channel, Patch_num, Patch_len]
        
        # Patch Channel GLU
        s = self.patch_channel_glu(s) # [Batch*Channel, Patch_num, d_model]
        
        # Patch Embedding
        s = self.patch_embed(s) # [Batch*Channel, Patch_num, d_model]
        
        # Causal Masking
        device = s.device
        patch_num = s.shape[1]
        mask = torch.triu(torch.ones(patch_num, patch_num, device=device), diagonal=1).bool()
        
        # Transformer Encoder with mask
        s = self.transformer_encoder(s, mask=mask) # [Batch*Channel, Patch_num, d_model]
        
        # Flatten Head
        s = self.flatten(s) # [Batch*Channel, Patch_num*d_model]
        s = self.fc_out(s) # [Batch*Channel, pred_len]


        # Linear Stream (trend)
        t = self.fc_trend2(t)
        t = self.avgpool2(t)
        t = self.ln2(t)
        t = self.fc_trend3(t)

        # Streams Concatination
        x = torch.cat((s, t), dim=1) # [Batch*Channel, pred_len*2]
        x = self.fc_concat(x) # [Batch*Channel, pred_len]

        # x = s + t
        # Channel concatination
        x = torch.reshape(x, (B, C, self.pred_len)) # [Batch, Channel, Output]
        x = x.permute(0,2,1) # [Batch, Output, Channel]
        return x