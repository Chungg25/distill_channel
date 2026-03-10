import torch
import torch.nn as nn


class CausalGraphLearner(nn.Module):

    def __init__(self, channels, max_lag):

        super().__init__()

        self.C = channels
        self.K = max_lag

        self.W = nn.Parameter(
            torch.randn(max_lag, channels, channels)
        )

    def forward(self):

        G = torch.softmax(self.W, dim=1)

        return G


def causal_propagation(x, G):

    """
    x : B T C
    G : K C C
    """

    B, T, C = x.shape
    K = G.shape[0]

    out = torch.zeros_like(x)

    for k in range(1, K + 1):

        lag_x = x[:, :-k, :]

        Gk = G[k - 1]

        agg = torch.einsum(
            "ij,btj->bti",
            Gk,
            lag_x
        )

        out[:, k:, :] += agg

    return out


class TeacherModel(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.pred_len = args.pred_len
        self.seq_len = args.seq_len
        self.channels = args.enc_in
        self.max_lag = 5

        self.graph = CausalGraphLearner(
            self.channels,
            self.max_lag
        )

        self.linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):

        """
        x : B T C
        """

        G = self.graph()

        causal_feat = causal_propagation(x, G)

        x = x + causal_feat

        x = x.permute(0, 2, 1)

        y = self.linear(x)

        y = y.permute(0, 2, 1)

        return y, G