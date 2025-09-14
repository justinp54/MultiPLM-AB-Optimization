import torch
import torch.nn.functional as F
from torch import nn

class TransposeAndLN(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.ln(x)
        x = x.transpose(1, 2)
        return x

def make_conv_branch(in_channels, hidden_dim, kernel_size, pooling_size):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=hidden_dim * 2, kernel_size=kernel_size),
        TransposeAndLN(hidden_dim * 2),
        nn.LeakyReLU(0.3),
        nn.MaxPool1d(kernel_size=pooling_size),
        nn.AdaptiveMaxPool1d(1)
    )

class AbAgNet(nn.Module):
    def __init__(
        self,
        seq_size=2000,
        dim=20,
        hidden_dim=64,
        kernel_sizes=(3, 5, 7),
        pooling_size=2,
        dense_dim=128,
        dropout_rate=0.5
    ):
        super(AbAgNet, self).__init__()
        self.branches = nn.ModuleList([
            make_conv_branch(dim, hidden_dim, ks, pooling_size) 
            for ks in kernel_sizes
        ])
        self.attn_ab = nn.Linear(hidden_dim * 2, 1, bias=False)
        self.attn_ag = nn.Linear(hidden_dim * 2, 1, bias=False)
        self.fc_ab = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.ln_fc_ab = nn.LayerNorm(hidden_dim * 2)
        self.fc_ag = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.ln_fc_ag = nn.LayerNorm(hidden_dim * 2)
        self.fc_wt = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.ln_fc_wt = nn.LayerNorm(hidden_dim * 2)
        self.fc_mut = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.ln_fc_mut = nn.LayerNorm(hidden_dim * 2)
        self.fc1 = nn.Linear(hidden_dim * 4, dense_dim)
        self.ln_fc1 = nn.LayerNorm(dense_dim)
        self.fc2 = nn.Linear(dense_dim, dense_dim // 2)
        self.ln_fc2 = nn.LayerNorm(dense_dim // 2)
        self.fc3 = nn.Linear(dense_dim // 2, 1)
        self.leaky_relu = nn.LeakyReLU(0.3)
        self.dropout = nn.Dropout(dropout_rate)

    def _attention_aggregate(self, x, branches, attn_proj):
        outs = [branch(x).squeeze(-1) for branch in branches]
        stacked = torch.stack(outs, dim=1)
        attn_scores = attn_proj(stacked)
        alpha = F.softmax(attn_scores, dim=1)
        weighted_out = (stacked * alpha).sum(dim=1)
        return weighted_out

    def forward(self, seq1, seq2, seq3, seq4):
        seq1 = seq1.permute(0, 2, 1)
        seq2 = seq2.permute(0, 2, 1)
        seq3 = seq3.permute(0, 2, 1)
        seq4 = seq4.permute(0, 2, 1)
        x1 = self._attention_aggregate(seq1, self.branches, self.attn_ab)
        x2 = self._attention_aggregate(seq2, self.branches, self.attn_ag)
        x3 = self._attention_aggregate(seq3, self.branches, self.attn_ab)
        x4 = self._attention_aggregate(seq4, self.branches, self.attn_ag)
        x1 = self.leaky_relu(self.ln_fc_ab(self.fc_ab(x1)))
        x2 = self.leaky_relu(self.ln_fc_ag(self.fc_ag(x2)))
        x3 = self.leaky_relu(self.ln_fc_ab(self.fc_ab(x3)))
        x4 = self.leaky_relu(self.ln_fc_ag(self.fc_ag(x4)))
        wt = torch.cat([x1, x2], dim=1)
        wt = self.leaky_relu(self.ln_fc_wt(self.fc_wt(wt)))
        mut = torch.cat([x3, x4], dim=1)
        mut = self.leaky_relu(self.ln_fc_mut(self.fc_mut(mut)))
        merge_vector = torch.cat([wt, mut], dim=1)
        x = self.leaky_relu(self.ln_fc1(self.fc1(merge_vector)))
        x = self.leaky_relu(self.fc2(x))
        ddg_output = self.fc3(x)
        return ddg_output, x1, x2, x3, x4
