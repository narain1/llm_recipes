import torch
from torch import nn
import torch.nn.functional as F
import math

class MHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.depth = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        pass


class MHA(nn.Module):
    def __init__(self, d_model, n_heads, qkv_bias=False, qk_norm=False, attn_drop: float=0.0, proj_drop: float = 0.0, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.depth = d_model // n_heads

        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.depth).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, rate=0.1):
        super().__init__()
        self.mha = MHA(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(rate)
        self.drop2 = nn.Dropout(rate)

    def forward(self, x, mask):
        attn_output = self.mha(x)
        x = self.ln1(x + self.drop1(attn_output))
        ffn_output = self.ffn(x)
        x = self.ln2(x + self.drop2(ffn_output))
        return x

class Bert(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers):
        super().__init__()
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff)])

    def forward(self, x, mask):
        for layer in self.enc_layers:
            x = layer(x, mask)
        return x
