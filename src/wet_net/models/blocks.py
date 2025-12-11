"""
Reusable attention blocks with continuous RoPE (from the original notebook).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousRoPE(nn.Module):
    def __init__(self, d_model: int, max_freq: float = 10000.0):
        super().__init__()
        half_dim = d_model // 2
        emb = math.log(max_freq) / (half_dim - 1)
        self.register_buffer("freqs", torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        args = t.unsqueeze(-1) * self.freqs.view(1, 1, -1)
        cos = torch.cos(args).repeat(1, 1, 2)
        sin = torch.sin(args).repeat(1, 1, 2)
        return cos, sin


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CustomSelfAttentionBlock(nn.Module):
    """Manual Transformer block that injects RoPE."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head**-0.5

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.rope = ContinuousRoPE(self.d_head)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        b, T, d = x.shape
        shortcut = x
        x = self.norm1(x)

        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t_: t_.view(b, T, self.n_heads, self.d_head).transpose(1, 2), qkv)

        cos, sin = self.rope(x, t)
        cos = cos.view(b, 1, T, self.d_head)
        sin = sin.view(b, 1, T, self.d_head)
        q, k = apply_rope(q, k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(b, T, d)
        x = self.proj(x)
        x = x + shortcut
        x = x + self.mlp(self.norm2(x))
        return x
