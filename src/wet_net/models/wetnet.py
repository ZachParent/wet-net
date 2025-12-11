"""
Tri-task model renamed to WetNet.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from wet_net.models.blocks import CustomSelfAttentionBlock


class CyclicUniTSBackbone(nn.Module):
    def __init__(self, input_dim: int, d_model: int, n_heads: int, n_layers: int, dropout: float, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([CustomSelfAttentionBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("base_time", torch.linspace(0, 1, steps=seq_len).unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        time_index = self.base_time.to(h.device).expand(h.size(0), -1)
        for block in self.layers:
            h = block(h, time_index)
        return self.norm(self.dropout(h))


class WetNet(nn.Module):
    """
    Multi-head network: reconstruction + 24h forecast + two anomaly heads.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        forecast_horizon: int,
        short_count: int,
        long_count: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.backbone = CyclicUniTSBackbone(input_dim, d_model, n_heads, n_layers, dropout, seq_len)
        self.reconstruction = nn.Linear(d_model, input_dim)
        self.forecast = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, forecast_horizon)
        )
        self.short_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, short_count))
        self.long_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, long_count))

    def forward(self, x: torch.Tensor):
        latent = self.backbone(x)
        pooled = latent.mean(dim=1)
        return {
            "latent": latent,
            "reconstruction": self.reconstruction(latent),
            "forecast": self.forecast(pooled),
            "short_logits": self.short_head(pooled),
            "long_logits": self.long_head(pooled),
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latent = self.backbone(x)
            return latent.mean(dim=1)
