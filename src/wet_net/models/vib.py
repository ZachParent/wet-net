from __future__ import annotations

import torch
import torch.nn as nn


class DualVIB(nn.Module):
    def __init__(self, d_model: int, d_content: int, d_style: int):
        super().__init__()
        self.content_mu = nn.Linear(d_model, d_content)
        self.content_logvar = nn.Linear(d_model, d_content)
        self.style_mu = nn.Linear(d_model, d_style)
        self.style_logvar = nn.Linear(d_model, d_style)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, z):
        c_mu = self.content_mu(z)
        c_logvar = self.content_logvar(z)
        s_mu = self.style_mu(z)
        s_logvar = self.style_logvar(z)
        c = self.reparameterize(c_mu, c_logvar)
        s = self.reparameterize(s_mu, s_logvar)
        return c, s, c_mu, c_logvar, s_mu, s_logvar


class VIBTransformer(nn.Module):
    def __init__(
        self, input_dim: int, seq_len: int, d_model: int, d_content: int, d_style: int, nhead: int, layers: int
    ):
        super().__init__()
        self.seq_len = seq_len
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.vib = DualVIB(d_model, d_content, d_style)
        self.decoder = nn.Sequential(nn.Linear(d_content + d_style, d_model), nn.GELU(), nn.Linear(d_model, input_dim))
        self.cls_head = nn.Linear(d_content, 1)

    def forward(self, x):
        emb = self.proj(x) + self.pos_embed[:, : x.size(1), :]
        z = self.encoder(emb)
        c, s, c_mu, c_logvar, s_mu, s_logvar = self.vib(z)
        recon = self.decoder(torch.cat([c, s], dim=-1))
        logits = self.cls_head(c[:, -1, :])
        return recon, logits, c_mu, c_logvar, s_mu, s_logvar
