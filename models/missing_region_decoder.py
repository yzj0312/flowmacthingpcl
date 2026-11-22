"""Decode fixed-slot missing-region latents into XYZ; no observed-point input."""
import torch
from torch import nn


class MissingRegionDecoder(nn.Module):
    def __init__(self, dim=256, heads=8, points=1024):
        super().__init__()
        if points < 1 or dim < 1 or heads < 1 or dim % heads:
            raise ValueError('points/dim/heads must be positive; dim divisible by heads')
        self.dim = dim
        self.queries = nn.Parameter(torch.randn(1, points, dim) * 0.02)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0, batch_first=True)
        self.xyz = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim),
                                 nn.GELU(), nn.Linear(dim, 3))

    def forward(self, latent):
        if (latent.ndim != 3 or latent.shape[-1] != self.dim
                or min(latent.shape[:2]) < 1 or not latent.is_floating_point()):
            raise ValueError('latent must be floating point [B>0,L>0,dim]')
        queries = self.queries.expand(latent.shape[0], -1, -1)
        update, _ = self.attention(queries, latent, latent, need_weights=False)
        return self.xyz(queries + update)


