"""Missing-region target encoder, independent of multimodal condition weights.

Input: sampled missing points [B, N, 3], normally N=1024.
Output: fixed learned-slot latent [B, 64, 256] by default.
Pretrain with a reconstruction decoder before freezing for flow matching.
"""
import torch
from torch import nn

from models.condition_encoders import PointNet2Encoder


class TargetEncoder(nn.Module):
    def __init__(self, dim=256, latent_tokens=64, heads=8,
                 point_counts=(256, 128), point_radii=(0.1, 0.2),
                 neighbors=32, point_depth=2, latent_depth=2):
        super().__init__()
        if dim < 1 or heads < 1 or dim % heads:
            raise ValueError('dim must be positive and divisible by heads')
        if latent_tokens < 1 or latent_depth < 0:
            raise ValueError('latent_tokens must be positive; latent_depth nonnegative')
        self.dim = dim
        self.latent_tokens = latent_tokens
        # Construct a new backbone; never reuse condition-encoder parameters.
        self.point_encoder = PointNet2Encoder(
            dim=dim, counts=point_counts, radii=point_radii,
            neighbors=neighbors, depth=point_depth, heads=heads)
        self.queries = nn.Parameter(torch.empty(1, latent_tokens, dim))
        nn.init.normal_(self.queries, std=0.02)
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            dim, heads, dropout=0, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                 nn.Linear(dim * 4, dim))
        self.latent_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0,
                                       batch_first=True, norm_first=True)
            for _ in range(latent_depth)])
        self.output_norm = nn.LayerNorm(dim)
        self._frozen = False

    def forward(self, missing_points, return_details=False):
        if (missing_points.ndim != 3 or missing_points.shape[-1] != 3
                or missing_points.shape[0] < 1 or missing_points.shape[1] < 1):
            raise ValueError('missing_points must have shape [B>0,N>0,3]')
        if not missing_points.is_floating_point():
            raise ValueError('missing_points must be floating point')
        features, centers = self.point_encoder(missing_points)
        queries = self.queries.expand(missing_points.shape[0], -1, -1)
        memory = self.memory_norm(features)
        update, _ = self.cross_attention(
            self.query_norm(queries), memory, memory, need_weights=False)
        latent = queries + update
        latent = latent + self.ffn(self.ffn_norm(latent))
        for block in self.latent_blocks:
            latent = block(latent)
        latent = self.output_norm(latent)
        if return_details:
            return {'latent': latent, 'local_features': features, 'centers': centers}
        return latent

    def freeze(self):
        """After reconstruction pretraining, freeze weights AND BN statistics."""
        self._frozen = True
        self.requires_grad_(False)
        self.eval()
        return self

    def unfreeze(self):
        """Enable reconstruction training; caller may then select train/eval mode."""
        self._frozen = False
        self.requires_grad_(True)
        return self

    def train(self, mode=True):
        return super().train(False if self._frozen else mode)
