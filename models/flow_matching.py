"""Conditional rectified flow over missing-region latent sets (B, K, D)."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def missing_region(partial, complete, count):
    """Approximate missing GT by largest nearest-observed distances, in chunks."""
    if not 0 < count <= complete.shape[1]:
        raise ValueError('missing point count must be in [1, number of GT points]')
    with torch.no_grad():
        distances = torch.cat([
            torch.cdist(chunk.float(), partial.float()).amin(dim=-1)
            for chunk in complete.split(256, dim=1)
        ], dim=1)
        indices = distances.topk(count, dim=1, sorted=True).indices
    return complete.gather(1, indices.unsqueeze(-1).expand(-1, -1, 3))


class MissingSetEncoder(nn.Module):
    """Learned slots give permutation-invariant, consistently indexed targets."""
    def __init__(self, dim, tokens, heads):
        super().__init__()
        self.point_embed = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.queries = nn.Parameter(torch.randn(1, tokens, dim) * 0.02)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, points):
        features = self.point_embed(points)
        queries = self.queries.expand(points.shape[0], -1, -1)
        slots = self.attention(queries, features, features, need_weights=False)[0]
        return self.norm(queries + slots)


class ConditionalVelocity(nn.Module):
    def __init__(self, dim, heads, depth):
        super().__init__()
        self.dim = dim
        self.time = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(dim, heads, dim * 4, dropout=0.,
                                       activation='gelu', batch_first=True)
            for _ in range(depth)
        ])
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, z, t, condition):
        frequencies = torch.exp(torch.arange(0, self.dim, 2, device=z.device,
                                              dtype=z.dtype) * (-math.log(10000.) / self.dim))
        angles = t[:, None] * frequencies[None] * 1000.
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)[:, :self.dim]
        z = z + self.time(embedding).unsqueeze(1)
        for block in self.blocks:
            z = block(z, condition)
        return self.output(z)


class LatentFlowMatching(nn.Module):
    def __init__(self, dim, tokens, heads, depth):
        super().__init__()
        self.tokens, self.dim = tokens, dim
        self.modality = nn.Parameter(torch.randn(2, 1, dim) * 0.02)
        self.condition_norm = nn.LayerNorm(dim)
        self.velocity = ConditionalVelocity(dim, heads, depth)
        self.slot_embed = nn.Parameter(torch.randn(1, tokens, dim) * 0.02)

    def condition(self, point_tokens, image_tokens):
        return self.condition_norm(torch.cat((point_tokens + self.modality[0],
                                               image_tokens + self.modality[1]), dim=1))

    def field(self, z, t, condition):
        return self.velocity(z + self.slot_embed, t, condition)

    def loss(self, target, condition):
        target = target.detach()
        noise = torch.randn_like(target)
        t = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
        z = (1 - t[:, None, None]) * noise + t[:, None, None] * target
        return F.mse_loss(self.field(z, t, condition), target - noise)

    def sample(self, condition, steps, noise=None):
        if steps < 1:
            raise ValueError('flow integration steps must be positive')
        shape = (condition.shape[0], self.tokens, self.dim)
        z = torch.randn(shape, device=condition.device, dtype=condition.dtype) if noise is None else noise
        if z.shape != shape or z.device != condition.device or z.dtype != condition.dtype:
            raise ValueError('noise must match latent shape, device and dtype')
        for step in range(steps):
            t = z.new_full((z.shape[0],), step / steps)
            z = z + self.field(z, t, condition) / steps
        return z


class MissingSetDecoder(nn.Module):
    def __init__(self, dim, tokens, points):
        super().__init__()
        if points % tokens:
            raise ValueError('missing_points must be divisible by latent_tokens')
        self.points_per_token = points // tokens
        self.decode = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(),
                                    nn.Linear(dim * 2, self.points_per_token * 3))

    def forward(self, z):
        return self.decode(z).reshape(z.shape[0], -1, 3).contiguous()
