"""Two-stage missing-region autoencoding and conditional latent flow matching."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.target_encoder import TargetEncoder
from models.missing_region_decoder import MissingRegionDecoder
from models.condition_encoders import MultimodalConditionEncoder, fps_indices, index_points


def chamfer_loss(prediction, target):
    """Symmetric squared Chamfer distance (CD-L2), averaged over both sets."""
    distance = torch.cdist(prediction, target).square()
    return (distance.amin(1).mean() + distance.amin(2).mean()) / 2


@torch.no_grad()
def extract_missing_region(gt, partial, threshold=0.03, count=1024, chunk_size=512):
    """Threshold GT-to-observation distance, FPS downsample or repeat to fixed size.

    Coordinates MUST share scale/frame. Empty candidate sets raise instead of
    silently inventing missing geometry. Threshold is in dataset coordinate units.
    """
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer')
    if threshold <= 0 or count < 1:
        raise ValueError('threshold/count must be positive')
    if (gt.ndim != 3 or partial.ndim != 3 or gt.shape[-1] != 3
            or partial.shape[-1] != 3 or gt.shape[0] != partial.shape[0]
            or min(gt.shape[0], gt.shape[1], partial.shape[1]) < 1):
        raise ValueError('expected nonempty paired [B,N,3] clouds')
    regions = []
    # Never allocate a dataset-sized pairwise matrix. Bound both point axes.
    for sample_index, (points, observed) in enumerate(zip(gt, partial)):
        minimums = []
        for query in points.split(chunk_size):
            nearest = query.new_full((query.shape[0],), float('inf'))
            for support in observed.split(chunk_size):
                block = torch.cdist(query[None], support[None],
                                    compute_mode='donot_use_mm_for_euclid_dist')[0]
                nearest = torch.minimum(nearest, block.amin(-1))
            minimums.append(nearest)
        distances = torch.cat(minimums)
        candidates = points[distances > threshold]
        if candidates.shape[0] == 0:
            raise ValueError(f'No missing candidates for batch sample {sample_index}; inspect threshold/alignment or filter sample')
        if candidates.shape[0] >= count:
            region = index_points(candidates[None], fps_indices(candidates[None], count))[0]
        else:
            region = candidates.repeat(math.ceil(count / candidates.shape[0]), 1)[:count]
        regions.append(region)
    return torch.stack(regions)


class MissingRegionAutoencoder(nn.Module):
    def __init__(self, dim=256, heads=8, latent_tokens=64, points=1024,
                 encoder_options=None):
        super().__init__()
        self.encoder = TargetEncoder(dim, latent_tokens, heads, **(encoder_options or {}))
        self.decoder = MissingRegionDecoder(dim, heads, points)

    def forward(self, missing):
        return self.decoder(self.encoder(missing))

    def reconstruction_loss(self, missing):
        return chamfer_loss(self(missing), missing)


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        frequency = torch.exp(-math.log(10000) * torch.arange(dim // 2) / max(dim // 2 - 1, 1))
        self.register_buffer('frequency', frequency)
        self.mlp = nn.Sequential(nn.Linear(2 * (dim // 2), dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        phase = t[:, None] * self.frequency[None] * 1000
        return self.mlp(torch.cat((phase.sin(), phase.cos()), -1))


class ConditionalVelocity(nn.Module):
    def __init__(self, dim=256, heads=8, depth=4, latent_tokens=64):
        super().__init__()
        if depth < 1:
            raise ValueError('velocity depth must be positive')
        self.slots = nn.Parameter(torch.randn(1, latent_tokens, dim) * 0.02)
        self.time = TimeEmbedding(dim)
        self.blocks = nn.ModuleList([nn.TransformerDecoderLayer(
            dim, heads, 4 * dim, dropout=0, batch_first=True, norm_first=True)
            for _ in range(depth)])
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, z, t, context):
        if z.shape[1:] != self.slots.shape[1:] or t.shape != (z.shape[0],):
            raise ValueError('expected z [B,slots,dim], t [B]')
        hidden = z + self.slots
        time = self.time(t)[:, None]
        for block in self.blocks:
            hidden = block(hidden + time, context)
        return self.output(hidden)


class LatentCompletion(nn.Module):
    """Requires a RECONSTRUCTION-PRETRAINED autoencoder, supplied explicitly."""
    def __init__(self, pretrained_autoencoder, condition_encoder=None, heads=8, depth=4):
        super().__init__()
        self.autoencoder = pretrained_autoencoder.requires_grad_(False).eval()
        self.autoencoder.encoder.freeze()
        dim = self.autoencoder.encoder.dim
        slots = self.autoencoder.encoder.latent_tokens
        self.condition_encoder = (condition_encoder if condition_encoder is not None
                                  else MultimodalConditionEncoder(dim=dim, heads=heads))
        self.velocity = ConditionalVelocity(dim, heads, depth, slots)

    def train(self, mode=True):
        super().train(mode)
        self.autoencoder.eval()
        return self

    def forward(self, partial, image, missing):
        """Training loss; target missing geometry is never given to condition branch."""
        with torch.no_grad():
            z1 = self.autoencoder.encoder(missing)
        z0 = torch.randn_like(z1)
        t = torch.rand(z1.shape[0], device=z1.device, dtype=z1.dtype)
        zt = (1 - t[:, None, None]) * z0 + t[:, None, None] * z1
        context = self.condition_encoder(partial, image)['context']
        predicted = self.velocity(zt, t, context)
        return F.mse_loss(predicted, z1 - z0)

    @torch.no_grad()
    def sample(self, partial, image, steps=32, noise=None):
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError('steps must be a positive integer')
        was_training = self.training
        self.eval()
        try:
            context = self.condition_encoder(partial, image)['context']
            shape = (partial.shape[0], self.autoencoder.encoder.latent_tokens,
                     self.autoencoder.encoder.dim)
            z = context.new_empty(shape).normal_() if noise is None else noise.clone()
            if z.shape != shape or z.device != context.device or z.dtype != context.dtype:
                raise ValueError('noise shape/device/dtype must match conditional latent')
            for step in range(steps):
                t = z.new_full((shape[0],), step / steps)
                z = z + self.velocity(z, t, context) / steps
            missing = self.autoencoder.decoder(z)
            return {'latent': z, 'missing': missing,
                    'complete': torch.cat((partial, missing), dim=1)}
        finally:
            self.train(was_training)
