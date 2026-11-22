"""DINOv2 and PointNet++ SSG condition encoders; no custom CUDA ops required."""
from pathlib import Path

import torch
from torch import nn


def index_points(points, indices):
    batch = torch.arange(points.shape[0], device=points.device)
    batch = batch.reshape((-1,) + (1,) * (indices.ndim - 1))
    return points[batch, indices]


@torch.no_grad()
def fps_indices(xyz, count):
    count = min(count, xyz.shape[1])
    distance = xyz.new_full(xyz.shape[:2], float('inf'))
    farthest = (xyz - xyz.mean(1, keepdim=True)).square().sum(-1).argmax(1)
    indices = torch.empty(xyz.shape[0], count, device=xyz.device, dtype=torch.long)
    for i in range(count):
        indices[:, i] = farthest
        center = index_points(xyz, farthest)
        distance = torch.minimum(distance, (xyz - center[:, None]).square().sum(-1))
        farthest = distance.argmax(1)
    return indices


class SetAbstraction(nn.Module):
    """PointNet++ single-scale grouping: FPS, radius query, PointNet, max."""

    def __init__(self, count, radius, neighbors, in_channels, channels):
        super().__init__()
        if min(count, radius, neighbors) <= 0:
            raise ValueError('count, radius and neighbors must be positive')
        self.count, self.radius, self.neighbors = count, radius, neighbors
        layers = []
        for out_channels in channels:
            layers.extend([nn.Conv2d(in_channels, out_channels, 1, bias=False),
                           nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)])
            in_channels = out_channels
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, features=None):
        centers = index_points(xyz, fps_indices(xyz, self.count))
        with torch.no_grad():
            distances = torch.cdist(centers, xyz)
            nearest = distances.argmin(-1, keepdim=True)
            masked = distances.masked_fill(distances > self.radius, float('inf'))
            values, indices = masked.topk(min(self.neighbors, xyz.shape[1]), largest=False)
            indices = torch.where(torch.isfinite(values), indices, nearest)
        relative = index_points(xyz, indices) - centers[:, :, None]
        grouped = relative if features is None else torch.cat(
            (relative, index_points(features, indices)), dim=-1)
        encoded = self.mlp(grouped.permute(0, 3, 1, 2).contiguous()).amax(-1)
        return centers, encoded.transpose(1, 2).contiguous()


class PointNet2Encoder(nn.Module):
    def __init__(self, dim=256, counts=(512, 128), radii=(0.1, 0.2),
                 neighbors=32, depth=2, heads=8):
        super().__init__()
        if len(counts) != 2 or len(radii) != 2 or depth < 0:
            raise ValueError('two abstraction stages and nonnegative depth required')
        self.sa1 = SetAbstraction(counts[0], radii[0], neighbors, 3, (64, 64, 128))
        self.sa2 = SetAbstraction(counts[1], radii[1], neighbors, 131, (128, 128, dim))
        self.position = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(
            dim, heads, dim * 4, dropout=0, batch_first=True, norm_first=True
        ) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, points):
        if points.ndim != 3 or points.shape[-1] != 3 or points.shape[1] < 1:
            raise ValueError('points must be [B,N>0,3]')
        xyz, features = self.sa1(points)
        xyz, features = self.sa2(xyz, features)
        features = features + self.position(xyz)
        for block in self.blocks:
            features = block(features)
        return self.norm(features), xyz


class DINOv2Encoder(nn.Module):
    """Accept RGB [0,1]; normalize here once. Local checkout/weights supported."""

    def __init__(self, dim=256, frozen=True, repo='facebookresearch/dinov2',
                 source='github', weights=None, backbone=None):
        super().__init__()
        if source not in ('github', 'local'):
            raise ValueError('DINO source must be github or local')
        if weights and not Path(weights).is_file():
            raise FileNotFoundError(weights)
        self.backbone = backbone if backbone is not None else torch.hub.load(
            repo, 'dinov2_vits14', source=source, pretrained=weights is None)
        if weights:
            state = torch.load(weights, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state, strict=True)
        self.frozen = frozen
        self.backbone.requires_grad_(not frozen)
        self.projection = nn.Sequential(nn.Linear(384, dim), nn.LayerNorm(dim))
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        if frozen:
            self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3 or min(image.shape[-2:]) < 14:
            raise ValueError('image must be RGB [B,3,H>=14,W>=14]')
        if image.shape[-2] % 14 or image.shape[-1] % 14:
            raise ValueError('image height and width must be divisible by 14')
        image = (image - self.mean) / self.std
        if self.frozen:
            with torch.no_grad():
                tokens = self.backbone.forward_features(image)['x_norm_patchtokens']
        else:
            tokens = self.backbone.forward_features(image)['x_norm_patchtokens']
        return self.projection(tokens)


class MultimodalConditionEncoder(nn.Module):
    def __init__(self, dim=256, heads=8, point_counts=(512, 128),
                 point_radii=(0.1, 0.2), neighbors=32, point_depth=2,
                 dino_frozen=True, dino_repo='facebookresearch/dinov2',
                 dino_source='github', dino_weights=None, image_backbone=None):
        super().__init__()
        self.image_encoder = DINOv2Encoder(dim, dino_frozen, dino_repo,
                                          dino_source, dino_weights, image_backbone)
        self.point_encoder = PointNet2Encoder(dim, point_counts, point_radii,
                                             neighbors, point_depth, heads)
        self.modality = nn.Parameter(torch.randn(2, 1, dim) * 0.02)

    def forward(self, pc, im):
        if pc.shape[0] != im.shape[0]:
            raise ValueError('image and point batch sizes differ')
        point, centers = self.point_encoder(pc)
        image = self.image_encoder(im)
        context = torch.cat((point + self.modality[0], image + self.modality[1]), dim=1)
        return {'point': point, 'image': image, 'centers': centers, 'context': context}
