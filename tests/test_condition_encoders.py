"""Run: python -m unittest discover -s tests -p test_condition_encoders.py"""
import unittest
import torch
from torch import nn
from models.condition_encoders import DINOv2Encoder, MultimodalConditionEncoder, SetAbstraction


class FakeDINO(nn.Module):
    """Test the wrapper without downloading the real pretrained backbone."""
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 384, 14, stride=14)
        self.seen = None

    def forward_features(self, x):
        self.seen = x.detach()
        return {'x_norm_patchtokens': self.patch(x).flatten(2).transpose(1, 2)}


class EncoderTests(unittest.TestCase):
    def test_conditions_and_gradients(self):
        model = MultimodalConditionEncoder(
            dim=32, heads=4, point_counts=(16, 8), neighbors=8,
            point_depth=1, image_backbone=FakeDINO())
        model.train()
        out = model(torch.rand(2, 32, 3), torch.rand(2, 3, 28, 28))
        self.assertEqual(out['point'].shape, (2, 8, 32))
        self.assertEqual(out['image'].shape, (2, 4, 32))
        self.assertEqual(out['context'].shape, (2, 12, 32))
        self.assertEqual(out['centers'].shape, (2, 8, 3))
        out['context'].square().mean().backward()
        self.assertIsNotNone(model.point_encoder.sa1.mlp[0].weight.grad)
        self.assertIsNotNone(model.image_encoder.projection[0].weight.grad)
        self.assertIsNone(model.image_encoder.backbone.patch.weight.grad)
        self.assertFalse(model.image_encoder.backbone.training)

    def test_normalization_and_finetuning(self):
        backbone = FakeDINO()
        model = DINOv2Encoder(32, frozen=False, backbone=backbone)
        model(torch.zeros(1, 3, 28, 28)).square().mean().backward()
        self.assertTrue(torch.allclose(backbone.seen, (-model.mean / model.std).expand_as(backbone.seen)))
        self.assertIsNotNone(backbone.patch.weight.grad)
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 3, 27, 28))

    def test_radius_query_small_cloud(self):
        layer = SetAbstraction(8, 0.01, 32, 3, (8, 16)).eval()
        xyz = torch.tensor([[[0., 0., 0.], [10., 0., 0.]]])
        centers, features = layer(xyz)
        self.assertEqual(centers.shape, (1, 2, 3))
        self.assertTrue(torch.isfinite(features).all())
        # Each radius contains only its center; relative geometry is zero.
        self.assertTrue(torch.allclose(features[:, 0], features[:, 1]))


if __name__ == '__main__':
    unittest.main()
