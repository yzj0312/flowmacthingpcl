"""Target latent contract and training lifecycle; no downloads required."""
import unittest
import torch
from models.target_encoder import TargetEncoder


class TargetEncoderTests(unittest.TestCase):
    def make_model(self):
        return TargetEncoder(dim=32, latent_tokens=64, heads=4,
                             point_counts=(16, 8), neighbors=8,
                             point_depth=1, latent_depth=1)

    def test_output_and_reconstruction_gradient_path(self):
        model = self.make_model()
        out = model(torch.rand(2, 32, 3), return_details=True)
        self.assertEqual(out['latent'].shape, (2, 64, 32))
        self.assertEqual(out['local_features'].shape, (2, 8, 32))
        self.assertEqual(out['centers'].shape, (2, 8, 3))
        # A nonuniform loss avoids the constant squared norm of LayerNorm.
        (out['latent'] * torch.randn_like(out['latent'])).mean().backward()
        for parameter in (model.queries, model.cross_attention.in_proj_weight,
                          model.point_encoder.sa1.mlp[0].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_frozen_targets_are_stable_under_parent_train(self):
        model = self.make_model().freeze()
        parent = torch.nn.Sequential(model).train()
        points = torch.rand(2, 32, 3)
        a, b = parent(points), parent(points)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(model.training)
        self.assertFalse(a.requires_grad)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        model.unfreeze().train()
        self.assertTrue(model.point_encoder.sa1.mlp[1].training)

    def test_independent_weights_and_invalid_input(self):
        a, b = self.make_model(), self.make_model()
        self.assertNotEqual(a.queries.data_ptr(), b.queries.data_ptr())
        for points in (torch.zeros(2, 0, 3), torch.zeros(2, 3),
                       torch.zeros(2, 32, 3, dtype=torch.long)):
            with self.assertRaises(ValueError):
                a(points)


if __name__ == '__main__':
    unittest.main()
