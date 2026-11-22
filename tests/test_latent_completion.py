import unittest
import torch
from torch import nn
from models.latent_completion import MissingRegionAutoencoder, LatentCompletion, extract_missing_region


class Conditions(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 32)

    def forward(self, partial, image):
        return {'context': self.proj(partial)}


class FlowTests(unittest.TestCase):
    def test_training_freeze_sampling(self):
        ae = MissingRegionAutoencoder(dim=32, heads=4, latent_tokens=4, points=16,
             encoder_options={'point_counts': (16, 8), 'neighbors': 4,
                              'point_depth': 1, 'latent_depth': 1})
        missing, partial = torch.rand(2, 32, 3), torch.rand(2, 16, 3)
        image = torch.rand(2, 3, 28, 28)
        ae.reconstruction_loss(missing).backward()
        self.assertIsNotNone(ae.encoder.queries.grad)
        ae.zero_grad(set_to_none=True)
        model = LatentCompletion(ae, Conditions(), heads=4, depth=1).train()
        loss = model(partial, image, missing)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.condition_encoder.proj.weight.grad)
        self.assertTrue(all(p.grad is None for p in ae.parameters()))
        noise = torch.randn(2, 4, 32)
        a = model.sample(partial, image, steps=2, noise=noise)
        b = model.sample(partial, image, steps=2, noise=noise)
        self.assertTrue(torch.equal(a['missing'], b['missing']))
        self.assertEqual(a['complete'].shape, (2, 32, 3))
        self.assertTrue(model.training)
        self.assertFalse(ae.training)
        with self.assertRaises(ValueError):
            model.sample(partial, image, steps=0)

    def test_missing_extraction(self):
        gt = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]])
        partial = gt[:, :1]
        result = extract_missing_region(gt, partial, threshold=0.1, count=4)
        self.assertEqual(result.shape, (1, 4, 3))
        self.assertTrue((result[..., 0] > 0).all())
        with self.assertRaises(ValueError):
            extract_missing_region(gt, gt)


if __name__ == '__main__':
    unittest.main()
