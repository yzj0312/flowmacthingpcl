import unittest
from unittest.mock import patch
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

    def test_extraction_matches_dense_and_bounds_every_distance_call(self):
        torch.manual_seed(7)
        gt = torch.rand(3, 17, 3)
        partial = gt[:, :5].clone()
        dense_distances = torch.cdist(
            gt, partial, compute_mode='donot_use_mm_for_euclid_dist').amin(-1)
        # Large chunk is a per-sample reference; small chunks must match it.
        reference = extract_missing_region(gt, partial, threshold=0.01, count=4, chunk_size=64)
        original_cdist = torch.cdist
        shapes = []

        def bounded_cdist(left, right, **kwargs):
            shapes.append((tuple(left.shape), tuple(right.shape)))
            self.assertEqual(left.shape[0], 1)
            self.assertEqual(right.shape[0], 1)
            self.assertLessEqual(left.shape[1], 3)
            self.assertLessEqual(right.shape[1], 3)
            return original_cdist(left, right, **kwargs)

        with patch('torch.cdist', side_effect=bounded_cdist):
            actual = extract_missing_region(gt, partial, threshold=0.01, count=4, chunk_size=3)
        self.assertTrue(shapes)
        torch.testing.assert_close(actual, reference)
        for index in range(3):
            candidates = gt[index][dense_distances[index] > 0.01]
            self.assertTrue((original_cdist(actual[index], candidates).amin(-1) < 1e-5).all())
        with self.assertRaises(ValueError):
            extract_missing_region(gt, partial, chunk_size=0)

    def test_dataset_extracts_only_requested_sample(self):
        from main import TensorSamples
        gt = torch.rand(5, 12, 3)
        data = {'gt': gt, 'partial': gt[:, :3], 'image': torch.rand(5, 3, 28, 28)}
        fake_missing = torch.ones(1, 1024, 3)
        with patch('main.extract_missing_region', return_value=fake_missing) as extract:
            dataset = TensorSamples(data, 'ae', chunk_size=3)
            extract.assert_not_called()
            sample = dataset[2]
            self.assertEqual(set(sample), {'missing'})
            self.assertEqual(sample['missing'].shape, (1024, 3))
            extract.assert_called_once()
            torch.testing.assert_close(extract.call_args.args[0], gt[2:3])
            self.assertEqual(extract.call_args.kwargs['chunk_size'], 3)
        self.assertNotIn('missing', data)
        with patch('main.extract_missing_region', return_value=fake_missing):
            sample = TensorSamples(data, 'flow')[1]
            self.assertEqual(set(sample), {'missing', 'partial', 'image'})
        supplied = {'missing': torch.ones(5, 1024, 3)}
        with patch('main.extract_missing_region') as extract:
            TensorSamples(supplied, 'ae')[0]
            TensorSamples(data, 'sample')[0]
            extract.assert_not_called()


if __name__ == '__main__':
    unittest.main()
