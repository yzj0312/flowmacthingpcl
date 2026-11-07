"""Run with: python -m unittest discover -s tests -v (requires PyTorch)."""
import unittest

import torch

from models.flow_matching import (LatentFlowMatching, MissingSetEncoder,
                                  MissingSetDecoder, missing_region)


class FlowTests(unittest.TestCase):
    def test_missing_region(self):
        partial = torch.tensor([[[0., 0., 0.]]])
        gt = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [3., 0., 0.]]])
        self.assertTrue(torch.equal(missing_region(partial, gt, 2), gt[:, [2, 1]]))

    def test_set_encoder_permutation(self):
        torch.manual_seed(1)
        encoder = MissingSetEncoder(12, 4, 3).eval()
        points = torch.randn(2, 16, 3)
        torch.testing.assert_close(encoder(points), encoder(points[:, torch.randperm(16)]),
                                   atol=1e-5, rtol=1e-5)

    def test_flow_gradients_and_merge(self):
        torch.manual_seed(2)
        flow = LatentFlowMatching(12, 4, 3, 1)
        decoder = MissingSetDecoder(12, 4, 8)
        pc = torch.randn(2, 5, 12, requires_grad=True)
        image = torch.randn(2, 7, 12, requires_grad=True)
        target = torch.randn(2, 4, 12, requires_grad=True)
        condition = flow.condition(pc, image)
        loss = flow.loss(target, condition)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(pc.grad.abs().sum().item(), 0)
        self.assertGreater(image.grad.abs().sum().item(), 0)
        self.assertIsNone(target.grad)
        noise = torch.randn(2, 4, 12)
        a = flow.sample(condition.detach(), 2, noise)
        b = flow.sample(condition.detach(), 2, noise)
        torch.testing.assert_close(a, b)
        partial = torch.randn(2, 8, 3)
        complete = torch.cat((partial, decoder(a)), dim=1)
        self.assertEqual(complete.shape, (2, 16, 3))
        self.assertTrue(torch.equal(complete[:, :8], partial))
        complete.square().mean().backward()
        self.assertTrue(any(p.grad is not None for p in decoder.parameters()))


if __name__ == '__main__':
    unittest.main()
