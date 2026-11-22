import unittest
import torch
from models.missing_region_decoder import MissingRegionDecoder
from models.latent_completion import MissingRegionAutoencoder


class DecoderTests(unittest.TestCase):
    def test_shape_and_gradient_to_latent(self):
        decoder = MissingRegionDecoder(dim=32, heads=4, points=48)
        latent = torch.randn(2, 64, 32, requires_grad=True)
        prediction = decoder(latent)
        self.assertEqual(prediction.shape, (2, 48, 3))
        prediction.square().mean().backward()
        for value in (latent, decoder.queries, decoder.attention.in_proj_weight):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)

    def test_autoencoder_uses_new_decoder(self):
        model = MissingRegionAutoencoder(dim=32, heads=4, points=48)
        self.assertIsInstance(model.decoder, MissingRegionDecoder)

    def test_invalid_latents(self):
        decoder = MissingRegionDecoder(dim=32, heads=4)
        for latent in (torch.rand(2, 32), torch.rand(2, 0, 32), torch.rand(2, 64, 16)):
            with self.assertRaises(ValueError):
                decoder(latent)


if __name__ == '__main__':
    unittest.main()
