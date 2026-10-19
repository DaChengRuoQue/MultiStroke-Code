import unittest

import torch

from src.models.latent_roughness_corrector import (
    LatentRoughnessCorrector,
    LatentRoughnessCorrectorConfig,
    corrector_checkpoint,
)


class LightweightCorrectorTests(unittest.TestCase):
    def test_default_model_is_lightweight_and_zero_initialized(self):
        model = LatentRoughnessCorrector()
        self.assertEqual(model.config.architecture, "flat")
        self.assertEqual(model.config.latent_channels, 4)
        self.assertEqual(model.config.hidden_channels, 32)
        self.assertEqual(model.config.num_res_blocks, 4)
        latent = torch.randn(2, 4, 8, 8)
        output = model(latent, torch.tensor([1.0, 0.8]), torch.tensor([0.99, 0.79]))
        self.assertEqual(output.shape, latent.shape)
        torch.testing.assert_close(output, torch.zeros_like(output))

    def test_invalid_time_shape_is_rejected(self):
        model = LatentRoughnessCorrector()
        latent = torch.randn(2, 4, 8, 8)
        with self.assertRaises(ValueError):
            model(latent, torch.tensor([1.0]), torch.tensor([0.99, 0.79]))
        with self.assertRaises(ValueError):
            model(latent, torch.tensor([1.0, 0.8]), torch.tensor([0.99]))

    def test_checkpoint_round_trip_preserves_output_kind_and_step(self):
        model = LatentRoughnessCorrector()
        checkpoint = corrector_checkpoint(model, step=17)
        self.assertEqual(checkpoint["kind"], "pixel_multistroke_latent_roughness_corrector")
        self.assertEqual(checkpoint["step"], 17)
        restored = LatentRoughnessCorrector.from_checkpoint(checkpoint)
        latent = torch.randn(1, 4, 8, 8)
        time = torch.tensor([0.5])
        destination = torch.tensor([0.49])
        torch.testing.assert_close(model(latent, time, destination), restored(latent, time, destination))

    def test_existing_lightweight_checkpoint_config_remains_loadable(self):
        model = LatentRoughnessCorrector()
        checkpoint = corrector_checkpoint(model, step=10_000)
        checkpoint["config"]["time_embedding_channels"] = 32
        checkpoint["config"]["time_hidden_channels"] = 128
        restored = LatentRoughnessCorrector.from_checkpoint(checkpoint)
        self.assertEqual(restored.config.architecture, "flat")

    def test_non_lightweight_architecture_is_rejected(self):
        with self.assertRaises(ValueError):
            LatentRoughnessCorrector(LatentRoughnessCorrectorConfig(architecture="unsupported"))


if __name__ == "__main__":
    unittest.main()
