import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.train import TrainingConfig


class TrainingConfigTests(unittest.TestCase):
    def environment(self, dataset_root: str, **overrides):
        values = {
            "IMAGENET_DATASET_ROOT": dataset_root,
            "SAVE_DIR": "models/release-test",
            "WORLD_SIZE": "8",
        }
        values.update(overrides)
        return patch.dict(os.environ, values, clear=True)

    def test_reference_defaults_match_baseline_configuration(self):
        with tempfile.TemporaryDirectory() as dataset_root, self.environment(dataset_root):
            config = TrainingConfig.from_environment()
        self.assertEqual(config.mode, "baseline")
        self.assertEqual(config.total_steps, 100_000)
        self.assertEqual(config.global_batch_size, 512)
        self.assertEqual(config.batch_size_per_rank, 64)
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertFalse(config.use_lr_scheduler)
        self.assertEqual(config.save_steps, (20_000, 40_000, 60_000, 80_000, 100_000))
        self.assertEqual(config.num_blocks, 17)
        self.assertEqual(config.dimension, 1088)
        self.assertEqual(config.class_dimension, 2048)
        self.assertEqual(config.hidden_scale, 4.0)
        self.assertEqual(config.num_heads, 17)
        self.assertEqual(config.positional_encoding, "RoPE")

    def test_multistroke_mode_fixes_pixel_configuration(self):
        with tempfile.TemporaryDirectory() as dataset_root, self.environment(
            dataset_root, TRAINING_MODE="multistroke"
        ):
            config = TrainingConfig.from_environment()
        self.assertTrue(config.multistroke)
        self.assertEqual(config.coarsen_factor, 8)
        self.assertEqual(config.schedule_steps, 100)

    def test_global_batch_must_be_divisible_by_world_size(self):
        with tempfile.TemporaryDirectory() as dataset_root, self.environment(
            dataset_root, GLOBAL_BATCH_SIZE="513"
        ):
            with self.assertRaisesRegex(ValueError, "divisible"):
                TrainingConfig.from_environment()

    def test_missing_dataset_fails_before_runtime_initialization(self):
        missing = str(Path(tempfile.gettempdir()) / "multistroke-missing-imagenet")
        with self.environment(missing):
            with self.assertRaisesRegex(FileNotFoundError, missing):
                TrainingConfig.from_environment()


if __name__ == "__main__":
    unittest.main()
