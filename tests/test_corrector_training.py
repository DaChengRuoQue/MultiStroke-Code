import argparse
from pathlib import Path
import tempfile
import unittest

import torch

from src.train_corrector import (
    sample_adjacent_transition_indices,
    set_optimizer_learning_rate,
    validate_args,
)


class CorrectorTrainingTests(unittest.TestCase):
    def args(self, dataset_root: str, coarsen_factor: int = 8):
        return argparse.Namespace(
            dataset_root=dataset_root,
            output_dir="models/corrector",
            total_steps=10_000,
            batch_size=32,
            learning_rate=1e-4,
            weight_decay=1e-2,
            num_workers=10,
            coarsen_factor=coarsen_factor,
            save_every=10_000,
            log_every=10,
            resume=False,
        )

    def test_adjacent_indices_are_bounded_and_next_finer(self):
        source, destination = sample_adjacent_transition_indices(
            batch_size=4096, device=torch.device("cpu")
        )
        self.assertGreaterEqual(int(source.min()), 2)
        self.assertLessEqual(int(source.max()), 100)
        torch.testing.assert_close(destination, source - 1)

    def test_invalid_coarsening_factor_is_rejected(self):
        with tempfile.TemporaryDirectory() as dataset_root:
            with self.assertRaisesRegex(ValueError, "divisor"):
                validate_args(self.args(dataset_root, coarsen_factor=7))

    def test_resume_restores_requested_constant_learning_rate(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=9e-3)
        set_optimizer_learning_rate(optimizer, 1e-4)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-4)

    def test_missing_dataset_fails_early(self):
        missing = str(Path(tempfile.gettempdir()) / "multistroke-missing-corrector-data")
        with self.assertRaisesRegex(FileNotFoundError, missing):
            validate_args(self.args(missing))


if __name__ == "__main__":
    unittest.main()
