import argparse
from pathlib import Path
import tempfile
import unittest

import torch

from src.models.latent_roughness_corrector import LatentRoughnessCorrector, corrector_checkpoint
from src.sample import (
    id_conditioning,
    load_corrector,
    pending_image_ids,
    validate_sampling_request,
)


class SamplingTests(unittest.TestCase):
    def test_id_conditioning_is_independent_of_batch_partitioning(self):
        image_ids = [2, 7, 11]
        labels, noise = id_conditioning(image_ids, seed=19, device=torch.device("cpu"))
        for position, image_id in enumerate(image_ids):
            single_label, single_noise = id_conditioning(
                [image_id], seed=19, device=torch.device("cpu")
            )
            torch.testing.assert_close(labels[position], single_label[0])
            torch.testing.assert_close(noise[position], single_noise[0])

    def test_rank_partition_covers_each_id_once(self):
        with tempfile.TemporaryDirectory() as output:
            partitions = [
                pending_image_ids(17, rank, 4, Path(output), resume=False) for rank in range(4)
            ]
        flattened = [image_id for partition in partitions for image_id in partition]
        self.assertEqual(sorted(flattened), list(range(17)))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_resume_skips_only_existing_ids(self):
        with tempfile.TemporaryDirectory() as output:
            image_dir = Path(output)
            (image_dir / "000002.png").touch()
            (image_dir / "000008.png").touch()
            self.assertEqual(
                pending_image_ids(10, rank=0, world_size=2, image_dir=image_dir, resume=True),
                [0, 4, 6],
            )
            self.assertEqual(
                pending_image_ids(10, rank=0, world_size=2, image_dir=image_dir, resume=False),
                [0, 2, 4, 6, 8],
            )

    def sampling_args(self, load_dir: str, **overrides):
        values = {
            "load_dir": load_dir,
            "load_file": "model.pkl",
            "load_def_file": "params.json",
            "output_dir": "samples",
            "num_samples": 8,
            "batch_size": 4,
            "num_steps": 100,
            "guidance": 3.0,
            "sampler": "euler",
            "resume": True,
            "latent_roughness_corrector": None,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_missing_flow_checkpoint_files_fail_early(self):
        with tempfile.TemporaryDirectory() as load_dir:
            with self.assertRaisesRegex(FileNotFoundError, "model.pkl"):
                validate_sampling_request(self.sampling_args(load_dir))

    def test_missing_corrector_file_fails_early(self):
        with tempfile.TemporaryDirectory() as load_dir:
            Path(load_dir, "model.pkl").touch()
            Path(load_dir, "params.json").touch()
            args = self.sampling_args(load_dir, latent_roughness_corrector="missing.pt")
            with self.assertRaisesRegex(FileNotFoundError, "missing.pt"):
                validate_sampling_request(args)

    def test_wrong_corrector_kind_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "wrong.pt")
            torch.save({"kind": "wrong"}, path)
            with self.assertRaisesRegex(ValueError, "corrector"):
                load_corrector(path, torch.device("cpu"))

    def test_non_lightweight_corrector_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = corrector_checkpoint(LatentRoughnessCorrector(), step=1)
            checkpoint["config"]["architecture"] = "unsupported"
            path = Path(directory, "wrong.pt")
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "architecture"):
                load_corrector(path, torch.device("cpu"))

    def test_corrector_requires_euler_and_one_hundred_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory, "model.pkl")
            params = Path(directory, "params.json")
            corrector = Path(directory, "corrector.pt")
            model.touch()
            params.touch()
            corrector.touch()
            with self.assertRaisesRegex(ValueError, "Euler"):
                validate_sampling_request(
                    self.sampling_args(
                        directory,
                        sampler="heun",
                        latent_roughness_corrector=str(corrector),
                    )
                )
            with self.assertRaisesRegex(ValueError, "100"):
                validate_sampling_request(
                    self.sampling_args(
                        directory,
                        num_steps=50,
                        latent_roughness_corrector=str(corrector),
                    )
                )


if __name__ == "__main__":
    unittest.main()
