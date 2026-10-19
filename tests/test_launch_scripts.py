import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LaunchScriptTests(unittest.TestCase):
    def run_with_fake_torchrun(self, script_name: str, environment: dict[str, str]):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            capture = temporary_path / "capture.txt"
            fake = temporary_path / "torchrun"
            fake.write_text(
                "#!/usr/bin/env bash\n"
                "{\n"
                "  printf 'TRAINING_MODE=%s\\n' \"${TRAINING_MODE:-}\"\n"
                "  printf 'BATCH_SIZE_PER_RANK=%s\\n' \"${BATCH_SIZE_PER_RANK:-}\"\n"
                "  printf 'TRAIN_SEED=%s\\n' \"${TRAIN_SEED:-}\"\n"
                "  printf '%s\\n' \"$@\"\n"
                "} > \"$CAPTURE_PATH\"\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            env = os.environ.copy()
            env.update(environment)
            env["PATH"] = f"{temporary}{os.pathsep}{env['PATH']}"
            env["CAPTURE_PATH"] = str(capture)
            result = subprocess.run(
                ["bash", str(ROOT / "scripts" / script_name)],
                cwd=ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            captured = capture.read_text(encoding="utf-8") if capture.exists() else ""
            return result, captured

    def test_training_scripts_derive_per_process_batch_and_select_mode(self):
        with tempfile.TemporaryDirectory() as dataset, tempfile.TemporaryDirectory() as output:
            common = {
                "IMAGENET_DATASET_ROOT": dataset,
                "OUTPUT_DIR": output,
                "NUM_PROCESSES": "8",
                "GLOBAL_BATCH_SIZE": "512",
            }
            for script, mode in (
                ("train_baseline.sh", "baseline"),
                ("train_multistroke.sh", "multistroke"),
            ):
                result, captured = self.run_with_fake_torchrun(script, common)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"TRAINING_MODE={mode}\n", captured)
                self.assertIn("BATCH_SIZE_PER_RANK=64\n", captured)
                self.assertIn("TRAIN_SEED=0\n", captured)
                self.assertIn("src.train\n", captured)

    def test_nondivisible_global_batch_is_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as dataset, tempfile.TemporaryDirectory() as output:
            result, captured = self.run_with_fake_torchrun(
                "train_baseline.sh",
                {
                    "IMAGENET_DATASET_ROOT": dataset,
                    "OUTPUT_DIR": output,
                    "NUM_PROCESSES": "8",
                    "GLOBAL_BATCH_SIZE": "513",
                },
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("divisible", result.stderr)
        self.assertEqual(captured, "")

    def test_required_dataset_variable_is_checked(self):
        result, captured = self.run_with_fake_torchrun(
            "train_baseline.sh",
            {"OUTPUT_DIR": "output", "NUM_PROCESSES": "1", "IMAGENET_DATASET_ROOT": ""},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("IMAGENET_DATASET_ROOT", result.stderr)
        self.assertEqual(captured, "")

    def test_corrector_script_uses_derived_batch(self):
        with tempfile.TemporaryDirectory() as dataset, tempfile.TemporaryDirectory() as output:
            result, captured = self.run_with_fake_torchrun(
                "train_corrector.sh",
                {
                    "IMAGENET_DATASET_ROOT": dataset,
                    "OUTPUT_DIR": output,
                    "NUM_PROCESSES": "8",
                    "GLOBAL_BATCH_SIZE": "256",
                },
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("BATCH_SIZE_PER_RANK=32\n", captured)
        self.assertIn("src.train_corrector\n", captured)
        self.assertIn("--batch-size\n32\n", captured)

    def test_sampling_scripts_invoke_sampler_and_pass_corrector_only_when_requested(self):
        with tempfile.TemporaryDirectory() as checkpoint, tempfile.TemporaryDirectory() as output:
            Path(checkpoint, "model_latest.pkl").touch()
            Path(checkpoint, "model_params_latest.json").touch()
            corrector = Path(checkpoint, "corrector.pt")
            corrector.touch()
            common = {
                "CHECKPOINT_DIR": checkpoint,
                "OUTPUT_DIR": output,
                "NUM_PROCESSES": "2",
            }
            baseline, baseline_capture = self.run_with_fake_torchrun("sample_baseline.sh", common)
            corrected, corrected_capture = self.run_with_fake_torchrun(
                "sample_multistroke_corrector.sh",
                {**common, "CORRECTOR_CHECKPOINT": str(corrector)},
            )
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        self.assertEqual(corrected.returncode, 0, corrected.stderr)
        self.assertIn("src.sample\n", baseline_capture)
        self.assertNotIn("--latent-roughness-corrector", baseline_capture)
        self.assertIn("--latent-roughness-corrector", corrected_capture)


if __name__ == "__main__":
    unittest.main()
