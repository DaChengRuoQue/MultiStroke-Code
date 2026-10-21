from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReleaseScopeTests(unittest.TestCase):
    def test_source_model_package_is_not_ignored(self):
        result = subprocess.run(
            ["git", "check-ignore", "src/models/__init__.py"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)

    def test_scaffold_files_exist(self):
        required = {
            ".gitignore",
            "README.md",
            "configs/cifar10_ddpm.yaml",
            "multistroke_sd/ddpm_diffusion.py",
            "multistroke_sd/ddpm_sampler.py",
            "multistroke_sd/ddpm_unet.py",
            "multistroke_sd/multistroke.py",
            "requirements.txt",
            "scripts/sample_cifar.sh",
            "scripts/sample_cifar_ddpm.py",
            "scripts/sample_baseline.sh",
            "scripts/sample_multistroke_corrector.sh",
            "scripts/train_cifar_baseline.sh",
            "scripts/train_cifar_ddpm.py",
            "scripts/train_cifar_multistroke.sh",
            "scripts/train_baseline.sh",
            "scripts/train_corrector.sh",
            "scripts/train_multistroke.sh",
            "src/__init__.py",
            "src/blocks/__init__.py",
            "src/helpers/__init__.py",
            "src/models/__init__.py",
            "src/models/diff_model.py",
            "src/models/latent_roughness_corrector.py",
            "src/model_trainer.py",
            "src/multistroke.py",
            "src/sample.py",
            "src/train.py",
            "src/train_corrector.py",
            "tests/test_corrector.py",
            "tests/test_corrector_training.py",
            "tests/test_cifar_multistroke.py",
            "tests/test_launch_scripts.py",
            "tests/test_multistroke.py",
            "tests/test_release_scope.py",
            "tests/test_sampling.py",
            "tests/test_training_config.py",
        }
        missing = sorted(path for path in required if not (ROOT / path).is_file())
        self.assertFalse(missing, f"Missing release files: {missing}")

    def test_release_contains_no_private_or_excluded_artifacts(self):
        forbidden_suffixes = {".pkl", ".pt", ".pth", ".ckpt", ".npz"}
        forbidden_directories = {"runs", "runner_logs", "cache", "VAE"}
        forbidden_text = {
            "/home/",
            "/shared_temp",
            "PEEK",
            "StrokeGen",
            "Seqolor",
            "InputOnlyMS",
            "unet_1m",
        }
        allowed = {Path("tests/test_release_scope.py")}

        violations = []
        for path in ROOT.rglob("*"):
            relative = path.relative_to(ROOT)
            if ".git" in relative.parts or "__pycache__" in relative.parts:
                continue
            if path.is_dir():
                if relative.parts and relative.parts[0] in forbidden_directories:
                    violations.append(f"forbidden directory: {relative}")
                continue
            if path.suffix in forbidden_suffixes:
                violations.append(f"forbidden artifact: {relative}")
            if relative in allowed:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for token in forbidden_text:
                if token in text:
                    violations.append(f"forbidden text {token!r}: {relative}")

        self.assertFalse(violations, "\n".join(violations))

    def test_readme_has_no_citation_block(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
        self.assertNotIn("## citation", readme)
        self.assertNotIn("bibtex", readme)

    def test_all_release_files_are_tracked(self):
        result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        pending = {
            path
            for path in result.stdout.splitlines()
            if path not in {"README.md", "tests/test_launch_scripts.py", "tests/test_release_scope.py"}
            and not path.startswith("scripts/")
        }
        self.assertFalse(pending, f"Unexpected untracked release files: {sorted(pending)}")


if __name__ == "__main__":
    unittest.main()
