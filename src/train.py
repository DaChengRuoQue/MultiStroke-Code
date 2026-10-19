"""Train the matched ImageNet-256 flow baseline or pixel MultiStroke model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import random

import numpy as np
import torch

from src.model_trainer import model_trainer
from src.models.diff_model import diff_model
from src.multistroke import FLOW_MULTISTROKE_SCHEDULE_STEPS


def _enabled(value: str) -> bool:
    return value.lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class TrainingConfig:
    dataset_root: Path
    save_dir: Path
    mode: str
    total_steps: int
    global_batch_size: int
    world_size: int
    batch_size_per_rank: int
    learning_rate: float = 1e-4
    use_lr_scheduler: bool = False
    warmup_steps: int = 1000
    ema_update_frequency: int = 100
    ema_decay: float = 0.99
    unconditional_probability: float = 0.1
    log_steps: int = 10
    num_workers: int = 10
    save_steps: tuple[int, ...] = (20_000, 40_000, 60_000, 80_000, 100_000)
    auto_resume: bool = True
    coarsen_factor: int = 8
    schedule_steps: int = FLOW_MULTISTROKE_SCHEDULE_STEPS
    input_channels: int = 4
    num_classes: int = 1000
    patch_size: int = 2
    num_blocks: int = 17
    dimension: int = 1088
    class_dimension: int = 2048
    hidden_scale: float = 4.0
    num_heads: int = 17
    positional_encoding: str = "RoPE"

    @property
    def multistroke(self) -> bool:
        return self.mode == "multistroke"

    @classmethod
    def from_environment(cls) -> "TrainingConfig":
        dataset_value = os.environ.get("IMAGENET_DATASET_ROOT", "")
        if not dataset_value:
            raise ValueError("IMAGENET_DATASET_ROOT is required")
        dataset_root = Path(dataset_value).expanduser()
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"ImageNet dataset root not found: {dataset_root}")
        save_value = os.environ.get("SAVE_DIR", "")
        if not save_value:
            raise ValueError("SAVE_DIR is required")
        mode = os.environ.get("TRAINING_MODE", "baseline").lower()
        if mode not in {"baseline", "multistroke"}:
            raise ValueError("TRAINING_MODE must be 'baseline' or 'multistroke'")
        total_steps = int(os.environ.get("TOTAL_STEPS", "100000"))
        global_batch_size = int(os.environ.get("GLOBAL_BATCH_SIZE", "512"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if total_steps <= 0 or global_batch_size <= 0 or world_size <= 0:
            raise ValueError("TOTAL_STEPS, GLOBAL_BATCH_SIZE, and WORLD_SIZE must be positive")
        if global_batch_size % world_size:
            raise ValueError("GLOBAL_BATCH_SIZE must be divisible by WORLD_SIZE")
        default_save_steps = ",".join(str(step) for step in range(20_000, total_steps + 1, 20_000))
        save_steps = tuple(
            int(value) for value in os.environ.get("SAVE_STEPS", default_save_steps).split(",") if value.strip()
        )
        if not save_steps or any(step <= 0 or step > total_steps for step in save_steps):
            raise ValueError("SAVE_STEPS must contain positive steps no greater than TOTAL_STEPS")
        return cls(
            dataset_root=dataset_root,
            save_dir=Path(save_value).expanduser(),
            mode=mode,
            total_steps=total_steps,
            global_batch_size=global_batch_size,
            world_size=world_size,
            batch_size_per_rank=global_batch_size // world_size,
            learning_rate=float(os.environ.get("LEARNING_RATE", "1e-4")),
            log_steps=int(os.environ.get("LOG_STEPS", "10")),
            num_workers=int(os.environ.get("NUM_WORKERS", "10")),
            save_steps=save_steps,
            auto_resume=_enabled(os.environ.get("AUTO_RESUME", "1")),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def main() -> None:
    parse_args()
    config = TrainingConfig.from_environment()
    training_seed = os.environ.get("TRAIN_SEED")
    if training_seed is not None:
        rank_seed = int(training_seed) + int(os.environ.get("RANK", "0"))
        random.seed(rank_seed)
        np.random.seed(rank_seed)
        torch.manual_seed(rank_seed)

    model = diff_model(
        inCh=config.input_channels,
        num_classes=config.num_classes,
        patch_size=config.patch_size,
        dim=config.dimension,
        c_dim=config.class_dimension,
        hidden_scale=config.hidden_scale,
        num_heads=config.num_heads,
        num_blocks=config.num_blocks,
        positional_encoding=config.positional_encoding,
        device="gpu",
    )
    latest = {
        "model": config.save_dir / "model_latest.pkl",
        "params": config.save_dir / "model_params_latest.json",
        "optimizer": config.save_dir / "optim_latest.pkl",
        "scheduler": config.save_dir / "scheduler_latest.pkl",
        "scaler": config.save_dir / "scaler_latest.pkl",
    }
    resume = config.auto_resume and all(path.is_file() for path in latest.values())
    if resume:
        model.loadModel(str(config.save_dir), latest["model"].name, latest["params"].name)

    trainer = model_trainer(
        model=model,
        config=config,
        optimizer_path=latest["optimizer"] if resume else None,
        scheduler_path=latest["scheduler"] if resume else None,
        scaler_path=latest["scaler"] if resume else None,
    )
    trainer.train()


if __name__ == "__main__":
    main()
