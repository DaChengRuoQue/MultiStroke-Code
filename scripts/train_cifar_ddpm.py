#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from multistroke_sd.ddpm_diffusion import DDPMSchedule, DiffusionConfig
from multistroke_sd.ddpm_unet import DDPMUNet, UNetConfig
from multistroke_sd.multistroke import ddpm_linear_multistroke_forward_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the matched CIFAR-10 DDPM or MultiStroke model.")
    parser.add_argument("--method", choices=("ddpm", "multistroke"), required=True)
    parser.add_argument("--config", default="configs/cifar10_ddpm.yaml")
    parser.add_argument("--data-root", default="data/cifar10")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume", default=None, help="Checkpoint directory or trainer_state.pt.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if int(config["num_train_timesteps"]) != 100:
        raise ValueError("The matched CIFAR experiment uses 100 diffusion steps.")
    if str(config["beta_schedule"]).lower() != "linear":
        raise ValueError("MultiStroke requires the linear beta schedule.")
    return config


def make_unet(config: dict) -> DDPMUNet:
    return DDPMUNet(
        UNetConfig(
            image_size=32,
            in_channels=3,
            out_channels=3,
            base_channels=128,
            channel_mult=(1, 2, 2, 2),
            num_res_blocks=2,
            attn_resolutions=(16,),
            dropout=float(config.get("unet_dropout", 0.0)),
            resamp_with_conv=True,
            num_classes=10,
            target_timestep_conditioning=False,
        )
    )


def checkpoint_path(path: str) -> Path:
    candidate = Path(path)
    return candidate / "trainer_state.pt" if candidate.is_dir() else candidate


def atomic_save(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(state, temporary)
    os.replace(temporary, path)


def write_checkpoint(accelerator: Accelerator, model: DDPMUNet, optimizer, state: dict, target: Path) -> None:
    if not accelerator.is_main_process:
        return
    target.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    payload = dict(state)
    payload["unet"] = unwrapped.state_dict()
    payload["optimizer"] = optimizer.state_dict()
    atomic_save(payload, target / "trainer_state.pt")
    with open(target / "unet_config.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(unwrapped.cfg), handle, indent=2)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["method"] = args.method
    if args.method == "multistroke":
        config["multistroke_target_mode"] = "docs_linear"
        config["multistroke_input_only"] = False

    precision = str(config.get("mixed_precision", "no")).lower()
    accelerator = Accelerator(
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
        mixed_precision=precision,
    )
    set_seed(args.seed, device_specific=True)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    with accelerator.main_process_first():
        dataset = datasets.CIFAR10(
            root=str(Path(args.data_root).expanduser()),
            train=True,
            transform=transform,
            download=args.download,
        )
    loader = DataLoader(
        dataset,
        batch_size=int(config["train_batch_size"]),
        shuffle=True,
        num_workers=int(config.get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(config.get("num_workers", 4)) > 0,
    )

    model = make_unet(config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    global_step = 0
    if args.resume:
        saved = torch.load(checkpoint_path(args.resume), map_location="cpu", weights_only=False)
        model.load_state_dict(saved["unet"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        global_step = int(saved["global_step"])

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    schedule = DDPMSchedule(
        DiffusionConfig(
            num_timesteps=int(config["num_train_timesteps"]),
            beta_start=float(config["beta_start"]),
            beta_end=float(config["beta_end"]),
            beta_schedule=str(config["beta_schedule"]),
            model_mean_type=str(config.get("model_mean_type", "eps")),
            model_var_type=str(config.get("model_var_type", "fixedlarge")),
        ),
        device=accelerator.device,
    )

    output = Path(args.output_dir)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        with open(output / "training_config.json", "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
    accelerator.wait_for_everyone()

    max_steps = int(config["max_train_steps"])
    interval = int(config.get("checkpoint_interval", max_steps))
    model.train()
    while global_step < max_steps:
        for images, labels in loader:
            if global_step >= max_steps:
                break
            images = images.to(accelerator.device, non_blocking=True)
            labels = labels.to(accelerator.device, non_blocking=True)
            timesteps = torch.randint(
                0,
                int(config["num_train_timesteps"]),
                (images.shape[0],),
                device=accelerator.device,
            )
            noise = torch.randn_like(images)
            if args.method == "multistroke":
                model_input, target = ddpm_linear_multistroke_forward_sample(
                    schedule,
                    x_start=images.float(),
                    timesteps=timesteps,
                    noise=noise.float(),
                    block_size=int(config.get("multistroke_k_max", 2)),
                    input_only=False,
                )
            else:
                model_input = schedule.q_sample(images, timesteps, noise=noise)
                target = noise

            with accelerator.accumulate(model):
                prediction = model(model_input, timesteps, None, labels)
                loss = F.mse_loss(prediction.float(), target.float())
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 1.0)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % 200 == 0:
                    mean_loss = accelerator.reduce(loss.detach(), reduction="mean")
                    accelerator.print(f"step={global_step} loss={float(mean_loss):.6f}")
                if global_step % interval == 0 or global_step == max_steps:
                    accelerator.wait_for_everyone()
                    state = {"global_step": global_step, "config": config}
                    write_checkpoint(accelerator, model, optimizer, state, output / "checkpoints" / "latest")
                    write_checkpoint(
                        accelerator,
                        model,
                        optimizer,
                        state,
                        output / "checkpoints" / f"progress_{global_step:06d}",
                    )
                    accelerator.wait_for_everyone()

    accelerator.print("Training complete.")


if __name__ == "__main__":
    main()
