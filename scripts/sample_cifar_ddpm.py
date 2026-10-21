#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from accelerate import Accelerator
from torchvision.transforms.functional import to_pil_image

from multistroke_sd.ddpm_diffusion import DDPMSchedule, DiffusionConfig, resolve_sampling_model_var_type
from multistroke_sd.ddpm_sampler import SamplerConfig, ddpm_sample
from multistroke_sd.ddpm_unet import DDPMUNet, UNetConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CIFAR-10 samples from a matched checkpoint.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", choices=("auto", "ddpm", "multistroke"), default="auto")
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=128, help="Per-process batch size.")
    parser.add_argument("--num-inference-steps", type=int, default=100)
    parser.add_argument("--model-var-type", default="fixedsmall", choices=("fixedsmall", "fixedlarge"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    accelerator = Accelerator()
    checkpoint = Path(args.checkpoint_dir)
    with open(checkpoint / "unet_config.json", "r", encoding="utf-8") as handle:
        unet_config = json.load(handle)
    unet_config.pop("rgb_gen_frequency_band_edges", None)
    unet_config["channel_mult"] = tuple(unet_config["channel_mult"])
    unet_config["attn_resolutions"] = tuple(unet_config["attn_resolutions"])
    model = DDPMUNet(UNetConfig(**unet_config))
    saved = torch.load(checkpoint / "trainer_state.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(saved["unet"], strict=True)
    model = accelerator.prepare_model(model, evaluation_mode=True)
    model.eval()

    config = dict(saved["config"])
    stored_method = config.get("method")
    if args.method == "auto":
        if stored_method not in {"ddpm", "multistroke"}:
            raise ValueError("Checkpoint does not record its method; pass --method ddpm or --method multistroke.")
        method = str(stored_method)
    else:
        method = str(args.method)
        if stored_method in {"ddpm", "multistroke"} and stored_method != method:
            raise ValueError(f"Requested method {method!r} does not match checkpoint method {stored_method!r}.")
    variance = resolve_sampling_model_var_type(str(config.get("model_var_type", "fixedlarge")), args.model_var_type)
    schedule = DDPMSchedule(
        DiffusionConfig(
            num_timesteps=int(config["num_train_timesteps"]),
            beta_start=float(config["beta_start"]),
            beta_end=float(config["beta_end"]),
            beta_schedule=str(config["beta_schedule"]),
            model_mean_type=str(config.get("model_mean_type", "eps")),
            model_var_type=variance,
        ),
        device=accelerator.device,
    )
    train_steps = int(config["num_train_timesteps"])
    if args.num_inference_steps < 1 or args.num_inference_steps > train_steps:
        raise ValueError("num-inference-steps must be between 1 and the training step count.")
    timesteps = torch.linspace(
        train_steps - 1,
        0,
        steps=args.num_inference_steps,
        device=accelerator.device,
    ).round().long()
    if torch.unique(timesteps).numel() != timesteps.numel():
        raise ValueError("The requested inference grid contains duplicate timesteps.")

    smooth_config = None
    if method == "multistroke":
        if str(config.get("multistroke_target_mode", "docs_linear")) != "docs_linear":
            raise ValueError("This focused sampler supports the validated docs_linear MultiStroke mode.")
        smooth_config = SamplerConfig(
            num_train_timesteps=train_steps,
            block_size=int(config.get("multistroke_k_max", 2)),
        )

    image_dir = Path(args.output_dir) / "images"
    if accelerator.is_main_process:
        image_dir.mkdir(parents=True, exist_ok=True)
        with open(Path(args.output_dir) / "sampling_config.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "method": method,
                    "model_var_type": variance,
                    "num_inference_steps": args.num_inference_steps,
                    "checkpoint": str(checkpoint),
                },
                handle,
                indent=2,
            )
    accelerator.wait_for_everyone()

    rank = accelerator.process_index
    world_size = accelerator.num_processes
    generator = torch.Generator(device=accelerator.device)
    generator.manual_seed(args.seed + 100003 * rank)
    pending = list(range(rank, args.num_samples, world_size))
    if args.resume:
        pending = [index for index in pending if not (image_dir / f"{index:06d}.png").exists()]

    with torch.no_grad():
        for offset in range(0, len(pending), args.batch_size):
            indices = pending[offset : offset + args.batch_size]
            labels = torch.randint(0, 10, (len(indices),), device=accelerator.device, generator=generator)
            latents = torch.randn(
                (len(indices), 3, 32, 32),
                device=accelerator.device,
                generator=generator,
            )

            def model_fn(x: torch.Tensor, timestep: torch.Tensor, target_timestep=None) -> torch.Tensor:
                del target_timestep
                return model(x, timestep, None, labels)

            samples, _ = ddpm_sample(
                latents,
                timesteps=timesteps,
                schedule=schedule,
                model_fn=model_fn,
                generator=generator,
                smooth_cfg=smooth_config,
                clip_denoised=True,
            )
            samples = samples.detach().float().cpu().add(1.0).mul(0.5).clamp(0.0, 1.0)
            for index, image in zip(indices, samples):
                to_pil_image(image).save(image_dir / f"{index:06d}.png")

    accelerator.wait_for_everyone()
    accelerator.print(f"Samples written to {image_dir}")


if __name__ == "__main__":
    main()
