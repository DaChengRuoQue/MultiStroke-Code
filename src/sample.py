"""Generate deterministic ImageNet samples from native flow checkpoints."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torchvision

from src.models.diff_model import diff_model
from src.models.latent_roughness_corrector import LatentRoughnessCorrector
from src.multistroke import FLOW_MULTISTROKE_SCHEDULE_STEPS


_ID_SEED_MODULUS = 2**63 - 1


def _id_seed(seed: int, image_id: int, stream: int) -> int:
    return (int(seed) * 1_000_003 + int(image_id) * 2 + int(stream)) % _ID_SEED_MODULUS


def id_conditioning(
    image_ids: list[int], *, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = []
    noises = []
    for image_id in image_ids:
        label_generator = torch.Generator(device="cpu").manual_seed(_id_seed(seed, image_id, stream=0))
        noise_generator = torch.Generator(device="cpu").manual_seed(_id_seed(seed, image_id, stream=1))
        labels.append(torch.randint(0, 1000, (1,), generator=label_generator))
        noises.append(torch.randn((4, 32, 32), generator=noise_generator))
    return torch.cat(labels).to(device), torch.stack(noises).to(device)


def pending_image_ids(
    total: int,
    rank: int,
    world_size: int,
    image_dir: Path,
    resume: bool,
) -> list[int]:
    image_ids = list(range(rank, total, world_size))
    if resume:
        image_ids = [image_id for image_id in image_ids if not (image_dir / f"{image_id:06d}.png").is_file()]
    return image_ids


def load_corrector(path: Path, device: torch.device) -> LatentRoughnessCorrector:
    if not path.is_file():
        raise FileNotFoundError(f"Latent corrector checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("kind") != "pixel_multistroke_latent_roughness_corrector":
        raise ValueError(f"Not a pixel MultiStroke latent corrector checkpoint: {path}")
    model = LatentRoughnessCorrector.from_checkpoint(checkpoint).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-dir", required=True)
    parser.add_argument("--load-file", default="model_latest.pkl")
    parser.add_argument("--load-def-file", default="model_params_latest.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampler", choices=("euler", "euler_stochastic", "heun"), default="euler")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--latent-roughness-corrector")
    return parser.parse_args()


def validate_sampling_request(args: argparse.Namespace) -> None:
    load_dir = Path(args.load_dir).expanduser()
    model_path = load_dir / args.load_file
    definition_path = load_dir / args.load_def_file
    if not model_path.is_file():
        raise FileNotFoundError(f"Flow checkpoint not found: {model_path}")
    if not definition_path.is_file():
        raise FileNotFoundError(f"Flow checkpoint definition not found: {definition_path}")
    if args.num_samples <= 0 or args.batch_size <= 0 or args.num_steps <= 0:
        raise ValueError("num_samples, batch_size, and num_steps must be positive")
    if args.latent_roughness_corrector is not None:
        corrector_path = Path(args.latent_roughness_corrector).expanduser()
        if not corrector_path.is_file():
            raise FileNotFoundError(f"Latent corrector checkpoint not found: {corrector_path}")
        if args.sampler != "euler":
            raise ValueError("Latent correction requires the Euler sampler")
        if args.num_steps != FLOW_MULTISTROKE_SCHEDULE_STEPS:
            raise ValueError(
                f"Latent correction requires {FLOW_MULTISTROKE_SCHEDULE_STEPS} sampling steps"
            )


def _build_model(device: torch.device):
    return diff_model(
        inCh=4,
        num_classes=1000,
        patch_size=2,
        dim=1088,
        c_dim=2048,
        hidden_scale=4.0,
        num_heads=17,
        num_blocks=17,
        positional_encoding="RoPE",
        device=device,
    )


def main() -> None:
    args = parse_args()
    validate_sampling_request(args)
    corrector = None
    if args.latent_roughness_corrector is not None:
        corrector = load_corrector(
            Path(args.latent_roughness_corrector).expanduser(), torch.device("cpu")
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Sampling requires CUDA")
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group("nccl")

    image_dir = Path(args.output_dir).expanduser() / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    model = _build_model(device)
    model.loadModel(args.load_dir, args.load_file, args.load_def_file)
    model = model.to(device).eval()
    model.device = device
    if corrector is not None:
        corrector = corrector.to(device)

    image_ids = pending_image_ids(args.num_samples, rank, world_size, image_dir, args.resume)
    for offset in range(0, len(image_ids), args.batch_size):
        batch_ids = image_ids[offset : offset + args.batch_size]
        labels, initial_noise = id_conditioning(batch_ids, seed=args.seed, device=device)
        images = model.sample_imgs(
            len(batch_ids),
            args.num_steps,
            labels,
            args.guidance,
            save_intermediate=False,
            use_tqdm=False,
            sampler=args.sampler,
            initial_noise=initial_noise,
            latent_roughness_corrector=corrector,
        )
        images = ((images + 1) * 0.5).clamp(0, 1)
        for image, image_id in zip(images, batch_ids):
            torchvision.utils.save_image(image, image_dir / f"{image_id:06d}.png")
        if rank == 0:
            completed = min(offset + len(batch_ids), len(image_ids))
            print(f"generated {completed}/{len(image_ids)} rank-0 samples", flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
