"""Train the lightweight latent corrector for pixel MultiStroke sampling."""

from __future__ import annotations

import argparse
import csv
from glob import glob
import io
import os
from pathlib import Path

import datasets
from PIL import Image
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler
import torchvision

from src.models.latent_roughness_corrector import LatentRoughnessCorrector, corrector_checkpoint
from src.multistroke import FLOW_MULTISTROKE_SCHEDULE_STEPS, flow_multistroke_roughen_mix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=os.environ.get("IMAGENET_DATASET_ROOT", ""))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--total-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=32, help="Per-process batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--coarsen-factor", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=10_000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).expanduser()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"ImageNet dataset root not found: {dataset_root}")
    if not args.output_dir:
        raise ValueError("output_dir is required")
    if args.total_steps <= 0 or args.batch_size <= 0 or args.save_every <= 0 or args.log_every <= 0:
        raise ValueError("total_steps, batch_size, save_every, and log_every must be positive")
    if args.coarsen_factor <= 0 or 256 % args.coarsen_factor:
        raise ValueError("coarsen_factor must be a positive divisor of 256")


def _init_distributed() -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("Corrector training requires CUDA")
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    return rank, world_size, torch.device(f"cuda:{local_rank}")


def _load_dataset(dataset_root: str):
    arrow_shards = sorted(glob(os.path.join(dataset_root, "imagenet-1k-train-*.arrow")))
    if arrow_shards:
        shards = [datasets.Dataset.from_file(path) for path in arrow_shards]
        return shards[0] if len(shards) == 1 else datasets.concatenate_datasets(shards)
    return datasets.load_dataset("parquet", data_dir=dataset_root, cache_dir="./cache", split="train", num_proc=64)


def _make_collate_fn():
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop((256, 256)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda image: 2 * image - 1.0),
        ]
    )

    def decode(value):
        if isinstance(value, Image.Image):
            image = value.convert("RGB")
        elif isinstance(value, dict) and value.get("bytes") is not None:
            image = Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        else:
            image = Image.open(io.BytesIO(value)).convert("RGB")
        try:
            return transform(image)
        finally:
            image.close()

    def collate(batch):
        return torch.stack([decode(sample["image"]) for sample in batch])

    return collate


def _load_frozen_vae(device: torch.device):
    from diffusers.utils import import_utils

    import_utils._xformers_available = False
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", cache_dir="./VAE").to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    return vae


@torch.no_grad()
def encode_mean(vae, image: torch.Tensor) -> torch.Tensor:
    latent = vae.encode(image).latent_dist.mean
    if vae.config.shift_factor is not None:
        latent = latent + vae.config.shift_factor
    return latent * vae.config.scaling_factor


def sample_adjacent_transition_indices(
    *, batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    source = torch.randint(2, FLOW_MULTISTROKE_SCHEDULE_STEPS + 1, (batch_size,), device=device)
    return source, source - 1


def set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _save_checkpoint(model, optimizer, step: int, output_dir: Path) -> None:
    raw_model = model.module if isinstance(model, DDP) else model
    state = corrector_checkpoint(raw_model, step=step)
    state["optimizer"] = optimizer.state_dict()
    for name in ("corrector_latest.pt", f"corrector_{step:06d}.pt"):
        target = output_dir / name
        temporary = target.with_suffix(".tmp")
        torch.save(state, temporary)
        os.replace(temporary, target)


def main() -> None:
    args = parse_args()
    validate_args(args)
    rank, world_size, device = _init_distributed()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = LatentRoughnessCorrector().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay
    )
    start_step = 0
    latest_path = output_dir / "corrector_latest.pt"
    if args.resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"Corrector checkpoint not found: {latest_path}")
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        model = LatentRoughnessCorrector.from_checkpoint(checkpoint).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay
        )
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
    set_optimizer_learning_rate(optimizer, args.learning_rate)
    if world_size > 1:
        model = DDP(model, device_ids=[device.index], broadcast_buffers=False)
    vae = _load_frozen_vae(device)
    dataset = _load_dataset(args.dataset_root)
    sampler = RandomSampler(
        dataset,
        replacement=True,
        num_samples=(args.total_steps - start_step) * args.batch_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=True,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=10 if args.num_workers > 0 else None,
        collate_fn=_make_collate_fn(),
    )
    loss_path = output_dir / "training_loss.csv"
    if rank == 0 and start_step == 0:
        with loss_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(["step", "loss", "learning_rate"])
    if world_size > 1:
        dist.barrier()

    model.train()
    for step, image in enumerate(loader, start=start_step + 1):
        if step > args.total_steps:
            break
        image = image.to(device, non_blocking=True)
        source, destination = sample_adjacent_transition_indices(batch_size=image.shape[0], device=device)
        source_time = source.to(dtype=image.dtype) / FLOW_MULTISTROKE_SCHEDULE_STEPS
        destination_time = destination.to(dtype=image.dtype) / FLOW_MULTISTROKE_SCHEDULE_STEPS
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            source_latent = encode_mean(
                vae,
                flow_multistroke_roughen_mix(
                    image, source_time, FLOW_MULTISTROKE_SCHEDULE_STEPS, args.coarsen_factor
                ),
            )
            destination_latent = encode_mean(
                vae,
                flow_multistroke_roughen_mix(
                    image, destination_time, FLOW_MULTISTROKE_SCHEDULE_STEPS, args.coarsen_factor
                ),
            )
            target = destination_latent - source_latent
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(source_latent, source_time, destination_time)
            loss = torch.nn.functional.mse_loss(prediction.float(), target.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_every == 0 or step == args.total_steps:
            mean_loss = loss.detach()
            if world_size > 1:
                dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
                mean_loss /= world_size
            if rank == 0:
                with loss_path.open("a", newline="", encoding="utf-8") as handle:
                    csv.writer(handle).writerow([step, mean_loss.item(), args.learning_rate])
                print(f"step {step}/{args.total_steps} loss={mean_loss.item():.7f}", flush=True)
        if step % args.save_every == 0 or step == args.total_steps:
            if world_size > 1:
                dist.barrier()
            if rank == 0:
                _save_checkpoint(model, optimizer, step, output_dir)
            if world_size > 1:
                dist.barrier()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
