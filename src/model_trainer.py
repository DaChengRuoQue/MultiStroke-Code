"""Distributed trainer shared by the flow baseline and pixel MultiStroke."""

from __future__ import annotations

import copy
import csv
from glob import glob
import io
import os
from pathlib import Path
from typing import TYPE_CHECKING

import datasets
from PIL import Image
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler
import torchvision
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

from src.helpers.TimeSampler import TimeSampler
from src.helpers.multi_gpu_helpers import is_main_process
from src.multistroke import flow_multistroke_roughen_mix

if TYPE_CHECKING:
    from src.train import TrainingConfig


def _init_distributed() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return torch.device(f"cuda:{local_rank}")


def _load_dataset(root: Path):
    arrow_shards = sorted(glob(str(root / "imagenet-1k-train-*.arrow")))
    if arrow_shards:
        shards = [datasets.Dataset.from_file(path) for path in arrow_shards]
        return shards[0] if len(shards) == 1 else datasets.concatenate_datasets(shards)
    return datasets.load_dataset("parquet", data_dir=str(root), cache_dir="./cache", split="train", num_proc=64)


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
        images = torch.stack([decode(sample["image"]) for sample in batch])
        labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.long)
        return images, labels

    return collate


class model_trainer:
    """Train one matched flow configuration on ImageNet."""

    def __init__(
        self,
        *,
        model,
        config: "TrainingConfig",
        optimizer_path: Path | None = None,
        scheduler_path: Path | None = None,
        scaler_path: Path | None = None,
    ) -> None:
        self.config = config
        self.device = _init_distributed()
        self.model = DDP(
            model.to(self.device),
            device_ids=[self.device.index],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        self.ema_model_cpu = copy.deepcopy(self.model.module).cpu().eval()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            eps=1e-8,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )
        scheduler_factory = (
            get_cosine_schedule_with_warmup if config.use_lr_scheduler else get_constant_schedule_with_warmup
        )
        scheduler_kwargs = {
            "optimizer": self.optimizer,
            "num_warmup_steps": config.warmup_steps,
        }
        if config.use_lr_scheduler:
            scheduler_kwargs["num_training_steps"] = config.total_steps
            scheduler_kwargs["num_cycles"] = 0.5
        self.scheduler = scheduler_factory(**scheduler_kwargs)
        self.scaler = torch.amp.GradScaler("cuda")
        if optimizer_path is not None:
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
        if scheduler_path is not None:
            self.scheduler.load_state_dict(torch.load(scheduler_path, map_location=self.device))
        if scaler_path is not None:
            self.scaler.load_state_dict(torch.load(scaler_path, map_location=self.device))
        self.start_step = int(self.model.module.start_step)
        self.time_sampler = TimeSampler(weighted=False)

    def _save(self, step: int) -> None:
        self.ema_model_cpu.wandb_id = self.model.module.wandb_id
        self.ema_model_cpu.saveModel(
            str(self.config.save_dir),
            self.optimizer,
            self.scheduler,
            self.scaler,
            step=step,
            tag="latest",
        )
        if step in self.config.save_steps:
            self.ema_model_cpu.saveModel(
                str(self.config.save_dir),
                self.optimizer,
                self.scheduler,
                self.scaler,
                step=step,
            )

    def train(self) -> None:
        if self.start_step >= self.config.total_steps:
            if is_main_process():
                print(f"Checkpoint already reached {self.start_step} updates; training complete")
            return
        self.config.save_dir.mkdir(parents=True, exist_ok=True)
        loss_path = self.config.save_dir / "training_loss.csv"
        if is_main_process() and (self.start_step == 0 or not loss_path.exists()):
            with loss_path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(["step", "loss", "learning_rate"])

        dataset = _load_dataset(self.config.dataset_root)
        remaining = self.config.total_steps - self.start_step
        sampler = RandomSampler(
            dataset,
            replacement=True,
            num_samples=remaining * self.config.batch_size_per_rank,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size_per_rank,
            sampler=sampler,
            drop_last=True,
            pin_memory=True,
            num_workers=self.config.num_workers,
            persistent_workers=self.config.num_workers > 0,
            prefetch_factor=10 if self.config.num_workers > 0 else None,
            collate_fn=_make_collate_fn(),
        )
        self.model.train()
        accumulated_loss = 0.0
        for step, (images, labels) in enumerate(loader, start=self.start_step + 1):
            if step > self.config.total_steps:
                break
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            if self.config.multistroke:
                times = torch.rand(images.shape[0], device=self.device, dtype=images.dtype).clamp_min(
                    1.0 / self.config.schedule_steps
                )
                images = flow_multistroke_roughen_mix(
                    images,
                    times,
                    num_steps=self.config.schedule_steps,
                    block_size=self.config.coarsen_factor,
                )
            else:
                times = self.time_sampler(images.shape[0]).to(self.device)

            with torch.no_grad():
                vae = self.model.module.VAE
                latents = vae.encode(images).latent_dist.sample()
                if vae.config.shift_factor is not None:
                    latents = latents + vae.config.shift_factor
                latents = latents * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                time_view = times[:, None, None, None]
                model_input = (1.0 - time_view) * latents + time_view * noise
                target = noise - latents
                null_class = (
                    torch.rand(images.shape[0], device=self.device) < self.config.unconditional_probability
                )

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = self.model(model_input.detach(), times, labels, null_class)
                loss = nn.functional.mse_loss(prediction.float(), target.float())
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            accumulated_loss += loss.detach().item()

            if step % self.config.ema_update_frequency == 0:
                with torch.no_grad():
                    for ema_parameter, parameter in zip(
                        self.ema_model_cpu.parameters(), self.model.module.parameters()
                    ):
                        if parameter.requires_grad:
                            ema_parameter.mul_(self.config.ema_decay).add_(
                                parameter.detach().cpu(), alpha=1.0 - self.config.ema_decay
                            )
            if step % self.config.log_steps == 0 or step == self.config.total_steps:
                mean_loss = torch.tensor(accumulated_loss / self.config.log_steps, device=self.device)
                dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
                mean_loss /= dist.get_world_size()
                if is_main_process():
                    learning_rate = self.optimizer.param_groups[0]["lr"]
                    with loss_path.open("a", newline="", encoding="utf-8") as handle:
                        csv.writer(handle).writerow([step, mean_loss.item(), learning_rate])
                    print(f"step {step}/{self.config.total_steps} loss={mean_loss.item():.7f}", flush=True)
                accumulated_loss = 0.0
            if step in self.config.save_steps or step == self.config.total_steps:
                dist.barrier()
                if is_main_process():
                    self._save(step)
                dist.barrier()

        dist.destroy_process_group()
