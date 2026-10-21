from __future__ import annotations

import torch
import torch.nn.functional as F

from .ddpm_diffusion import DDPMSchedule


def _validate_timesteps(timesteps: torch.Tensor, batch_size: int, num_train_timesteps: int) -> None:
    if timesteps.ndim != 1 or timesteps.shape[0] != batch_size:
        raise ValueError(f"Expected timesteps shape ({batch_size},), got {tuple(timesteps.shape)}")
    if num_train_timesteps < 1:
        raise ValueError(f"num_train_timesteps must be positive, got {num_train_timesteps}")


def linear_stroke_weight(timesteps: torch.Tensor, *, num_train_timesteps: int) -> torch.Tensor:
    """Return the linear MultiStroke weight for one-based timesteps."""
    if timesteps.ndim != 1:
        raise ValueError(f"Expected one-dimensional timesteps, got {tuple(timesteps.shape)}")
    if num_train_timesteps < 1:
        raise ValueError(f"num_train_timesteps must be positive, got {num_train_timesteps}")
    values = timesteps.to(dtype=torch.float32).clamp(min=1.0, max=float(num_train_timesteps))
    return torch.maximum(values - 1.0, torch.ones_like(values)) / float(num_train_timesteps)


def _block_projection(value: torch.Tensor, block_size: int) -> torch.Tensor:
    if value.ndim != 4:
        raise ValueError(f"Expected a BCHW tensor, got {tuple(value.shape)}")
    if block_size < 1:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if block_size == 1:
        return value
    height, width = value.shape[-2:]
    if height % block_size or width % block_size:
        raise ValueError(f"Image size {(height, width)} must be divisible by block_size={block_size}")
    pooled = F.avg_pool2d(value, kernel_size=block_size, stride=block_size)
    return F.interpolate(pooled, size=(height, width), mode="nearest")


def linear_roughen_mix(
    value: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    num_train_timesteps: int,
    block_size: int = 2,
) -> torch.Tensor:
    """Blend a tensor with its block-averaged projection."""
    _validate_timesteps(timesteps, value.shape[0], num_train_timesteps)
    projection = _block_projection(value, int(block_size))
    weight = linear_stroke_weight(
        timesteps,
        num_train_timesteps=num_train_timesteps,
    ).to(device=value.device, dtype=value.dtype).view(-1, 1, 1, 1)
    return (1.0 - weight) * value + weight * projection


def linear_inverse_roughen(
    value: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    num_train_timesteps: int,
    block_size: int = 2,
) -> torch.Tensor:
    """Invert the linear roughening operator."""
    _validate_timesteps(timesteps, value.shape[0], num_train_timesteps)
    projection = _block_projection(value, int(block_size))
    weight = linear_stroke_weight(
        timesteps,
        num_train_timesteps=num_train_timesteps,
    ).to(device=value.device, dtype=value.dtype).view(-1, 1, 1, 1)
    fine_scale = (1.0 - weight).clamp_min(torch.finfo(value.dtype).eps)
    return projection + (value - projection) / fine_scale


def ddpm_linear_multistroke_forward_sample(
    schedule: DDPMSchedule,
    *,
    x_start: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    block_size: int = 2,
    input_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct the MultiStroke noisy input and epsilon target."""
    if x_start.shape != noise.shape:
        raise ValueError(
            f"Expected x_start and noise to have matching shapes, got {tuple(x_start.shape)} and {tuple(noise.shape)}"
        )
    _validate_timesteps(timesteps, x_start.shape[0], schedule.cfg.num_timesteps)
    noisy = schedule.q_sample(x_start, timesteps.to(device=x_start.device), noise=noise)
    one_based_timesteps = timesteps.to(device=x_start.device, dtype=torch.long) + 1
    model_input = linear_roughen_mix(
        noisy,
        one_based_timesteps,
        num_train_timesteps=schedule.cfg.num_timesteps,
        block_size=block_size,
    )
    if input_only:
        return model_input, noise
    target = linear_roughen_mix(
        noise,
        one_based_timesteps,
        num_train_timesteps=schedule.cfg.num_timesteps,
        block_size=block_size,
    )
    return model_input, target
