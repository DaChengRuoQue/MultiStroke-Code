"""Pixel-roughening schedule used by ImageNet MultiStroke flow matching."""

import torch
import torch.nn.functional as F


FLOW_MULTISTROKE_SCHEDULE_STEPS = 100


def flow_multistroke_roughen_mix(
    image: torch.Tensor,
    timesteps,
    num_steps: int = FLOW_MULTISTROKE_SCHEDULE_STEPS,
    block_size: int = 8,
) -> torch.Tensor:
    """Blend each image with its block-averaged representation at flow time."""
    if image.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got {tuple(image.shape)}")
    times = torch.as_tensor(timesteps, device=image.device, dtype=image.dtype)
    if times.ndim == 0:
        times = times.expand(image.shape[0])
    if times.ndim != 1 or times.shape[0] not in (1, image.shape[0]):
        raise ValueError(f"Expected a scalar or one timestep per sample, got {tuple(times.shape)}")
    if times.shape[0] == 1:
        times = times.expand(image.shape[0])
    num_steps = int(num_steps)
    block_size = int(block_size)
    if num_steps <= 0 or block_size < 1:
        raise ValueError("num_steps and block_size must be positive")
    height, width = image.shape[-2:]
    if height % block_size or width % block_size:
        raise ValueError(f"Image size {(height, width)} must be divisible by block_size={block_size}")
    coarse = F.interpolate(
        F.avg_pool2d(image, kernel_size=block_size, stride=block_size),
        size=(height, width),
        mode="nearest",
    )
    weight = (times - 1.0 / num_steps).view(-1, 1, 1, 1)
    if torch.any(weight < 0) or torch.any(weight > 1):
        raise ValueError("timesteps must satisfy 1/num_steps <= t <= 1")
    return (1.0 - weight) * image + weight * coarse

