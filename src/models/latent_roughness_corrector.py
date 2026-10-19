"""Lightweight residual conversion between adjacent pixel-roughness latents."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class LatentRoughnessCorrectorConfig:
    architecture: str = "flat"
    latent_channels: int = 4
    hidden_channels: int = 32
    num_res_blocks: int = 4
    num_groups: int = 8
    time_embedding_channels: int = 32
    time_hidden_channels: int = 128


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        x = self.conv2(torch.nn.functional.silu(self.norm2(x)))
        return x + residual


class LatentRoughnessCorrector(nn.Module):
    """Predict the destination-minus-source residual from a source latent."""

    def __init__(self, config: LatentRoughnessCorrectorConfig | None = None) -> None:
        super().__init__()
        self.config = config or LatentRoughnessCorrectorConfig()
        c = self.config
        if c.architecture != "flat":
            raise ValueError(f"Unsupported corrector architecture: {c.architecture}")
        if c.latent_channels <= 0 or c.hidden_channels <= 0 or c.num_res_blocks <= 0:
            raise ValueError("Corrector channel counts and num_res_blocks must be positive")
        if c.hidden_channels % c.num_groups:
            raise ValueError("hidden_channels must be divisible by num_groups")
        self.stem = nn.Conv2d(c.latent_channels + 2, c.hidden_channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(
            *[_ResidualBlock(c.hidden_channels, c.num_groups) for _ in range(c.num_res_blocks)]
        )
        self.out_norm = nn.GroupNorm(c.num_groups, c.hidden_channels)
        self.head = nn.Conv2d(c.hidden_channels, c.latent_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _time_map(value: torch.Tensor, *, name: str, z: torch.Tensor) -> torch.Tensor:
        if value.ndim != 1 or value.shape[0] != z.shape[0]:
            raise ValueError(
                f"{name} must provide one timestep per latent; got {tuple(value.shape)} "
                f"for batch {z.shape[0]}"
            )
        return value.to(device=z.device, dtype=z.dtype).view(-1, 1, 1, 1).expand(
            -1, 1, z.shape[-2], z.shape[-1]
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor, dst_t: torch.Tensor) -> torch.Tensor:
        if z.ndim != 4 or z.shape[1] != self.config.latent_channels:
            raise ValueError(
                f"z must have shape (B,{self.config.latent_channels},H,W), got {tuple(z.shape)}"
            )
        t_map = self._time_map(t, name="t", z=z)
        dst_map = self._time_map(dst_t, name="dst_t", z=z)
        x = self.stem(torch.cat((z, t_map, dst_map), dim=1))
        x = self.blocks(x)
        return self.head(torch.nn.functional.silu(self.out_norm(x)))

    def checkpoint_config(self) -> dict[str, Any]:
        return asdict(self.config)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Mapping[str, Any],
        *,
        map_location: str | torch.device = "cpu",
    ) -> "LatentRoughnessCorrector":
        del map_location
        config = LatentRoughnessCorrectorConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model"])
        return model


def corrector_checkpoint(model: LatentRoughnessCorrector, *, step: int) -> dict[str, Any]:
    return {
        "config": model.checkpoint_config(),
        "model": model.state_dict(),
        "step": int(step),
        "kind": "pixel_multistroke_latent_roughness_corrector",
    }
