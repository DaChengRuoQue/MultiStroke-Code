from __future__ import annotations

"""DDPM-style UNet (PyTorch) aligned to the structure of Ho et al. (2020) official TF repo.

Key design:
- Swish/SiLU activations
- GroupNorm
- Residual blocks with timestep embedding injection
- Single-head spatial attention blocks at specified resolutions

This module is intentionally dependency-light (torch only).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def swish(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding."""
    half = dim // 2
    device = timesteps.device
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=device, dtype=torch.float32) / float(half)
    )
    args = timesteps.to(torch.float32)[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=device, dtype=torch.float32)], dim=1)
    return emb


class GroupNorm32(nn.GroupNorm):
    def __init__(self, num_channels: int, num_groups: int = 32, eps: float = 1e-5):
        g = num_groups
        while num_channels % g != 0 and g > 1:
            g //= 2
        super().__init__(g, num_channels, eps=eps, affine=True)


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_ch: int, dropout: float):
        super().__init__()
        self.norm1 = GroupNorm32(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.temb_proj = nn.Linear(temb_ch, out_ch)
        self.target_temb_proj = zero_module(nn.Linear(temb_ch, out_ch))

        self.norm2 = GroupNorm32(out_ch)
        self.dropout = nn.Dropout(p=float(dropout))
        self.conv2 = zero_module(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(swish(self.norm1(x)))
        temb = swish(temb)
        if temb.shape[1] == self.temb_proj.in_features * 2:
            current_temb, target_temb = torch.split(temb, self.temb_proj.in_features, dim=1)
            temb_proj = self.temb_proj(current_temb) + self.target_temb_proj(target_temb)
        else:
            temb_proj = self.temb_proj(temb)
        h = h + temb_proj[:, :, None, None]
        h = self.conv2(self.dropout(swish(self.norm2(h))))
        return self.shortcut(x) + h


class AttnBlock(nn.Module):
    """Single-head attention over spatial positions."""
    def __init__(self, ch: int):
        super().__init__()
        self.norm = GroupNorm32(ch)
        self.q = nn.Conv2d(ch, ch, kernel_size=1)
        self.k = nn.Conv2d(ch, ch, kernel_size=1)
        self.v = nn.Conv2d(ch, ch, kernel_size=1)
        self.proj_out = zero_module(nn.Conv2d(ch, ch, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        h_ = self.norm(x)
        q = self.q(h_).reshape(b, c, h * w)
        k = self.k(h_).reshape(b, c, h * w)
        v = self.v(h_).reshape(b, c, h * w)

        attn = torch.bmm(q.permute(0, 2, 1), k) * (c ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1)).reshape(b, c, h, w)
        out = self.proj_out(out)
        return x + out


class Downsample(nn.Module):
    def __init__(self, ch: int, with_conv: bool = True):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1) if with_conv else nn.AvgPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int, with_conv: bool = True):
        super().__init__()
        self.with_conv = with_conv
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1) if with_conv else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


@dataclass
class UNetConfig:
    image_size: int
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 128
    channel_mult: Tuple[int, ...] = (1, 2, 2, 2)
    num_res_blocks: int = 2
    attn_resolutions: Tuple[int, ...] = (16,)
    dropout: float = 0.1
    resamp_with_conv: bool = True
    num_classes: int = 1  # >1 enables conventional class-conditional embedding
    target_timestep_conditioning: bool = False


class DDPMUNet(nn.Module):
    def __init__(self, cfg: UNetConfig):
        super().__init__()
        self.cfg = cfg

        ch = cfg.base_channels
        temb_ch = ch * 4

        self.temb_dense0 = nn.Linear(ch, temb_ch)
        self.temb_dense1 = nn.Linear(temb_ch, temb_ch)

        self.label_emb = nn.Embedding(cfg.num_classes, temb_ch) if cfg.num_classes and cfg.num_classes > 1 else None

        self.conv_in = nn.Conv2d(cfg.in_channels, ch, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        hs_channels = [ch]
        curr_res = cfg.image_size
        in_ch = ch
        for i_level, mult in enumerate(cfg.channel_mult):
            out_ch = ch * mult
            for _ in range(cfg.num_res_blocks):
                self.down_blocks.append(ResBlock(in_ch, out_ch, temb_ch, cfg.dropout))
                in_ch = out_ch
                self.down_attn.append(AttnBlock(in_ch) if curr_res in cfg.attn_resolutions else nn.Identity())
                hs_channels.append(in_ch)
            if i_level != len(cfg.channel_mult) - 1:
                self.downsamples.append(Downsample(in_ch, with_conv=cfg.resamp_with_conv))
                curr_res //= 2
                hs_channels.append(in_ch)
            else:
                self.downsamples.append(nn.Identity())

        self.mid_block1 = ResBlock(in_ch, in_ch, temb_ch, cfg.dropout)
        self.mid_attn = AttnBlock(in_ch)
        self.mid_block2 = ResBlock(in_ch, in_ch, temb_ch, cfg.dropout)

        self.up_blocks = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i_level, mult in reversed(list(enumerate(cfg.channel_mult))):
            out_ch = ch * mult
            for _ in range(cfg.num_res_blocks + 1):
                skip_ch = hs_channels.pop()
                self.up_blocks.append(ResBlock(in_ch + skip_ch, out_ch, temb_ch, cfg.dropout))
                in_ch = out_ch
                self.up_attn.append(AttnBlock(in_ch) if curr_res in cfg.attn_resolutions else nn.Identity())
            if i_level != 0:
                self.upsamples.append(Upsample(in_ch, with_conv=cfg.resamp_with_conv))
                curr_res *= 2
            else:
                self.upsamples.append(nn.Identity())

        assert len(hs_channels) == 0

        self.norm_out = GroupNorm32(in_ch)
        self.conv_out = zero_module(nn.Conv2d(in_ch, cfg.out_channels, kernel_size=3, padding=1))

        if not bool(cfg.target_timestep_conditioning):
            for module in self.modules():
                if isinstance(module, ResBlock):
                    module.target_temb_proj.requires_grad_(False)

    def _encode_timestep_signal(self, timesteps: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        temb = timestep_embedding(timesteps, self.cfg.base_channels).to(dtype=dtype, device=device)
        temb = self.temb_dense0(temb)
        temb = self.temb_dense1(swish(temb))
        return temb

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        target_t: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temb = self._encode_timestep_signal(t, dtype=x.dtype, device=x.device)
        if self.label_emb is not None:
            if y is None:
                raise ValueError('Model is class-conditional but y is None.')
            temb = temb + self.label_emb(y)
        if target_t is not None:
            target_temb = self._encode_timestep_signal(target_t, dtype=x.dtype, device=x.device)
            temb = torch.cat([temb, target_temb], dim=1)

        hs = []
        h = self.conv_in(x)
        hs.append(h)

        down_block_idx = 0
        for i_level in range(len(self.cfg.channel_mult)):
            for _ in range(self.cfg.num_res_blocks):
                h = self.down_blocks[down_block_idx](h, temb)
                h = self.down_attn[down_block_idx](h)
                down_block_idx += 1
                hs.append(h)
            h_ds = self.downsamples[i_level](h)
            if not isinstance(self.downsamples[i_level], nn.Identity):
                h = h_ds
                hs.append(h)

        h = self.mid_block1(h, temb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, temb)

        up_block_idx = 0
        for i_level in range(len(self.cfg.channel_mult)):
            for _ in range(self.cfg.num_res_blocks + 1):
                h = torch.cat([h, hs.pop()], dim=1)
                h = self.up_blocks[up_block_idx](h, temb)
                h = self.up_attn[up_block_idx](h)
                up_block_idx += 1
            h = self.upsamples[i_level](h)

        assert len(hs) == 0
        h = swish(self.norm_out(h))
        return self.conv_out(h)
