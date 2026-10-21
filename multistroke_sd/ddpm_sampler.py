from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch

from .ddpm_diffusion import DDPMSchedule, DiffusionConfig
from .multistroke import linear_inverse_roughen, linear_roughen_mix


@dataclass(frozen=True)
class SamplerConfig:
    num_train_timesteps: int
    block_size: int = 2


def _randn_like(value: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
    if generator is None:
        return torch.randn_like(value)
    return torch.randn(value.shape, device=value.device, dtype=value.dtype, generator=generator)


def _build_respaced_schedule(
    base_schedule: DDPMSchedule,
    timesteps: torch.Tensor,
) -> tuple[DDPMSchedule, dict[int, int]]:
    selected = sorted({int(timestep) for timestep in timesteps.tolist()})
    if len(selected) != timesteps.numel():
        raise ValueError("Inference timesteps must be unique.")
    base_alpha = base_schedule.alphas_cumprod.detach().cpu()
    previous_alpha = 1.0
    betas = []
    for timestep in selected:
        alpha = float(base_alpha[timestep].item())
        betas.append(1.0 - alpha / previous_alpha)
        previous_alpha = alpha
    config = DiffusionConfig(
        num_timesteps=len(selected),
        trained_betas=betas,
        beta_start=base_schedule.cfg.beta_start,
        beta_end=base_schedule.cfg.beta_end,
        beta_schedule=base_schedule.cfg.beta_schedule,
        model_mean_type=base_schedule.cfg.model_mean_type,
        model_var_type=base_schedule.cfg.model_var_type,
    )
    schedule = DDPMSchedule(config, device=base_schedule.betas.device)
    return schedule, {timestep: index for index, timestep in enumerate(selected)}


def _validate_inputs(
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: DDPMSchedule,
    smooth_config: Optional[SamplerConfig],
) -> None:
    if latents.ndim != 4:
        raise ValueError(f"Expected BCHW latents, got {tuple(latents.shape)}")
    if timesteps.ndim != 1 or timesteps.numel() < 1:
        raise ValueError("timesteps must be a non-empty one-dimensional tensor.")
    if not bool(torch.all(timesteps[:-1] > timesteps[1:])):
        raise ValueError("timesteps must be strictly decreasing.")
    if int(timesteps.min()) < 0 or int(timesteps.max()) >= schedule.cfg.num_timesteps:
        raise ValueError("Inference timesteps fall outside the training schedule.")
    if smooth_config is None:
        return
    if smooth_config.num_train_timesteps != schedule.cfg.num_timesteps:
        raise ValueError("MultiStroke and DDPM training schedules must have the same length.")
    if smooth_config.block_size < 1:
        raise ValueError("MultiStroke block size must be positive.")


def ddpm_sample(
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: DDPMSchedule,
    model_fn: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    *,
    generator: Optional[torch.Generator] = None,
    smooth_cfg: Optional[SamplerConfig] = None,
    capture_frames: bool = False,
    num_frames: int = 10,
    clip_denoised: bool = True,
    stop_after_step_index: Optional[int] = None,
    trace_recorder: Optional[dict] = None,
) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
    """Run baseline or linear MultiStroke ancestral DDPM sampling."""
    _validate_inputs(latents, timesteps, schedule, smooth_cfg)
    frames: Optional[List[torch.Tensor]] = [] if capture_frames else None
    capture_indices = set()
    if capture_frames:
        capture_indices = set(
            torch.linspace(0, timesteps.numel() - 1, steps=min(num_frames, timesteps.numel()))
            .long()
            .tolist()
        )

    multistroke = smooth_cfg is not None
    value = latents
    if multistroke:
        first_timestep = torch.full(
            (value.shape[0],),
            int(timesteps[0].item()) + 1,
            device=value.device,
            dtype=torch.long,
        )
        value = linear_roughen_mix(
            value,
            first_timestep,
            num_train_timesteps=smooth_cfg.num_train_timesteps,
            block_size=smooth_cfg.block_size,
        )

    respaced_schedule, original_to_dense = _build_respaced_schedule(schedule, timesteps)
    with torch.inference_mode():
        for index, timestep in enumerate(timesteps.tolist()):
            original_timestep = torch.full(
                (value.shape[0],), timestep, device=value.device, dtype=torch.long
            )
            dense_timestep = torch.full(
                (value.shape[0],), original_to_dense[timestep], device=value.device, dtype=torch.long
            )
            next_timestep_value = int(timesteps[index + 1].item()) if index + 1 < timesteps.numel() else 0
            next_timestep = torch.full(
                (value.shape[0],), next_timestep_value, device=value.device, dtype=torch.long
            )
            model_output = model_fn(value, original_timestep)
            output = respaced_schedule.p_mean_variance_from_output(
                value,
                dense_timestep,
                model_output,
                clip_denoised=bool(clip_denoised and not multistroke),
            )

            one_based_timestep = original_timestep + 1
            if multistroke and clip_denoised:
                clean_prediction = linear_inverse_roughen(
                    output["pred_xstart"],
                    one_based_timestep,
                    num_train_timesteps=smooth_cfg.num_train_timesteps,
                    block_size=smooth_cfg.block_size,
                ).clamp(-1.0, 1.0)
                rough_prediction = linear_roughen_mix(
                    clean_prediction,
                    one_based_timestep,
                    num_train_timesteps=smooth_cfg.num_train_timesteps,
                    block_size=smooth_cfg.block_size,
                )
                output["pred_xstart"] = rough_prediction
                output["mean"] = respaced_schedule.q_posterior_mean(
                    rough_prediction,
                    value,
                    dense_timestep,
                )

            if trace_recorder is not None:
                trace_recorder.setdefault("pred_xstart_track", []).append(
                    {
                        "timestep": timestep,
                        "state": value.detach().clone(),
                        "pred_xstart": output["pred_xstart"].detach().clone(),
                    }
                )

            nonzero = (dense_timestep != 0).to(dtype=value.dtype).view(-1, 1, 1, 1)
            if multistroke:
                if bool(nonzero.any().item()):
                    noise = linear_roughen_mix(
                        _randn_like(value, generator),
                        one_based_timestep,
                        num_train_timesteps=smooth_cfg.num_train_timesteps,
                        block_size=smooth_cfg.block_size,
                    )
                    value = output["mean"] + nonzero * torch.exp(0.5 * output["log_variance"]) * noise
                    value = linear_inverse_roughen(
                        value,
                        one_based_timestep,
                        num_train_timesteps=smooth_cfg.num_train_timesteps,
                        block_size=smooth_cfg.block_size,
                    )
                    value = linear_roughen_mix(
                        value,
                        next_timestep + 1,
                        num_train_timesteps=smooth_cfg.num_train_timesteps,
                        block_size=smooth_cfg.block_size,
                    )
                else:
                    value = linear_inverse_roughen(
                        output["mean"],
                        one_based_timestep,
                        num_train_timesteps=smooth_cfg.num_train_timesteps,
                        block_size=smooth_cfg.block_size,
                    )
            else:
                noise = _randn_like(value, generator)
                value = output["mean"] + nonzero * torch.exp(0.5 * output["log_variance"]) * noise

            if capture_frames and index in capture_indices:
                frames.append(value.detach().float().cpu())
            if stop_after_step_index is not None and index >= stop_after_step_index:
                break

    return value, frames
