
from __future__ import annotations

"""Diffusion schedules, posterior updates, and training losses for DDPM."""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import torch


def resolve_sampling_model_var_type(configured: str, override: Optional[str] = None) -> str:
    """Resolve an optional sampling-only variance override."""
    value = str(configured if override is None else override).strip().lower()
    aliases = {
        "fixed_small": "fixedsmall",
        "fixed_large": "fixedlarge",
    }
    value = aliases.get(value, value)
    if value not in {"fixedsmall", "fixedlarge", "learned_range"}:
        raise ValueError(f"Unsupported sampling model_var_type {value!r}")
    return value


def linear_beta_schedule(num_timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)


def scaled_linear_beta_schedule(num_timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps, dtype=torch.float64) ** 2


def squaredcos_cap_v2_beta_schedule(num_timesteps: int, max_beta: float = 0.999) -> torch.Tensor:
    import math

    def alpha_bar(t: float) -> float:
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_timesteps):
        t1 = i / num_timesteps
        t2 = (i + 1) / num_timesteps
        betas.append(min(1.0 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float64)


def _as_float64_tensor(x: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(dtype=torch.float64)
    return torch.tensor(list(x), dtype=torch.float64)


def _mean_flat(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=tuple(range(1, x.ndim)))


def _normal_kl(mean1: torch.Tensor, logvar1: torch.Tensor, mean2: torch.Tensor, logvar2: torch.Tensor) -> torch.Tensor:
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2).pow(2) * torch.exp(-logvar2))


def _approx_standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    import math

    return 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def _discretized_gaussian_log_likelihood(
    x: torch.Tensor,
    *,
    means: torch.Tensor,
    log_scales: torch.Tensor,
) -> torch.Tensor:
    if x.shape != means.shape or x.shape != log_scales.shape:
        raise ValueError(
            f"Expected matching shapes for x/means/log_scales, got {tuple(x.shape)}, {tuple(means.shape)}, {tuple(log_scales.shape)}"
        )
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = _approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = _approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(torch.clamp(cdf_plus, min=1e-12))
    log_one_minus_cdf_min = torch.log(torch.clamp(1.0 - cdf_min, min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    mid = torch.log(torch.clamp(cdf_delta, min=1e-12))
    return torch.where(x < -0.999, log_cdf_plus, torch.where(x > 0.999, log_one_minus_cdf_min, mid))


@dataclass
class DiffusionConfig:
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    beta_schedule: str = 'linear'
    trained_betas: Optional[Iterable[float]] = None
    model_mean_type: str = 'eps'
    model_var_type: str = 'fixedlarge'  # fixedsmall, fixedlarge, learned_range


def resolve_diffusion_betas(cfg: DiffusionConfig) -> torch.Tensor:
    """Resolve a diffusion configuration to its exact float64 beta sequence."""
    if cfg.trained_betas is not None:
        betas = _as_float64_tensor(cfg.trained_betas)
        if betas.numel() != int(cfg.num_timesteps):
            raise ValueError(
                f"trained_betas length mismatch: got {betas.numel()} but num_timesteps={cfg.num_timesteps}"
            )
        return betas

    sched = str(cfg.beta_schedule).lower()
    if sched == "linear":
        return linear_beta_schedule(cfg.num_timesteps, cfg.beta_start, cfg.beta_end)
    if sched == "scaled_linear":
        return scaled_linear_beta_schedule(cfg.num_timesteps, cfg.beta_start, cfg.beta_end)
    if sched == "squaredcos_cap_v2":
        return squaredcos_cap_v2_beta_schedule(cfg.num_timesteps)
    raise ValueError(
        f"Unsupported beta_schedule='{cfg.beta_schedule}'. Supported: linear, scaled_linear, squaredcos_cap_v2, or provide trained_betas."
    )


class DDPMSchedule:
    def __init__(self, cfg: DiffusionConfig, device: torch.device):
        betas = resolve_diffusion_betas(cfg)

        if betas.ndim != 1:
            raise ValueError("betas must be 1-D")
        if not ((betas > 0).all() and (betas <= 1).all()):
            raise ValueError("betas must be in (0, 1]")

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas.cpu().numpy(), axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        posterior_variance = betas.cpu().numpy() * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # Improved-diffusion handling at t=0 reuses posterior_variance[1].
        # For respaced schedules with a single timestep, fall back to that only entry.
        if posterior_variance.shape[0] == 1:
            posterior_log_variance_clipped = np.log(np.maximum(posterior_variance, 1e-20))
        else:
            posterior_log_variance_clipped = np.log(np.append(posterior_variance[1], posterior_variance[1:]))
        posterior_mean_coef1 = betas.cpu().numpy() * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(alphas.cpu().numpy()) / (1.0 - alphas_cumprod)

        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)

        self.betas = torch.tensor(betas.cpu().numpy(), device=device, dtype=torch.float32)
        self.alphas_cumprod = torch.tensor(alphas_cumprod, device=device, dtype=torch.float32)
        self.alphas_cumprod_prev = torch.tensor(alphas_cumprod_prev, device=device, dtype=torch.float32)
        self.sqrt_alphas_cumprod = torch.tensor(sqrt_alphas_cumprod, device=device, dtype=torch.float32)
        self.sqrt_one_minus_alphas_cumprod = torch.tensor(sqrt_one_minus_alphas_cumprod, device=device, dtype=torch.float32)
        self.sqrt_recip_alphas_cumprod = torch.tensor(sqrt_recip_alphas_cumprod, device=device, dtype=torch.float32)
        self.sqrt_recipm1_alphas_cumprod = torch.tensor(sqrt_recipm1_alphas_cumprod, device=device, dtype=torch.float32)
        self.posterior_variance = torch.tensor(posterior_variance, device=device, dtype=torch.float32)
        self.posterior_log_variance_clipped = torch.tensor(posterior_log_variance_clipped, device=device, dtype=torch.float32)
        self.posterior_mean_coef1 = torch.tensor(posterior_mean_coef1, device=device, dtype=torch.float32)
        self.posterior_mean_coef2 = torch.tensor(posterior_mean_coef2, device=device, dtype=torch.float32)
        if posterior_variance.shape[0] == 1:
            fixed_large_variance = np.maximum(betas.cpu().numpy(), 1e-20)
        else:
            fixed_large_variance = np.append(posterior_variance[1], betas.cpu().numpy()[1:])
        self.fixed_large_variance = torch.tensor(fixed_large_variance, device=device, dtype=torch.float32)
        self.fixed_large_log_variance = torch.log(self.fixed_large_variance)

        self.cfg = cfg

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        out = a.gather(0, t)
        while len(out.shape) < len(x_shape):
            out = out.unsqueeze(-1)
        return out

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def predict_xstart_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps

    def predict_eps_from_xstart(self, x_t: torch.Tensor, t: torch.Tensor, pred_xstart: torch.Tensor) -> torch.Tensor:
        return (self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def q_posterior_mean(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t

    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mean = self.q_posterior_mean(x_start, x_t, t)
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def split_model_output(self, model_output: torch.Tensor, x_shape: torch.Size) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, c = x_shape[:2]
        model_var_type = str(self.cfg.model_var_type).lower()
        if model_var_type in {"learned", "learned_range"}:
            expected = (b, c * 2, *x_shape[2:])
            if tuple(model_output.shape) != expected:
                raise ValueError(f"Expected learned-variance output shape {expected}, got {tuple(model_output.shape)}")
            return torch.split(model_output, c, dim=1)
        expected = (b, c, *x_shape[2:])
        if tuple(model_output.shape) != expected:
            raise ValueError(f"Expected model output shape {expected}, got {tuple(model_output.shape)}")
        return model_output, None

    def variance_from_values(self, x: torch.Tensor, t: torch.Tensor, model_var_values: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        model_var_type = str(self.cfg.model_var_type).lower()
        if model_var_type == "learned_range":
            if model_var_values is None:
                raise ValueError("learned_range requires model_var_values")
            min_log = self._extract(self.posterior_log_variance_clipped, t, x.shape).to(model_var_values.dtype)
            max_log = self._extract(torch.log(torch.clamp(self.betas, min=1e-20)), t, x.shape).to(model_var_values.dtype)
            frac = (model_var_values + 1.0) / 2.0
            model_log_variance = frac * max_log + (1.0 - frac) * min_log
            model_variance = torch.exp(model_log_variance)
            return model_variance, model_log_variance
        if model_var_type == "fixedsmall":
            model_variance = self._extract(self.posterior_variance, t, x.shape)
            model_log_variance = self._extract(self.posterior_log_variance_clipped, t, x.shape)
            return model_variance, model_log_variance
        if model_var_type == "fixedlarge":
            model_variance = self._extract(self.fixed_large_variance, t, x.shape)
            model_log_variance = self._extract(self.fixed_large_log_variance, t, x.shape)
            return model_variance, model_log_variance
        raise ValueError(f"Unsupported model_var_type {self.cfg.model_var_type}")

    def p_mean_variance_from_output(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_output: torch.Tensor,
        *,
        clip_denoised: bool = True,
    ) -> Dict[str, torch.Tensor]:
        eps_pred, model_var_values = self.split_model_output(model_output, x_t.shape)
        model_variance, model_log_variance = self.variance_from_values(x_t, t, model_var_values)
        pred_xstart = self.predict_xstart_from_eps(x_t, t, eps_pred)
        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1.0, 1.0)
        model_mean, _, _ = self.q_posterior_mean_variance(pred_xstart, x_t, t)
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "eps": eps_pred,
        }

    def vb_terms_bpd(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_output: torch.Tensor,
        *,
        clip_denoised: bool = False,
    ) -> Dict[str, torch.Tensor]:
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start, x_t, t)
        out = self.p_mean_variance_from_output(x_t, t, model_output, clip_denoised=clip_denoised)
        kl = _normal_kl(true_mean, true_log_variance_clipped, out["mean"], out["log_variance"])
        kl = _mean_flat(kl) / np.log(2.0)
        decoder_nll = -_discretized_gaussian_log_likelihood(
            x_start,
            means=out["mean"],
            log_scales=0.5 * out["log_variance"],
        )
        decoder_nll = _mean_flat(decoder_nll) / np.log(2.0)
        output = torch.where(t == 0, decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(
        self,
        model_output: torch.Tensor,
        x_start: torch.Tensor,
        t: torch.Tensor,
        *,
        noise: torch.Tensor,
        x_t: Optional[torch.Tensor] = None,
        clip_denoised: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if x_t is None:
            x_t = self.q_sample(x_start, t, noise=noise)
        terms: Dict[str, torch.Tensor] = {}
        eps_pred, model_var_values = self.split_model_output(model_output, x_t.shape)
        if model_var_values is not None:
            frozen_out = torch.cat([eps_pred.detach(), model_var_values], dim=1)
            vb = self.vb_terms_bpd(x_start, x_t, t, frozen_out, clip_denoised=clip_denoised)["output"]
            # Retain the improved-diffusion variational-bound scaling rule.
            vb = vb * (self.cfg.num_timesteps / 1000.0)
            terms["vb"] = vb
        terms["mse"] = _mean_flat((noise - eps_pred) ** 2)
        terms["loss"] = terms["mse"] + terms.get("vb", torch.zeros_like(terms["mse"]))
        return terms
