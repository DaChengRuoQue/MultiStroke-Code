import unittest

import torch

from multistroke_sd.ddpm_diffusion import DDPMSchedule, DiffusionConfig, resolve_sampling_model_var_type
from multistroke_sd.ddpm_sampler import SamplerConfig, _build_respaced_schedule, ddpm_sample
from multistroke_sd.multistroke import (
    ddpm_linear_multistroke_forward_sample,
    linear_inverse_roughen,
    linear_roughen_mix,
)


class CifarMultiStrokeTests(unittest.TestCase):
    def test_forward_process_roughens_state_and_epsilon(self):
        schedule = DDPMSchedule(DiffusionConfig(num_timesteps=10), device=torch.device("cpu"))
        image = torch.randn(2, 3, 8, 8)
        noise = torch.randn_like(image)
        timesteps = torch.tensor([0, 9])
        state, target = ddpm_linear_multistroke_forward_sample(
            schedule,
            x_start=image,
            timesteps=timesteps,
            noise=noise,
            block_size=2,
        )
        ordinary = schedule.q_sample(image, timesteps, noise=noise)
        docs_timesteps = timesteps + 1
        torch.testing.assert_close(
            state,
            linear_roughen_mix(ordinary, docs_timesteps, num_train_timesteps=10, block_size=2),
        )
        torch.testing.assert_close(
            target,
            linear_roughen_mix(noise, docs_timesteps, num_train_timesteps=10, block_size=2),
        )

    def test_roughening_inverse_round_trip(self):
        value = torch.randn(3, 3, 8, 8)
        timesteps = torch.tensor([1, 5, 10])
        rough = linear_roughen_mix(value, timesteps, num_train_timesteps=10, block_size=2)
        recovered = linear_inverse_roughen(rough, timesteps, num_train_timesteps=10, block_size=2)
        torch.testing.assert_close(recovered, value, atol=2e-6, rtol=2e-6)

    def test_identity_operator_matches_baseline_sampler(self):
        schedule = DDPMSchedule(
            DiffusionConfig(num_timesteps=4, model_var_type="fixedlarge"),
            device=torch.device("cpu"),
        )
        latents = torch.randn((1, 1, 4, 4), generator=torch.Generator().manual_seed(17))
        timesteps = torch.tensor([3, 2], dtype=torch.long)

        def model_fn(value, timestep):
            del timestep
            return torch.zeros_like(value)

        baseline, _ = ddpm_sample(
            latents.clone(),
            timesteps,
            schedule,
            model_fn,
            generator=torch.Generator().manual_seed(23),
            clip_denoised=False,
            stop_after_step_index=0,
        )
        multistroke, _ = ddpm_sample(
            latents.clone(),
            timesteps,
            schedule,
            model_fn,
            generator=torch.Generator().manual_seed(23),
            smooth_cfg=SamplerConfig(num_train_timesteps=4, block_size=1),
            clip_denoised=False,
            stop_after_step_index=0,
        )
        torch.testing.assert_close(multistroke, baseline)

    def test_clean_prediction_is_clipped_after_inverse_roughening(self):
        schedule = DDPMSchedule(
            DiffusionConfig(num_timesteps=4, model_var_type="fixedsmall"),
            device=torch.device("cpu"),
        )
        latents = torch.tensor(
            [[[[2.0, -2.0, 2.0, -2.0],
               [-2.0, 2.0, -2.0, 2.0],
               [2.0, -2.0, 2.0, -2.0],
               [-2.0, 2.0, -2.0, 2.0]]]],
            dtype=torch.float32,
        )
        timesteps = torch.tensor([3, 2], dtype=torch.long)
        one_based_timestep = torch.tensor([4], dtype=torch.long)
        initial = linear_roughen_mix(
            latents,
            one_based_timestep,
            num_train_timesteps=4,
            block_size=2,
        )
        respaced, mapping = _build_respaced_schedule(schedule, timesteps)
        raw_prediction = respaced.p_mean_variance_from_output(
            initial,
            torch.tensor([mapping[3]], dtype=torch.long),
            torch.zeros_like(initial),
            clip_denoised=False,
        )["pred_xstart"]
        expected = linear_roughen_mix(
            linear_inverse_roughen(
                raw_prediction,
                one_based_timestep,
                num_train_timesteps=4,
                block_size=2,
            ).clamp(-1.0, 1.0),
            one_based_timestep,
            num_train_timesteps=4,
            block_size=2,
        )
        trace = {}
        ddpm_sample(
            latents,
            timesteps,
            schedule,
            lambda value, timestep: torch.zeros_like(value),
            smooth_cfg=SamplerConfig(num_train_timesteps=4, block_size=2),
            clip_denoised=True,
            stop_after_step_index=0,
            trace_recorder=trace,
        )
        torch.testing.assert_close(trace["pred_xstart_track"][0]["pred_xstart"], expected)

    def test_respaced_sampling_uses_original_training_coordinates(self):
        schedule = DDPMSchedule(
            DiffusionConfig(num_timesteps=10, model_var_type="fixedsmall"),
            device=torch.device("cpu"),
        )
        timesteps = torch.tensor([9, 4])
        latent = torch.randn(1, 1, 8, 8)
        seen = {}

        def model_fn(x, timestep):
            seen.setdefault("inputs", []).append(x.detach().clone())
            return torch.zeros_like(x)

        sample, _ = ddpm_sample(
            latent,
            timesteps=timesteps,
            schedule=schedule,
            model_fn=model_fn,
            generator=torch.Generator().manual_seed(37),
            smooth_cfg=SamplerConfig(num_train_timesteps=10, block_size=2),
            clip_denoised=False,
            stop_after_step_index=0,
        )
        expected = linear_roughen_mix(
            latent,
            torch.tensor([10]),
            num_train_timesteps=10,
            block_size=2,
        )
        torch.testing.assert_close(seen["inputs"][0], expected)

        respaced, mapping = _build_respaced_schedule(schedule, timesteps)
        posterior = respaced.p_mean_variance_from_output(
            expected,
            torch.tensor([mapping[9]], dtype=torch.long),
            torch.zeros_like(expected),
            clip_denoised=False,
        )
        raw_noise = torch.randn(latent.shape, generator=torch.Generator().manual_seed(37))
        rough_noise = linear_roughen_mix(
            raw_noise,
            torch.tensor([10]),
            num_train_timesteps=10,
            block_size=2,
        )
        current = posterior["mean"] + torch.exp(0.5 * posterior["log_variance"]) * rough_noise
        expected_next = linear_roughen_mix(
            linear_inverse_roughen(
                current,
                torch.tensor([10]),
                num_train_timesteps=10,
                block_size=2,
            ),
            torch.tensor([5]),
            num_train_timesteps=10,
            block_size=2,
        )
        torch.testing.assert_close(sample, expected_next)

    def test_sampling_variance_override(self):
        self.assertEqual(resolve_sampling_model_var_type("fixedlarge", "fixedsmall"), "fixedsmall")
        with self.assertRaises(ValueError):
            resolve_sampling_model_var_type("fixedlarge", "unsupported")


if __name__ == "__main__":
    unittest.main()
