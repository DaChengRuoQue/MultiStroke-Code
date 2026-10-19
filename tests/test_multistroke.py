import unittest

import torch

from src.multistroke import FLOW_MULTISTROKE_SCHEDULE_STEPS, flow_multistroke_roughen_mix


class PixelRougheningTests(unittest.TestCase):
    def setUp(self):
        self.image = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)

    def test_first_schedule_level_preserves_input(self):
        time = torch.tensor([1.0 / FLOW_MULTISTROKE_SCHEDULE_STEPS])
        actual = flow_multistroke_roughen_mix(self.image, time, block_size=2)
        torch.testing.assert_close(actual, self.image)

    def test_maximum_schedule_level_matches_expected_block_average(self):
        expected = torch.tensor(
            [[[[2.5, 2.5, 4.5, 4.5], [2.5, 2.5, 4.5, 4.5],
               [10.5, 10.5, 12.5, 12.5], [10.5, 10.5, 12.5, 12.5]]]]
        )
        actual = flow_multistroke_roughen_mix(self.image, torch.tensor([1.0]), block_size=2)
        weight = 1.0 - 1.0 / FLOW_MULTISTROKE_SCHEDULE_STEPS
        expected_mix = (1.0 - weight) * self.image + weight * expected
        torch.testing.assert_close(actual, expected_mix)

    def test_scalar_and_batch_times_agree(self):
        batch = self.image.repeat(2, 1, 1, 1)
        scalar = flow_multistroke_roughen_mix(batch, 0.5, block_size=2)
        vector = flow_multistroke_roughen_mix(batch, torch.tensor([0.5, 0.5]), block_size=2)
        torch.testing.assert_close(scalar, vector)

    def test_invalid_arguments_are_rejected(self):
        cases = [
            lambda: flow_multistroke_roughen_mix(self.image[0], 0.5, block_size=2),
            lambda: flow_multistroke_roughen_mix(self.image, torch.tensor([0.5, 0.6]), block_size=2),
            lambda: flow_multistroke_roughen_mix(self.image, 0.5, num_steps=0, block_size=2),
            lambda: flow_multistroke_roughen_mix(self.image, 0.5, block_size=0),
            lambda: flow_multistroke_roughen_mix(self.image, 0.0, block_size=2),
            lambda: flow_multistroke_roughen_mix(self.image, 0.5, block_size=3),
        ]
        for call in cases:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
