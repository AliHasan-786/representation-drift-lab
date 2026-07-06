from __future__ import annotations

import unittest

import torch

from driftlab.wise_ft import interpolate_state_dict


class WiseFTTests(unittest.TestCase):
    def test_weight_interpolation_uses_requested_convex_coefficient(self) -> None:
        baseline = {"weight": torch.tensor([0.0, 2.0])}
        tuned = {"weight": torch.tensor([2.0, 6.0])}
        result = interpolate_state_dict(baseline, tuned, 0.25)
        self.assertTrue(torch.allclose(result["weight"], torch.tensor([0.5, 3.0])))

    def test_mismatched_states_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must align"):
            interpolate_state_dict({"a": torch.ones(1)}, {"b": torch.ones(1)}, 0.5)


if __name__ == "__main__":
    unittest.main()
