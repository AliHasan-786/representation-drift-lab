from __future__ import annotations

import unittest

from driftlab.interpolation import set_lora_output_scale


class _Layer:
    def __init__(self) -> None:
        self.scaling = {"default": 2.0}


class InterpolationTests(unittest.TestCase):
    def test_adapter_output_scaling_uses_fixed_trained_scale(self) -> None:
        layer = _Layer()
        controls = [(layer, "default", 2.0)]
        set_lora_output_scale(None, 0.25, controls)
        self.assertEqual(layer.scaling["default"], 0.5)
        set_lora_output_scale(None, 1.0, controls)
        self.assertEqual(layer.scaling["default"], 2.0)

    def test_out_of_range_alpha_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            set_lora_output_scale(None, 1.1, [])


if __name__ == "__main__":
    unittest.main()
