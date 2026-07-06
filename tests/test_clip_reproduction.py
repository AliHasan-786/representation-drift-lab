from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from driftlab.clip_reproduction import _feature_tensor


class ClipReproductionTests(unittest.TestCase):
    def test_feature_tensor_supports_transformers_4_and_5_outputs(self) -> None:
        tensor = torch.randn(2, 3)
        self.assertIs(_feature_tensor(tensor), tensor)
        structured = SimpleNamespace(pooler_output=tensor)
        self.assertIs(_feature_tensor(structured), tensor)

    def test_feature_tensor_rejects_unknown_outputs(self) -> None:
        with self.assertRaises(TypeError):
            _feature_tensor(SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
