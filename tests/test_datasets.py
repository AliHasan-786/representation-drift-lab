from __future__ import annotations

import unittest

import numpy as np

from driftlab.datasets import make_smoke_splits


class DatasetTests(unittest.TestCase):
    def test_splits_are_deterministic_and_isolated(self) -> None:
        first = make_smoke_splits(42)
        second = make_smoke_splits(42)
        for key in first:
            self.assertEqual(first[key].fingerprint, second[key].fingerprint)
            np.testing.assert_array_equal(first[key].features, second[key].features)
        self.assertNotEqual(
            first["adaptation_train"].fingerprint,
            first["adaptation_eval"].fingerprint,
        )


if __name__ == "__main__":
    unittest.main()
