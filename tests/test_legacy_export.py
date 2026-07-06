from __future__ import annotations

import unittest

from driftlab.legacy_export import _infer_baseline


class LegacyExportTests(unittest.TestCase):
    def test_retention_and_learning_use_opposite_delta_conventions(self) -> None:
        rows = [
            {"retained": 0.8, "forgetting": 0.1, "adapted": 0.7, "learning": 0.2},
            {"retained": 0.7, "forgetting": 0.2, "adapted": 0.8, "learning": 0.3},
        ]
        retained = _infer_baseline(
            rows,
            "retained",
            "forgetting",
            delta_is_baseline_minus_current=True,
        )
        adapted = _infer_baseline(
            rows,
            "adapted",
            "learning",
            delta_is_baseline_minus_current=False,
        )
        self.assertAlmostEqual(retained, 0.9)
        self.assertAlmostEqual(adapted, 0.5)


if __name__ == "__main__":
    unittest.main()
