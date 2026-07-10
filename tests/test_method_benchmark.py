from __future__ import annotations

import unittest

from driftlab.method_benchmark import comparable_fingerprints, complement_interval


class MethodBenchmarkTests(unittest.TestCase):
    def test_reference_only_split_does_not_change_score_comparison(self) -> None:
        fingerprints = {
            "41": {
                "adaptation_train": "train",
                "adaptation_eval": "adapt-eval",
                "retained_eval": "retain-eval",
                "retained_reference": "disjoint-training-reference",
            }
        }
        self.assertEqual(
            comparable_fingerprints(fingerprints),
            {
                "41": {
                    "adaptation_train": "train",
                    "adaptation_eval": "adapt-eval",
                    "retained_eval": "retain-eval",
                }
            },
        )

    def test_complement_reverses_interval_bounds(self) -> None:
        result = complement_interval(
            {"n": 3, "mean": 0.8, "ci_low": 0.7, "ci_high": 0.9}
        )
        self.assertAlmostEqual(result["mean"], 0.2)
        self.assertAlmostEqual(result["ci_low"], 0.1)
        self.assertAlmostEqual(result["ci_high"], 0.3)


if __name__ == "__main__":
    unittest.main()
