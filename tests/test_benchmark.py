from __future__ import annotations

import unittest

from driftlab.benchmark import aggregate_run_artifacts


def _run(seed: int, retained: float) -> dict:
    return {
        "run_id": f"run-{seed}",
        "checkpoints": [
            {
                "step": 0,
                "retained": {"top1_accuracy": retained},
                "adaptation": {"top1_accuracy": 0.5 + seed / 100.0},
                "geometry": {"retained": {"linear_cka": 1.0}},
                "cross_modal": {},
                "layerwise": {},
                "optimization": {"trainable_parameters": 10},
            }
        ],
    }


class BenchmarkTests(unittest.TestCase):
    def test_independent_runs_are_aggregated_with_uncertainty(self) -> None:
        aggregate = aggregate_run_artifacts(
            [_run(1, 0.7), _run(2, 0.8), _run(3, 0.9)]
        )
        retained = aggregate[0]["retained"]["top1_accuracy"]
        self.assertEqual(retained["n"], 3)
        self.assertAlmostEqual(retained["mean"], 0.8)
        self.assertLess(retained["ci_low"], retained["mean"])
        self.assertGreater(retained["ci_high"], retained["mean"])

    def test_misaligned_checkpoint_sequences_are_rejected(self) -> None:
        first = _run(1, 0.7)
        second = _run(2, 0.8)
        second["checkpoints"][0]["step"] = 5
        with self.assertRaisesRegex(ValueError, "do not align"):
            aggregate_run_artifacts([first, second])


if __name__ == "__main__":
    unittest.main()
