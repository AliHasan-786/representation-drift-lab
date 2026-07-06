from __future__ import annotations

import unittest

import numpy as np

from driftlab.early_warning import evaluate_early_warning


def _examples() -> list[dict]:
    rows = []
    for index in range(30):
        value = index / 29
        split = "train" if index < 18 else "validation" if index < 24 else "test"
        rows.append(
            {
                "scenario_id": f"scenario-{index}",
                "split": split,
                "observation_fraction": 0.25,
                "features": {
                    "early_forgetting": 0.2 * value,
                    "drift": 0.5 * value,
                },
                "final_forgetting": 0.1 + 0.7 * value,
            }
        )
    return rows


class EarlyWarningTests(unittest.TestCase):
    def test_held_out_model_reports_baselines_calibration_and_predictions(self) -> None:
        result = evaluate_early_warning(
            _examples(), feature_names=("early_forgetting", "drift")
        )
        self.assertEqual(
            result["protocol"]["split_counts"],
            {"train": 18, "validation": 6, "test": 6},
        )
        self.assertLess(
            result["test_metrics"]["early_warning_model"]["rmse"],
            result["test_metrics"]["train_mean_baseline"]["rmse"],
        )
        self.assertEqual(len(result["test_predictions"]), 6)
        self.assertIsNotNone(
            result["test_metrics"]["early_warning_model"]["calibration_slope"]
        )

    def test_future_observations_and_duplicate_scenarios_are_rejected(self) -> None:
        rows = _examples()
        rows[0]["observation_fraction"] = 1.0
        with self.assertRaisesRegex(ValueError, "strictly before"):
            evaluate_early_warning(rows, feature_names=("drift",))
        rows = _examples()
        rows[1]["scenario_id"] = rows[0]["scenario_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            evaluate_early_warning(rows, feature_names=("drift",))


if __name__ == "__main__":
    unittest.main()
