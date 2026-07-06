from __future__ import annotations

import unittest

import numpy as np

from driftlab.metrics import (
    association_with_bootstrap,
    baseline_fixed_projection,
    class_centroid_movement,
    classwise_diagnostics,
    continual_learning_metrics,
    cosine_centroid_drift,
    cross_modal_diagnostics,
    effective_rank,
    layerwise_representation_diagnostics,
    linear_cka,
    local_neighborhood_overlap,
    mean_confidence_interval,
    stable_frechet_distance,
    task_metrics,
)


class MetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = np.array(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=float
        )
        self.labels = np.array([0, 0, 1, 1])

    def test_identical_representations_have_zero_drift(self) -> None:
        self.assertAlmostEqual(cosine_centroid_drift(self.base, self.base), 0.0)
        self.assertAlmostEqual(stable_frechet_distance(self.base, self.base), 0.0, places=8)
        self.assertAlmostEqual(linear_cka(self.base, self.base), 1.0)
        self.assertAlmostEqual(
            local_neighborhood_overlap(self.base, self.base, neighbors=2), 1.0
        )

    def test_sample_space_cka_matches_feature_space_definition(self) -> None:
        rng = np.random.default_rng(19)
        first = rng.normal(size=(7, 11))
        second = rng.normal(size=(7, 13))
        centered_first = first - first.mean(axis=0)
        centered_second = second - second.mean(axis=0)
        expected = np.linalg.norm(centered_first.T @ centered_second, "fro") ** 2
        expected /= np.linalg.norm(centered_first.T @ centered_first, "fro")
        expected /= np.linalg.norm(centered_second.T @ centered_second, "fro")
        self.assertAlmostEqual(linear_cka(first, second), expected)

    def test_frechet_is_stable_for_rank_deficient_covariance(self) -> None:
        first = np.column_stack([np.linspace(0, 1, 8)] * 5)
        second = first + 0.05
        distance = stable_frechet_distance(first, second)
        self.assertTrue(np.isfinite(distance))
        self.assertGreaterEqual(distance, 0.0)

    def test_task_metrics_match_perfect_predictions(self) -> None:
        logits = np.array([[10.0, -2.0], [8.0, -1.0], [-3.0, 9.0], [-1.0, 7.0]])
        result = task_metrics(logits, self.labels)
        self.assertEqual(result.top1_accuracy, 1.0)
        self.assertEqual(result.macro_f1, 1.0)
        self.assertLess(result.negative_log_likelihood, 0.001)

    def test_classwise_diagnostics_preserve_error_counts(self) -> None:
        logits = np.array([[3.0, 0.0], [0.0, 3.0], [2.0, 1.0], [0.0, 2.0]])
        result = classwise_diagnostics(logits, self.labels, ("left", "right"))
        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(result["support"], {"left": 2, "right": 2})
        self.assertEqual(result["per_class_accuracy"], {"left": 0.5, "right": 0.5})

    def test_geometry_helpers(self) -> None:
        moved = self.base + np.array([0.2, -0.1])
        movement = class_centroid_movement(self.base, moved, self.labels)
        self.assertEqual(set(movement), {"0", "1"})
        full_rank = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        self.assertGreater(effective_rank(full_rank), 1.0)

    def test_baseline_fixed_projection_does_not_fit_current_state(self) -> None:
        shifted = self.base + np.array([10.0, -4.0])
        projected_base, projected_shifted = baseline_fixed_projection(self.base, shifted, 1)
        self.assertAlmostEqual(float(projected_base.mean()), 0.0)
        self.assertNotAlmostEqual(float(projected_shifted.mean()), 0.0)

    def test_layerwise_and_cross_modal_diagnostics(self) -> None:
        moved = self.base + np.array([0.05, -0.02])
        layers = layerwise_representation_diagnostics(
            {"block_00": self.base, "block_01": self.base * 2.0},
            {"block_00": moved, "block_01": moved * 2.0},
            projection_dimension=1,
        )
        self.assertEqual(set(layers), {"block_00", "block_01"})
        self.assertEqual(layers["block_00"]["projection_dimension"], 1)
        self.assertTrue(layers["block_00"]["linear_cka_defined"])
        cross_modal = cross_modal_diagnostics(
            self.base, moved, self.base, self.base
        )
        self.assertAlmostEqual(cross_modal["text_cosine_centroid_drift"], 0.0)
        self.assertLess(cross_modal["alignment_change"], 0.0)

    def test_constant_layer_reports_undefined_cka_without_fabricating_value(self) -> None:
        constant = np.ones((4, 3))
        result = layerwise_representation_diagnostics(
            {"input_cls": constant}, {"input_cls": constant}
        )
        self.assertIsNone(result["input_cls"]["linear_cka"])
        self.assertFalse(result["input_cls"]["linear_cka_defined"])

    def test_constant_text_prototypes_report_undefined_cross_modal_cka(self) -> None:
        constant_text = np.ones_like(self.base)
        result = cross_modal_diagnostics(
            self.base, self.base, constant_text, constant_text
        )
        self.assertIsNone(result["text_linear_cka"])
        self.assertFalse(result["text_linear_cka_defined"])

    def test_mean_confidence_interval_exposes_run_count_and_spread(self) -> None:
        result = mean_confidence_interval(np.array([0.7, 0.8, 0.9]))
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["mean"], 0.8)
        self.assertGreater(result["ci_high"], result["mean"])
        self.assertLess(result["ci_low"], result["mean"])

    def test_continual_learning_summary(self) -> None:
        matrix = np.array(
            [[0.8, np.nan, np.nan], [0.7, 0.75, np.nan], [0.65, 0.70, 0.72]]
        )
        result = continual_learning_metrics(matrix)
        self.assertAlmostEqual(result["average_accuracy"], 0.69)
        self.assertLess(result["backward_transfer"], 0.0)
        self.assertGreater(result["average_forgetting"], 0.0)

    def test_association_reports_both_coefficients_and_interval(self) -> None:
        x = np.arange(1.0, 9.0)
        y = 2.0 * x + np.array([0.0, 0.1, -0.1, 0.0, 0.1, -0.1, 0.0, 0.1])
        result = association_with_bootstrap(x, y, seed=7, bootstrap_samples=100)
        self.assertGreater(result["pearson"], 0.99)
        self.assertGreater(result["spearman"], 0.99)
        self.assertLessEqual(result["pearson_ci_low"], result["pearson"])
        self.assertGreaterEqual(result["pearson_ci_high"], result["pearson"])


if __name__ == "__main__":
    unittest.main()
