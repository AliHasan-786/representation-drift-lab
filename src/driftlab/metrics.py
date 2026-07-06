from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import stats


def _as_2d(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2:
        raise ValueError(f"{name} must have shape [samples, features] with >=2 samples")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class TaskMetrics:
    top1_accuracy: float
    macro_f1: float
    negative_log_likelihood: float
    expected_calibration_error: float
    sample_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classwise_diagnostics(
    logits: np.ndarray, labels: np.ndarray, class_names: tuple[str, ...] | None = None
) -> dict[str, Any]:
    """Return raw confusion counts and per-class accuracy for error analysis."""
    scores = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if scores.ndim != 2 or targets.ndim != 1 or len(scores) != len(targets):
        raise ValueError("logits must be [samples, classes] and labels [samples]")
    if len(targets) == 0 or targets.min() < 0 or targets.max() >= scores.shape[1]:
        raise ValueError("labels fall outside the logits class range")
    if class_names is not None and len(class_names) != scores.shape[1]:
        raise ValueError("class_names must align with the logits class dimension")
    predictions = scores.argmax(axis=1)
    confusion = np.zeros((scores.shape[1], scores.shape[1]), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    counts = confusion.sum(axis=1)
    accuracy = np.divide(
        np.diag(confusion),
        counts,
        out=np.full(scores.shape[1], np.nan, dtype=np.float64),
        where=counts > 0,
    )
    names = class_names or tuple(str(index) for index in range(scores.shape[1]))
    return {
        "class_names": list(names),
        "confusion_matrix": confusion.tolist(),
        "support": {name: int(counts[index]) for index, name in enumerate(names)},
        "per_class_accuracy": {
            name: None if not np.isfinite(accuracy[index]) else float(accuracy[index])
            for index, name in enumerate(names)
        },
    }


def task_metrics(logits: np.ndarray, labels: np.ndarray, bins: int = 10) -> TaskMetrics:
    scores = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if scores.ndim != 2 or targets.ndim != 1 or len(scores) != len(targets):
        raise ValueError("logits must be [samples, classes] and labels [samples]")
    if len(targets) == 0 or scores.shape[1] < 2:
        raise ValueError("task metrics require samples and at least two classes")
    if targets.min() < 0 or targets.max() >= scores.shape[1]:
        raise ValueError("labels fall outside the logits class range")
    probabilities = _softmax(scores)
    predictions = probabilities.argmax(axis=1)
    accuracy = float(np.mean(predictions == targets))
    class_f1 = []
    for class_id in range(scores.shape[1]):
        true_positive = np.sum((predictions == class_id) & (targets == class_id))
        false_positive = np.sum((predictions == class_id) & (targets != class_id))
        false_negative = np.sum((predictions != class_id) & (targets == class_id))
        denominator = 2 * true_positive + false_positive + false_negative
        class_f1.append(0.0 if denominator == 0 else (2 * true_positive) / denominator)
    nll = float(
        -np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1.0)).mean()
    )
    confidence = probabilities.max(axis=1)
    correct = predictions == targets
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        upper_operator = confidence <= edges[index + 1] if index == bins - 1 else confidence < edges[index + 1]
        in_bin = (confidence >= edges[index]) & upper_operator
        if in_bin.any():
            ece += float(in_bin.mean()) * abs(
                float(correct[in_bin].mean()) - float(confidence[in_bin].mean())
            )
    return TaskMetrics(
        top1_accuracy=accuracy,
        macro_f1=float(np.mean(class_f1)),
        negative_log_likelihood=nll,
        expected_calibration_error=float(ece),
        sample_count=int(len(targets)),
    )


def cosine_centroid_drift(baseline: np.ndarray, current: np.ndarray) -> float:
    first = _as_2d("baseline", baseline).mean(axis=0)
    second = _as_2d("current", current).mean(axis=0)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-15:
        raise ValueError("centroid norm is zero")
    return float(1.0 - np.clip(np.dot(first, second) / denominator, -1.0, 1.0))


def _psd_sqrt(matrix: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    values = np.clip(values, 0.0, None)
    return (vectors * np.sqrt(values)) @ vectors.T


def stable_frechet_distance(
    baseline: np.ndarray,
    current: np.ndarray,
    *,
    covariance_regularization: float = 1e-6,
) -> float:
    """Bures/Frechet distance with PSD square roots and ridge covariance."""
    first = _as_2d("baseline", baseline)
    second = _as_2d("current", current)
    if first.shape[1] != second.shape[1]:
        raise ValueError("feature dimensions must match")
    if covariance_regularization <= 0:
        raise ValueError("covariance_regularization must be positive")
    mean_first = first.mean(axis=0)
    mean_second = second.mean(axis=0)
    identity = np.eye(first.shape[1], dtype=np.float64)
    covariance_first = np.cov(first, rowvar=False) + covariance_regularization * identity
    covariance_second = np.cov(second, rowvar=False) + covariance_regularization * identity
    root_first = _psd_sqrt(covariance_first)
    middle_root = _psd_sqrt(root_first @ covariance_second @ root_first)
    distance = np.sum((mean_first - mean_second) ** 2) + np.trace(
        covariance_first + covariance_second - 2.0 * middle_root
    )
    return float(max(distance, 0.0))


def linear_cka(baseline: np.ndarray, current: np.ndarray) -> float:
    first = _as_2d("baseline", baseline)
    second = _as_2d("current", current)
    if first.shape[0] != second.shape[0]:
        raise ValueError("CKA requires paired samples")
    first = first - first.mean(axis=0, keepdims=True)
    second = second - second.mean(axis=0, keepdims=True)
    # The sample-space form is algebraically equivalent to ||X^T Y||_F^2,
    # but avoids constructing hidden_width x hidden_width matrices.
    first_gram = first @ first.T
    second_gram = second @ second.T
    numerator = float(np.sum(first_gram * second_gram))
    denominator = np.linalg.norm(first_gram, ord="fro") * np.linalg.norm(
        second_gram, ord="fro"
    )
    if denominator <= 1e-15:
        raise ValueError("CKA is undefined for constant representations")
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def effective_rank(embeddings: np.ndarray) -> float:
    covariance = np.atleast_2d(
        np.cov(_as_2d("embeddings", embeddings), rowvar=False)
    )
    values = np.linalg.eigvalsh(covariance)
    values = np.clip(values, 0.0, None)
    total = values.sum()
    if total <= 1e-15:
        return 0.0
    probabilities = values[values > 0] / total
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def baseline_fixed_projection(
    baseline: np.ndarray, current: np.ndarray, maximum_dimension: int
) -> tuple[np.ndarray, np.ndarray]:
    """Project both states into PCA axes fit only on the baseline state."""
    first = _as_2d("baseline", baseline)
    second = _as_2d("current", current)
    if first.shape[1] != second.shape[1]:
        raise ValueError("feature dimensions must match")
    if maximum_dimension < 1:
        raise ValueError("maximum_dimension must be positive")
    center = first.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(first - center, full_matrices=False)
    dimension = min(maximum_dimension, len(right), first.shape[0] - 1)
    basis = right[:dimension].T
    return (first - center) @ basis, (second - center) @ basis


def layerwise_representation_diagnostics(
    baseline_layers: dict[str, np.ndarray],
    current_layers: dict[str, np.ndarray],
    *,
    projection_dimension: int = 32,
) -> dict[str, dict[str, float | int | bool | None]]:
    """Compute comparable drift summaries for every registered model layer."""
    if not baseline_layers or set(baseline_layers) != set(current_layers):
        raise ValueError("baseline and current layer names must be non-empty and identical")
    diagnostics: dict[str, dict[str, float | int | bool | None]] = {}
    for name in baseline_layers:
        baseline = _as_2d(f"baseline_layers[{name}]", baseline_layers[name])
        current = _as_2d(f"current_layers[{name}]", current_layers[name])
        if baseline.shape != current.shape:
            raise ValueError(f"layer {name} representations must be paired")
        projected_baseline, projected_current = baseline_fixed_projection(
            baseline, current, projection_dimension
        )
        try:
            cka: float | None = linear_cka(baseline, current)
        except ValueError as error:
            if "constant representations" not in str(error):
                raise
            cka = None
        diagnostics[name] = {
            "linear_cka": cka,
            "linear_cka_defined": cka is not None,
            "cosine_centroid_drift": cosine_centroid_drift(baseline, current),
            "frechet_distance": stable_frechet_distance(
                projected_baseline, projected_current
            ),
            "effective_rank": effective_rank(projected_current),
            "projection_dimension": int(projected_current.shape[1]),
        }
    return diagnostics


def cross_modal_diagnostics(
    baseline_images: np.ndarray,
    current_images: np.ndarray,
    baseline_text: np.ndarray,
    current_text: np.ndarray,
) -> dict[str, float | bool | None]:
    """Separate image/text drift from paired image-text alignment change."""
    base_image = _as_2d("baseline_images", baseline_images)
    now_image = _as_2d("current_images", current_images)
    base_text = _as_2d("baseline_text", baseline_text)
    now_text = _as_2d("current_text", current_text)
    if not (base_image.shape == now_image.shape == base_text.shape == now_text.shape):
        raise ValueError("cross-modal representations must have identical paired shapes")

    def paired_cosine(first: np.ndarray, second: np.ndarray) -> float:
        numerator = np.sum(first * second, axis=1)
        denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
        if np.any(denominator <= 1e-15):
            raise ValueError("cross-modal alignment is undefined for zero-norm rows")
        return float(np.mean(np.clip(numerator / denominator, -1.0, 1.0)))

    baseline_alignment = paired_cosine(base_image, base_text)
    current_alignment = paired_cosine(now_image, now_text)
    def safe_cka(first: np.ndarray, second: np.ndarray) -> float | None:
        try:
            return linear_cka(first, second)
        except ValueError as error:
            if "constant representations" not in str(error):
                raise
            return None

    image_cka = safe_cka(base_image, now_image)
    text_cka = safe_cka(base_text, now_text)
    return {
        "baseline_paired_cosine_alignment": baseline_alignment,
        "current_paired_cosine_alignment": current_alignment,
        "alignment_change": current_alignment - baseline_alignment,
        "image_cosine_centroid_drift": cosine_centroid_drift(base_image, now_image),
        "text_cosine_centroid_drift": cosine_centroid_drift(base_text, now_text),
        "image_linear_cka": image_cka,
        "image_linear_cka_defined": image_cka is not None,
        "text_linear_cka": text_cka,
        "text_linear_cka_defined": text_cka is not None,
    }


def mean_confidence_interval(
    values: np.ndarray, *, confidence: float = 0.95
) -> dict[str, float | int]:
    """Aggregate independent runs with a two-sided Student-t interval."""
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or len(samples) < 2 or not np.isfinite(samples).all():
        raise ValueError("confidence intervals require at least two finite values")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    mean = float(samples.mean())
    standard_deviation = float(samples.std(ddof=1))
    standard_error = standard_deviation / np.sqrt(len(samples))
    critical = float(stats.t.ppf((1.0 + confidence) / 2.0, df=len(samples) - 1))
    margin = critical * standard_error
    return {
        "n": int(len(samples)),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "confidence": float(confidence),
        "ci_low": mean - margin,
        "ci_high": mean + margin,
    }


def class_centroid_movement(
    baseline: np.ndarray, current: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    first = _as_2d("baseline", baseline)
    second = _as_2d("current", current)
    targets = np.asarray(labels)
    if first.shape != second.shape or len(targets) != len(first):
        raise ValueError("paired embeddings and labels must align")
    movement = {}
    for class_id in np.unique(targets):
        mask = targets == class_id
        movement[str(class_id)] = float(
            np.linalg.norm(first[mask].mean(axis=0) - second[mask].mean(axis=0))
        )
    return movement


def local_neighborhood_overlap(
    baseline: np.ndarray, current: np.ndarray, *, neighbors: int = 5
) -> float:
    first = _as_2d("baseline", baseline)
    second = _as_2d("current", current)
    if first.shape[0] != second.shape[0]:
        raise ValueError("neighborhood overlap requires paired samples")
    if not 1 <= neighbors < first.shape[0]:
        raise ValueError("neighbors must be between 1 and sample_count - 1")

    def nearest(matrix: np.ndarray) -> np.ndarray:
        normalized = matrix / np.clip(
            np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None
        )
        similarity = normalized @ normalized.T
        np.fill_diagonal(similarity, -np.inf)
        return np.argpartition(-similarity, neighbors - 1, axis=1)[:, :neighbors]

    before = nearest(first)
    after = nearest(second)
    overlaps = [
        len(set(row_a).intersection(row_b)) / neighbors
        for row_a, row_b in zip(before, after)
    ]
    return float(np.mean(overlaps))


def continual_learning_metrics(accuracy_matrix: np.ndarray) -> dict[str, float]:
    """Compute standard summaries from a stage-by-task accuracy matrix."""
    matrix = np.asarray(accuracy_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("accuracy_matrix must be square")
    final = matrix[-1]
    observed = np.isfinite(final)
    average_accuracy = float(np.mean(final[observed]))
    forgetting = []
    backward = []
    for task in range(matrix.shape[1] - 1):
        history = matrix[task:, task]
        history = history[np.isfinite(history)]
        forgetting.append(
            float(np.max(history[:-1]) - history[-1]) if len(history) > 1 else 0.0
        )
        backward.append(float(history[-1] - matrix[task, task]))
    return {
        "average_accuracy": average_accuracy,
        "final_accuracy": float(final[observed][-1]),
        "average_forgetting": float(np.mean(forgetting)) if forgetting else 0.0,
        "backward_transfer": float(np.mean(backward)) if backward else 0.0,
    }


def association_with_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 0,
    bootstrap_samples: int = 1000,
) -> dict[str, float]:
    """Report Pearson/Spearman association and a paired bootstrap Pearson CI."""
    first = np.asarray(x, dtype=np.float64)
    second = np.asarray(y, dtype=np.float64)
    if (
        first.ndim != 1
        or second.ndim != 1
        or len(first) != len(second)
        or len(first) < 4
    ):
        raise ValueError("association requires paired vectors with at least four values")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("association inputs must be finite")
    pearson = float(stats.pearsonr(first, second).statistic)
    spearman = float(stats.spearmanr(first, second).statistic)
    rng = np.random.default_rng(seed)
    bootstrapped = []
    for _ in range(bootstrap_samples):
        indices = rng.integers(0, len(first), size=len(first))
        sample_x = first[indices]
        sample_y = second[indices]
        if np.std(sample_x) > 0 and np.std(sample_y) > 0:
            bootstrapped.append(float(stats.pearsonr(sample_x, sample_y).statistic))
    if not bootstrapped:
        raise ValueError("all bootstrap resamples were degenerate")
    low, high = np.quantile(bootstrapped, [0.025, 0.975])
    return {
        "pearson": pearson,
        "spearman": spearman,
        "pearson_ci_low": float(low),
        "pearson_ci_high": float(high),
        "bootstrap_samples": int(bootstrap_samples),
    }
