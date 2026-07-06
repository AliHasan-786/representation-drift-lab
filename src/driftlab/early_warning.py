from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .artifacts import publish_manifest, read_json, sha256_file, write_json_atomic
from .provenance import ARTIFACT_SCHEMA_VERSION, environment_snapshot, git_revision, utc_now


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    residual = predicted - actual
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    r_squared = None if denominator <= 1e-15 else 1.0 - float(np.sum(residual**2)) / denominator
    if np.std(predicted) <= 1e-15:
        calibration_slope = None
        calibration_intercept = float(actual.mean())
    else:
        design = np.column_stack([np.ones(len(predicted)), predicted])
        calibration_intercept, calibration_slope = np.linalg.lstsq(
            design, actual, rcond=None
        )[0]
        calibration_intercept = float(calibration_intercept)
        calibration_slope = float(calibration_slope)
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mean_bias": float(np.mean(residual)),
        "r_squared": r_squared,
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
    }


def evaluate_early_warning(
    examples: Iterable[Mapping[str, Any]],
    *,
    feature_names: tuple[str, ...],
    ridge_candidates: tuple[float, ...] = (0.0, 0.01, 0.1, 1.0, 10.0),
    interval_coverage: float = 0.9,
) -> dict[str, Any]:
    """Fit on train scenarios, tune on validation, and report only held-out test error."""
    rows = [dict(example) for example in examples]
    if not rows or not feature_names:
        raise ValueError("early-warning evaluation requires examples and features")
    if any(not 0.0 < float(row["observation_fraction"]) < 1.0 for row in rows):
        raise ValueError("all predictors must be observed strictly before the final target")
    identifiers = [str(row["scenario_id"]) for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("scenario_id values must be unique across all splits")
    partitions: dict[str, list[dict[str, Any]]] = {
        split: [row for row in rows if row.get("split") == split]
        for split in ("train", "validation", "test")
    }
    if any(len(partitions[split]) < 3 for split in partitions):
        raise ValueError("train, validation, and test each require at least three scenarios")
    if any(alpha < 0 for alpha in ridge_candidates):
        raise ValueError("ridge candidates must be non-negative")

    def arrays(split: str) -> tuple[np.ndarray, np.ndarray]:
        selected = partitions[split]
        features = np.asarray(
            [[float(row["features"][name]) for name in feature_names] for row in selected],
            dtype=np.float64,
        )
        targets = np.asarray([float(row["final_forgetting"]) for row in selected])
        if not np.isfinite(features).all() or not np.isfinite(targets).all():
            raise ValueError("early-warning values must be finite")
        return features, targets

    train_x, train_y = arrays("train")
    validation_x, validation_y = arrays("validation")
    test_x, test_y = arrays("test")
    location = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale <= 1e-12] = 1.0

    def standardized(values: np.ndarray) -> np.ndarray:
        return (values - location) / scale

    def fit(alpha: float) -> np.ndarray:
        design = np.column_stack([np.ones(len(train_x)), standardized(train_x)])
        if alpha == 0.0:
            return np.linalg.lstsq(design, train_y, rcond=None)[0]
        penalty = np.eye(design.shape[1]) * alpha
        penalty[0, 0] = 0.0
        return np.linalg.solve(design.T @ design + penalty, design.T @ train_y)

    candidates = []
    for alpha in ridge_candidates:
        coefficients = fit(float(alpha))
        validation_design = np.column_stack(
            [np.ones(len(validation_x)), standardized(validation_x)]
        )
        predicted = validation_design @ coefficients
        candidates.append(
            {
                "alpha": float(alpha),
                "validation_rmse": _regression_metrics(validation_y, predicted)["rmse"],
            }
        )
    selected_alpha = min(
        candidates, key=lambda item: (float(item["validation_rmse"]), item["alpha"])
    )["alpha"]
    coefficients = fit(float(selected_alpha))

    train_design = np.column_stack([np.ones(len(train_x)), standardized(train_x)])
    test_design = np.column_stack([np.ones(len(test_x)), standardized(test_x)])
    model_train_prediction = train_design @ coefficients
    model_test_prediction = test_design @ coefficients
    mean_prediction = np.full(len(test_y), train_y.mean())
    if "early_forgetting" in feature_names:
        persistence_prediction = test_x[:, feature_names.index("early_forgetting")]
    else:
        persistence_prediction = np.zeros(len(test_y))
    residual_radius = float(
        np.quantile(
            np.abs(model_train_prediction - train_y), interval_coverage, method="higher"
        )
    )
    lower = model_test_prediction - residual_radius
    upper = model_test_prediction + residual_radius
    model_metrics = _regression_metrics(test_y, model_test_prediction)
    model_metrics.update(
        {
            "prediction_interval_nominal_coverage": interval_coverage,
            "prediction_interval_empirical_coverage": float(
                np.mean((test_y >= lower) & (test_y <= upper))
            ),
            "prediction_interval_radius": residual_radius,
        }
    )
    return {
        "protocol": {
            "split_strategy": "disjoint scenario-level train/validation/test",
            "temporal_rule": "features observed before final forgetting target",
            "feature_names": list(feature_names),
            "split_counts": {split: len(values) for split, values in partitions.items()},
            "maximum_observation_fraction": max(
                float(row["observation_fraction"]) for row in rows
            ),
            "selected_ridge_alpha": selected_alpha,
            "validation_candidates": candidates,
            "interval_method": "absolute train-residual quantile",
        },
        "test_metrics": {
            "early_warning_model": model_metrics,
            "train_mean_baseline": _regression_metrics(test_y, mean_prediction),
            "early_forgetting_persistence_baseline": _regression_metrics(
                test_y, persistence_prediction
            ),
        },
        "model": {
            "standardized_intercept": float(coefficients[0]),
            "standardized_coefficients": {
                name: float(value)
                for name, value in zip(feature_names, coefficients[1:])
            },
            "training_feature_mean": {
                name: float(value) for name, value in zip(feature_names, location)
            },
            "training_feature_scale": {
                name: float(value) for name, value in zip(feature_names, scale)
            },
        },
        "test_predictions": [
            {
                "scenario_id": str(row["scenario_id"]),
                "actual_final_forgetting": float(actual),
                "predicted_final_forgetting": float(predicted),
                "interval_low": float(low),
                "interval_high": float(high),
            }
            for row, actual, predicted, low, high in zip(
                partitions["test"], test_y, model_test_prediction, lower, upper
            )
        ],
    }


def build_synthetic_methodology_artifact(
    output: str | Path = "public/data/early-warning-methodology.json",
) -> Path:
    """Create a labeled synthetic artifact that exercises the held-out protocol."""
    rng = np.random.default_rng(5787)
    examples = []
    split_cycle = ("test", "validation", "train", "train", "train")
    for index in range(60):
        severity = rng.uniform(0.0, 1.0)
        plasticity = rng.uniform(0.0, 1.0)
        early_forgetting = np.clip(
            0.04 + 0.20 * severity + 0.07 * plasticity + rng.normal(0.0, 0.015),
            0.0,
            1.0,
        )
        centroid_drift = np.clip(
            0.02 + 0.45 * severity + 0.08 * plasticity + rng.normal(0.0, 0.025),
            0.0,
            1.0,
        )
        cka_loss = np.clip(
            0.01 + 0.25 * severity + rng.normal(0.0, 0.02), 0.0, 1.0
        )
        final_forgetting = np.clip(
            0.02
            + 0.45 * severity
            + 0.15 * plasticity
            + 0.30 * early_forgetting
            + rng.normal(0.0, 0.025),
            0.0,
            1.0,
        )
        examples.append(
            {
                "scenario_id": f"synthetic-{index:03d}",
                "split": split_cycle[index % len(split_cycle)],
                "observation_fraction": 0.25,
                "features": {
                    "early_forgetting": float(early_forgetting),
                    "centroid_drift": float(centroid_drift),
                    "cka_loss": float(cka_loss),
                    "plasticity_gain": float(0.20 * plasticity),
                },
                "final_forgetting": float(final_forgetting),
            }
        )
    feature_names = (
        "early_forgetting",
        "centroid_drift",
        "cka_loss",
        "plasticity_gain",
    )
    evaluation = evaluate_early_warning(examples, feature_names=feature_names)
    identity = {
        "name": "synthetic-early-warning-methodology",
        "seed": 5787,
        "scenario_count": len(examples),
        "feature_names": feature_names,
    }
    config_hash = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    run_id = f"synthetic-early-warning-methodology-{config_hash}"
    artifact_dir = Path("artifacts/analysis") / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    existing_manifest_path = artifact_dir / "manifest.json"
    started_at = utc_now()
    if existing_manifest_path.exists():
        started_at = str(read_json(existing_manifest_path).get("started_at", started_at))
    evaluation_path = artifact_dir / "evaluation.json"
    write_json_atomic(evaluation_path, evaluation)
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "complete",
        "config_hash": config_hash,
        "config": identity,
        "seed": 5787,
        "git": git_revision(),
        "environment": environment_snapshot(),
        "started_at": started_at,
        "completed_steps": [0],
        "artifacts": [
            {
                "path": "evaluation.json",
                "sha256": sha256_file(evaluation_path),
                "bytes": evaluation_path.stat().st_size,
            }
        ],
    }
    manifest_path = artifact_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    destination = Path(output)
    public_manifest_path, public_manifest_sha = publish_manifest(
        manifest, destination.parent / "manifests" / f"{run_id}.json"
    )
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_id": run_id,
        "config_hash": config_hash,
        "evidence_status": "synthetic-methodology-validation",
        "publication_caveat": (
            "This validates temporal splitting, baselines, calibration, and error "
            "reporting. It is not evidence that CLIP forgetting is predictable."
        ),
        "source_manifest": {
            "run_id": run_id,
            "public_path": f"/data/manifests/{public_manifest_path.name}",
            "sha256": public_manifest_sha,
        },
        "evaluation": evaluation,
    }
    write_json_atomic(destination, payload)
    return destination
