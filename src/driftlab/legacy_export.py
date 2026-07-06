from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from .artifacts import publish_manifest, sha256_file, write_json_atomic
from .metrics import association_with_bootstrap
from .provenance import ARTIFACT_SCHEMA_VERSION


METRICS_RELATIVE_PATH = Path(
    "drift_analysis_final_submission_v2/results/final_metrics.csv"
)
SUMMARY_RELATIVE_PATH = Path(
    "drift_analysis_final_submission_v2/results/final_report_text.txt"
)


def _manifest_artifacts(manifest_path: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {artifact["path"]: artifact for artifact in manifest["artifacts"]}


def _verified_path(
    root: Path, relative: Path, registered: dict[str, dict[str, Any]]
) -> tuple[Path, dict[str, Any]]:
    artifact = registered.get(str(relative))
    if artifact is None:
        raise ValueError(f"legacy artifact is not registered: {relative}")
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(artifact["bytes"]):
        raise ValueError(f"legacy artifact size mismatch: {relative}")
    if sha256_file(path) != artifact["sha256"]:
        raise ValueError(f"legacy artifact checksum mismatch: {relative}")
    return path, artifact


def _load_metrics(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = {key: float(value) for key, value in raw.items()}
            row["step"] = int(row["step"])
            rows.append(row)
    rows.sort(key=lambda row: int(row["step"]))
    if len(rows) < 4:
        raise ValueError("legacy metrics require at least four checkpoints")
    steps = [int(row["step"]) for row in rows]
    if steps != sorted(set(steps)):
        raise ValueError("legacy metric steps must be sorted and unique")
    return rows


def _infer_baseline(
    rows: list[dict[str, float]],
    value: str,
    delta: str,
    *,
    delta_is_baseline_minus_current: bool,
) -> float:
    sign = 1.0 if delta_is_baseline_minus_current else -1.0
    candidates = [row[value] + sign * row[delta] for row in rows]
    baseline = float(median(candidates))
    if max(abs(candidate - baseline) for candidate in candidates) > 1e-8:
        raise ValueError(f"legacy {value}/{delta} columns imply inconsistent baselines")
    return baseline


def export_legacy_web_artifact(
    *,
    root: str | Path,
    output: str | Path = "public/data/legacy-historical.json",
    manifest_path: str | Path = "data/legacy/manifest.json",
) -> Path:
    """Convert the registered metric table without upgrading its evidence status."""
    root_path = Path(root)
    registry_path = Path(manifest_path)
    registered = _manifest_artifacts(registry_path)
    metrics_path, metrics_artifact = _verified_path(
        root_path, METRICS_RELATIVE_PATH, registered
    )
    summary_path, summary_artifact = _verified_path(
        root_path, SUMMARY_RELATIVE_PATH, registered
    )
    rows = _load_metrics(metrics_path)
    cifar_baseline = _infer_baseline(
        rows,
        "cifar_acc",
        "forgetting",
        delta_is_baseline_minus_current=True,
    )
    food_baseline = _infer_baseline(
        rows,
        "food_acc",
        "learning",
        delta_is_baseline_minus_current=False,
    )
    checkpoints: list[dict[str, Any]] = [
        {
            "step": 0,
            "retained": {
                "top1_accuracy": cifar_baseline,
                "accuracy_change": 0.0,
                "sample_count": 2000,
            },
            "adaptation": {
                "top1_accuracy": food_baseline,
                "accuracy_change": 0.0,
                "sample_count": 1000,
            },
            "geometry": {
                "retained": {"cosine_centroid_drift": 0.0, "frechet_distance": 0.0},
                "adaptation": {"cosine_centroid_drift": 0.0, "frechet_distance": 0.0},
            },
        }
    ]
    for row in rows:
        checkpoints.append(
            {
                "step": int(row["step"]),
                "retained": {
                    "top1_accuracy": row["cifar_acc"],
                    "accuracy_change": -row["forgetting"],
                    "sample_count": 2000,
                },
                "adaptation": {
                    "top1_accuracy": row["food_acc"],
                    "accuracy_change": row["learning"],
                    "sample_count": 1000,
                },
                "geometry": {
                    "retained": {
                        "cosine_centroid_drift": row["cifar_drift"],
                        "frechet_distance": row["cifar_fd"],
                    },
                    "adaptation": {
                        "cosine_centroid_drift": row["food_drift"],
                        "frechet_distance": row["food_fd"],
                    },
                },
            }
        )
    association = association_with_bootstrap(
        np.array([row["cifar_fd"] for row in rows]),
        np.array([row["forgetting"] for row in rows]),
        seed=42,
        bootstrap_samples=2000,
    )
    final = rows[-1]
    destination = Path(output)
    public_manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": "legacy-family-b-23000-steps",
        "status": "historical-unreproduced",
        "config_hash": metrics_artifact["sha256"][:16],
        "datasets": {
            "retained": {"name": "cifar10", "sample_count": 2000},
            "adaptation": {"name": "food101", "sample_count": 1000},
        },
        "artifacts": [metrics_artifact, summary_artifact],
    }
    public_manifest_path, public_manifest_sha256 = publish_manifest(
        public_manifest,
        destination.parent / "manifests" / "legacy-family-b-23000-steps.json",
    )
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": "legacy-family-b-23000-steps",
        "config_hash": metrics_artifact["sha256"][:16],
        "config_hash_kind": "artifact-identity-not-training-config",
        "evidence_status": "historical-unreproduced",
        "publication_treatment": "Historical evidence only; producing training code and exact configuration are absent.",
        "source_manifest": {
            "registry_relative_path": str(registry_path),
            "public_path": f"/data/manifests/{public_manifest_path.name}",
            "sha256": public_manifest_sha256,
            "metrics": {
                "relative_path": str(METRICS_RELATIVE_PATH),
                "sha256": metrics_artifact["sha256"],
            },
            "summary": {
                "relative_path": str(SUMMARY_RELATIVE_PATH),
                "sha256": summary_artifact["sha256"],
            },
        },
        "experiment": {
            "name": "Legacy Family B - saved 23,000-step artifacts",
            "model": {"family": "openai-clip-vit-b-32", "revision": "unverified"},
            "method": {"name": "lora", "fidelity": "legacy-unverified"},
            "seed": "reported-42-not-reproducible-from-bundle",
        },
        "datasets": {
            "retained": {"name": "cifar10", "sample_count": 2000, "fingerprint": "unavailable"},
            "adaptation": {"name": "food101", "sample_count": 1000, "fingerprint": "unavailable"},
        },
        "checkpoints": checkpoints,
        "analysis": {
            "drift_forgetting_association": association,
            "interpretation": "Association only; checkpoints from one run are autocorrelated and do not establish causation.",
            "early_warning": {
                "checkpoint_step": 8000,
                "predicted_final_drift": 0.6647,
                "actual_final_drift": 0.3055,
                "absolute_error": 0.3592,
                "status": "failed-legacy-predictor",
            },
        },
        "summary": {
            "baseline_retained_accuracy": cifar_baseline,
            "baseline_adaptation_accuracy": food_baseline,
            "final_retained_accuracy": final["cifar_acc"],
            "final_adaptation_accuracy": final["food_acc"],
            "final_retained_accuracy_change": -final["forgetting"],
            "final_adaptation_accuracy_change": final["learning"],
            "registered_summary_sha256": sha256_file(summary_path),
        },
        "unavailable_metrics": [
            "macro_f1",
            "negative_log_likelihood",
            "expected_calibration_error",
            "linear_cka",
            "effective_rank",
            "local_neighborhood_overlap",
            "layerwise_drift",
            "confidence_intervals",
        ],
    }
    write_json_atomic(destination, payload)
    return destination
