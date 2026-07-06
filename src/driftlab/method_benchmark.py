from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml
import numpy as np

from .artifacts import (
    publish_manifest,
    read_json,
    sha256_file,
    validate_web_artifact,
    write_json_atomic,
)
from .provenance import ARTIFACT_SCHEMA_VERSION, environment_snapshot, git_revision, utc_now
from .metrics import association_with_bootstrap


def complement_interval(interval: Mapping[str, Any]) -> dict[str, Any]:
    """Transform an uncertainty interval for x into the interval for 1-x."""
    return {
        **dict(interval),
        "mean": 1.0 - float(interval["mean"]),
        "ci_low": 1.0 - float(interval["ci_high"]),
        "ci_high": 1.0 - float(interval["ci_low"]),
    }


def build_method_comparison(config_path: str | Path) -> Path:
    config_file = Path(config_path)
    raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("methods"), list):
        raise ValueError("method comparison config must define a method list")
    if len(raw["methods"]) < 2:
        raise ValueError("method comparison requires at least two methods")
    records = []
    sources = []
    reference_fingerprints = None
    for method in raw["methods"]:
        artifact_path = Path(str(method["artifact"]))
        artifact = read_json(artifact_path)
        validate_web_artifact(artifact)
        if int(artifact["experiment"]["run_count"]) < 3:
            raise ValueError(f"{method['id']} has fewer than three valid runs")
        fingerprints_by_run = artifact["datasets"]["per_run_fingerprints"]
        fingerprints = {
            str(source["seed"]): fingerprints_by_run[source["run_id"]]
            for source in artifact["runs"]
        }
        if reference_fingerprints is None:
            reference_fingerprints = fingerprints
        elif fingerprints != reference_fingerprints:
            raise ValueError("method artifacts do not use identical dataset selections")
        baseline = artifact["checkpoints"][0]
        final = artifact["checkpoints"][-1]
        strategy = artifact["experiment"]["method"].get("strategy", "standard")
        if str(method["id"]) == "linear-probe":
            strategy = "linear-probe"
        final_step = int(final["step"])
        if strategy == "linear-probe":
            probe_steps, joint_steps = final_step, 0
        else:
            probe_steps = int(artifact["experiment"]["method"].get("probe_steps", 0))
            joint_steps = final_step
        records.append(
            {
                "id": str(method["id"]),
                "label": str(method["label"]),
                "category": str(method["category"]),
                "fidelity": artifact["experiment"]["method"]["fidelity"],
                "strategy": strategy,
                "seeds": artifact["experiment"]["seeds"],
                "checkpoints": [item["step"] for item in artifact["checkpoints"]],
                "training_budget": {
                    "probe_steps": probe_steps,
                    "joint_or_adaptation_steps": joint_steps,
                    "total_optimizer_steps": probe_steps + joint_steps,
                },
                "metrics": {
                    "baseline_adaptation_accuracy": baseline["adaptation"][
                        "top1_accuracy"
                    ],
                    "final_adaptation_accuracy": final["adaptation"]["top1_accuracy"],
                    "adaptation_accuracy_change": final["adaptation"][
                        "accuracy_change"
                    ],
                    "baseline_retained_accuracy": baseline["retained"][
                        "top1_accuracy"
                    ],
                    "final_retained_accuracy": final["retained"]["top1_accuracy"],
                    "retained_accuracy_change": final["retained"]["accuracy_change"],
                    "retained_cka_loss": complement_interval(
                        final["geometry"]["retained"]["linear_cka"]
                    ),
                    "trainable_parameters": final["optimization"][
                        "trainable_parameters"
                    ],
                    "adapter_l2_delta": final["optimization"][
                        "adapter_l2_delta_from_initial"
                    ],
                },
                "source": {
                    "benchmark_run_id": artifact["run_id"],
                    "benchmark_config_hash": artifact["config_hash"],
                    "artifact_path": str(artifact_path),
                    "artifact_sha256": sha256_file(artifact_path),
                    "manifest": artifact["source_manifest"],
                },
                "limitations": (
                    [
                        "The head reached 100% on a tiny six-class evaluation subset; "
                        "this is a separability diagnostic, not a realistic Food-101 score."
                    ]
                    if str(method["id"]) == "linear-probe"
                    else (
                        [
                            "The random-head full fine-tune remained below the zero-shot "
                            "adaptation baseline on this sparse local subset."
                        ]
                        if str(method["id"]) == "full-finetune"
                        else []
                    )
                ),
            }
        )
        sources.append(
            {
                "run_id": artifact["run_id"],
                "config_hash": artifact["config_hash"],
                "artifact_path": str(artifact_path),
                "artifact_sha256": sha256_file(artifact_path),
                "source_manifest": artifact["source_manifest"],
            }
        )
    identity = {
        "name": str(raw["name"]),
        "evidence_status": str(raw["evidence_status"]),
        "methods": [
            {
                "id": record["id"],
                "source_config_hash": record["source"]["benchmark_config_hash"],
            }
            for record in records
        ],
    }
    config_hash = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    run_id = f"{identity['name']}-{config_hash}"
    run_dir = Path("artifacts/benchmarks") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    started_at = utc_now()
    if manifest_path.exists():
        started_at = str(read_json(manifest_path).get("started_at", started_at))
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "complete",
        "config_hash": config_hash,
        "config": identity,
        "git": git_revision(),
        "environment": environment_snapshot(),
        "started_at": started_at,
        "completed_at": utc_now(),
        "completed_steps": [0],
        "source_runs": sources,
    }
    write_json_atomic(manifest_path, manifest)
    output = Path(str(raw["output"]))
    public_manifest, public_manifest_sha = publish_manifest(
        manifest, output.parent / "manifests" / f"{run_id}.json"
    )
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "config_hash": config_hash,
        "evidence_status": identity["evidence_status"],
        "publication_caveat": (
            "Small local subsets and three seeds validate comparison machinery; "
            "they do not establish general method rankings."
        ),
        "source_manifest": {
            "run_id": run_id,
            "public_path": f"/data/manifests/{public_manifest.name}",
            "sha256": public_manifest_sha,
        },
        "dataset_fingerprints": reference_fingerprints,
        "methods": records,
        "analysis": {
            "retained_cka_loss_vs_mean_forgetting": association_with_bootstrap(
                np.asarray(
                    [record["metrics"]["retained_cka_loss"]["mean"] for record in records]
                ),
                np.asarray(
                    [
                        -record["metrics"]["retained_accuracy_change"]["mean"]
                        for record in records
                    ]
                ),
                seed=5787,
                bootstrap_samples=2000,
            ),
            "caveat": (
                f"Exploratory association across {len(records)} interventions on one bounded "
                "scenario; it is neither causal nor a general predictor validation."
            ),
        },
        "failure_cases": [
            {
                "id": "distillation-drift-without-mean-forgetting",
                "method_id": "zscl-inspired-distillation",
                "observation": (
                    "Mean retained accuracy returned to baseline while retained "
                    "CKA still changed materially."
                ),
            },
            {
                "id": "nullspace-lower-drift-worse-retention",
                "method_id": "retention-gradient-nullspace",
                "observation": (
                    "Lower mean CKA loss than standard LoRA did not produce better "
                    "mean retained accuracy."
                ),
            },
        ],
    }
    write_json_atomic(output, payload)
    return output
