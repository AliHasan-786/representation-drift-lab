from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .artifacts import publish_manifest, read_json, sha256_file, validate_web_artifact, write_json_atomic
from .method_benchmark import complement_interval
from .provenance import ARTIFACT_SCHEMA_VERSION, environment_snapshot, git_revision, utc_now


def build_domain_comparison(config_path: str | Path) -> Path:
    config_file = Path(config_path)
    raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or len(raw.get("scenarios", [])) < 3:
        raise ValueError("domain comparison requires at least three scenarios")
    scenarios = []
    sources = []
    for scenario in raw["scenarios"]:
        artifact_path = Path(str(scenario["artifact"]))
        artifact = read_json(artifact_path)
        validate_web_artifact(artifact)
        if int(artifact["experiment"]["run_count"]) < 3:
            raise ValueError(f"{scenario['id']} has fewer than three runs")
        baseline = artifact["checkpoints"][0]
        final = artifact["checkpoints"][-1]
        policy = artifact["datasets"]["selection_policy"]
        adaptation_spec = policy["adaptation"]
        retained_spec = policy["retained"]
        adaptation_samples = int(adaptation_spec["class_count"]) * int(
            adaptation_spec["eval_per_class"]
        )
        retained_samples = (
            int(retained_spec["eval_size"])
            if "eval_size" in retained_spec
            else int(retained_spec["class_count"])
            * int(retained_spec["eval_per_class"])
        )
        scenarios.append(
            {
                "id": str(scenario["id"]),
                "label": str(scenario["label"]),
                "adaptation_domain": str(scenario["adaptation_domain"]),
                "retained_domain": str(scenario["retained_domain"]),
                "seeds": artifact["experiment"]["seeds"],
                "sample_counts_per_seed": {
                    "adaptation_eval": adaptation_samples,
                    "retained_eval": retained_samples,
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
                },
                "source": {
                    "run_id": artifact["run_id"],
                    "config_hash": artifact["config_hash"],
                    "artifact_path": str(artifact_path),
                    "artifact_sha256": sha256_file(artifact_path),
                    "manifest": artifact["source_manifest"],
                },
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
        "scenarios": [
            {"id": item["id"], "config_hash": item["source"]["config_hash"]}
            for item in scenarios
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
        "evidence_status": str(raw["evidence_status"]),
        "publication_caveat": (
            "Three small local scenarios establish domain coverage and expose failure "
            "modes; they do not estimate population-level model behavior."
        ),
        "source_manifest": {
            "run_id": run_id,
            "public_path": f"/data/manifests/{public_manifest.name}",
            "sha256": public_manifest_sha,
        },
        "scenarios": scenarios,
    }
    write_json_atomic(output, payload)
    return output
