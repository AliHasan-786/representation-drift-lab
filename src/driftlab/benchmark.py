from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .artifacts import (
    publish_manifest,
    read_json,
    sha256_file,
    validate_web_artifact,
    write_json_atomic,
)
from .clip_reproduction import regenerate_clip_metrics, run_clip_reproduction
from .config import load_config
from .metrics import mean_confidence_interval
from .provenance import ARTIFACT_SCHEMA_VERSION, environment_snapshot, git_revision, utc_now


def _aggregate_scalar_mappings(
    mappings: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recursively aggregate numeric leaves shared by every independent run."""
    if len(mappings) < 2:
        raise ValueError("aggregation requires at least two runs")
    shared = set.intersection(*(set(mapping) for mapping in mappings))
    aggregate: dict[str, Any] = {}
    for key in sorted(shared):
        values = [mapping[key] for mapping in mappings]
        if all(isinstance(value, Mapping) for value in values):
            nested = _aggregate_scalar_mappings(values)  # type: ignore[arg-type]
            if nested:
                aggregate[key] = nested
        elif all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and np.isfinite(value)
            for value in values
        ):
            aggregate[key] = mean_confidence_interval(np.asarray(values, dtype=float))
    return aggregate


def aggregate_run_artifacts(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Align checkpoints and aggregate all shared numeric diagnostics."""
    if len(runs) < 2:
        raise ValueError("a benchmark needs at least two run artifacts")
    step_sequences = [
        [int(checkpoint["step"]) for checkpoint in run["checkpoints"]] for run in runs
    ]
    if any(steps != step_sequences[0] for steps in step_sequences[1:]):
        raise ValueError("run checkpoint steps do not align")
    checkpoints = []
    for index, step in enumerate(step_sequences[0]):
        source = [run["checkpoints"][index] for run in runs]
        checkpoints.append(
            {
                "step": step,
                "retained": _aggregate_scalar_mappings(
                    [checkpoint["retained"] for checkpoint in source]
                ),
                "adaptation": _aggregate_scalar_mappings(
                    [checkpoint["adaptation"] for checkpoint in source]
                ),
                "geometry": _aggregate_scalar_mappings(
                    [checkpoint["geometry"] for checkpoint in source]
                ),
                "cross_modal": _aggregate_scalar_mappings(
                    [checkpoint.get("cross_modal", {}) for checkpoint in source]
                ),
                "layerwise": _aggregate_scalar_mappings(
                    [checkpoint.get("layerwise", {}) for checkpoint in source]
                ),
                "optimization": _aggregate_scalar_mappings(
                    [checkpoint.get("optimization", {}) for checkpoint in source]
                ),
            }
        )
    return checkpoints


def run_clip_benchmark_suite(
    suite_path: str | Path,
    *,
    resume: bool = True,
    regenerate_metrics: bool = False,
) -> dict[str, Path]:
    suite_file = Path(suite_path)
    raw = yaml.safe_load(suite_file.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("benchmark suite root must be a mapping")
    name = str(raw.get("name", ""))
    seeds = tuple(int(seed) for seed in raw.get("seeds", ()))
    if not name or len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("suite requires a name and at least three unique seeds")
    base_path = (suite_file.parent / str(raw["base_config"])).resolve()
    base = load_config(base_path)
    web_output = Path(str(raw.get("web_output", "public/data/benchmark-local.json")))
    run_web_dir = web_output.parent / "runs"
    run_artifacts: list[dict[str, Any]] = []
    run_sources: list[dict[str, Any]] = []
    started_at = utc_now()

    for seed in seeds:
        config = replace(base, seed=seed)
        run_web_path = run_web_dir / f"{config.run_id}.json"
        outputs = (
            regenerate_clip_metrics(config, web_output=run_web_path)
            if regenerate_metrics
            else run_clip_reproduction(
                config,
                resume=resume,
                web_output=run_web_path,
            )
        )
        artifact = read_json(outputs["web_artifact"])
        validate_web_artifact(artifact)
        run_artifacts.append(artifact)
        run_sources.append(
            {
                "run_id": artifact["run_id"],
                "seed": seed,
                "config_hash": artifact["config_hash"],
                "web_artifact": str(run_web_path),
                "web_artifact_sha256": sha256_file(run_web_path),
                "source_manifest": artifact["source_manifest"],
            }
        )

    suite_identity = {
        "name": name,
        "base_config": str(raw["base_config"]),
        "base_scientific_payload": base.scientific_payload(),
        "seeds": list(seeds),
    }
    config_hash = hashlib.sha256(
        json.dumps(suite_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    run_id = f"{name}-{config_hash}"
    benchmark_dir = Path("artifacts/benchmarks") / run_id
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    existing_manifest_path = benchmark_dir / "manifest.json"
    if existing_manifest_path.exists():
        existing_manifest = read_json(existing_manifest_path)
        started_at = str(existing_manifest.get("started_at", started_at))
    aggregate_checkpoints = aggregate_run_artifacts(run_artifacts)
    manifest_path = benchmark_dir / "manifest.json"
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "complete",
        "config_hash": config_hash,
        "config": suite_identity,
        "git": git_revision(),
        "environment": environment_snapshot(),
        "started_at": started_at,
        "completed_at": utc_now(),
        "completed_steps": [item["step"] for item in aggregate_checkpoints],
        "source_runs": run_sources,
    }
    write_json_atomic(manifest_path, manifest)
    public_manifest_path, public_manifest_sha256 = publish_manifest(
        manifest,
        web_output.parent / "manifests" / f"{run_id}.json",
    )
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "config_hash": config_hash,
        "evidence_status": "local-multiseed-preliminary",
        "source_manifest": {
            "run_id": run_id,
            "public_path": f"/data/manifests/{public_manifest_path.name}",
            "sha256": public_manifest_sha256,
        },
        "experiment": {
            "name": name,
            "model": run_artifacts[0]["experiment"]["model"],
            "method": run_artifacts[0]["experiment"]["method"],
            "seeds": list(seeds),
            "run_count": len(seeds),
            "uncertainty": "two-sided 95% Student-t confidence interval",
        },
        "datasets": {
            "selection_policy": base.datasets,
            "per_run_fingerprints": {
                run["run_id"]: {
                    role: details["fingerprint"]
                    for role, details in run["datasets"].items()
                }
                for run in run_artifacts
            },
        },
        "runs": run_sources,
        "checkpoints": aggregate_checkpoints,
    }
    write_json_atomic(web_output, payload)
    validate_web_artifact(payload)
    return {
        "benchmark_dir": benchmark_dir,
        "manifest": manifest_path,
        "web_artifact": web_output,
    }
