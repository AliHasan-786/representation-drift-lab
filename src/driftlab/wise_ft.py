from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from .artifacts import publish_manifest, read_json, sha256_file, validate_web_artifact, write_json_atomic
from .benchmark import aggregate_run_artifacts
from .clip_data import load_reproduction_datasets
from .clip_reproduction import (
    _checkpoint_record,
    _device,
    _evaluate_both,
    _resolve_base_model,
    _save_outputs,
    _text_prototypes,
)
from .config import load_config
from .provenance import ARTIFACT_SCHEMA_VERSION, environment_snapshot, git_revision, utc_now


def interpolate_state_dict(
    baseline: Mapping[str, torch.Tensor],
    tuned: Mapping[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("weight interpolation alpha must be between zero and one")
    if set(baseline) != set(tuned):
        raise ValueError("weight interpolation state dictionaries must align")
    result = {}
    for key, initial in baseline.items():
        final = tuned[key].to(dtype=initial.dtype, device=initial.device)
        result[key] = (
            initial * (1.0 - alpha) + final * alpha
            if initial.is_floating_point()
            else initial.clone()
        )
    return result


def run_wise_ft_suite(config_path: str | Path) -> Path:
    suite_file = Path(config_path)
    raw = yaml.safe_load(suite_file.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("WiSE-FT suite root must be a mapping")
    alpha = float(raw["alpha"])
    seeds = tuple(int(seed) for seed in raw["seeds"])
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("WiSE-FT suite requires at least three unique seeds")
    source_base = load_config(
        (suite_file.parent / str(raw["source_config"])).resolve()
    )
    identity = {
        "name": str(raw["name"]),
        "source_config": str(raw["source_config"]),
        "source_scientific_payload": source_base.scientific_payload(),
        "seeds": list(seeds),
        "alpha": alpha,
        "fidelity": str(raw["fidelity"]),
    }
    config_hash = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    run_id = f"{identity['name']}-{config_hash}"
    run_dir = Path("artifacts/benchmarks") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_artifacts = []
    source_runs = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        config = replace(source_base, seed=seed)
        source_dir = config.output_dir / config.run_id
        checkpoint_path = source_dir / f"checkpoints/step_{config.total_steps:06d}.pt"
        source_manifest_path = source_dir / "manifest.json"
        if not checkpoint_path.exists() or not source_manifest_path.exists():
            raise FileNotFoundError(f"source full fine-tune run missing: {config.run_id}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["config_hash"] != config.config_hash:
            raise ValueError("WiSE-FT source checkpoint configuration mismatch")
        device = _device(str(config.runtime.get("device", "auto")))
        datasets = load_reproduction_datasets(config.datasets, seed)
        model, processor, resolved_revision, _ = _resolve_base_model(config, device)
        prototypes = {
            "retained": _text_prototypes(
                model, processor, datasets["retained_eval"].class_names, device
            ),
            "adaptation": _text_prototypes(
                model, processor, datasets["adaptation_eval"].class_names, device
            ),
        }
        batch_size = int(config.training.get("evaluation_batch_size", config.batch_size))
        baseline = _evaluate_both(
            model, processor, datasets, prototypes, device, batch_size
        )
        vision_state = interpolate_state_dict(
            model.vision_model.state_dict(), checkpoint["vision_state"], alpha
        )
        projection_state = interpolate_state_dict(
            model.visual_projection.state_dict(),
            checkpoint["visual_projection_state"],
            alpha,
        )
        model.vision_model.load_state_dict(vision_state)
        model.visual_projection.load_state_dict(projection_state)
        current = _evaluate_both(
            model, processor, datasets, prototypes, device, batch_size
        )
        zero_shot_weight = (
            model.logit_scale.detach().exp().cpu()
            * prototypes["adaptation"].detach().cpu()
        )
        tuned_weight = checkpoint["classifier_state"]["weight"].float()
        tuned_bias = checkpoint["classifier_state"]["bias"].float()
        classifier_weight = (1.0 - alpha) * zero_shot_weight + alpha * tuned_weight
        classifier_bias = alpha * tuned_bias
        current["adaptation"]["logits"] = (
            torch.as_tensor(current["adaptation"]["image_embeddings"])
            @ classifier_weight.T
            + classifier_bias
        ).numpy()
        class_names = {
            "retained": datasets["retained_eval"].class_names,
            "adaptation": datasets["adaptation_eval"].class_names,
        }
        records = [
            _checkpoint_record(
                0,
                baseline,
                baseline,
                int(config.analysis.get("covariance_projection_dim", 32)),
                class_names,
            ),
            _checkpoint_record(
                config.total_steps,
                baseline,
                current,
                int(config.analysis.get("covariance_projection_dim", 32)),
                class_names,
            ),
        ]
        trainable_parameters = int(
            read_json(source_manifest_path)["model"]["encoder_trainable_parameters"]
            + read_json(source_manifest_path)["model"]["head_trainable_parameters"]
        )
        for record in records:
            record["optimization"] = {
                "adapter_l2_delta_from_initial": 0.0,
                "trainable_parameters": trainable_parameters,
            }
        for role in ("retained", "adaptation"):
            _save_outputs(
                run_dir / f"outputs/seed_{seed}_alpha_{alpha:.2f}_{role}.npz",
                current[role],
            )
        per_run = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "run_id": f"{run_id}-s{seed}",
            "config_hash": config_hash,
            "source_manifest": {"run_id": config.run_id},
            "experiment": {
                "name": identity["name"],
                "model": {
                    "requested_revision": config.model["revision"],
                    "resolved_revision": resolved_revision,
                    "trainable_parameters": trainable_parameters,
                },
                "method": {
                    "name": "wise-ft",
                    "strategy": "wise-ft",
                    "fidelity": identity["fidelity"],
                    "alpha": alpha,
                    "source_training_steps": config.total_steps,
                },
                "seed": seed,
            },
            "datasets": {
                key: {
                    "repository": value.name,
                    "split": value.split,
                    "revision": value.revision,
                    "selection": value.selection,
                    "fingerprint": value.fingerprint,
                    "samples": len(value.labels),
                }
                for key, value in datasets.items()
            },
            "checkpoints": records,
        }
        run_artifacts.append(per_run)
        source_runs.append(
            {
                "run_id": config.run_id,
                "seed": seed,
                "config_hash": config.config_hash,
                "manifest_sha256": sha256_file(source_manifest_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
            }
        )

    checkpoints = aggregate_run_artifacts(run_artifacts)
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
        "completed_steps": [item["step"] for item in checkpoints],
        "source_runs": source_runs,
        "artifacts": [
            {
                "path": str(path.relative_to(run_dir)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted((run_dir / "outputs").glob("*.npz"))
        ],
    }
    write_json_atomic(manifest_path, manifest)
    output = Path(str(raw["web_output"]))
    public_manifest, public_manifest_sha = publish_manifest(
        manifest, output.parent / "manifests" / f"{run_id}.json"
    )
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "config_hash": config_hash,
        "evidence_status": "local-multiseed-preliminary",
        "source_manifest": {
            "run_id": run_id,
            "public_path": f"/data/manifests/{public_manifest.name}",
            "sha256": public_manifest_sha,
        },
        "experiment": {
            "name": identity["name"],
            "model": run_artifacts[0]["experiment"]["model"],
            "method": run_artifacts[0]["experiment"]["method"],
            "seeds": list(seeds),
            "run_count": len(seeds),
            "uncertainty": "two-sided 95% Student-t confidence interval",
        },
        "datasets": {
            "selection_policy": source_base.datasets,
            "per_run_fingerprints": {
                run["run_id"]: {
                    role: details["fingerprint"]
                    for role, details in run["datasets"].items()
                }
                for run in run_artifacts
            },
        },
        "runs": [
            {
                "run_id": run["run_id"],
                "seed": seed,
                "config_hash": config_hash,
            }
            for run, seed in zip(run_artifacts, seeds)
        ],
        "checkpoints": checkpoints,
    }
    write_json_atomic(output, payload)
    validate_web_artifact(payload)
    return output
