from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import yaml
import torch

from .artifacts import publish_manifest, read_json, sha256_file, write_json_atomic
from .benchmark import _aggregate_scalar_mappings
from .clip_data import load_reproduction_datasets
from .clip_reproduction import (
    _checkpoint_record,
    _device,
    _evaluate_both,
    _resolve_and_load_model,
    _save_outputs,
    _text_prototypes,
)
from .config import load_config
from .provenance import ARTIFACT_SCHEMA_VERSION, environment_snapshot, git_revision, utc_now


def set_lora_output_scale(model: Any, alpha: float, base_scales: list[tuple[Any, str, float]]) -> None:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("interpolation alpha must be between zero and one")
    for module, adapter, base_scale in base_scales:
        module.scaling[adapter] = base_scale * alpha


def _lora_scales(model: Any) -> list[tuple[Any, str, float]]:
    scales = []
    for module in model.modules():
        values = getattr(module, "scaling", None)
        if not isinstance(values, dict):
            continue
        for adapter, scale in values.items():
            scales.append((module, str(adapter), float(scale)))
    if not scales:
        raise RuntimeError("no LoRA scaling controls found")
    return scales


def run_interpolation_suite(config_path: str | Path) -> Path:
    suite_file = Path(config_path)
    raw = yaml.safe_load(suite_file.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("interpolation suite root must be a mapping")
    seeds = tuple(int(seed) for seed in raw["seeds"])
    alphas = tuple(float(alpha) for alpha in raw["alphas"])
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("interpolation requires at least three unique seeds")
    if not alphas or tuple(sorted(set(alphas))) != alphas or alphas[0] != 0.0 or alphas[-1] != 1.0:
        raise ValueError("alphas must be unique, sorted, and span zero through one")
    base = load_config((suite_file.parent / str(raw["base_config"])).resolve())
    identity = {
        "name": str(raw["name"]),
        "base_config": str(raw["base_config"]),
        "base_scientific_payload": base.scientific_payload(),
        "seeds": list(seeds),
        "alphas": list(alphas),
        "fidelity": str(raw["fidelity"]),
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
    per_alpha: dict[float, list[dict[str, Any]]] = {alpha: [] for alpha in alphas}
    source_runs = []

    for seed in seeds:
        torch.manual_seed(seed)
        config = replace(base, seed=seed)
        source_dir = config.output_dir / config.run_id
        source_manifest_path = source_dir / "manifest.json"
        final_checkpoint_path = source_dir / f"checkpoints/step_{config.total_steps:06d}.pt"
        if not source_manifest_path.exists() or not final_checkpoint_path.exists():
            raise FileNotFoundError(f"source LoRA run is incomplete: {config.run_id}")
        source_manifest = read_json(source_manifest_path)
        device = _device(str(config.runtime.get("device", "auto")))
        datasets = load_reproduction_datasets(config.datasets, seed)
        model, processor, resolved_revision, imports = _resolve_and_load_model(config, device)
        prototypes = {
            "retained": _text_prototypes(
                model, processor, datasets["retained_eval"].class_names, device
            ),
            "adaptation": _text_prototypes(
                model, processor, datasets["adaptation_eval"].class_names, device
            ),
        }
        baseline = _evaluate_both(
            model,
            processor,
            datasets,
            prototypes,
            device,
            int(config.training.get("evaluation_batch_size", config.batch_size)),
        )
        state = torch.load(final_checkpoint_path, map_location="cpu", weights_only=True)
        if state["config_hash"] != config.config_hash:
            raise ValueError("source interpolation checkpoint configuration mismatch")
        imports["set_peft_model_state_dict"](
            model.vision_model, state["adapter_state"]
        )
        base_scales = _lora_scales(model.vision_model)
        class_names = {
            "retained": datasets["retained_eval"].class_names,
            "adaptation": datasets["adaptation_eval"].class_names,
        }
        for alpha in alphas:
            set_lora_output_scale(model.vision_model, alpha, base_scales)
            current = _evaluate_both(
                model,
                processor,
                datasets,
                prototypes,
                device,
                int(config.training.get("evaluation_batch_size", config.batch_size)),
            )
            for role in ("retained", "adaptation"):
                _save_outputs(
                    run_dir / f"outputs/seed_{seed}_alpha_{alpha:.2f}_{role}.npz",
                    current[role],
                )
            record = _checkpoint_record(
                int(round(alpha * 1000)),
                baseline,
                current,
                int(config.analysis.get("covariance_projection_dim", 32)),
                class_names,
            )
            record.pop("step")
            record["alpha"] = alpha
            per_alpha[alpha].append(record)
        source_runs.append(
            {
                "run_id": config.run_id,
                "seed": seed,
                "config_hash": config.config_hash,
                "manifest_sha256": sha256_file(source_manifest_path),
                "checkpoint_sha256": sha256_file(final_checkpoint_path),
                "model_revision": resolved_revision,
            }
        )

    curve = []
    for alpha in alphas:
        records = per_alpha[alpha]
        retained = _aggregate_scalar_mappings(
            [record["retained"] for record in records]
        )
        adaptation = _aggregate_scalar_mappings(
            [record["adaptation"] for record in records]
        )
        geometry = _aggregate_scalar_mappings(
            [record["geometry"]["retained"] for record in records]
        )
        curve.append(
            {
                "alpha": alpha,
                "retained": {
                    key: retained[key]
                    for key in ("top1_accuracy", "accuracy_change", "macro_f1")
                },
                "adaptation": {
                    key: adaptation[key]
                    for key in ("top1_accuracy", "accuracy_change", "macro_f1")
                },
                "geometry": {
                    key: geometry[key]
                    for key in (
                        "linear_cka",
                        "cosine_centroid_drift",
                        "frechet_distance",
                        "neighborhood_overlap_at_5",
                    )
                },
            }
        )
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
        "completed_steps": list(range(len(alphas))),
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
        "evidence_status": "local-posthoc-recovery-preliminary",
        "publication_caveat": (
            "The curve scales a trained LoRA output on a small local scenario. "
            "It is WiSE-FT-inspired and is not an exact WiSE-FT reproduction."
        ),
        "source_manifest": {
            "run_id": run_id,
            "public_path": f"/data/manifests/{public_manifest.name}",
            "sha256": public_manifest_sha,
        },
        "experiment": {
            "fidelity": identity["fidelity"],
            "seeds": list(seeds),
            "run_count": len(seeds),
            "alphas": list(alphas),
        },
        "curve": curve,
    }
    write_json_atomic(output, payload)
    return output
