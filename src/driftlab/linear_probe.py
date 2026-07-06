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
    _feature_tensor,
    _image_embeddings,
    _resolve_base_model,
    _save_outputs,
    _text_prototypes,
)
from .config import ExperimentConfig, load_config
from .provenance import ARTIFACT_SCHEMA_VERSION, environment_snapshot, git_revision, utc_now


def run_linear_probe(
    config: ExperimentConfig, *, web_output: str | Path
) -> dict[str, Path]:
    run_dir = config.output_dir / config.run_id
    manifest_path = run_dir / "manifest.json"
    web_path = Path(web_output)
    if manifest_path.exists() and web_path.exists():
        manifest = read_json(manifest_path)
        artifact = read_json(web_path)
        if manifest.get("config_hash") != config.config_hash:
            raise ValueError("existing classification run configuration mismatch")
        validate_web_artifact(artifact)
        if artifact.get("run_id") == config.run_id:
            return {
                "run_dir": run_dir,
                "manifest": manifest_path,
                "web_artifact": web_path,
            }
    started_at = utc_now()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = _device(str(config.runtime.get("device", "auto")))
    datasets = load_reproduction_datasets(config.datasets, config.seed)
    model, processor, model_revision, _ = _resolve_base_model(config, device)
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
    train_outputs = _image_embeddings(
        model,
        processor,
        datasets["adaptation_train"],
        prototypes["adaptation"],
        device,
        batch_size,
    )
    features = torch.as_tensor(
        train_outputs["image_embeddings"], dtype=torch.float32, device=device
    )
    labels = torch.as_tensor(train_outputs["labels"], dtype=torch.long, device=device)
    classifier = torch.nn.Linear(
        features.shape[1], len(datasets["adaptation_eval"].class_names)
    ).to(device)
    if bool(config.method.get("initialize_head_from_text", False)):
        with torch.no_grad():
            classifier.weight.copy_(
                model.logit_scale.detach().exp()
                * prototypes["adaptation"].detach()
            )
            classifier.bias.zero_()
    strategy = str(config.method.get("strategy", "linear-probe"))
    if strategy not in {"linear-probe", "full-finetune", "lp-ft"}:
        raise ValueError(f"unsupported classification strategy: {strategy}")
    probe_steps = (
        config.total_steps
        if strategy == "linear-probe"
        else int(config.method.get("probe_steps", 0))
    )
    if probe_steps:
        optimizer = torch.optim.AdamW(
            classifier.parameters(),
            lr=float(config.method.get("probe_learning_rate", config.learning_rate)),
            weight_decay=float(config.training.get("weight_decay", 0.0)),
        )
        classifier.train()
        for _ in range(probe_steps):
            loss = torch.nn.functional.cross_entropy(classifier(features), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    encoder_trainable_parameters = 0
    if strategy != "linear-probe":
        for parameter in model.vision_model.parameters():
            parameter.requires_grad = True
        for parameter in model.visual_projection.parameters():
            parameter.requires_grad = True
        encoder_parameters = [
            parameter
            for module in (model.vision_model, model.visual_projection)
            for parameter in module.parameters()
        ]
        encoder_trainable_parameters = sum(
            parameter.numel() for parameter in encoder_parameters
        )
        optimizer = torch.optim.SGD(
            [
                {
                    "params": encoder_parameters,
                    "lr": float(config.method.get("encoder_learning_rate", 1e-5)),
                },
                {
                    "params": classifier.parameters(),
                    "lr": float(config.method.get("head_learning_rate", 1e-2)),
                },
            ],
            momentum=float(config.training.get("momentum", 0.9)),
            weight_decay=float(config.training.get("weight_decay", 0.0)),
        )
        generator = torch.Generator().manual_seed(config.seed + 1700)
        train = datasets["adaptation_train"]
        model.train()
        classifier.train()
        for _ in range(config.total_steps):
            indices = torch.randint(
                0, len(train.labels), (config.batch_size,), generator=generator
            ).tolist()
            pixels = processor(
                images=[train.images[index] for index in indices],
                return_tensors="pt",
            )["pixel_values"].to(device)
            batch_labels = torch.tensor(
                [train.labels[index] for index in indices], device=device
            )
            batch_features = torch.nn.functional.normalize(
                _feature_tensor(model.get_image_features(pixel_values=pixels)), dim=-1
            )
            loss = torch.nn.functional.cross_entropy(
                classifier(batch_features), batch_labels
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    classifier.eval()
    current = _evaluate_both(
        model, processor, datasets, prototypes, device, batch_size
    )
    with torch.no_grad():
        adaptation_features = torch.as_tensor(
            current["adaptation"]["image_embeddings"],
            dtype=torch.float32,
            device=device,
        )
        current["adaptation"]["logits"] = classifier(
            adaptation_features
        ).cpu().numpy()
    class_names = {
        "retained": datasets["retained_eval"].class_names,
        "adaptation": datasets["adaptation_eval"].class_names,
    }
    projection_dimension = int(config.analysis.get("covariance_projection_dim", 32))
    records = [
        _checkpoint_record(0, baseline, baseline, projection_dimension, class_names),
        _checkpoint_record(
            config.total_steps, baseline, current, projection_dimension, class_names
        ),
    ]
    head_trainable_parameters = sum(
        parameter.numel() for parameter in classifier.parameters()
    )
    trainable_parameters = head_trainable_parameters + encoder_trainable_parameters
    records[0]["optimization"] = {
        "adapter_l2_delta_from_initial": 0.0,
        "trainable_parameters": trainable_parameters,
    }
    records[1]["optimization"] = {
        "adapter_l2_delta_from_initial": float(
            torch.sqrt(
                sum(torch.sum(parameter.detach() ** 2) for parameter in classifier.parameters())
            ).cpu()
        ),
        "trainable_parameters": trainable_parameters,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(run_dir / "metrics/checkpoints.json", records)
    for role in ("retained", "adaptation"):
        _save_outputs(run_dir / f"outputs/step_000000_{role}.npz", baseline[role])
        _save_outputs(
            run_dir / f"outputs/step_{config.total_steps:06d}_{role}.npz",
            current[role],
        )
    checkpoint_path = run_dir / f"checkpoints/step_{config.total_steps:06d}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "config_hash": config.config_hash,
            "step": config.total_steps,
            "classifier_state": classifier.state_dict(),
        }
    if strategy != "linear-probe":
        checkpoint["vision_state"] = {
            key: value.detach().cpu().half()
            for key, value in model.vision_model.state_dict().items()
        }
        checkpoint["visual_projection_state"] = {
            key: value.detach().cpu().half()
            for key, value in model.visual_projection.state_dict().items()
        }
    torch.save(checkpoint, checkpoint_path)
    artifact_files = sorted(
        path
        for pattern in ("metrics/*", "outputs/*", "checkpoints/*")
        for path in run_dir.glob(pattern)
        if path.is_file()
    )
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": config.run_id,
        "status": "complete",
        "config_hash": config.config_hash,
        "config": config.scientific_payload(),
        "git": git_revision(),
        "environment": environment_snapshot(),
        "model": {
            "requested_revision": config.model["revision"],
            "resolved_revision": model_revision,
            "encoder_trainable_parameters": encoder_trainable_parameters,
            "head_trainable_parameters": head_trainable_parameters,
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
        "started_at": started_at,
        "completed_at": utc_now(),
        "completed_steps": [0, config.total_steps],
        "artifacts": [
            {
                "path": str(path.relative_to(run_dir)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in artifact_files
        ],
    }
    write_json_atomic(manifest_path, manifest)
    public_manifest, public_manifest_sha = publish_manifest(
        manifest, web_path.parent / "manifests" / f"{config.run_id}.json"
    )
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": config.run_id,
        "config_hash": config.config_hash,
        "evidence_status": "local-linear-probe-preliminary",
        "source_manifest": {
            "run_id": config.run_id,
            "public_path": f"/data/manifests/{public_manifest.name}",
            "sha256": public_manifest_sha,
        },
        "experiment": {
            "name": config.name,
            "model": manifest["model"],
            "method": config.method,
            "seed": config.seed,
        },
        "datasets": manifest["datasets"],
        "checkpoints": records,
    }
    write_json_atomic(web_path, payload)
    validate_web_artifact(payload)
    return {"run_dir": run_dir, "manifest": manifest_path, "web_artifact": web_path}


def run_linear_probe_suite(suite_path: str | Path) -> Path:
    suite_file = Path(suite_path)
    raw = yaml.safe_load(suite_file.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("linear-probe suite root must be a mapping")
    seeds = tuple(int(seed) for seed in raw["seeds"])
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("linear-probe suite requires three unique seeds")
    base = load_config((suite_file.parent / str(raw["base_config"])).resolve())
    output = Path(str(raw["web_output"]))
    runs = []
    sources = []
    for seed in seeds:
        config = replace(base, seed=seed)
        run_path = output.parent / "runs" / f"{config.run_id}.json"
        result = run_linear_probe(config, web_output=run_path)
        artifact = read_json(result["web_artifact"])
        runs.append(artifact)
        sources.append(
            {
                "run_id": config.run_id,
                "seed": seed,
                "config_hash": config.config_hash,
                "web_artifact": str(run_path),
                "web_artifact_sha256": sha256_file(run_path),
                "source_manifest": artifact["source_manifest"],
            }
        )
    identity = {
        "name": str(raw["name"]),
        "base_config": str(raw["base_config"]),
        "base_scientific_payload": base.scientific_payload(),
        "seeds": list(seeds),
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
    checkpoints = aggregate_run_artifacts(runs)
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
        "source_runs": sources,
    }
    write_json_atomic(manifest_path, manifest)
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
            "model": runs[0]["experiment"]["model"],
            "method": runs[0]["experiment"]["method"],
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
                for run in runs
            },
        },
        "runs": sources,
        "checkpoints": checkpoints,
    }
    write_json_atomic(output, payload)
    validate_web_artifact(payload)
    return output
