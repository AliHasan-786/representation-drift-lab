from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import publish_manifest, sha256_file, write_json_atomic
from .clip_data import ImageDataset, load_reproduction_datasets
from .config import ExperimentConfig
from .losses import multi_positive_clip_loss, orthogonalize_gradient
from .metrics import (
    baseline_fixed_projection,
    class_centroid_movement,
    classwise_diagnostics,
    cosine_centroid_drift,
    cross_modal_diagnostics,
    effective_rank,
    layerwise_representation_diagnostics,
    linear_cka,
    local_neighborhood_overlap,
    stable_frechet_distance,
    task_metrics,
)
from .provenance import ARTIFACT_SCHEMA_VERSION, environment_snapshot, git_revision, utc_now


def _research_imports() -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi
        from peft import (
            LoraConfig,
            get_peft_model,
            get_peft_model_state_dict,
            set_peft_model_state_dict,
        )
        from transformers import AutoProcessor, CLIPModel
    except ImportError as error:
        raise RuntimeError(
            "research dependencies are required; install requirements.research.lock"
        ) from error
    return {
        "HfApi": HfApi,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "get_peft_model_state_dict": get_peft_model_state_dict,
        "set_peft_model_state_dict": set_peft_model_state_dict,
        "AutoProcessor": AutoProcessor,
        "CLIPModel": CLIPModel,
    }


def _device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _prompt(class_name: str) -> str:
    return f"a photo of {class_name.replace('_', ' ')}"


def _feature_tensor(output: Any) -> torch.Tensor:
    """Normalize Transformers 4 tensor and Transformers 5 structured outputs."""
    if isinstance(output, torch.Tensor):
        return output
    pooled = getattr(output, "pooler_output", None)
    if isinstance(pooled, torch.Tensor):
        return pooled
    raise TypeError(f"unsupported feature output type: {type(output).__name__}")


def _resolve_base_model(config: ExperimentConfig, device: torch.device):
    imports = _research_imports()
    requested_revision = str(config.model["revision"])
    resolved_revision = str(imports["HfApi"]().model_info(requested_revision).sha)
    model = imports["CLIPModel"].from_pretrained(
        requested_revision, revision=resolved_revision
    )
    processor = imports["AutoProcessor"].from_pretrained(
        requested_revision, revision=resolved_revision
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.to(device)
    return model, processor, resolved_revision, imports


def _resolve_and_load_model(config: ExperimentConfig, device: torch.device):
    model, processor, resolved_revision, imports = _resolve_base_model(config, device)
    lora = imports["LoraConfig"](
        r=int(config.method["rank"]),
        lora_alpha=int(config.method["alpha"]),
        lora_dropout=float(config.method["dropout"]),
        target_modules=list(config.method["target_modules"]),
        bias="none",
    )
    model.vision_model = imports["get_peft_model"](model.vision_model, lora)
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    expected = config.method.get("expected_trainable_parameters")
    if expected is not None and trainable != int(expected):
        raise RuntimeError(
            f"trainable parameter invariant failed: expected {expected}, got {trainable}"
        )
    model.to(device)
    return model, processor, resolved_revision, imports


def _text_prototypes(model, processor, names: tuple[str, ...], device: torch.device):
    inputs = processor(
        text=[_prompt(name) for name in names], padding=True, return_tensors="pt"
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        features = _feature_tensor(model.get_text_features(**inputs))
        features = torch.nn.functional.normalize(features, dim=-1)
    return features


def _image_embeddings(
    model,
    processor,
    dataset: ImageDataset,
    prototypes: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    model.eval()
    chunks: list[torch.Tensor] = []
    layer_chunks: dict[str, list[torch.Tensor]] = {}
    for start in range(0, len(dataset.images), batch_size):
        images = list(dataset.images[start : start + batch_size])
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            vision_outputs = model.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
            features = model.visual_projection(_feature_tensor(vision_outputs))
            chunks.append(torch.nn.functional.normalize(features, dim=-1).cpu())
            hidden_states = getattr(vision_outputs, "hidden_states", None)
            if hidden_states is None:
                raise RuntimeError("vision model did not return requested hidden states")
            for index, hidden_state in enumerate(hidden_states):
                name = (
                    "vision_embeddings"
                    if index == 0
                    else f"vision_block_{index - 1:02d}"
                )
                layer_chunks.setdefault(name, []).append(
                    hidden_state[:, 0, :].detach().cpu()
                )
    embeddings = torch.cat(chunks, dim=0)
    scale = float(model.logit_scale.detach().exp().cpu())
    logits = scale * embeddings @ prototypes.detach().cpu().T
    labels = np.asarray(dataset.labels, dtype=np.int64)
    result = {
        "image_embeddings": embeddings.numpy(),
        "text_embeddings": prototypes.detach().cpu().numpy()[labels],
        "logits": logits.numpy(),
        "labels": labels,
    }
    result.update(
        {
            f"layer__{name}": torch.cat(values, dim=0).numpy()
            for name, values in layer_chunks.items()
        }
    )
    return result


def _baseline_projection(
    baseline: np.ndarray, current: np.ndarray, maximum_dimension: int
) -> tuple[np.ndarray, np.ndarray]:
    return baseline_fixed_projection(baseline, current, maximum_dimension)


def _geometry(
    baseline: dict[str, np.ndarray],
    current: dict[str, np.ndarray],
    projection_dimension: int,
) -> dict[str, Any]:
    base = baseline["image_embeddings"]
    now = current["image_embeddings"]
    projected_base, projected_now = _baseline_projection(
        base, now, projection_dimension
    )
    return {
        "cosine_centroid_drift": cosine_centroid_drift(base, now),
        "frechet_distance": stable_frechet_distance(projected_base, projected_now),
        "frechet_projection_dimension": int(projected_base.shape[1]),
        "linear_cka": linear_cka(base, now),
        "effective_rank": effective_rank(projected_now),
        "neighborhood_overlap_at_5": local_neighborhood_overlap(
            base, now, neighbors=min(5, len(base) - 1)
        ),
        "class_centroid_movement": class_centroid_movement(
            base, now, current["labels"]
        ),
    }


def _projection_points(
    baseline: dict[str, np.ndarray], current: dict[str, np.ndarray], limit: int = 40
) -> list[dict[str, Any]]:
    base, now = _baseline_projection(
        baseline["image_embeddings"], current["image_embeddings"], 2
    )
    return [
        {
            "id": index,
            "label": int(current["labels"][index]),
            "baseline": [float(value) for value in base[index]],
            "current": [float(value) for value in now[index]],
        }
        for index in range(min(limit, len(base)))
    ]


def _evaluate_both(
    model,
    processor,
    datasets: dict[str, ImageDataset],
    prototypes: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    return {
        "retained": _image_embeddings(
            model,
            processor,
            datasets["retained_eval"],
            prototypes["retained"],
            device,
            batch_size,
        ),
        "adaptation": _image_embeddings(
            model,
            processor,
            datasets["adaptation_eval"],
            prototypes["adaptation"],
            device,
            batch_size,
        ),
    }


def _checkpoint_record(
    step: int,
    baseline: dict[str, dict[str, np.ndarray]],
    current: dict[str, dict[str, np.ndarray]],
    projection_dimension: int,
    class_names: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "step": step,
        "geometry": {},
        "layerwise": {},
        "cross_modal": {},
        "classwise": {},
        "samples": {},
    }
    for role in ("retained", "adaptation"):
        metrics = task_metrics(current[role]["logits"], current[role]["labels"]).to_dict()
        baseline_accuracy = task_metrics(
            baseline[role]["logits"], baseline[role]["labels"]
        ).top1_accuracy
        metrics["accuracy_change"] = metrics["top1_accuracy"] - baseline_accuracy
        record[role] = metrics
        record["geometry"][role] = _geometry(
            baseline[role], current[role], projection_dimension
        )
        baseline_layers = {
            key.removeprefix("layer__"): value
            for key, value in baseline[role].items()
            if key.startswith("layer__")
        }
        current_layers = {
            key.removeprefix("layer__"): value
            for key, value in current[role].items()
            if key.startswith("layer__")
        }
        record["layerwise"][role] = layerwise_representation_diagnostics(
            baseline_layers,
            current_layers,
            projection_dimension=projection_dimension,
        )
        record["cross_modal"][role] = cross_modal_diagnostics(
            baseline[role]["image_embeddings"],
            current[role]["image_embeddings"],
            baseline[role]["text_embeddings"],
            current[role]["text_embeddings"],
        )
        record["classwise"][role] = classwise_diagnostics(
            current[role]["logits"],
            current[role]["labels"],
            class_names[role],
        )
        record["samples"][role] = _projection_points(baseline[role], current[role])
    return record


def _save_outputs(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)


def _load_outputs(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as values:
        return {key: values[key] for key in values.files}


def _write_summary_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "step",
        "retained_accuracy",
        "adaptation_accuracy",
        "retained_accuracy_change",
        "adaptation_accuracy_change",
        "retained_frechet_distance",
        "retained_linear_cka",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "step": record["step"],
                    "retained_accuracy": record["retained"]["top1_accuracy"],
                    "adaptation_accuracy": record["adaptation"]["top1_accuracy"],
                    "retained_accuracy_change": record["retained"]["accuracy_change"],
                    "adaptation_accuracy_change": record["adaptation"]["accuracy_change"],
                    "retained_frechet_distance": record["geometry"]["retained"]["frechet_distance"],
                    "retained_linear_cka": record["geometry"]["retained"]["linear_cka"],
                }
            )


def _artifact_files(run_dir: Path) -> list[Path]:
    return sorted(
        path
        for pattern in ("metrics/*", "outputs/*", "checkpoints/*")
        for path in run_dir.glob(pattern)
        if path.is_file()
    )


def _web_payload(
    config: ExperimentConfig,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    public_manifest_path: Path,
    public_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": config.run_id,
        "config_hash": config.config_hash,
        "evidence_status": "local-reproduction-preliminary",
        "source_manifest": {
            "run_id": config.run_id,
            "public_path": f"/data/manifests/{public_manifest_path.name}",
            "sha256": public_manifest_sha256,
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


def regenerate_clip_metrics(
    config: ExperimentConfig,
    *,
    web_output: str | Path = "public/data/reproduction-local.json",
) -> dict[str, Path]:
    """Recompute all diagnostics deterministically from saved model outputs."""
    run_dir = config.output_dir / config.run_id
    manifest_path = run_dir / "manifest.json"
    records_path = run_dir / "metrics/checkpoints.json"
    if not manifest_path.exists() or not records_path.exists():
        raise FileNotFoundError(f"completed run not found at {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("config_hash") != config.config_hash:
        raise ValueError("run manifest configuration mismatch")
    previous_records = json.loads(records_path.read_text(encoding="utf-8"))
    previous_by_step = {int(record["step"]): record for record in previous_records}
    steps = sorted(previous_by_step)
    baseline = {
        role: _load_outputs(run_dir / f"outputs/step_000000_{role}.npz")
        for role in ("retained", "adaptation")
    }
    class_names = {
        role: tuple(previous_records[0]["classwise"][role]["class_names"])
        for role in ("retained", "adaptation")
    }
    projection_dimension = int(config.analysis.get("covariance_projection_dim", 32))
    records = []
    for step in steps:
        current = (
            baseline
            if step == 0
            else {
                role: _load_outputs(
                    run_dir / f"outputs/step_{step:06d}_{role}.npz"
                )
                for role in ("retained", "adaptation")
            }
        )
        record = _checkpoint_record(
            step, baseline, current, projection_dimension, class_names
        )
        record["optimization"] = previous_by_step[step]["optimization"]
        records.append(record)
    write_json_atomic(records_path, records)
    _write_summary_csv(run_dir / "metrics/summary.csv", records)
    # Regeneration must be byte-stable; do not add a wall-clock field.
    manifest.pop("metrics_regenerated_at", None)
    manifest["environment"] = environment_snapshot()
    manifest["artifacts"] = [
        {
            "path": str(path.relative_to(run_dir)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in _artifact_files(run_dir)
    ]
    write_json_atomic(manifest_path, manifest)
    web_path = Path(web_output)
    public_manifest_path, public_manifest_sha256 = publish_manifest(
        manifest,
        web_path.parent / "manifests" / f"{config.run_id}.json",
    )
    write_json_atomic(
        web_path,
        _web_payload(
            config,
            manifest,
            records,
            public_manifest_path,
            public_manifest_sha256,
        ),
    )
    return {"run_dir": run_dir, "manifest": manifest_path, "web_artifact": web_path}


def run_clip_reproduction(
    config: ExperimentConfig,
    *,
    resume: bool = False,
    web_output: str | Path = "public/data/reproduction-local.json",
) -> dict[str, Path]:
    invocation_started_at = utc_now()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = _device(str(config.runtime.get("device", "auto")))
    datasets = load_reproduction_datasets(config.datasets, config.seed)
    model, processor, model_revision, imports = _resolve_and_load_model(config, device)
    initial_adapter_state = {
        key: value.detach().cpu().clone()
        for key, value in imports["get_peft_model_state_dict"](
            model.vision_model
        ).items()
    }
    prototypes = {
        "retained": _text_prototypes(
            model, processor, datasets["retained_eval"].class_names, device
        ),
        "adaptation": _text_prototypes(
            model, processor, datasets["adaptation_eval"].class_names, device
        ),
    }
    run_dir = config.output_dir / config.run_id
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists() and not resume:
        raise FileExistsError(f"run exists at {run_dir}; pass --resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = invocation_started_at
    if manifest_path.exists():
        import json

        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        started_at = str(existing_manifest.get("started_at", started_at))
    records_path = run_dir / "metrics/checkpoints.json"
    records = []
    if records_path.exists():
        import json

        records = json.loads(records_path.read_text(encoding="utf-8"))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=0.01,
    )
    generator = torch.Generator().manual_seed(config.seed + 700)
    start_step = 0
    checkpoint_paths = sorted(
        run_dir.glob("checkpoints/step_*.pt"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if checkpoint_paths:
        if not resume:
            raise FileExistsError(checkpoint_paths[-1])
        state = torch.load(checkpoint_paths[-1], map_location="cpu", weights_only=True)
        if state["config_hash"] != config.config_hash:
            raise ValueError("checkpoint configuration mismatch")
        imports["set_peft_model_state_dict"](model.vision_model, state["adapter_state"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        generator.set_state(state["generator_state"])
        start_step = int(state["step"])
    baseline_paths = {
        role: run_dir / f"outputs/step_000000_{role}.npz"
        for role in ("retained", "adaptation")
    }
    if all(path.exists() for path in baseline_paths.values()):
        baseline = {role: _load_outputs(path) for role, path in baseline_paths.items()}
    else:
        baseline = _evaluate_both(
            model,
            processor,
            datasets,
            prototypes,
            device,
            int(config.training.get("evaluation_batch_size", config.batch_size)),
        )
        for role, path in baseline_paths.items():
            _save_outputs(path, baseline[role])
    completed = {int(record["step"]) for record in records}
    projection_dimension = int(config.analysis.get("covariance_projection_dim", 32))
    class_names = {
        "retained": datasets["retained_eval"].class_names,
        "adaptation": datasets["adaptation_eval"].class_names,
    }

    def adapter_delta() -> float:
        current_state = imports["get_peft_model_state_dict"](model.vision_model)
        squared = 0.0
        for key, initial in initial_adapter_state.items():
            difference = current_state[key].detach().cpu().float() - initial.float()
            squared += float(torch.sum(difference * difference))
        return squared**0.5

    def capture(step: int) -> None:
        if step in completed:
            return
        current = _evaluate_both(
            model,
            processor,
            datasets,
            prototypes,
            device,
            int(config.training.get("evaluation_batch_size", config.batch_size)),
        )
        for role in ("retained", "adaptation"):
            if step:
                _save_outputs(
                    run_dir / f"outputs/step_{step:06d}_{role}.npz", current[role]
                )
        record = _checkpoint_record(
            step, baseline, current, projection_dimension, class_names
        )
        record["optimization"] = {
            "adapter_l2_delta_from_initial": adapter_delta(),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        }
        records.append(record)
        records.sort(key=lambda record: int(record["step"]))
        completed.add(step)
        write_json_atomic(records_path, records)

    if 0 not in completed:
        record_zero = _checkpoint_record(
            0, baseline, baseline, projection_dimension, class_names
        )
        record_zero["optimization"] = {
            "adapter_l2_delta_from_initial": 0.0,
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        }
        records.append(record_zero)
        completed.add(0)
        write_json_atomic(records_path, records)
    train = datasets["adaptation_train"]
    strategy = str(config.method.get("strategy", "standard"))
    supported_strategies = {
        "standard",
        "zscl-inspired-distillation",
        "retention-gradient-nullspace",
    }
    if strategy not in supported_strategies:
        raise ValueError(f"unsupported adaptation strategy: {strategy}")
    retained_train = datasets.get("retained_reference")
    retained_baseline = None
    if strategy != "standard":
        if retained_train is None:
            raise ValueError(
                f"{strategy} requires a disjoint datasets.retained reference split"
            )
        retained_baseline_path = run_dir / "outputs/step_000000_retained_reference.npz"
        if retained_baseline_path.exists():
            retained_baseline = _load_outputs(retained_baseline_path)
        else:
            retained_baseline = _image_embeddings(
                model,
                processor,
                retained_train,
                prototypes["retained"],
                device,
                int(config.training.get("evaluation_batch_size", config.batch_size)),
            )
            _save_outputs(retained_baseline_path, retained_baseline)
    checkpoint_steps = set(config.checkpoint_steps)
    for step in range(start_step + 1, config.total_steps + 1):
        model.train()
        indices = torch.randint(
            0, len(train.labels), (config.batch_size,), generator=generator
        ).tolist()
        images = [train.images[index] for index in indices]
        labels = torch.tensor([train.labels[index] for index in indices], device=device)
        captions = [_prompt(train.class_names[label]) for label in labels.tolist()]
        image_inputs = processor(images=images, return_tensors="pt")
        text_inputs = processor(text=captions, padding=True, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to(device)
        text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
        image_features = _feature_tensor(
            model.get_image_features(pixel_values=pixel_values)
        )
        with torch.no_grad():
            text_features = _feature_tensor(model.get_text_features(**text_inputs))
        adaptation_loss = multi_positive_clip_loss(image_features, text_features, labels)
        optimizer.zero_grad()
        if strategy == "standard":
            adaptation_loss.backward()
        else:
            retained_indices = torch.randint(
                0,
                len(retained_train.labels),
                (config.batch_size,),
                generator=generator,
            ).tolist()
            retained_images = [retained_train.images[index] for index in retained_indices]
            retained_labels = torch.tensor(
                [retained_train.labels[index] for index in retained_indices],
                device=device,
            )
            retained_pixels = processor(
                images=retained_images, return_tensors="pt"
            )["pixel_values"].to(device)
            retained_features = torch.nn.functional.normalize(
                _feature_tensor(model.get_image_features(pixel_values=retained_pixels)),
                dim=-1,
            )
            retained_logits = (
                model.logit_scale.detach().exp()
                * retained_features
                @ prototypes["retained"].T
            )
            if strategy == "zscl-inspired-distillation":
                temperature = float(config.method.get("distillation_temperature", 2.0))
                weight = float(config.method.get("retention_weight", 1.0))
                baseline_logits = torch.as_tensor(
                    retained_baseline["logits"][retained_indices],
                    device=device,
                    dtype=retained_logits.dtype,
                )
                distillation_loss = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(
                        retained_logits / temperature, dim=-1
                    ),
                    torch.nn.functional.softmax(
                        baseline_logits / temperature, dim=-1
                    ),
                    reduction="batchmean",
                ) * (temperature**2)
                (adaptation_loss + weight * distillation_loss).backward()
            else:
                adaptation_loss.backward()
                primary_gradients = {
                    name: parameter.grad.detach().clone()
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad and parameter.grad is not None
                }
                optimizer.zero_grad()
                retention_loss = torch.nn.functional.cross_entropy(
                    retained_logits, retained_labels
                )
                retention_loss.backward()
                for name, parameter in model.named_parameters():
                    if not parameter.requires_grad or name not in primary_gradients:
                        continue
                    constraint = parameter.grad
                    parameter.grad = (
                        primary_gradients[name]
                        if constraint is None
                        else orthogonalize_gradient(
                            primary_gradients[name], constraint
                        )
                    )
        optimizer.step()
        if step in checkpoint_steps:
            capture(step)
            checkpoint = run_dir / f"checkpoints/step_{step:06d}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "config_hash": config.config_hash,
                    "step": step,
                    "adapter_state": imports["get_peft_model_state_dict"](
                        model.vision_model
                    ),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "generator_state": generator.get_state(),
                },
                checkpoint,
            )
    _write_summary_csv(run_dir / "metrics/summary.csv", records)
    artifact_files = _artifact_files(run_dir)
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
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
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
        "completed_steps": sorted(completed),
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
    web_path = Path(web_output)
    public_manifest_path, public_manifest_sha256 = publish_manifest(
        manifest,
        web_path.parent / "manifests" / f"{config.run_id}.json",
    )
    web_payload = _web_payload(
        config,
        manifest,
        records,
        public_manifest_path,
        public_manifest_sha256,
    )
    write_json_atomic(web_path, web_payload)
    return {"run_dir": run_dir, "manifest": manifest_path, "web_artifact": web_path}
