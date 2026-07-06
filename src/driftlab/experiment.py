from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from .artifacts import publish_manifest, sha256_file, write_json_atomic
from .config import ExperimentConfig
from .datasets import ArrayDataset, make_smoke_splits
from .metrics import (
    class_centroid_movement,
    cosine_centroid_drift,
    effective_rank,
    linear_cka,
    local_neighborhood_overlap,
    stable_frechet_distance,
    task_metrics,
)
from .provenance import ARTIFACT_SCHEMA_VERSION, environment_snapshot, git_revision, utc_now


def _prototypes() -> torch.Tensor:
    angles = torch.deg2rad(torch.tensor([15.0, 135.0, 255.0], dtype=torch.float32))
    return torch.column_stack((torch.cos(angles), torch.sin(angles)))


def _normalize(array: torch.Tensor) -> torch.Tensor:
    return functional.normalize(array, dim=-1)


def _evaluate(model: torch.nn.Module, dataset: ArrayDataset) -> dict[str, np.ndarray]:
    model.eval()
    features = torch.from_numpy(dataset.features)
    with torch.no_grad():
        embeddings = _normalize(model(features))
        logits = embeddings @ _prototypes().T / 0.1
    return {
        "embeddings": embeddings.numpy(),
        "logits": logits.numpy(),
        "labels": dataset.labels,
    }


def _latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted(
        run_dir.glob("checkpoints/step_*.pt"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    return checkpoints[-1] if checkpoints else None


def _checkpoint_step(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _save_outputs(path: Path, outputs: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **outputs)


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("metrics checkpoint file must contain a list")
    return payload


def _geometry(
    baseline: dict[str, np.ndarray], current: dict[str, np.ndarray]
) -> dict[str, Any]:
    base_embeddings = baseline["embeddings"]
    current_embeddings = current["embeddings"]
    labels = current["labels"]
    return {
        "cosine_centroid_drift": cosine_centroid_drift(base_embeddings, current_embeddings),
        "frechet_distance": stable_frechet_distance(base_embeddings, current_embeddings),
        "linear_cka": linear_cka(base_embeddings, current_embeddings),
        "effective_rank": effective_rank(current_embeddings),
        "neighborhood_overlap_at_5": local_neighborhood_overlap(
            base_embeddings, current_embeddings, neighbors=5
        ),
        "class_centroid_movement": class_centroid_movement(
            base_embeddings, current_embeddings, labels
        ),
    }


def _sample_points(
    baseline: dict[str, np.ndarray], current: dict[str, np.ndarray], limit: int = 36
) -> list[dict[str, Any]]:
    count = min(limit, len(current["labels"]))
    points = []
    for index in range(count):
        points.append(
            {
                "id": index,
                "label": int(current["labels"][index]),
                "baseline": [float(value) for value in baseline["embeddings"][index, :2]],
                "current": [float(value) for value in current["embeddings"][index, :2]],
            }
        )
    return points


def _record_checkpoint(
    *,
    step: int,
    datasets: dict[str, ArrayDataset],
    baseline: dict[str, dict[str, np.ndarray]],
    current: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    retained_task = task_metrics(
        current["retained"]["logits"], current["retained"]["labels"]
    ).to_dict()
    adaptation_task = task_metrics(
        current["adaptation"]["logits"], current["adaptation"]["labels"]
    ).to_dict()
    base_retained = task_metrics(
        baseline["retained"]["logits"], baseline["retained"]["labels"]
    ).top1_accuracy
    base_adaptation = task_metrics(
        baseline["adaptation"]["logits"], baseline["adaptation"]["labels"]
    ).top1_accuracy
    retained_task["accuracy_change"] = retained_task["top1_accuracy"] - base_retained
    adaptation_task["accuracy_change"] = adaptation_task["top1_accuracy"] - base_adaptation
    return {
        "step": step,
        "retained": retained_task,
        "adaptation": adaptation_task,
        "geometry": {
            "retained": _geometry(baseline["retained"], current["retained"]),
            "adaptation": _geometry(baseline["adaptation"], current["adaptation"]),
        },
        "samples": {
            "retained": _sample_points(baseline["retained"], current["retained"]),
            "adaptation": _sample_points(baseline["adaptation"], current["adaptation"]),
        },
        "dataset_fingerprints": {
            "retained": datasets["retained_eval"].fingerprint,
            "adaptation": datasets["adaptation_eval"].fingerprint,
        },
    }


def _write_metrics_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "step",
        "retained_accuracy",
        "retained_accuracy_change",
        "adaptation_accuracy",
        "adaptation_accuracy_change",
        "retained_cosine_drift",
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
                    "retained_accuracy_change": record["retained"]["accuracy_change"],
                    "adaptation_accuracy": record["adaptation"]["top1_accuracy"],
                    "adaptation_accuracy_change": record["adaptation"]["accuracy_change"],
                    "retained_cosine_drift": record["geometry"]["retained"]["cosine_centroid_drift"],
                    "retained_frechet_distance": record["geometry"]["retained"]["frechet_distance"],
                    "retained_linear_cka": record["geometry"]["retained"]["linear_cka"],
                }
            )


def _write_manifest(
    *,
    run_dir: Path,
    config: ExperimentConfig,
    datasets: dict[str, ArrayDataset],
    started_at: str,
    status: str,
    completed_steps: list[int],
) -> Path:
    artifacts = []
    for relative in ("metrics/checkpoints.json", "metrics/summary.csv"):
        path = run_dir / relative
        if path.exists():
            artifacts.append(
                {"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size}
            )
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": config.run_id,
        "status": status,
        "config_hash": config.config_hash,
        "config": config.scientific_payload(),
        "seed": config.seed,
        "git": git_revision(),
        "environment": environment_snapshot(),
        "datasets": {
            key: {
                "name": dataset.name,
                "split": dataset.split,
                "fingerprint": dataset.fingerprint,
                "samples": len(dataset.labels),
            }
            for key, dataset in datasets.items()
        },
        "started_at": started_at,
        "updated_at": utc_now(),
        "completed_steps": completed_steps,
        "artifacts": artifacts,
    }
    path = run_dir / "manifest.json"
    write_json_atomic(path, manifest)
    return path


def _export_web_artifact(
    *,
    path: Path,
    config: ExperimentConfig,
    public_manifest_path: Path,
    public_manifest_sha256: str,
    datasets: dict[str, ArrayDataset],
    records: list[dict[str, Any]],
) -> None:
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": config.run_id,
        "config_hash": config.config_hash,
        "source_manifest": {
            "run_id": config.run_id,
            "public_path": f"/data/manifests/{public_manifest_path.name}",
            "sha256": public_manifest_sha256,
        },
        "experiment": {
            "name": config.name,
            "tier": config.tier,
            "model": config.model,
            "method": config.method,
            "seed": config.seed,
        },
        "datasets": {
            "retained": {
                "name": datasets["retained_eval"].name,
                "fingerprint": datasets["retained_eval"].fingerprint,
                "sample_count": len(datasets["retained_eval"].labels),
            },
            "adaptation": {
                "name": datasets["adaptation_eval"].name,
                "fingerprint": datasets["adaptation_eval"].fingerprint,
                "sample_count": len(datasets["adaptation_eval"].labels),
            },
        },
        "checkpoints": records,
    }
    write_json_atomic(path, payload)


def run_smoke_experiment(
    config: ExperimentConfig,
    *,
    resume: bool = False,
    web_output: str | Path = "public/data/smoke.json",
    stop_after: int | None = None,
) -> dict[str, Path]:
    if config.tier != "smoke":
        raise ValueError("run_smoke_experiment requires a smoke-tier config")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    datasets = make_smoke_splits(
        config.seed,
        adaptation_rotation_degrees=float(
            config.datasets.get("adaptation_rotation_degrees", 85.0)
        ),
    )
    run_dir = config.output_dir / config.run_id
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists() and not resume:
        raise FileExistsError(
            f"run already exists at {run_dir}; pass resume=True to continue safely"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        started_at = str(existing_manifest.get("started_at", started_at))

    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed + 101)
    records_path = run_dir / "metrics/checkpoints.json"
    records = _load_records(records_path)
    start_step = 0
    latest = _latest_checkpoint(run_dir)
    if latest is not None:
        if not resume:
            raise FileExistsError(f"checkpoint exists at {latest}")
        state = torch.load(latest, map_location="cpu", weights_only=True)
        if state["config_hash"] != config.config_hash:
            raise ValueError("checkpoint config hash does not match requested config")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        generator.set_state(state["generator_state"])
        start_step = int(state["step"])

    baseline_dir = run_dir / "outputs"
    baseline: dict[str, dict[str, np.ndarray]] = {}
    baseline_files = {
        "retained": baseline_dir / "step_000000_retained.npz",
        "adaptation": baseline_dir / "step_000000_adaptation.npz",
    }
    if all(path.exists() for path in baseline_files.values()):
        for key, path in baseline_files.items():
            with np.load(path) as loaded:
                baseline[key] = {name: loaded[name] for name in loaded.files}
    else:
        baseline = {
            "retained": _evaluate(model, datasets["retained_eval"]),
            "adaptation": _evaluate(model, datasets["adaptation_eval"]),
        }
        _save_outputs(baseline_files["retained"], baseline["retained"])
        _save_outputs(baseline_files["adaptation"], baseline["adaptation"])

    completed_steps = {int(record["step"]) for record in records}

    def capture(step: int) -> None:
        nonlocal records
        if step in completed_steps:
            return
        current = {
            "retained": _evaluate(model, datasets["retained_eval"]),
            "adaptation": _evaluate(model, datasets["adaptation_eval"]),
        }
        if step != 0:
            _save_outputs(
                baseline_dir / f"step_{step:06d}_retained.npz", current["retained"]
            )
            _save_outputs(
                baseline_dir / f"step_{step:06d}_adaptation.npz", current["adaptation"]
            )
        records.append(
            _record_checkpoint(
                step=step,
                datasets=datasets,
                baseline=baseline,
                current=current,
            )
        )
        records.sort(key=lambda item: int(item["step"]))
        completed_steps.add(step)

    def persist_records() -> None:
        destination = records_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(records, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

    if 0 not in completed_steps:
        current_zero = {"retained": baseline["retained"], "adaptation": baseline["adaptation"]}
        records.append(
            _record_checkpoint(
                step=0,
                datasets=datasets,
                baseline=baseline,
                current=current_zero,
            )
        )
        completed_steps.add(0)
        persist_records()

    train_features = torch.from_numpy(datasets["adaptation_train"].features)
    train_labels = torch.from_numpy(datasets["adaptation_train"].labels)
    checkpoint_steps = set(config.checkpoint_steps)
    final_step = config.total_steps if stop_after is None else min(stop_after, config.total_steps)
    for step in range(start_step + 1, final_step + 1):
        model.train()
        indices = torch.randint(
            0,
            len(train_labels),
            (config.batch_size,),
            generator=generator,
        )
        embeddings = _normalize(model(train_features[indices]))
        logits = embeddings @ _prototypes().T / 0.1
        loss = functional.cross_entropy(logits, train_labels[indices])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        should_checkpoint = step in checkpoint_steps or step == final_step
        if should_checkpoint:
            capture(step)
            persist_records()
            checkpoint_path = run_dir / f"checkpoints/step_{step:06d}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "config_hash": config.config_hash,
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "generator_state": generator.get_state(),
                },
                checkpoint_path,
            )

    persist_records()
    csv_path = run_dir / "metrics/summary.csv"
    _write_metrics_csv(csv_path, records)
    status = "complete" if final_step == config.total_steps else "interrupted"
    manifest_path = _write_manifest(
        run_dir=run_dir,
        config=config,
        datasets=datasets,
        started_at=started_at,
        status=status,
        completed_steps=sorted(completed_steps),
    )
    web_path = Path(web_output)
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    public_manifest_path, public_manifest_sha256 = publish_manifest(
        manifest_payload,
        web_path.parent / "manifests" / f"{config.run_id}.json",
    )
    _export_web_artifact(
        path=web_path,
        config=config,
        public_manifest_path=public_manifest_path,
        public_manifest_sha256=public_manifest_sha256,
        datasets=datasets,
        records=records,
    )
    return {"run_dir": run_dir, "manifest": manifest_path, "web_artifact": web_path}
