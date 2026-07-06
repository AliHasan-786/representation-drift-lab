from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .provenance import ARTIFACT_SCHEMA_VERSION


class ArtifactValidationError(ValueError):
    """Raised when an artifact does not meet the publication contract."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ArtifactValidationError("artifact root must be an object")
    return payload


def validate_web_artifact(payload: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "run_id",
        "config_hash",
        "source_manifest",
        "datasets",
        "checkpoints",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ArtifactValidationError(f"missing fields: {', '.join(missing)}")
    if payload["schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactValidationError("unsupported schema_version")
    if not isinstance(payload["checkpoints"], list) or not payload["checkpoints"]:
        raise ArtifactValidationError("checkpoints must be a non-empty list")
    steps = [checkpoint.get("step") for checkpoint in payload["checkpoints"]]
    if steps != sorted(set(steps)):
        raise ArtifactValidationError("checkpoint steps must be sorted and unique")
    for checkpoint in payload["checkpoints"]:
        for field in ("step", "retained", "adaptation", "geometry"):
            if field not in checkpoint:
                raise ArtifactValidationError(f"checkpoint missing {field}")


def publish_manifest(
    manifest: Mapping[str, Any], destination: str | Path
) -> tuple[Path, str]:
    """Publish the stable, non-secret provenance subset used by web artifacts."""
    fields = (
        "schema_version",
        "run_id",
        "status",
        "config_hash",
        "config",
        "seed",
        "git",
        "environment",
        "model",
        "datasets",
        "started_at",
        "completed_steps",
        "artifacts",
        "source_runs",
    )
    payload = {field: manifest[field] for field in fields if field in manifest}
    path = Path(destination)
    write_json_atomic(path, payload)
    return path, sha256_file(path)
