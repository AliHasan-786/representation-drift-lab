from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ARTIFACT_SCHEMA_VERSION = "1.0.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git_revision(workdir: str | Path = ".") -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=workdir, check=False, capture_output=True, text=True
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""

    commit = run("rev-parse", "HEAD") or "uncommitted"
    dirty = bool(run("status", "--porcelain"))
    return {"commit": commit, "dirty": dirty}


def _package_versions(names: tuple[str, ...]) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def environment_snapshot() -> dict[str, Any]:
    accelerator = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        accelerator = "cuda"
    return {
        "python": sys.version.split()[0],
        "platform": platform.system(),
        "platform_release": platform.release(),
        "architecture": platform.machine(),
        "accelerator": accelerator,
        "packages": _package_versions(
            (
                "numpy",
                "scipy",
                "torch",
                "PyYAML",
                "transformers",
                "datasets",
                "peft",
                "accelerate",
                "Pillow",
                "huggingface-hub",
            )
        ),
    }
