from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .artifacts import sha256_file


def verify_legacy_manifest(
    manifest_path: str | Path = "data/legacy/manifest.json",
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if root is None:
        variable = manifest["storage"]["root_environment_variable"]
        value = os.environ.get(variable)
        if not value:
            raise ValueError(f"set {variable} or pass root explicitly")
        root = value
    root_path = Path(root)
    results = []
    for artifact in manifest["artifacts"]:
        relative = Path(artifact["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe legacy path: {relative}")
        path = root_path / relative
        result = {"path": str(relative), "exists": path.is_file()}
        if path.is_file():
            result["bytes_match"] = path.stat().st_size == int(artifact["bytes"])
            result["sha256_match"] = sha256_file(path) == artifact["sha256"]
        else:
            result["bytes_match"] = False
            result["sha256_match"] = False
        result["valid"] = all(
            result[field] for field in ("exists", "bytes_match", "sha256_match")
        )
        results.append(result)
    return {
        "schema_version": manifest["schema_version"],
        "checked": len(results),
        "valid": sum(result["valid"] for result in results),
        "all_valid": all(result["valid"] for result in results),
        "results": results,
    }
