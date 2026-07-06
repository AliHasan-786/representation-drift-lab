from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from driftlab.artifacts import sha256_file
from driftlab.legacy import verify_legacy_manifest


class LegacyManifestTests(unittest.TestCase):
    def test_manifest_contains_only_safe_relative_paths(self) -> None:
        for manifest_path in (
            Path("data/legacy/manifest.json"),
            Path("data/legacy/code-archive-manifest.json"),
        ):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for artifact in manifest["artifacts"]:
                path = Path(artifact["path"])
                self.assertFalse(path.is_absolute(), manifest_path)
                self.assertNotIn("..", path.parts, manifest_path)
                self.assertEqual(len(artifact["sha256"]), 64, manifest_path)

    def test_verifier_checks_size_and_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.txt"
            artifact.write_text("verified\n", encoding="utf-8")
            manifest = {
                "schema_version": "1.0.0",
                "storage": {"root_environment_variable": "UNUSED"},
                "artifacts": [
                    {
                        "path": "artifact.txt",
                        "bytes": artifact.stat().st_size,
                        "sha256": sha256_file(artifact),
                    }
                ],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = verify_legacy_manifest(manifest_path, root=root)
            self.assertTrue(result["all_valid"])


if __name__ == "__main__":
    unittest.main()
