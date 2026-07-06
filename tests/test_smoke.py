from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from driftlab.artifacts import read_json, validate_web_artifact
from driftlab.config import load_config
from driftlab.experiment import run_smoke_experiment


class SmokeExperimentTests(unittest.TestCase):
    def _config(self, directory: str):
        content = Path("configs/smoke.yaml").read_text(encoding="utf-8")
        config_path = Path(directory) / "smoke.yaml"
        config_path.write_text(
            content.replace("artifacts/runs", str(Path(directory) / "runs")),
            encoding="utf-8",
        )
        return load_config(config_path)

    def test_end_to_end_and_safe_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory)
            web_path = Path(directory) / "web.json"
            partial = run_smoke_experiment(config, web_output=web_path, stop_after=5)
            partial_manifest = json.loads(
                partial["manifest"].read_text(encoding="utf-8")
            )
            self.assertEqual(partial_manifest["status"], "interrupted")
            completed = run_smoke_experiment(
                config, resume=True, web_output=web_path
            )
            manifest = json.loads(completed["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["completed_steps"], [0, 5, 10, 20])
            artifact = read_json(web_path)
            validate_web_artifact(artifact)
            public_manifest = (
                web_path.parent
                / "manifests"
                / Path(artifact["source_manifest"]["public_path"]).name
            )
            self.assertTrue(public_manifest.is_file())
            self.assertEqual(
                hashlib.sha256(public_manifest.read_bytes()).hexdigest(),
                artifact["source_manifest"]["sha256"],
            )
            self.assertEqual(
                [checkpoint["step"] for checkpoint in artifact["checkpoints"]],
                [0, 5, 10, 20],
            )

    def test_existing_run_requires_explicit_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory)
            web_path = Path(directory) / "web.json"
            run_smoke_experiment(config, web_output=web_path)
            with self.assertRaises(FileExistsError):
                run_smoke_experiment(config, web_output=web_path)


if __name__ == "__main__":
    unittest.main()
