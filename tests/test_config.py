from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from driftlab.config import ConfigError, load_config


class ConfigTests(unittest.TestCase):
    def test_hash_is_stable_and_ignores_output_location(self) -> None:
        first = load_config("configs/smoke.yaml")
        content = Path("configs/smoke.yaml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "other.yaml"
            path.write_text(content.replace("artifacts/runs", directory), encoding="utf-8")
            second = load_config(path)
        self.assertEqual(first.config_hash, second.config_hash)
        self.assertEqual(first.run_id, second.run_id)

    def test_checkpoint_steps_must_be_sorted_and_unique(self) -> None:
        content = Path("configs/smoke.yaml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(
                content.replace("[0, 5, 10, 20]", "[0, 10, 5, 20]"),
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
