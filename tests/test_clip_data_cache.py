from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from driftlab.clip_data import _dataset_server_rows, _ranges_overlap, _server_class_names


class DatasetServerCacheTests(unittest.TestCase):
    def test_reference_and_evaluation_ranges_must_be_disjoint(self) -> None:
        self.assertFalse(_ranges_overlap(0, 30, 1000, 30))
        self.assertTrue(_ranges_overlap(0, 30, 29, 30))
        self.assertFalse(_ranges_overlap(0, 30, 30, 30))
        with self.assertRaises(ValueError):
            _ranges_overlap(0, -1, 30, 30)

    def test_nonstandard_label_field_names_are_supported(self) -> None:
        payload = {
            "features": [
                {"name": "image", "type": {"_type": "Image"}},
                {"name": "fine_label", "type": {"names": ["apple", "bear"]}},
            ]
        }
        self.assertEqual(
            _server_class_names(payload, "fine_label"), ("apple", "bear")
        )

    def test_identical_row_request_is_served_from_local_cache(self) -> None:
        payload = {"features": [], "rows": [], "num_rows_total": 10}
        response = Mock(status_code=200)
        response.json.return_value = payload
        response.raise_for_status.return_value = None
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                request = Mock(return_value=response)
                fake_requests = SimpleNamespace(get=request)
                with patch.dict(sys.modules, {"requests": fake_requests}):
                    first = _dataset_server_rows(
                        "example/data", "train", config_name="default", offset=2, length=3
                    )
                    second = _dataset_server_rows(
                        "example/data", "train", config_name="default", offset=2, length=3
                    )
                self.assertEqual(first, payload)
                self.assertEqual(second, payload)
                request.assert_called_once()
                cached = list(Path("artifacts/cache/huggingface/rows").glob("*.json"))
                self.assertEqual(len(cached), 1)
                self.assertEqual(json.loads(cached[0].read_text()), payload)
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
