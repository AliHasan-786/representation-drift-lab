from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from driftlab.clip_data import (
    _dataset_server_rows,
    _ranges_overlap,
    _server_class_names,
    _server_dataset,
)


class DatasetServerCacheTests(unittest.TestCase):
    def test_server_dataset_returns_constructed_dataset(self) -> None:
        decoded = [{"image": object(), "label": 1, "row_idx": 7}]
        expected = object()
        with (
            patch("driftlab.clip_data._decode_server_records", return_value=decoded),
            patch("driftlab.clip_data._from_records", return_value=expected) as construct,
        ):
            actual = _server_dataset(
                spec={"repository": "example/data", "split": "test", "image_field": "image"},
                revision="abc123",
                rows=[{"row_idx": 7, "row": {}}],
                class_names=("zero", "one"),
                selection={"role": "eval"},
            )
        self.assertIs(actual, expected)
        self.assertEqual(construct.call_args.kwargs["selection"]["row_indices"], [7])

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

    def test_wide_row_request_is_reconstructed_from_api_compliant_chunks(self) -> None:
        def response_for(params: dict[str, object]) -> Mock:
            offset = int(params["offset"])
            length = int(params["length"])
            response = Mock(status_code=200)
            response.raise_for_status.return_value = None
            response.json.return_value = {
                "features": [{"name": "label", "type": {"names": ["zero"]}}],
                "rows": [{"row_idx": index, "row": {"label": 0}} for index in range(offset, offset + length)],
                "num_rows_total": 1000,
            }
            return response

        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                request = Mock(side_effect=lambda _url, params, timeout: response_for(params))
                fake_requests = SimpleNamespace(get=request)
                with patch.dict(sys.modules, {"requests": fake_requests}):
                    payload = _dataset_server_rows(
                        "example/data", "train", config_name="default", offset=22596, length=192
                    )
                self.assertEqual([item["row_idx"] for item in payload["rows"]], list(range(22596, 22788)))
                self.assertEqual(request.call_count, 2)
                self.assertEqual(request.call_args_list[0].kwargs["params"]["length"], 100)
                self.assertEqual(request.call_args_list[1].kwargs["params"]["offset"], 22696)
                self.assertEqual(request.call_args_list[1].kwargs["params"]["length"], 92)
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
