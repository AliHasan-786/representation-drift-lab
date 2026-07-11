from __future__ import annotations

import unittest

from driftlab.provenance import is_generated_artifact_path, source_status_is_dirty


class ProvenanceTests(unittest.TestCase):
    def test_generated_outputs_do_not_count_as_source_changes(self) -> None:
        self.assertTrue(is_generated_artifact_path("artifacts/runs/example/manifest.json"))
        self.assertTrue(is_generated_artifact_path("public/data/runs/example.json"))
        self.assertTrue(is_generated_artifact_path("public/report/representation-drift-lab-report.pdf"))
        self.assertTrue(is_generated_artifact_path("output/pdf/report.pdf"))
        self.assertFalse(is_generated_artifact_path("src/driftlab/benchmark.py"))
        self.assertFalse(is_generated_artifact_path("configs/reproduction-local.yaml"))

    def test_first_porcelain_line_keeps_its_status_columns(self) -> None:
        status = " M public/report/representation-drift-lab-report.pdf\n?? public/data/run.json\n"
        self.assertFalse(source_status_is_dirty(status))
        self.assertTrue(source_status_is_dirty(status + " M src/driftlab/provenance.py\n"))


if __name__ == "__main__":
    unittest.main()
