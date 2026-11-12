"""Saved flow records remain usable after moving the evaluation package."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from check.scripts import export_paper_scopf_flows as flows


class PortableFlowExportTests(unittest.TestCase):
    def test_original_machine_path_uses_packaged_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "check/results/run/logs/case"
            target.mkdir(parents=True)
            (target / "adapter_result.json").write_text(json.dumps({
                "rating_normalization_validation": {
                    "maximum_absolute_flow_by_branch_mw": {"a": 20, "b": 0}
                }
            }))
            row = {"summary_file": "/old/server/check/results/run/logs/case/summary.json",
                   "noncoupling_mean_absolute_flow_mw": "10",
                   "coupling_mean_absolute_flow_mw": "10",
                   "coupling_p95_absolute_flow_mw": "20",
                   "coupling_maximum_absolute_flow_mw": "20",
                   "expected_coupling_branch_count": "2"}
            with patch.object(flows, "ROOT", root):
                self.assertEqual(flows.positive_relative_flows(row), 1)


if __name__ == "__main__":
    unittest.main()
