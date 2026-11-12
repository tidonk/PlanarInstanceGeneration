import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from check.scripts import build_partial_pglib_report as report


class SamplingInputTests(unittest.TestCase):
    def discover(self, configured, methods, *, use_configured=True):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "sample_pglib_job_i001"
            (run / "tables").mkdir(parents=True)
            (run / "instances").mkdir()
            data = []
            for method in methods:
                instance = run / "instances" / f"{method}.m"
                instance.write_text("input")
                data.append(dict(strategy=method, artifact_stem=method,
                                 instance_file=str(instance), target_size="10",
                                 source_artifact_stems="source"))
            with (run / "tables/accepted_instances.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            (run / "manifest.json").write_text(json.dumps({
                "run": {"strategies": configured},
            }))
            with patch.object(report, "RESULTS_DIRECTORY", root):
                return report._discover_cohorts(
                    "sample", index=1, use_configured_strategies=use_configured,
                )

    def test_complete_sampling_input_is_accepted(self):
        found = self.discover(["exact_outer_face"], ["exact_outer_face"])
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].rows[0]["strategy"], "exact_outer_face")

    def test_incomplete_full_comparison_is_rejected(self):
        self.assertEqual(self.discover(list(report.STRATEGIES), ["exact_outer_face"]), [])

    def test_duplicate_rows_are_rejected(self):
        self.assertEqual(self.discover(["exact_outer_face"], ["exact_outer_face"] * 2), [])

    def test_default_report_still_requires_all_methods(self):
        self.assertEqual(self.discover(["exact_outer_face"], ["exact_outer_face"],
                                       use_configured=False), [])
        self.assertEqual(len(self.discover(list(report.STRATEGIES),
                                          list(report.STRATEGIES))), 1)


if __name__ == "__main__":
    unittest.main()
