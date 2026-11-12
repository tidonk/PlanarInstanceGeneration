from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from check.eval.make_application_jobs import (
    _materialize_application_input,
    _materialize_pglib_metadata,
    _strategy_rows,
)
from check.eval.postprocess import (
    _Job,
    _archive_jobs,
    _discover_jobs,
    _validate_completed_evaluation,
)


class CohortArchiveTests(unittest.TestCase):
    def test_application_jobs_accept_a_requested_strategy_subset(self) -> None:
        rows = [
            {"strategy": strategy}
            for strategy in (
                "random",
                "high_degree",
                "kamada_kawai",
                "heuristic_outer_face",
                "exact_outer_face",
            )
        ]

        selected = _strategy_rows(
            rows,
            ("heuristic_outer_face", "exact_outer_face"),
            Path("accepted_instances.csv"),
        )

        self.assertEqual(set(selected), {row["strategy"] for row in rows})

    def test_completed_evaluation_validation_accepts_populated_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            combined = Path(directory) / "combined"
            evaluation = combined / "evaluation"
            evaluation.mkdir(parents=True)
            (combined / "manifest.json").write_text("{}\n", encoding="utf-8")
            (evaluation / "manifest.json").write_text("{}\n", encoding="utf-8")
            for name in ("skewness", "structural", "paired_strategy_outcomes"):
                (evaluation / f"{name}.csv").write_text(
                    "instance,value\nexample,1\n", encoding="utf-8"
                )

            _validate_completed_evaluation(combined)

    def test_completed_evaluation_validation_rejects_missing_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            combined = Path(directory) / "combined"
            evaluation = combined / "evaluation"
            evaluation.mkdir(parents=True)
            (combined / "manifest.json").write_text("{}\n", encoding="utf-8")
            (evaluation / "manifest.json").write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(SystemExit, "missing required files"):
                _validate_completed_evaluation(combined)

    def test_archive_move_updates_manifest_and_remains_discoverable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results"
            cohort = results / "experiment_pglib_job_i001"
            cohort.mkdir(parents=True)
            (cohort / "manifest.json").write_text("{}\n", encoding="utf-8")
            combined = results / "experiment_pglib"
            combined.mkdir()
            manifest = combined / "manifest.json"
            manifest.write_text(
                json.dumps({"consolidated": {"source_job_manifests": []}}),
                encoding="utf-8",
            )

            _archive_jobs(
                [_Job(1, cohort)],
                results,
                "experiment_pglib",
                manifest,
            )

            archived = (
                results
                / "_cohorts"
                / "experiment_pglib"
                / "experiment_pglib_job_i001"
            )
            self.assertFalse(cohort.exists())
            self.assertTrue((archived / "manifest.json").is_file())
            self.assertEqual(
                _discover_jobs(results, "experiment", "pglib")[0].directory,
                archived,
            )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["consolidated"]["source_job_manifests"],
                [str(archived / "manifest.json")],
            )

    def test_application_input_survives_source_removal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cohort" / "example.xml"
            source.parent.mkdir()
            source.write_text("network\n", encoding="utf-8")
            target = root / "application" / "instances" / source.name

            materialized = _materialize_application_input(source, target)
            source.unlink()

            self.assertEqual(materialized.read_text(encoding="utf-8"), "network\n")


    def test_pglib_metadata_survives_source_removal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort = root / "cohort"
            source = (
                cohort
                / "plots"
                / "composite"
                / "case"
                / "metadata.json"
            )
            source.parent.mkdir(parents=True)
            source.write_text('{"artifact_stem": "case"}\n', encoding="utf-8")

            materialized = _materialize_pglib_metadata(
                cohort, root / "application", "case"
            )
            source.unlink()

            self.assertEqual(
                json.loads(materialized.read_text(encoding="utf-8"))["artifact_stem"],
                "case",
            )


if __name__ == "__main__":
    unittest.main()
