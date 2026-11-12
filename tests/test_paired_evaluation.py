from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from planar_instance_generation.evaluation.paired import (
    SNDLIB_PAIRED_SCHEMA,
    write_paired_strategy_report,
)


class PairedStrategyEvaluationTests(unittest.TestCase):
    def test_writes_only_complete_shared_extraction_outcomes(self) -> None:
        accepted = [
            _accepted(1, 1, "40", "random"),
            _accepted(1, 1, "40_2", "exact_outer_face"),
            _accepted(2, 1, "50", "random"),
        ]
        attempts = [
            _attempt(1, 1, "random", "accepted"),
            _attempt(1, 1, "exact_outer_face", "accepted"),
            _attempt(2, 1, "random", "accepted"),
            _attempt(2, 1, "exact_outer_face", "rejected"),
        ]
        extractions = [
            _extraction("40", 1, seed=71),
            _extraction("40_2", 1, seed=71),
            _extraction("50", 1, seed=82),
        ]
        structural = [
            _structural("Composite: Random", "40", 0.2),
            _structural("Composite: Exact face", "40_2", 0.3),
            _structural("Composite: Random", "50", 0.4),
        ]
        skewness = [
            _skewness("Composite: Random", "40", 1),
            _skewness("Composite: Exact face", "40_2", 2),
            _skewness("Composite: Random", "50", 3),
        ]

        with tempfile.TemporaryDirectory() as directory:
            artifacts = write_paired_strategy_report(
                ("random", "exact_outer_face"),
                accepted,
                attempts,
                extractions,
                metadata_strategy_by_artifact={
                    "40": "random",
                    "40_2": "exact_outer_face",
                    "50": "random",
                },
                strategy_groups={
                    "random": "Composite: Random",
                    "exact_outer_face": "Composite: Exact face",
                },
                structural_rows=structural,
                skewness_rows=skewness,
                output_directory=directory,
            )

            outcomes = _rows(artifacts.outcomes_csv)
            self.assertEqual([row["artifact_stem"] for row in outcomes], ["40", "40_2"])
            self.assertTrue(all(row["shared_extraction_verified"] == "True" for row in outcomes))
            self.assertEqual(outcomes[0]["merge_node_selection_seconds"], "0.25")
            self.assertEqual(outcomes[0]["structural_density"], "0.2")
            self.assertEqual(outcomes[1]["skewness_lower_bound"], "2")

            cohorts = _rows(artifacts.cohorts_csv)
            self.assertEqual([row["status"] for row in cohorts], ["included", "excluded"])
            self.assertEqual(cohorts[1]["missing_strategies"], "exact_outer_face")
            self.assertEqual(cohorts[1]["reason"], "incomplete_requested_strategies")
            report = json.loads(artifacts.report_json.read_text(encoding="utf-8"))
            self.assertEqual(report["included_cohort_count"], 1)
            self.assertEqual(report["excluded_cohort_count"], 1)

    def test_excludes_nominally_complete_cohort_when_extractions_differ(self) -> None:
        accepted = [
            _accepted(1, 1, "40", "random"),
            _accepted(1, 1, "40_2", "exact_outer_face"),
        ]
        attempts = [
            _attempt(1, 1, "random", "accepted"),
            _attempt(1, 1, "exact_outer_face", "accepted"),
        ]
        extractions = [
            _extraction("40", 1, seed=71),
            _extraction("40_2", 1, seed=72),
        ]
        with tempfile.TemporaryDirectory() as directory:
            artifacts = write_paired_strategy_report(
                ("random", "exact_outer_face"),
                accepted,
                attempts,
                extractions,
                metadata_strategy_by_artifact={
                    "40": "random",
                    "40_2": "exact_outer_face",
                },
                strategy_groups={
                    "random": "Composite: Random",
                    "exact_outer_face": "Composite: Exact face",
                },
                structural_rows=[],
                skewness_rows=[],
                output_directory=directory,
            )
            self.assertEqual(_rows(artifacts.outcomes_csv), [])
            cohort = _rows(artifacts.cohorts_csv)[0]
            self.assertEqual(cohort["status"], "excluded")
            self.assertEqual(cohort["reason"], "shared_extraction_mismatch")

    def test_rejects_accepted_status_disagreement(self) -> None:
        accepted = [_accepted(1, 1, "40", "random")]
        attempts = [_attempt(1, 1, "random", "rejected")]
        with tempfile.TemporaryDirectory() as directory:
            artifacts = write_paired_strategy_report(
                ("random",),
                accepted,
                attempts,
                [_extraction("40", 1, seed=71)],
                metadata_strategy_by_artifact={"40": "random"},
                strategy_groups={"random": "Composite: Random"},
                structural_rows=[],
                skewness_rows=[],
                output_directory=directory,
            )
            cohort = _rows(artifacts.cohorts_csv)[0]
            self.assertEqual(cohort["status"], "excluded")
            self.assertIn("acceptance_table_mismatch", cohort["reason"])

    def test_snd_schema_uses_network_fields_and_accepts_unitless_sources(self) -> None:
        accepted = [
            {**_accepted(1, 1, "12", "random"), "node_count": 12, "link_count": 15},
            {
                **_accepted(1, 1, "12_2", "exact_outer_face"),
                "node_count": 12,
                "link_count": 16,
            },
        ]
        attempts = [
            _attempt(1, 1, "random", "accepted"),
            _attempt(1, 1, "exact_outer_face", "accepted"),
        ]
        extractions = [
            _snd_extraction("12", 1),
            _snd_extraction("12_2", 1),
        ]
        structural = [
            _structural("Composite: Random", "12", 0.2),
            _structural("Composite: Exact face", "12_2", 0.3),
        ]
        skewness = [
            _skewness("Composite: Random", "12", 1),
            _skewness("Composite: Exact face", "12_2", 0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            artifacts = write_paired_strategy_report(
                ("random", "exact_outer_face"),
                accepted,
                attempts,
                extractions,
                metadata_strategy_by_artifact={
                    "12": "random",
                    "12_2": "exact_outer_face",
                },
                strategy_groups={
                    "random": "Composite: Random",
                    "exact_outer_face": "Composite: Exact face",
                },
                structural_rows=structural,
                skewness_rows=skewness,
                output_directory=directory,
                schema=SNDLIB_PAIRED_SCHEMA,
            )
            outcomes = _rows(artifacts.outcomes_csv)
            self.assertEqual(outcomes[0]["node_count"], "12")
            self.assertIn("link_count", outcomes[0])
            self.assertNotIn("bus_count", outcomes[0])
            report = json.loads(artifacts.report_json.read_text(encoding="utf-8"))
            self.assertEqual(report["domain_schema"], "sndlib")


def _accepted(output: int, attempt: int, artifact: str, strategy: str) -> dict[str, object]:
    return {
        "output_index": output,
        "attempt_index": attempt,
        "artifact_stem": artifact,
        "strategy": strategy,
        "merge_node_selection_seconds": 0.25,
        "composition_seconds": 0.3,
        "merge_seconds": 0.55,
        "generation_and_gate_seconds": 0.8,
        "gate_feasible": True,
        "gate_status": "FEASIBLE",
        "objective": 10.0,
        "gate_seconds": 0.2,
        "gate_optimizer_seconds": 0.1,
    }


def _attempt(output: int, attempt: int, strategy: str, status: str) -> dict[str, object]:
    return {
        "output_index": output,
        "attempt_index": attempt,
        "strategy": strategy,
        "status": status,
        "draw_seed": 11,
        "generation_seed": 12,
        "source_names": "source-a",
    }


def _extraction(artifact: str, position: int, *, seed: int) -> dict[str, object]:
    return {
        "final_artifact_stem": artifact,
        "source_position": position,
        "source_name": "source-a",
        "source_node_count": 80,
        "target_nodes": 40,
        "realized_fraction": 0.5,
        "seed": seed,
        "bus_count": 40,
        "generator_count": 5,
        "branch_count": 60,
        "cut_branch_count": 2,
        "compensated_bus_count": 1,
        "compensation_generator_count": 1,
    }


def _snd_extraction(artifact: str, position: int) -> dict[str, object]:
    return {
        "final_artifact_stem": artifact,
        "source_position": position,
        "source_name": "source-a",
        "routing_unit": None,
        "source_coordinates_type": "pixel",
        "source_node_count": 12,
        "target_nodes": 6,
        "realized_fraction": 0.5,
        "seed": 71,
        "extraction_seconds": 0.01,
        "node_count": 6,
        "link_count": 7,
        "commodity_count": 2,
        "cut_link_count": 1,
        "compensated_node_count": 1,
        "compensated_commodity_count": 1,
        "reference_flow_nonzero_count": 4,
    }


def _structural(group: str, artifact: str, density: float) -> dict[str, object]:
    return {"group": group, "instance": artifact, "density": density}


def _skewness(group: str, artifact: str, value: int) -> dict[str, object]:
    return {
        "group": group,
        "instance": artifact,
        "skewness_lower_bound": value,
    }


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


if __name__ == "__main__":
    unittest.main()
