import csv
import json
from pathlib import Path
import tempfile
import unittest

from check.scripts.summarize_ogdf_progress import (
    B,
    H,
    STRATEGIES,
    collect,
    result_row,
    summarize,
)


def sample(strategy=B, lower=3, upper=3):
    def bounds(lo, hi):
        return dict(
            skewness_lower_bound=lo,
            skewness_upper_bound=hi,
            edge_count=20,
            runtime_seconds=1.0,
            status="TIME_LIMIT",
        )

    payload = dict(
        experiment="trial",
        cohort=1,
        strategy=strategy,
        settings={"time_limit_seconds": 3600},
        graph_results={
            "final:1:graph": bounds(lower, upper),
            "source:1:a": bounds(1, 3),
            "source:1:b": bounds(0, 0),
        },
    )
    accepted = dict(
        artifact_stem="graph",
        source_artifact_stems="a;b",
        coupling_branch_count="4",
        merge_node_selection_seconds="1.5",
        strategy=strategy,
        attempt_index="1",
        seed="7",
        gate_feasible="True",
    )
    return payload, accepted


class OgdfProgressTests(unittest.TestCase):
    def test_shared_uncertain_source_bounds_cancel_in_method_comparison(self):
        b, _ = result_row(*sample())
        h, _ = result_row(*sample(H, 4, 5))
        self.assertGreaterEqual(b["inflation_upper_bound"], h["inflation_lower_bound"])
        result = summarize([b, h])
        self.assertEqual(result["b_proven_better"], [1])
        self.assertEqual(result["complete_inputs"], [])
        self.assertEqual(result["methods"][B]["matched"], 0)
        self.assertEqual(result["outer_face_pairs"], [1])

    def test_smaller_upper_bound_is_not_a_proven_improvement(self):
        b, _ = result_row(*sample(B, 2, 3))
        h, _ = result_row(*sample(H, 2, 4))
        result = summarize([b, h])
        self.assertEqual(result["b_smaller_upper_bound"], [1])
        self.assertEqual(result["b_proven_better"], [])
        self.assertEqual(result["proven_equal"], [])

    def test_zero_added_and_exact_final_skewness_are_distinct(self):
        payload, accepted = sample(B, 1, 1)
        row, _ = result_row(payload, accepted)
        self.assertTrue(row["exact"])
        self.assertEqual(row["inflation_upper_bound"], 0)
        payload["graph_results"]["final:1:graph"]["skewness_lower_bound"] = 0
        payload["graph_results"]["final:1:graph"]["skewness_upper_bound"] = 0
        with self.assertRaises(ValueError):
            result_row(payload, accepted)

    def test_empty_run_is_reported_without_creating_results(self):
        with tempfile.TemporaryDirectory() as directory:
            report, rows = collect(Path(directory), "trial", 2)
            self.assertEqual(rows, [])
            self.assertEqual(report["errors"], [])
            self.assertEqual(report["datasets"]["pglib"]["generated_inputs"], [])
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_inconsistent_settings_exclude_input_from_comparisons(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "trial_pglib_job_i001"
            (run / "tables").mkdir(parents=True)
            (run / "manifest.json").write_text("{}")
            accepted = [sample(s)[1] for s in STRATEGIES]
            with (run / "tables/accepted_instances.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(accepted[0]))
                writer.writeheader()
                writer.writerows(accepted)
            checkpoints = root / "trial_pglib_exact/evaluation/cohort_checkpoints/i001"
            checkpoints.mkdir(parents=True)
            for strategy in STRATEGIES:
                payload, _ = sample(strategy)
                if strategy == B:
                    payload["settings"]["time_limit_seconds"] = 10
                (checkpoints / f"{strategy}.json").write_text(json.dumps(payload))
            report, rows = collect(root, "trial", 1)
            self.assertEqual(rows, [])
            self.assertEqual(len(report["errors"]), 1)
            self.assertIn("inconsistent solver settings", report["errors"][0])
            self.assertEqual(report["datasets"]["pglib"]["generated_inputs"], [1])


if __name__ == "__main__":
    unittest.main()
