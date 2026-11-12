from pathlib import Path
import unittest

from check.eval.make_sampling_sensitivity_dag import build_dag


class SamplingSensitivityDagTests(unittest.TestCase):
    def test_focused_job_counts_and_limits(self) -> None:
        dag = build_dag(
            "sampling_test",
            Path("/tmp/settings.env"),
            Path("/tmp/workflow"),
            instances=2,
        )

        self.assertEqual(sum(line.startswith("JOB gen_") for line in dag.splitlines()), 6)
        self.assertEqual(sum(line.startswith("JOB skew_") for line in dag.splitlines()), 6)
        self.assertEqual(sum(line.startswith("JOB app_") for line in dag.splitlines()), 3)
        self.assertEqual(dag.count('time_limit="3600"'), 6)
        self.assertEqual(dag.count('application_time_limit="0"'), 3)
        self.assertIn('sampling="equidistant" prefer_non_leaf="0"', dag)
        self.assertIn('sampling="adaptive_skip_one" prefer_non_leaf="0"', dag)
        self.assertIn('sampling="consecutive" prefer_non_leaf="1"', dag)
        self.assertIn("skew_equidistant_i002_exact_outer_face", dag)
        self.assertNotIn("heuristic_outer_face", dag)
        self.assertNotIn(".venv/bin/python", dag)
        self.assertEqual(dag.count("/usr/bin/python3"), 6)


if __name__ == "__main__":
    unittest.main()
