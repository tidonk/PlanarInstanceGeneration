import importlib.util
import math
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "check/scripts" / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


betweenness = module("recompute_exact_betweenness")
tables = module("export_paper_exact_tables")
source_skewness = module("evaluate_pglib_source_skewness")


class ExactPaperEvaluationTests(unittest.TestCase):
    def test_long_source_job_transformation_flags(self):
        for option, expected in (("--voltage-separated", True),
                                 ("--no-voltage-separated", False)):
            with self.subTest(option=option), patch.object(
                    sys, "argv", ["evaluate", "pglib_opf_case1354_pegase", option]):
                args = source_skewness.parse_args()
                self.assertIs(args.voltage_separated, expected)
                self.assertEqual(args.time_limit, 50400)
                self.assertFalse(hasattr(args, "warm_start_iterations"))
                self.assertIsNone(args.output_directory)

    def test_shifted_mean_includes_zeros(self):
        self.assertAlmostEqual(tables.shifted_mean([0, 3]), 1)
        self.assertEqual(tables.shifted_mean([0, 0]), 0)
        self.assertAlmostEqual(tables.shifted_mean([9, 9]), 9)

    def test_shifted_mean_rejects_invalid_times(self):
        for values in ([], [-1], [math.inf], [math.nan]):
            with self.assertRaises(AssertionError):
                tables.shifted_mean(values)

    def test_exact_measurement_and_digest_validated_cache(self):
        graph = nx.path_graph(5)
        class Case:
            def topology(self, active_only=True):
                return graph

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "check/data/pglib/path.m"
            source.parent.mkdir(parents=True)
            source.write_text("fixture")
            run = root / "run"
            (run / "evaluation").mkdir(parents=True)
            row = {"instance": "path", "group": "Original sources",
                   "node_count": "5", "edge_count": "4"}
            with patch.object(betweenness, "ROOT", root), patch.object(
                    betweenness, "read_pglib", return_value=Case()):
                actual = betweenness.measure(("pglib", run, row))
                self.assertTrue(actual["betweenness_exact"])
                self.assertEqual(actual["betweenness_sample_count"], "")
                expected = nx.betweenness_centrality(graph, normalized=True, weight=None)
                self.assertAlmostEqual(actual["mean_betweenness"], sum(expected.values()) / 5)
                with patch.object(betweenness.nx, "betweenness_centrality",
                                  side_effect=AssertionError("cache should be reused")):
                    self.assertEqual(betweenness.measure(("pglib", run, row)), actual)
                source.write_text("changed fixture")
                with patch.object(betweenness.nx, "betweenness_centrality",
                                  wraps=nx.betweenness_centrality) as solve:
                    betweenness.measure(("pglib", run, row))
                    self.assertEqual(solve.call_count, 1)


if __name__ == "__main__":
    unittest.main()
