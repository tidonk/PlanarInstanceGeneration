from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx
from check.scripts.evaluate_pglib import _compile_review_stage_pdf
from planar_instance_generation.plt import LatexCompilationError, PlotArtifacts

from planar_instance_generation.core.planarization import maximum_planar_subgraph
from planar_instance_generation.evaluation.reporting import (
    EvaluationGraph,
    SkewnessBounds,
    SkewnessObservation,
    write_evaluation_report,
)


class EvaluationReportingTests(unittest.TestCase):
    def test_review_stage_pdf_failure_is_recorded_and_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plot = PlotArtifacts(
                tikz=root / "stage.tikz",
                standalone_tex=root / "stage.tex",
                coordinates_json=root / "stage.json",
            )
            with patch(
                "check.scripts.evaluate_pglib.compile_plot_pdf",
                side_effect=LatexCompilationError("stage too large"),
            ):
                pdf, failure = _compile_review_stage_pdf(
                    plot,
                    output_pdf=root / "stage.pdf",
                    log_directory=root / "logs",
                    overwrite=True,
                )

            self.assertIsNone(pdf)
            self.assertIsNotNone(failure)
            failure_path = Path(failure["error_file"])
            self.assertTrue(failure_path.is_file())
            self.assertIn("stage too large", failure_path.read_text(encoding="utf-8"))

    def test_review_stage_pdf_reuses_complete_cached_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "stage.pdf"
            logs = root / "logs"
            logs.mkdir()
            output.write_bytes(b"pdf")
            (logs / "stage.stdout.log").write_text("", encoding="utf-8")
            (logs / "stage.latex.log").write_text("", encoding="utf-8")
            plot = PlotArtifacts(
                tikz=root / "stage.tikz",
                standalone_tex=root / "stage.tex",
                coordinates_json=root / "stage.json",
            )
            with patch(
                "check.scripts.evaluate_pglib.compile_plot_pdf",
                return_value=SimpleNamespace(pdf=output),
            ) as mocked:
                pdf, failure = _compile_review_stage_pdf(
                    plot,
                    output_pdf=output,
                    log_directory=logs,
                    overwrite=True,
                )

            self.assertEqual(pdf, output)
            self.assertIsNone(failure)
            self.assertFalse(mocked.call_args.kwargs["overwrite"])

    def test_writes_all_paper_metrics_and_plot_inputs_without_pdf(self) -> None:
        source_path = nx.path_graph(4)
        source_cycle = nx.cycle_graph(5)
        composite_exact = nx.complete_graph(5)
        composite_bounded = nx.complete_bipartite_graph(3, 3)
        groups = {
            "Sources": (
                EvaluationGraph("path", source_path),
                EvaluationGraph("cycle", source_cycle),
            ),
            "Generated": (
                EvaluationGraph("exact", composite_exact, (0, 1)),
                EvaluationGraph("bounded", composite_bounded, (0, 3)),
            ),
        }
        skewness = {
            ("Sources", "path"): SkewnessBounds(0, 0, "OPTIMAL", 0.01),
            ("Sources", "cycle"): SkewnessBounds(0, 0, "OPTIMAL", 0.02),
            ("Generated", "exact"): SkewnessObservation(
                SkewnessBounds(1, 1, "OPTIMAL", 0.5),
                source_floor_lower_bound=0,
                source_floor_upper_bound=0,
                coupling_edge_count=2,
            ),
            ("Generated", "bounded"): SkewnessObservation(
                SkewnessBounds(1, 3, "TIME_LIMIT", 5.0),
                source_floor_lower_bound=0,
                source_floor_upper_bound=1,
                coupling_edge_count=3,
            ),
        }

        with tempfile.TemporaryDirectory() as directory:
            artifacts = write_evaluation_report(
                groups,
                Path(directory) / "evaluation",
                skewness=skewness,
                seed=17,
            )

            structural = _csv_rows(artifacts.structural_csv)
            self.assertEqual(len(structural), 4)
            expected_metrics = {
                "node_count",
                "edge_count",
                "density",
                "mean_degree",
                "maximum_degree",
                "average_clustering",
                "degeneracy",
                "treewidth_upper_bound",
                "diameter",
                "algebraic_connectivity",
                "mean_betweenness",
                "betweenness_p95",
                "coupling_betweenness_mean",
                "coupling_betweenness_p95",
                "coupling_betweenness_max",
                "degree_assortativity",
            }
            self.assertTrue(expected_metrics.issubset(structural[0]))

            degree_rows = _csv_rows(artifacts.degree_distribution_csv)
            self.assertTrue(
                {"group", "instance", "degree", "node_count", "probability"}
                .issubset(degree_rows[0])
            )
            coupling_rows = _csv_rows(artifacts.coupling_betweenness_csv)
            self.assertEqual(len(coupling_rows), 4)

            skewness_rows = {
                (row["group"], row["instance"]): row
                for row in json.loads(
                    artifacts.skewness_json.read_text(encoding="utf-8")
                )["instances"]
            }
            exact = skewness_rows[("Generated", "exact")]
            self.assertTrue(exact["inflation_exact"])
            self.assertEqual(exact["inflation"], 1)
            self.assertEqual(exact["normalized_inflation"], 0.5)
            self.assertEqual(exact["normalized_inflation_lower_bound"], 0.5)
            self.assertEqual(exact["normalized_inflation_upper_bound"], 0.5)
            bounded = skewness_rows[("Generated", "bounded")]
            self.assertIsNone(bounded["normalized_inflation"])
            self.assertEqual(bounded["normalized_inflation_lower_bound"], 0.0)
            self.assertEqual(bounded["normalized_inflation_upper_bound"], 1.0)

            required_plots = {
                "node_count",
                "edge_count",
                "density",
                "degree",
                "mean_degree",
                "maximum_degree",
                "average_clustering",
                "degeneracy",
                "treewidth_upper_bound",
                "diameter",
                "algebraic_connectivity",
                "mean_betweenness",
                "betweenness_p95",
                "coupling_betweenness",
                "degree_assortativity",
                "skewness_lower_bound",
                "skewness_upper_bound",
                "normalized_inflation_lower_bound",
                "normalized_inflation_upper_bound",
                "normalized_inflation",
            }
            self.assertTrue(required_plots.issubset(artifacts.boxplots))
            for plot in artifacts.boxplots.values():
                self.assertTrue(plot.tikz.is_file())
                self.assertTrue(plot.standalone_tex.is_file())
                self.assertTrue(plot.coordinates_json.is_file())
                self.assertIn(
                    "\\input{\\jobname.tikz}",
                    plot.standalone_tex.read_text(encoding="utf-8"),
                )
            degree_plot = artifacts.boxplots["maximum_degree"].tikz.read_text(
                encoding="utf-8"
            )
            self.assertIn("{Sources}", degree_plot)
            self.assertIn("{Generated}", degree_plot)
            self.assertIn("0072B2", degree_plot)
            self.assertIn("E69F00", degree_plot)
            self.assertFalse(list(Path(directory).rglob("*.pdf")))
            manifest = json.loads(
                artifacts.boxplot_manifest_json.read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["compiler_invoked"])

    def test_defines_inflation_when_derived_interval_collapses(self) -> None:
        observation = SkewnessObservation(
            SkewnessBounds(0, 1),
            source_floor_lower_bound=1,
            source_floor_upper_bound=2,
            coupling_edge_count=1,
        )
        groups = {"Generated": {"one": nx.path_graph(3)}}
        with tempfile.TemporaryDirectory() as directory:
            artifacts = write_evaluation_report(
                groups,
                Path(directory) / "evaluation",
                skewness={("Generated", "one"): observation},
            )
            row = json.loads(
                artifacts.skewness_json.read_text(encoding="utf-8")
            )["instances"][0]
            self.assertTrue(row["inflation_exact"])
            self.assertEqual(row["inflation"], 0)
            self.assertTrue(row["normalized_inflation_exact"])
            self.assertEqual(row["normalized_inflation"], 0.0)

    def test_betweenness_sample_count_is_a_per_graph_cap(self) -> None:
        groups = {
            "Graphs": {
                "small": nx.path_graph(3),
                "large": nx.path_graph(5),
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            artifacts = write_evaluation_report(
                groups,
                Path(directory) / "evaluation",
                betweenness_samples=3,
                seed=23,
            )
            rows = {
                row["instance"]: row
                for row in json.loads(
                    artifacts.structural_json.read_text(encoding="utf-8")
                )["instances"]
            }
            self.assertTrue(rows["small"]["betweenness_exact"])
            self.assertIsNone(rows["small"]["betweenness_sample_count"])
            self.assertFalse(rows["large"]["betweenness_exact"])
            self.assertEqual(rows["large"]["betweenness_sample_count"], 3)
            self.assertEqual(rows["large"]["betweenness_seed"], 23)

    def test_accepts_planarization_result_and_validates_dimensions(self) -> None:
        graph = nx.path_graph(4)
        result = maximum_planar_subgraph(graph)
        groups = {"Sources": {"path": graph}}

        with tempfile.TemporaryDirectory() as directory:
            artifacts = write_evaluation_report(
                groups,
                Path(directory) / "good",
                skewness={("Sources", "path"): result},
            )
            row = _csv_rows(artifacts.skewness_csv)[0]
            self.assertEqual(row["skewness"], "0")
            self.assertEqual(row["status"], "OPTIMAL")

            with self.assertRaisesRegex(ValueError, "dimensions"):
                write_evaluation_report(
                    {"Sources": (EvaluationGraph("other", nx.path_graph(3)),)},
                    Path(directory) / "bad",
                    skewness={("Sources", "other"): result},
                )
            with self.assertRaisesRegex(ValueError, "certificate"):
                write_evaluation_report(
                    {"Sources": {"other": nx.star_graph(3)}},
                    Path(directory) / "wrong_graph",
                    skewness={("Sources", "other"): result},
                )

    def test_rejects_inconsistent_inflation_and_existing_output(self) -> None:
        groups = {"Generated": (EvaluationGraph("one", nx.path_graph(3)),)}
        inconsistent = SkewnessObservation(
            SkewnessBounds(0, 0),
            source_floor_lower_bound=2,
            source_floor_upper_bound=2,
            coupling_edge_count=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation"
            with self.assertRaisesRegex(ValueError, "contradict"):
                write_evaluation_report(
                    groups,
                    output,
                    skewness={("Generated", "one"): inconsistent},
                )

            write_evaluation_report(
                groups,
                output,
                skewness={("Generated", "one"): SkewnessBounds(0, 0)},
            )
            stale_plot = output / "boxplots" / "skewness_lower_bound.tikz"
            self.assertTrue(stale_plot.exists())
            with self.assertRaises(FileExistsError):
                write_evaluation_report(groups, output)
            write_evaluation_report(groups, output, overwrite=True)
            self.assertFalse(stale_plot.exists())


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


if __name__ == "__main__":
    unittest.main()
