from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import networkx as nx

from planar_instance_generation.core import random_merge, random_merge_sequence
from planar_instance_generation.plt import (
    PlotConfig,
    make_source_styles,
    write_graph_plot,
    write_merge_process,
)


class TikzPlottingTests(unittest.TestCase):
    def test_single_graph_can_include_source_edge_legend(self) -> None:
        graph = nx.path_graph(3)
        nx.set_node_attributes(graph, "G_1", "source_graph")
        nx.set_edge_attributes(graph, "G_1", "source_graph")
        styles = make_source_styles({"G_1": "source one"})
        with tempfile.TemporaryDirectory() as directory:
            artifacts = write_graph_plot(
                graph,
                Path(directory) / "with_legend",
                source_styles=styles,
                include_legend=True,
            )
            text = artifacts.tikz.read_text(encoding="utf-8")

        self.assertIn("Source graphs", text)
        self.assertIn("source one", text)
        self.assertIn("draw=pigSrc1", text)

    def test_graph_plot_is_deterministic_and_has_paper_fragment(self) -> None:
        graph = nx.cycle_graph(5)
        with tempfile.TemporaryDirectory() as directory:
            first = write_graph_plot(graph, Path(directory) / "first")
            second = write_graph_plot(graph, Path(directory) / "second")
            self.assertEqual(
                first.tikz.read_text(encoding="utf-8"),
                second.tikz.read_text(encoding="utf-8"),
            )
            self.assertIn("\\input{\\jobname.tikz}", first.standalone_tex.read_text())
            data = json.loads(first.coordinates_json.read_text())
            self.assertEqual(data["layout"], "kamada_kawai")
            self.assertEqual(len(data["nodes"]), 5)
            tikz = first.tikz.read_text(encoding="utf-8")
            self.assertLess(tikz.index("\\coordinate (n0)"), tikz.index("\\draw[graph edge]"))

    def test_labels_are_latex_escaped(self) -> None:
        graph = nx.Graph()
        graph.add_node("bus_1%")
        with tempfile.TemporaryDirectory() as directory:
            artifact = write_graph_plot(
                graph,
                Path(directory) / "labels",
                config=PlotConfig(show_labels=True),
            )
            tikz = artifact.tikz.read_text(encoding="utf-8")
            self.assertIn(r"bus\_1\%", tikz)

    def test_merge_process_has_separate_source_step_and_final_windows(self) -> None:
        result = random_merge(nx.path_graph(4), nx.cycle_graph(5), 3, seed=4)
        with tempfile.TemporaryDirectory() as directory:
            artifacts = write_merge_process(result, directory)
            self.assertEqual(len(artifacts.steps), 3)
            for artifact in (
                artifacts.document,
                artifacts.original,
                *artifacts.steps,
                artifacts.final,
            ):
                self.assertTrue(artifact.tikz.is_file())
                self.assertTrue(artifact.standalone_tex.is_file())
                self.assertTrue(artifact.coordinates_json.is_file())
            source_data = json.loads(artifacts.sources.coordinates_json.read_text())
            left_x = [node["x_cm"] for node in source_data["nodes"] if node["source"] == "left"]
            right_x = [node["x_cm"] for node in source_data["nodes"] if node["source"] == "right"]
            self.assertLess(max(left_x), min(right_x))
            self.assertIn("\\draw[new merge edge]", artifacts.steps[0].tikz.read_text())
            self.assertNotIn("\\draw[new merge edge]", artifacts.final.tikz.read_text())
            self.assertEqual(artifacts.final.tikz.read_text().count("\\draw[merge edge]"), 3)
            figure = artifacts.document.tikz.read_text()
            original_title = r"{\small\bfseries Original}"
            step_title = r"{\small\bfseries Step 1}"
            final_title = r"{\small\bfseries Final}"
            self.assertLess(figure.index(original_title), figure.index(step_title))
            self.assertLess(figure.index(step_title), figure.index(final_title))
            self.assertIn("Source graphs", figure)

    def test_source_node_and_edge_colors_persist_across_pairwise_stages(self) -> None:
        sequence = random_merge_sequence(
            [nx.path_graph(3), nx.cycle_graph(4), nx.star_graph(3)],
            [2, 2],
            seed=8,
        )
        styles = make_source_styles(
            {"G_1": "First_grid", "G_2": "Second grid", "G_3": "Third grid"}
        )
        with tempfile.TemporaryDirectory() as directory:
            first = write_merge_process(
                sequence.stages[0],
                directory,
                stem="stage_001",
                source_styles=styles,
            )
            second = write_merge_process(
                sequence.stages[1],
                directory,
                stem="stage_002",
                source_styles=styles,
            )
            first_figure = first.document.tikz.read_text()
            second_figure = second.document.tikz.read_text()
            for definition in (
                r"\definecolor{pigSrc1}{HTML}{0072B2}",
                r"\definecolor{pigSrc2}{HTML}{E69F00}",
            ):
                self.assertIn(definition, first_figure)
                self.assertIn(definition, second_figure)
            self.assertIn(r"\definecolor{pigSrc3}{HTML}{009E73}", second_figure)
            self.assertIn(r"\definecolor{pigMerge}{HTML}{7F7F7F}", second_figure)
            self.assertIn("densely dashed", second_figure)
            self.assertIn(r"draw=pigSrc1", second.final.tikz.read_text())
            self.assertIn(r"draw=pigSrc2", second.final.tikz.read_text())
            self.assertIn(r"draw=pigSrc3", second.final.tikz.read_text())
            self.assertIn("First\\_grid", first_figure)
            self.assertNotIn("Third grid", first_figure)
            self.assertIn("Third grid", second_figure)

    def test_existing_complete_artifacts_are_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "cached"
            first = write_graph_plot(nx.path_graph(3), base)
            before = first.tikz.read_text()
            second = write_graph_plot(nx.complete_graph(4), base)
            self.assertEqual(first, second)
            self.assertEqual(second.tikz.read_text(), before)


if __name__ == "__main__":
    unittest.main()
