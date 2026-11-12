from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import networkx as nx

from planar_instance_generation.evaluation import (
    EmbeddedSourceGraph,
    audit_preserved_properties,
    write_preservation_report,
)


class PreservationTests(unittest.TestCase):
    def test_reports_paper_bounds_and_true_instance_values(self) -> None:
        left = nx.path_graph(3)
        right = nx.path_graph(2)
        composite = nx.disjoint_union(left, right)
        composite.add_edge(2, 3)
        sources = (
            EmbeddedSourceGraph("left", left, {0: 0, 1: 1, 2: 2}),
            EmbeddedSourceGraph("right", right, {0: 3, 1: 4}),
        )

        audit = audit_preserved_properties(
            sources,
            composite,
            [(2, 3)],
            noncrossing_matching_certified=True,
            composition_strategy="exact_outer_face",
        )

        rows = {row.property: row for row in audit.properties}
        self.assertEqual(rows["node_count"].paper_lower_bound, 5)
        self.assertEqual(rows["node_count"].observed_value, 5)
        self.assertTrue(rows["node_count"].empirically_within_bound)
        self.assertEqual(rows["maximum_degree"].paper_lower_bound, 2)
        self.assertEqual(rows["maximum_degree"].paper_upper_bound, 3)
        self.assertEqual(rows["maximum_degree"].observed_value, 2)
        self.assertTrue(all(row.within_bound for row in audit.degree_distribution))
        self.assertTrue(audit.sequential_matching)
        self.assertTrue(audit.noncrossing_matching_certified)
        self.assertEqual(audit.composition_strategy, "exact_outer_face")
        self.assertTrue(
            rows["average_clustering_difference"].paper_hypothesis_satisfied
        )
        self.assertEqual(
            audit.degree_distribution[0].paper_bound,
            "|p_C(q) - p_src(q)| <= |B| / |V(C)| for every q",
        )
        self.assertTrue(
            all(
                row.equality_holds
                for row in audit.node_clustering
                if row.equality_required
            )
        )
        self.assertEqual(rows["treewidth"].observed_lower_bound, 1)
        self.assertEqual(rows["treewidth"].observed_upper_bound, 1)
        self.assertEqual(rows["treewidth"].observed_value, 1)
        self.assertEqual(rows["treewidth"].paper_upper_bound, 3)

    def test_conditional_bounds_are_not_claimed_without_a_certificate(self) -> None:
        source = nx.path_graph(2)
        composite = nx.disjoint_union(source, source)
        composite.add_edge(1, 2)
        audit = audit_preserved_properties(
            (
                EmbeddedSourceGraph("a", source, {0: 0, 1: 1}),
                EmbeddedSourceGraph("b", source, {0: 2, 1: 3}),
            ),
            composite,
            [(1, 2)],
            composition_strategy="random",
        )

        rows = {row.property: row for row in audit.properties}
        for name in ("average_clustering_difference", "degeneracy"):
            self.assertIsNone(rows[name].paper_hypothesis_satisfied)
            self.assertIsNone(rows[name].empirically_within_bound)
            self.assertTrue(rows[name].numerically_within_bound)
            self.assertIn("not certified", rows[name].note)

    def test_detects_when_coupling_edges_are_not_a_sequential_matching(self) -> None:
        source = nx.empty_graph(1)
        composite = nx.empty_graph(3)
        composite.add_edges_from(((0, 2), (1, 2)))
        audit = audit_preserved_properties(
            tuple(
                EmbeddedSourceGraph(str(index), source, {0: index})
                for index in range(3)
            ),
            composite,
            [(0, 2), (1, 2)],
            noncrossing_matching_certified=True,
        )

        rows = {row.property: row for row in audit.properties}
        self.assertFalse(audit.sequential_matching)
        self.assertFalse(
            rows["average_clustering_difference"].paper_hypothesis_satisfied
        )
        self.assertIsNone(
            rows["average_clustering_difference"].empirically_within_bound
        )

    def test_treewidth_uses_brackets_when_the_true_values_are_not_exact(self) -> None:
        grid = nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3))
        singleton = nx.empty_graph(1)
        composite = nx.disjoint_union(grid, singleton)
        composite.add_edge(0, 9)
        audit = audit_preserved_properties(
            (
                EmbeddedSourceGraph("grid", grid, {node: node for node in grid}),
                EmbeddedSourceGraph("single", singleton, {0: 9}),
            ),
            composite,
            [(0, 9)],
            noncrossing_matching_certified=True,
        )

        treewidth = next(
            row for row in audit.properties if row.property == "treewidth"
        )
        self.assertIsNone(treewidth.paper_upper_bound)
        self.assertIsNone(treewidth.observed_value)
        self.assertEqual(treewidth.paper_rhs_lower_estimate, 4)
        self.assertEqual(treewidth.paper_rhs_upper_estimate, 5)
        self.assertLess(treewidth.observed_lower_bound, treewidth.observed_upper_bound)
        self.assertIn("paper RHS in [4, 5]", treewidth.note)

    def test_detects_a_malformed_composite(self) -> None:
        left = nx.path_graph(2)
        right = nx.path_graph(2)
        composite = nx.disjoint_union(left, right)
        composite.add_edges_from([(1, 2), (0, 3)])
        sources = (
            EmbeddedSourceGraph("left", left, {0: 0, 1: 1}),
            EmbeddedSourceGraph("right", right, {0: 2, 1: 3}),
        )
        with self.assertRaisesRegex(ValueError, "exactly source plus"):
            audit_preserved_properties(sources, composite, [(1, 2)])

    def test_writes_tidy_instancewise_artifacts(self) -> None:
        source = nx.path_graph(2)
        composite = nx.disjoint_union(source, source)
        composite.add_edge(1, 2)
        audit = audit_preserved_properties(
            (
                EmbeddedSourceGraph("a", source, {0: 0, 1: 1}),
                EmbeddedSourceGraph("b", source, {0: 2, 1: 3}),
            ),
            composite,
            [(1, 2)],
        )
        with tempfile.TemporaryDirectory() as directory:
            artifacts = write_preservation_report({"four": audit}, directory)
            with artifacts.properties_csv.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row["instance"] for row in rows}, {"four"})
            self.assertIn("paper_bound", rows[0])
            self.assertIn("paper_hypothesis_satisfied", rows[0])
            self.assertIn("composition_strategy", rows[0])
            self.assertTrue(artifacts.json.is_file())
            with self.assertRaises(FileExistsError):
                write_preservation_report({"four": audit}, directory)


if __name__ == "__main__":
    unittest.main()
