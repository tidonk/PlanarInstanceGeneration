from __future__ import annotations

import math
import unittest

import networkx as nx

from planar_instance_generation.evaluation import measure_structure


class StructuralStatisticsTests(unittest.TestCase):
    def test_path_properties_and_coupling_values(self) -> None:
        graph = nx.path_graph(4)

        result = measure_structure(graph, coupling_nodes=[1, 2])

        self.assertEqual(result.node_count, 4)
        self.assertEqual(result.edge_count, 3)
        self.assertEqual(result.degree_distribution, {1: 0.5, 2: 0.5})
        self.assertEqual(result.mean_degree, 1.5)
        self.assertEqual(result.maximum_degree, 2)
        self.assertEqual(result.average_clustering, 0.0)
        self.assertEqual(result.degeneracy, 1)
        self.assertEqual(result.treewidth_upper_bound, 1)
        self.assertEqual(result.diameter, 3)
        self.assertAlmostEqual(result.algebraic_connectivity, 2 - math.sqrt(2), places=7)
        self.assertAlmostEqual(result.mean_betweenness, 1 / 3)
        self.assertAlmostEqual(result.betweenness_p95, 2 / 3)
        self.assertIsNone(result.betweenness_sample_count)
        self.assertEqual(set(result.coupling_betweenness), {1, 2})
        self.assertGreaterEqual(result.measurement_seconds, 0.0)

    def test_disconnected_graph_reports_undefined_diameter(self) -> None:
        result = measure_structure(nx.disjoint_union(nx.path_graph(2), nx.path_graph(2)))

        self.assertFalse(result.connected)
        self.assertIsNone(result.diameter)
        self.assertEqual(result.algebraic_connectivity, 0.0)

    def test_rejects_invalid_sampling_and_unknown_coupling_nodes(self) -> None:
        graph = nx.path_graph(3)
        sampled = measure_structure(graph, betweenness_samples=2)
        self.assertEqual(sampled.betweenness_sample_count, 2)
        with self.assertRaises(ValueError):
            measure_structure(graph, betweenness_samples=4)
        with self.assertRaisesRegex(ValueError, "unknown"):
            measure_structure(graph, coupling_nodes=[99])


if __name__ == "__main__":
    unittest.main()
