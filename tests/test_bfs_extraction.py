from __future__ import annotations

import math
import random
import unittest

import networkx as nx

from planar_instance_generation.core.extraction import (
    ComponentTooSmallError,
    randomized_bfs_extraction,
)


class _FixedStartRandom(random.Random):
    def __init__(self, start_node: int) -> None:
        super().__init__(0)
        self.start_node = start_node

    def choice(self, sequence):  # type: ignore[no-untyped-def]
        if self.start_node not in sequence:
            raise AssertionError("fixed start node is absent")
        return self.start_node


class RandomizedBfsExtractionTests(unittest.TestCase):
    def test_selects_exact_ceiling_target_in_bfs_order(self) -> None:
        graph = nx.path_graph(10)

        result = randomized_bfs_extraction(graph, 0.21, rng=_FixedStartRandom(0))

        self.assertEqual(result.target_size, math.ceil(0.21 * len(graph)))
        self.assertEqual(result.selected_nodes, (0, 1, 2))
        self.assertEqual(result.cut_edges, ((2, 3),))

    def test_seed_and_equivalent_random_instances_are_reproducible(self) -> None:
        graph = nx.complete_graph(12)

        seeded_first = randomized_bfs_extraction(graph, 0.5, seed=123)
        seeded_second = randomized_bfs_extraction(graph, 0.5, seed=123)
        rng_first = randomized_bfs_extraction(graph, 0.5, rng=random.Random(123))
        rng_second = randomized_bfs_extraction(graph, 0.5, rng=random.Random(123))

        self.assertEqual(seeded_first, seeded_second)
        self.assertEqual(rng_first, rng_second)
        self.assertEqual(seeded_first, rng_first)

    def test_reports_every_and_only_cut_edge(self) -> None:
        graph = nx.Graph([(0, 1), (1, 2), (2, 3), (0, 3), (1, 3)])
        result = randomized_bfs_extraction(graph, 0.5, rng=_FixedStartRandom(0))
        selected = set(result.selected_nodes)

        expected = {
            frozenset((first, second))
            for first, second in graph.edges
            if (first in selected) != (second in selected)
        }
        actual = {frozenset(edge) for edge in result.cut_edges}

        self.assertEqual(actual, expected)
        self.assertTrue(all((u in selected) != (v in selected) for u, v in result.cut_edges))

    def test_rejects_sampled_component_smaller_than_target(self) -> None:
        graph = nx.disjoint_union(nx.path_graph(2), nx.path_graph(4))

        with self.assertRaises(ComponentTooSmallError) as caught:
            randomized_bfs_extraction(graph, 0.5, rng=_FixedStartRandom(0))

        self.assertEqual(caught.exception.start_node, 0)
        self.assertEqual(caught.exception.component_size, 2)
        self.assertEqual(caught.exception.target_size, 3)
        self.assertIn("component of size 2", str(caught.exception))

    def test_full_fraction_returns_entire_connected_graph_without_cut(self) -> None:
        graph = nx.cycle_graph(7)

        result = randomized_bfs_extraction(graph, 1.0, seed=8)

        self.assertEqual(len(result.selected_nodes), len(graph))
        self.assertEqual(set(result.selected_nodes), set(graph))
        self.assertEqual(result.cut_edges, ())

    def test_integer_target_avoids_fraction_rounding(self) -> None:
        graph = nx.path_graph(10)

        result = randomized_bfs_extraction(
            graph, target_nodes=4, rng=_FixedStartRandom(0)
        )

        self.assertEqual(result.target_size, 4)
        self.assertEqual(result.selected_nodes, (0, 1, 2, 3))

    def test_requires_exactly_one_size_input(self) -> None:
        graph = nx.path_graph(3)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            randomized_bfs_extraction(graph)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            randomized_bfs_extraction(graph, 0.5, target_nodes=2)
        with self.assertRaisesRegex(ValueError, "integer in"):
            randomized_bfs_extraction(graph, target_nodes=4)

    def test_does_not_modify_graph(self) -> None:
        graph = nx.path_graph(5)
        graph.nodes[2]["label"] = "middle"
        before = graph.copy()

        randomized_bfs_extraction(graph, 0.6, seed=4)

        self.assertTrue(nx.utils.graphs_equal(graph, before))

    def test_rejects_invalid_fraction_and_graph(self) -> None:
        graph = nx.path_graph(3)
        for fraction in (0, -0.1, 1.1, math.inf, math.nan):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                randomized_bfs_extraction(graph, fraction)
        with self.assertRaises(TypeError):
            randomized_bfs_extraction(graph, True)
        with self.assertRaises(ValueError):
            randomized_bfs_extraction(nx.Graph(), 1.0)
        with self.assertRaises(TypeError):
            randomized_bfs_extraction(nx.DiGraph([(0, 1)]), 1.0)
        with self.assertRaises(TypeError):
            randomized_bfs_extraction(nx.MultiGraph([(0, 1)]), 1.0)

    def test_rejects_conflicting_or_invalid_random_sources(self) -> None:
        graph = nx.path_graph(3)
        with self.assertRaises(ValueError):
            randomized_bfs_extraction(graph, 0.5, seed=1, rng=random.Random(1))
        with self.assertRaises(TypeError):
            randomized_bfs_extraction(graph, 0.5, rng=object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
