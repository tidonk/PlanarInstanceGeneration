from __future__ import annotations

import unittest
from unittest.mock import patch

import networkx as nx

from planar_instance_generation.core.selection import (
    coupling_pair_count,
    select_high_degree_pairs,
    select_kamada_kawai_pairs,
    select_merge_node_pairs,
    select_random_pairs,
)


class MergeNodeSelectionTests(unittest.TestCase):
    def test_coupling_count_uses_smaller_node_count_and_logarithm(self) -> None:
        self.assertEqual(
            coupling_pair_count(nx.path_graph(8), nx.cycle_graph(10)),
            3,
        )
        self.assertEqual(
            coupling_pair_count(nx.empty_graph(1), nx.empty_graph(1)),
            1,
        )
        self.assertEqual(
            coupling_pair_count(nx.path_graph(2), nx.path_graph(7)),
            2,
        )
        self.assertEqual(
            coupling_pair_count(nx.path_graph(16), nx.path_graph(40)),
            4,
        )
        self.assertEqual(
            coupling_pair_count(nx.path_graph(33), nx.path_graph(40)),
            6,
        )

    def test_random_selection_is_reproducible_and_endpoint_disjoint(self) -> None:
        left = nx.path_graph(12)
        right = nx.cycle_graph(range(20, 31))
        left_before = left.copy()
        right_before = right.copy()

        first = select_random_pairs(left, right, 8, seed=73)
        second = select_random_pairs(left, right, 8, seed=73)

        self.assertEqual(first.pairs, second.pairs)
        self.assertEqual(len(set(first.left_nodes)), 8)
        self.assertEqual(len(set(first.right_nodes)), 8)
        self.assertEqual(first.strategy, "random")
        self.assertEqual(first.left_candidate_count, 12)
        self.assertEqual(first.right_candidate_count, 11)
        self.assertGreaterEqual(first.selection_seconds, 0.0)
        self.assertTrue(nx.utils.graphs_equal(left, left_before))
        self.assertTrue(nx.utils.graphs_equal(right, right_before))

    def test_high_degree_selects_largest_degrees_and_seeds_ties(self) -> None:
        left = nx.star_graph(8)
        right = nx.complete_bipartite_graph(3, 6)

        first = select_high_degree_pairs(left, right, 4, seed=19)
        second = select_high_degree_pairs(left, right, 4, seed=19)

        self.assertEqual(first.pairs, second.pairs)
        self.assertEqual([left.degree[node] for node in first.left_nodes], [8, 1, 1, 1])
        self.assertEqual([right.degree[node] for node in first.right_nodes], [6, 6, 6, 3])
        self.assertEqual(len(set(first.left_nodes)), 4)
        self.assertEqual(len(set(first.right_nodes)), 4)

        tied = nx.empty_graph(12)
        seed_one = select_high_degree_pairs(tied, tied.copy(), 6, seed=1)
        seed_two = select_high_degree_pairs(tied, tied.copy(), 6, seed=2)
        self.assertNotEqual(seed_one.pairs, seed_two.pairs)

    def test_kamada_kawai_greedily_selects_nearest_available_pairs(self) -> None:
        left = nx.path_graph(["a", "b", "c"])
        right = nx.path_graph(["x", "y", "z"])
        layouts = [
            {"a": (0.0, 0.0), "b": (0.0, 1.0), "c": (0.0, 2.0)},
            {"x": (0.0, 0.0), "y": (0.0, 1.0), "z": (0.0, 2.0)},
        ]

        with patch(
            "planar_instance_generation.core.selection.nx.kamada_kawai_layout",
            side_effect=layouts,
        ) as layout:
            result = select_kamada_kawai_pairs(left, right, 2, seed=5)

        self.assertEqual(len(result.pairs), 2)
        self.assertTrue(set(result.pairs) <= {("a", "x"), ("b", "y"), ("c", "z")})
        self.assertEqual(layout.call_count, 2)
        self.assertEqual(result.strategy, "kamada_kawai")
        self.assertEqual(result.seed, 5)
        self.assertGreaterEqual(result.selection_seconds, 0.0)

    def test_kamada_kawai_handles_single_node_graphs(self) -> None:
        result = select_kamada_kawai_pairs(nx.empty_graph(1), nx.empty_graph(1), 1)
        self.assertEqual(result.pairs, ((0, 0),))

    def test_kamada_kawai_rejects_smaller_candidate_set(self) -> None:
        left = nx.path_graph(5)
        right = nx.path_graph(range(10, 15))
        layouts = [
            {
                0: (0.0, 0.0),
                1: (1.0, 0.0),
                2: (1.0, 1.0),
                3: (0.0, 1.0),
                4: (0.5, 0.5),
            },
            {
                10: (0.0, 0.0),
                11: (1.0, 0.0),
                12: (0.5, 1.0),
                13: (0.5, 0.4),
                14: (0.5, 0.6),
            },
        ]
        with patch(
            "planar_instance_generation.core.selection.nx.kamada_kawai_layout",
            side_effect=layouts,
        ):
            with self.assertRaisesRegex(
                ValueError, "requested 4 distinct boundary nodes.*found 4 and 3"
            ):
                select_kamada_kawai_pairs(left, right, 4)

    def test_kamada_kawai_uses_seed_for_equal_distance_ties(self) -> None:
        left = nx.empty_graph(5)
        right = nx.empty_graph(range(10, 15))
        tied_layouts = [
            {node: (0.0, 0.0) for node in left},
            {node: (0.0, 0.0) for node in right},
        ]
        with patch(
            "planar_instance_generation.core.selection.nx.kamada_kawai_layout",
            side_effect=tied_layouts * 2,
        ):
            first = select_kamada_kawai_pairs(left, right, 3, seed=1)
            second = select_kamada_kawai_pairs(left, right, 3, seed=2)
        self.assertNotEqual(first.pairs, second.pairs)

    def test_dispatch_and_validation(self) -> None:
        left = nx.path_graph(3)
        right = nx.path_graph(4)
        dispatched = select_merge_node_pairs(left, right, 2, strategy="random", seed=11)
        direct = select_random_pairs(left, right, 2, seed=11)
        self.assertEqual(dispatched.pairs, direct.pairs)

        with self.assertRaisesRegex(ValueError, "unknown selection strategy"):
            select_merge_node_pairs(left, right, 1, strategy="missing")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            select_random_pairs(left, right, 4)
        with self.assertRaises(ValueError):
            select_high_degree_pairs(left, right, 0)
        with self.assertRaises(TypeError):
            select_random_pairs(left, right, True)
        with self.assertRaises(TypeError):
            select_random_pairs(nx.DiGraph([(0, 1)]), right, 1)
        with self.assertRaises(TypeError):
            select_random_pairs(nx.MultiGraph([(0, 1)]), right, 1)


if __name__ == "__main__":
    unittest.main()
