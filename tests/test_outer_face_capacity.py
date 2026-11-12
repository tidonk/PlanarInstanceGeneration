"""Tests for logarithmic capacity on retained planar joins."""

from __future__ import annotations

import unittest

import networkx as nx

from planar_instance_generation.evaluation.outer_face_capacity import (
    evaluate_retained_planar_joins,
    logarithmic_pair_count,
)


class OuterFaceCapacityTests(unittest.TestCase):
    def test_pair_count_has_floor_and_small_graph_cap(self) -> None:
        self.assertEqual(logarithmic_pair_count(1), 1)
        self.assertEqual(logarithmic_pair_count(2), 2)
        self.assertEqual(logarithmic_pair_count(4), 3)
        self.assertEqual(logarithmic_pair_count(16), 4)
        self.assertEqual(logarithmic_pair_count(33), 6)

    def test_retained_join_measures_actual_left_and_right_faces(self) -> None:
        result = evaluate_retained_planar_joins(
            (nx.cycle_graph(8), nx.path_graph(5), nx.cycle_graph(3)),
            seed=7,
        )

        self.assertEqual(len(result.joins), 2)
        first = result.joins[0]
        self.assertEqual(first.left_node_count, 8)
        self.assertEqual(first.right_node_count, 5)
        self.assertEqual(first.n_min, 5)
        self.assertEqual(first.logarithmic_denominator, 3)
        self.assertEqual(first.available_distinct_boundary_nodes, 5)
        self.assertAlmostEqual(first.rho, 5 / 3)
        self.assertTrue(first.logarithmic_request_feasible)
        self.assertTrue(first.proposed_request_feasible)
        self.assertFalse(first.request_was_capped)
        self.assertEqual(result.final_graph.number_of_nodes(), 16)
        self.assertTrue(nx.check_planarity(result.final_graph)[0])
        self.assertTrue(result.every_logarithmic_request_feasible)

    def test_rejected_join_is_retained_and_later_joins_are_measured(self) -> None:
        triangulation = nx.complete_graph(4)
        previous = 3
        for node in range(4, 12):
            triangulation.add_edges_from(
                ((node, 0), (node, 1), (node, previous))
            )
            previous = node
        self.assertTrue(nx.check_planarity(triangulation)[0])

        result = evaluate_retained_planar_joins(
            (triangulation, triangulation.copy(), nx.path_graph(5)),
            seed=11,
        )

        self.assertEqual(len(result.joins), 2)
        first = result.joins[0]
        self.assertEqual(first.logarithmic_denominator, 4)
        self.assertEqual(first.available_distinct_boundary_nodes, 3)
        self.assertAlmostEqual(first.rho, 0.75)
        self.assertFalse(first.logarithmic_request_feasible)
        self.assertFalse(first.proposed_request_feasible)
        self.assertEqual(first.realized_pair_count, 3)
        self.assertTrue(first.request_was_capped)
        self.assertFalse(result.every_logarithmic_request_feasible)
        self.assertTrue(nx.check_planarity(result.final_graph)[0])


if __name__ == "__main__":
    unittest.main()
