"""Certificates must survive every join, including reused source node labels."""

import unittest

import networkx as nx
from planar_instance_generation.core.ogdf import direct_solution

from planar_instance_generation.core.planarization import (
    CertifiedPlanarPiece,
    ogdf_planar_subgraph,
    relabel_planar_piece,
    select_exact_outer_face_pairs,
    select_heuristic_outer_face_pairs,
)


class SequentialPlanarizationTests(unittest.TestCase):
    def _five_piece_sequence(self, exact=False):
        current = nx.complete_graph(5)
        carried = ()
        expected_lower = 0
        expected_upper = None
        for stage in range(1, 5):
            right = nx.complete_graph(5)  # Same labels as the first source.
            selector = (
                select_exact_outer_face_pairs
                if exact
                else select_heuristic_outer_face_pairs
            )
            options = dict(time_limit_seconds=2) if exact else {}
            selected = selector(
                current,
                right,
                3,
                seed=stage,
                left_certified_pieces=carried,
                **options,
            )
            left_result, right_result = (
                selected.left_planarization,
                selected.right_planarization,
            )
            self.assertEqual(left_result.shared_piece_count, 0 if stage == 1 else 2)
            self.assertEqual(right_result.shared_piece_count, 0)
            if carried:
                self.assertEqual(left_result.shared_piece_lower_bound, expected_lower)
                self.assertGreaterEqual(left_result.omitted_lower_bound, expected_lower)
                self.assertLessEqual(
                    left_result.piece_start_upper_bound, expected_upper
                )
                self.assertLessEqual(
                    left_result.omitted_upper_bound,
                    left_result.greedy_start_upper_bound,
                )
                self.assertLessEqual(
                    left_result.omitted_upper_bound, left_result.piece_start_upper_bound
                )
            left_map = {node: (stage, "left", node) for node in current}
            right_map = {node: (stage, "right", node) for node in right}
            carried = (
                relabel_planar_piece(current, left_result, left_map),
                relabel_planar_piece(right, right_result, right_map),
            )
            current = nx.compose(carried[0].graph, carried[1].graph)
            coupling = [(left_map[u], right_map[v]) for u, v in selected.pairs]
            current.add_edges_from(coupling)
            expected_lower = (
                left_result.omitted_lower_bound + right_result.omitted_lower_bound
            )
            expected_upper = (
                left_result.omitted_upper_bound + right_result.omitted_upper_bound
            )
            incumbent = direct_solution(current, carried)
            self.assertTrue(nx.check_planarity(incumbent)[0])
            self.assertTrue(all(incumbent.has_edge(*edge) for edge in coupling))
            self.assertEqual(current.number_of_nodes(), 5 * (stage + 1))
        self.assertGreaterEqual(expected_lower, 5)

    def test_heuristic_reuses_four_successive_nonplanar_joins(self):
        self._five_piece_sequence()

    def test_exact_reuses_four_successive_nonplanar_joins(self):
        self._five_piece_sequence(exact=True)

    def test_relabeling_is_bijective_and_does_not_mutate_the_old_result(self):
        graph = nx.complete_graph(5)
        result = ogdf_planar_subgraph(graph, threads=1)
        mapping = {node: f"bus_{node}" for node in graph}
        piece = relabel_planar_piece(graph, result, mapping)
        self.assertEqual(set(piece.graph), set(mapping.values()))
        self.assertEqual(piece.lower_bound, result.omitted_lower_bound)
        piece.graph.remove_node("bus_0")
        piece.planar_subgraph.remove_node("bus_0")
        self.assertEqual(len(graph), 5)
        self.assertEqual(len(result.planar_subgraph), 5)
        for invalid in ({0: "x"}, {node: "x" for node in graph}):
            with self.assertRaisesRegex(ValueError, "bijection"):
                relabel_planar_piece(graph, result, invalid)

    def test_open_certificate_remains_usable_without_claiming_optimality(self):
        graph = nx.complete_graph(5)
        # A spanning path has six omissions, but only one is certified necessary.
        pieces = (CertifiedPlanarPiece(graph, nx.path_graph(5), 1),)
        result = ogdf_planar_subgraph(graph, threads=1, certified_pieces=pieces)
        self.assertEqual(result.shared_piece_lower_bound, 1)
        self.assertEqual(result.omitted_upper_bound, 1)
        self.assertTrue(result.exact)  # Proven by the new incumbent, not assumed.

    def test_reused_heuristic_rejects_invalid_certificate_even_on_planar_fast_path(
        self,
    ):
        graph = nx.cycle_graph(3)
        invalid = CertifiedPlanarPiece(graph, nx.path_graph(3), 1)
        with self.assertRaisesRegex(ValueError, "contradicts"):
            ogdf_planar_subgraph(graph, threads=1, certified_pieces=(invalid,))


if __name__ == "__main__":
    unittest.main()
