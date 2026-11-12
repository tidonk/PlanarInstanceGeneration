from __future__ import annotations

import unittest

import networkx as nx
from planar_instance_generation.core.ogdf import static_euler_rows

from planar_instance_generation.core.planarization import (
    exact_outer_face_pairs,
    largest_outer_face,
    maximum_planar_subgraph,
    select_exact_outer_face_pairs,
    select_heuristic_outer_face_pairs,
)


class PlanarizationTests(unittest.TestCase):

    def test_euler_subset_separator_finds_triangle_and_triangle_free_cuts(self) -> None:
        for graph, expected_rhs in (
            (nx.complete_graph(7), 6),
            (nx.complete_bipartite_graph(3, 3), 1),
        ):
            cuts = static_euler_rows(graph)
            self.assertTrue(cuts)
            indices, rhs = cuts[0]
            self.assertEqual(rhs, expected_rhs)
            self.assertEqual(len(indices), graph.number_of_edges())

    def test_planar_fast_path_does_not_need_gurobi(self) -> None:
        graph = nx.complete_graph(4)
        graph.nodes[0]["source"] = "kept"
        graph.edges[0, 1]["capacity"] = 7.0

        result = maximum_planar_subgraph(graph, time_limit_seconds=0.1)

        self.assertTrue(result.exact)
        self.assertEqual(result.skewness, 0)
        self.assertEqual(result.omitted_edges, ())
        self.assertEqual(result.termination_status, "OPTIMAL")
        self.assertIsNone(result.optimizer)
        self.assertEqual(result.planar_subgraph.nodes[0]["source"], "kept")
        self.assertEqual(result.planar_subgraph.edges[0, 1]["capacity"], 7.0)

    def test_exact_solver_uses_default_ogdf_without_custom_passes(self):
        result = maximum_planar_subgraph(nx.complete_graph(5), seed=17, threads=1)
        self.assertEqual(result.solver_metadata["heuristic_runs"], 10)
        self.assertEqual(result.solver_metadata["extra_greedy_passes"], 0)
        self.assertLessEqual(
            result.omitted_upper_bound, result.greedy_start_upper_bound
        )

    def test_largest_outer_face_deduplicates_a_boundary_walk(self) -> None:
        graph = nx.path_graph(["a", "b", "c", "d"])
        face = largest_outer_face(graph)
        self.assertEqual(set(face.candidates), set(graph))
        self.assertEqual(len(face.candidates), 4)
        self.assertGreater(len(face.boundary_walk), len(face.candidates))

    def test_outer_face_pairs_are_distinct_reproducible_and_planar(self) -> None:
        left = nx.relabel_nodes(nx.cycle_graph(8), lambda node: ("L", node))
        right = nx.relabel_nodes(nx.cycle_graph(7), lambda node: ("R", node))

        first = exact_outer_face_pairs(left, right, 5, seed=12)
        second = exact_outer_face_pairs(left, right, 5, seed=12)

        self.assertEqual(first.pairs, second.pairs)
        self.assertEqual(first.sampling, "consecutive")
        self.assertEqual(len({edge[0] for edge in first.pairs}), 5)
        self.assertEqual(len({edge[1] for edge in first.pairs}), 5)
        combined = nx.compose(left, right)
        combined.add_edges_from(first.pairs)
        self.assertTrue(nx.check_planarity(combined)[0])
        self.assertGreaterEqual(first.selection_seconds, 0.0)

    def test_outer_face_pairing_accepts_overlapping_source_identifiers(self) -> None:
        result = exact_outer_face_pairs(nx.cycle_graph(5), nx.path_graph(5), 3, seed=2)
        self.assertEqual(len(result.pairs), 3)

    def test_consecutive_outer_face_sampling_selects_cyclic_arcs(self) -> None:
        left = nx.relabel_nodes(nx.cycle_graph(11), lambda node: ("L", node))
        right = nx.relabel_nodes(nx.cycle_graph(13), lambda node: ("R", node))

        result = exact_outer_face_pairs(left, right, 5, seed=12, sampling="consecutive")

        selected_sides = (
            (tuple(left for left, _ in result.pairs), result.left_face.candidates),
            (
                tuple(reversed([right for _, right in result.pairs])),
                result.right_face.candidates,
            ),
        )
        for selected, face in selected_sides:
            positions = [face.index(node) for node in selected]
            self.assertTrue(
                all(
                    (second - first) % len(face) == 1
                    for first, second in zip(positions, positions[1:])
                )
            )
        self.assertEqual(result.sampling, "consecutive")

    def test_non_leaf_preference_uses_leaves_only_after_exhaustion(self) -> None:
        left = nx.relabel_nodes(nx.path_graph(6), lambda node: ("L", node))
        right = nx.relabel_nodes(nx.path_graph(6), lambda node: ("R", node))

        non_leaf_only = exact_outer_face_pairs(
            left,
            right,
            4,
            seed=12,
            sampling="consecutive",
            prefer_non_leaf=True,
        )
        self.assertTrue(non_leaf_only.prefer_non_leaf)
        self.assertEqual(non_leaf_only.left_non_leaf_candidate_count, 4)
        self.assertEqual(non_leaf_only.right_non_leaf_candidate_count, 4)
        self.assertEqual(non_leaf_only.left_selected_leaf_count, 0)
        self.assertEqual(non_leaf_only.right_selected_leaf_count, 0)
        self.assertTrue(all(left.degree[node] != 1 for node, _ in non_leaf_only.pairs))
        self.assertTrue(all(right.degree[node] != 1 for _, node in non_leaf_only.pairs))

        with_fallback = exact_outer_face_pairs(
            left,
            right,
            5,
            seed=12,
            sampling="consecutive",
            prefer_non_leaf=True,
        )
        self.assertEqual(with_fallback.left_selected_leaf_count, 1)
        self.assertEqual(with_fallback.right_selected_leaf_count, 1)

    def test_adaptive_skip_one_uses_all_available_balanced_gaps(self) -> None:
        for boundary_size, count, expected_skips in (
            (12, 4, 4),
            (7, 5, 2),
            (5, 5, 0),
        ):
            left = nx.relabel_nodes(
                nx.cycle_graph(boundary_size), lambda node: ("L", node)
            )
            right = nx.relabel_nodes(
                nx.cycle_graph(boundary_size), lambda node: ("R", node)
            )
            result = exact_outer_face_pairs(
                left, right, count, seed=12, sampling="adaptive_skip_one"
            )
            selected = tuple(node for node, _ in result.pairs)
            face = result.left_face.candidates
            positions = [face.index(node) for node in selected]
            skipped = [
                (second - first) % len(face) - 1
                for first, second in zip(
                    positions, (*positions[1:], positions[0]), strict=True
                )
            ]
            self.assertEqual(sum(value > 0 for value in skipped), expected_skips)
            self.assertTrue(all(value >= 0 for value in skipped))
            self.assertEqual(len(set(selected)), count)
            self.assertEqual(result.sampling, "adaptive_skip_one")

    def test_outer_face_sampling_rejects_unknown_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "sampling must be"):
            exact_outer_face_pairs(
                nx.cycle_graph(5), nx.cycle_graph(5), 3, sampling="unknown"
            )

    def test_end_to_end_selector_includes_planarization_times(self) -> None:
        result = select_exact_outer_face_pairs(
            nx.cycle_graph(6), nx.path_graph(5), 3, time_limit_seconds=0.1, seed=4
        )
        self.assertEqual(result.strategy, "exact_outer_face")
        self.assertEqual(len(result.pairs), 3)
        self.assertTrue(result.left_planarization.exact)
        self.assertTrue(result.right_planarization.exact)
        self.assertGreaterEqual(
            result.selection_seconds, result.faces.selection_seconds
        )

    def test_heuristic_face_selector_is_disjoint_noncrossing_and_fixed_iteration(
        self,
    ) -> None:
        result = select_heuristic_outer_face_pairs(
            nx.complete_graph(5),
            nx.complete_bipartite_graph(3, 3),
            3,
            seed=11,
        )
        self.assertEqual(result.strategy, "heuristic_outer_face")
        self.assertEqual(result.iterations_per_graph, 10)
        self.assertEqual(result.left_planarization.iteration_limit, 10)
        self.assertEqual(len({left for left, _ in result.pairs}), 3)
        self.assertEqual(len({right for _, right in result.pairs}), 3)
        self.assertGreaterEqual(
            result.selection_seconds,
            result.left_planarization.total_runtime_seconds
            + result.right_planarization.total_runtime_seconds,
        )
        certificate = nx.compose(
            nx.relabel_nodes(
                result.left_planarization.planar_subgraph,
                lambda node: ("left", node),
            ),
            nx.relabel_nodes(
                result.right_planarization.planar_subgraph,
                lambda node: ("right", node),
            ),
        )
        certificate.add_edges_from(
            (("left", left), ("right", right)) for left, right in result.pairs
        )
        self.assertTrue(nx.check_planarity(certificate)[0])
        self.assertTrue(result.statistics()["planarity_certificate"])

    def test_outer_face_pair_count_is_strict(self) -> None:
        left = nx.relabel_nodes(nx.path_graph(3), lambda node: ("L", node))
        right = nx.relabel_nodes(nx.path_graph(2), lambda node: ("R", node))
        with self.assertRaisesRegex(
            ValueError, "requested 3 distinct boundary nodes.*found 3 and 2"
        ):
            exact_outer_face_pairs(left, right, 3)

        with self.assertRaisesRegex(
            ValueError, "requested 3 distinct boundary nodes.*found 3 and 1"
        ):
            exact_outer_face_pairs(left, nx.empty_graph(1), 3)

    def test_rejects_nonplanar_face_and_unsupported_graphs(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a planar"):
            largest_outer_face(nx.complete_graph(5))
        with self.assertRaises(TypeError):
            maximum_planar_subgraph(nx.MultiGraph([(0, 1)]))
        loop = nx.Graph()
        loop.add_edge(0, 0)
        with self.assertRaisesRegex(ValueError, "self-loops"):
            maximum_planar_subgraph(loop)
        for options in (
            {"initial_witness_rounds": 1},
            {"root_euler_subset_cuts": True},
            {"lifted_euler_cuts": True},
            {"root_fractional_witness_cuts": True},
        ):
            with self.subTest(options=options), self.assertRaises(TypeError):
                maximum_planar_subgraph(nx.complete_graph(4), **options)
        for options in ({"time_limit_seconds": 0}, {"threads": 0}, {"seed": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                maximum_planar_subgraph(nx.complete_graph(4), **options)

    def test_statistics_are_flat_and_include_certified_bounds(self) -> None:
        result = maximum_planar_subgraph(nx.cycle_graph(5), time_limit_seconds=0.1)
        statistics = result.statistics()
        self.assertEqual(statistics["omitted_lower_bound"], 0)
        self.assertEqual(statistics["omitted_upper_bound"], 0)
        self.assertEqual(statistics["absolute_gap"], 0)
        self.assertTrue(statistics["exact"])


if __name__ == "__main__":
    unittest.main()
