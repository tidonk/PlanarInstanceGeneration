"""Regression checks for the production OGDF integration."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx

from planar_instance_generation.core.ogdf import bound_rows, direct_solution
from planar_instance_generation.core.planarization import (
    CertifiedPlanarPiece,
    maximum_planar_subgraph,
    ogdf_planar_subgraph,
    relabel_planar_piece,
    select_exact_outer_face_pairs,
    select_heuristic_outer_face_pairs,
)
from planar_instance_generation.evaluation.planarization_batch import (
    PlanarizationTask,
    solve_planarization_batch,
)


class OgdfProductionTests(unittest.TestCase):
    def test_selectors_accept_the_largest_supported_seed(self):
        for select in (
            select_exact_outer_face_pairs,
            select_heuristic_outer_face_pairs,
        ):
            result = select(
                nx.cycle_graph(4), nx.cycle_graph(4), 2, seed=2**31 - 1, threads=1
            )
            self.assertEqual(len(result.pairs), 2)
            self.assertEqual(result.left_planarization.seed, 2**31 - 1)
            self.assertEqual(result.right_planarization.seed, 0)

    def test_two_euler_families_can_select_different_triangle_free_subsets(self):
        from planar_instance_generation.core.ogdf import static_euler_rows

        graph = nx.complete_bipartite_graph(6, 6)
        core = frozenset(map(frozenset, graph.edges))
        graph.add_edges_from([(12, 0), (12, 1)])
        self.assertTrue(nx.is_biconnected(graph))
        self.assertFalse(any(nx.triangles(graph).values()))
        rows = {
            frozenset(map(frozenset, support)): lower
            for support, lower in static_euler_rows(graph)
        }
        self.assertEqual(rows, {frozenset(map(frozenset, graph.edges)): 16, core: 6})

    def test_peeling_keeps_triangle_free_subset_after_a_tied_invalid_score(self):
        from planar_instance_generation.core.ogdf import static_euler_rows

        graph = nx.complete_bipartite_graph(3, 3)
        core_edges = frozenset(map(frozenset, graph.edges))
        # The added degree-two vertex makes a triangle. The full graph and
        # K3,3 have the same coefficient-2 score, but only K3,3 permits that bound.
        graph.add_edges_from([(6, 0), (6, 3)])
        self.assertTrue(nx.is_biconnected(graph))
        rows = static_euler_rows(graph)
        self.assertEqual(len(rows), 1)
        support, lower = rows[0]
        self.assertEqual(frozenset(map(frozenset, support)), core_edges)
        self.assertEqual(lower, 1)
        self.assertEqual(static_euler_rows(nx.complete_graph(4)), ())

    def test_peeling_matches_exhaustive_checks_of_every_visited_subset(self):
        from planar_instance_generation.core.ogdf import static_euler_rows

        # This reference recomputes planarity-family eligibility from scratch.
        for graph in list(nx.graph_atlas_g()) + [nx.gnp_random_graph(30, 0.25, seed=7)]:
            indices = {v: i for i, v in enumerate(graph)}
            edge_indices = {frozenset(e): i for i, e in enumerate(graph.edges)}
            remaining = graph.copy()
            scores, best = {2: 0, 3: 0}, {}
            while len(remaining) >= 3:
                for c in scores:
                    if c == 2 and any(nx.triangles(remaining).values()):
                        continue
                    score = remaining.number_of_edges() - c * (len(remaining) - 2)
                    if score > scores[c]:
                        scores[c] = score
                        best[c] = tuple(
                            sorted(edge_indices[frozenset(e)] for e in remaining.edges)
                        )
                remaining.remove_node(
                    min(remaining, key=lambda v: (remaining.degree[v], indices[v]))
                )
            expected = {}
            for c, ids in best.items():
                expected[ids] = max(expected.get(ids, 0), scores[c])
            actual = {
                tuple(sorted(edge_indices[frozenset(e)] for e in support)): lower
                for support, lower in static_euler_rows(graph)
            }
            self.assertEqual(actual, expected)

    def test_static_rows_use_valid_euler_coefficient_and_preserve_input(self):
        from planar_instance_generation.core.ogdf import static_euler_rows

        for graph in (
            nx.complete_graph(7),
            nx.complete_bipartite_graph(4, 4),
            nx.petersen_graph(),
            nx.barbell_graph(7, 10),
        ):
            original = list(graph.edges)
            rows = static_euler_rows(graph)
            self.assertLessEqual(len(rows), 2)
            for support, lower in rows:
                induced = graph.subgraph({v for edge in support for v in edge})
                coefficients = (3,) if any(nx.triangles(induced).values()) else (2, 3)
                self.assertIn(
                    lower, {len(support) - c * (len(induced) - 2) for c in coefficients}
                )
                self.assertEqual(
                    set(map(frozenset, support)), set(map(frozenset, induced.edges))
                )
                self.assertGreater(lower, 0)
            self.assertEqual(list(graph.edges), original)

    def test_duplicate_bound_edges_are_rejected(self):
        graph = nx.complete_graph(5)
        kept = graph.copy()
        kept.remove_edge(0, 1)
        piece = CertifiedPlanarPiece(graph, kept, 1, ((((0, 1), (1, 0)), 1),))
        with self.assertRaisesRegex(ValueError, "duplicate edges"):
            bound_rows(graph, prior=(piece,))

    def test_invalid_cached_edge_indices_trigger_a_fresh_solve(self):
        import json

        graph = nx.complete_bipartite_graph(4, 4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = PlanarizationTask(graph, 7, root / "solve.log", "indices")
            settings = dict(
                time_limit_seconds=10,
                solver_threads=1,
                workers=1,
                checkpoint_directory=root,
            )
            solve_planarization_batch([task], **settings)
            (checkpoint,) = root.glob("*.json")
            payload = json.loads(checkpoint.read_text())
            payload["result"]["omitted_edge_indices"][0] = -1
            checkpoint.write_text(json.dumps(payload))
            with patch(
                "planar_instance_generation.evaluation.planarization_batch.maximum_planar_subgraph",
                wraps=maximum_planar_subgraph,
            ) as solve:
                solve_planarization_batch([task], **settings)
            self.assertEqual(solve.call_count, 1)

    def test_static_bounds_are_common_and_triangle_free_bound_is_used(self):
        graph = nx.complete_bipartite_graph(4, 4)
        heuristic = ogdf_planar_subgraph(graph, threads=1)
        exact = maximum_planar_subgraph(graph, threads=1, time_limit_seconds=10)
        self.assertEqual(heuristic.omitted_lower_bound, 4)
        self.assertEqual((exact.omitted_lower_bound, exact.omitted_upper_bound), (4, 4))
        for result in (heuristic, exact):
            self.assertLessEqual(result.solver_metadata["static_euler_rows"], 2)
            self.assertEqual(result.solver_metadata["extra_greedy_passes"], 0)
            self.assertEqual(result.solver_metadata["heuristic_runs"], 10)

    def test_direct_solution_falls_back_to_a_forest_without_extra_passes(self):
        graph = nx.complete_bipartite_graph(3, 3)
        pieces = tuple(
            CertifiedPlanarPiece(
                graph.subgraph([v]).copy(), graph.subgraph([v]).copy(), 0
            )
            for v in graph
        )
        direct = direct_solution(graph, pieces)
        self.assertEqual(direct.number_of_edges(), len(graph) - 1)
        self.assertTrue(nx.is_tree(direct))

    def test_fresh_result_wins_ties_and_direct_result_can_improve_it(self):
        graph = nx.complete_graph(5)
        direct = graph.copy()
        direct.remove_edge(0, 1)
        fresh = graph.copy()
        fresh.remove_edge(2, 3)
        piece = CertifiedPlanarPiece(graph, direct, 1)
        with patch(
            "planar_instance_generation.core.ogdf._run",
            return_value=(fresh, {"runs": 10}),
        ):
            tied = ogdf_planar_subgraph(graph, certified_pieces=(piece,))
        self.assertEqual(
            set(map(frozenset, tied.planar_subgraph.edges)),
            set(map(frozenset, fresh.edges)),
        )
        worse = fresh.copy()
        worse.remove_edge(0, 2)
        with patch(
            "planar_instance_generation.core.ogdf._run",
            return_value=(worse, {"runs": 10}),
        ):
            improved = maximum_planar_subgraph(graph, certified_pieces=(piece,))
        self.assertEqual(improved.omitted_upper_bound, 1)
        self.assertIsNone(improved.optimizer)
        self.assertEqual(improved.greedy_start_upper_bound, 2)

    def test_disjoint_block_bounds_survive_mapping_and_checkpoint(self):
        graph = nx.disjoint_union(
            nx.complete_bipartite_graph(3, 3), nx.complete_bipartite_graph(3, 3)
        )
        graph.add_edge(0, 6)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = PlanarizationTask(graph, 7, root / "solve.log", "blocks")
            settings = dict(
                time_limit_seconds=10,
                solver_threads=1,
                workers=1,
                checkpoint_directory=root,
            )
            result = solve_planarization_batch([task], **settings)[0]
            with patch(
                "planar_instance_generation.evaluation.planarization_batch.maximum_planar_subgraph",
                side_effect=AssertionError("cache was not used"),
            ):
                cached = solve_planarization_batch([task], **settings)[0]
            self.assertEqual(cached.bound_rows, result.bound_rows)
            self.assertEqual(
                (cached.omitted_lower_bound, cached.omitted_upper_bound), (2, 2)
            )
            labels = {v: f"bus_{v}" for v in graph}
            piece = relabel_planar_piece(graph, cached, labels)
            mapped = nx.relabel_nodes(graph, labels)
            rows, local, lower, _, _ = bound_rows(mapped, (piece,), (piece, piece))
            self.assertEqual(lower, 2)
            self.assertTrue(
                all(
                    set(map(frozenset, support)) <= set(map(frozenset, mapped.edges))
                    for support, _ in rows
                )
            )
            self.assertFalse(
                any(len(support) == mapped.number_of_edges() for support, _ in local)
            )

    def test_timeout_keeps_connected_feasible_solution_on_every_block(self):
        graph = nx.disjoint_union(nx.petersen_graph(), nx.petersen_graph())
        graph.add_edge(0, 10)
        result = maximum_planar_subgraph(graph, time_limit_seconds=0.001, threads=1)
        self.assertTrue(nx.is_connected(result.planar_subgraph))
        self.assertTrue(nx.check_planarity(result.planar_subgraph)[0])
        self.assertTrue(result.planar_subgraph.has_edge(0, 10))
        self.assertLessEqual(result.omitted_lower_bound, result.omitted_upper_bound)
        self.assertTrue(
            all(
                b["subproblems"] == 0
                for r in result.solver_metadata["exact"]
                for b in r["blocks"]
            )
        )

    def test_nonzero_propagated_rows_survive_cache_serialization(self):
        graph = nx.complete_bipartite_graph(4, 4)
        kept = graph.copy()
        kept.remove_edges_from((i, i + 4) for i in range(4))
        piece = CertifiedPlanarPiece(graph, kept, 4, ((tuple(graph.edges), 4),))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = PlanarizationTask(
                graph, 7, root / "solve.log", "bounds", prior_certificates=(piece,)
            )
            settings = dict(
                time_limit_seconds=10,
                solver_threads=1,
                workers=1,
                checkpoint_directory=root,
            )
            result = solve_planarization_batch([task], **settings)[0]
            with patch(
                "planar_instance_generation.evaluation.planarization_batch.maximum_planar_subgraph",
                side_effect=AssertionError("shared piece was solved again"),
            ):
                cached = solve_planarization_batch([task], **settings)[0]
            self.assertEqual(cached.total_runtime_seconds, result.total_runtime_seconds)
            self.assertEqual(cached.bound_rows, result.bound_rows)

    def test_timeout_statistics_can_be_exported_by_both_evaluators(self):
        import sys

        scripts = str(Path(__file__).resolve().parents[1] / "check/scripts")
        with patch.object(sys, "path", [scripts, *sys.path]):
            from check.scripts.evaluate_pglib_cohort_skewness import (
                _result_row as power_row,
            )
            from check.scripts.evaluate_sndlib_cohort_skewness import (
                _result_row as snd_row,
            )
        result = maximum_planar_subgraph(
            nx.petersen_graph(), time_limit_seconds=0.001, threads=1
        )
        self.assertIsNotNone(result.optimizer)
        for convert in (power_row, snd_row):
            row = convert(result)
            self.assertEqual(row["objective_upper_bound"], result.omitted_upper_bound)
            self.assertLessEqual(row["best_bound"], row["objective_upper_bound"])

    def test_overlapping_bounds_and_invalid_certificates(self):
        graph = nx.complete_graph(5)
        kept = graph.copy()
        kept.remove_edge(0, 1)
        piece = CertifiedPlanarPiece(graph, kept, 1)
        result = ogdf_planar_subgraph(
            graph, prior_certificates=(piece, piece), threads=1
        )
        self.assertEqual(result.omitted_lower_bound, 1)
        with self.assertRaises(ValueError):
            ogdf_planar_subgraph(
                graph, certified_pieces=(replace(piece, lower_bound=2),)
            )


if __name__ == "__main__":
    unittest.main()
