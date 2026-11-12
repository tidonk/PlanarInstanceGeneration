from types import SimpleNamespace
import unittest

import networkx as nx
from planar_instance_generation.core.ogdf import direct_solution

from planar_instance_generation.core.planarization import (
    CertifiedPlanarPiece,
    _piece_forest_bound,
    maximum_planar_subgraph,
)
from planar_instance_generation.evaluation.reporting import (
    SkewnessBounds,
    SkewnessObservation,
    propagate_composition_bounds,
    _add_inflation,
)
from check.scripts.refresh_composition_bounds import refresh_row
from check.scripts.build_partial_pglib_report import _skewness_rows as partial_rows


class CouplingForestTests(unittest.TestCase):
    def test_partial_report_closes_bridge_only_composition_and_keeps_raw_bound(self):
        cohort = SimpleNamespace(
            index=1,
            source_stems=("a", "b", "c"),
            target_size=9,
            rows=tuple(
                dict(artifact_stem=str(i), coupling_branch_count=2) for i in range(5)
            ),
        )
        results = {
            f"source:1:{stem}": dict(skewness_lower_bound=0, skewness_upper_bound=0)
            for stem in cohort.source_stems
        }
        results.update(
            {
                f"final:1:{i}": dict(
                    skewness_lower_bound=0,
                    skewness_upper_bound=12,
                    skewness_exact=False,
                    absolute_gap=12,
                )
                for i in range(5)
            }
        )
        for row in partial_rows((cohort,), results, 4):
            self.assertEqual(row["skewness_upper_bound"], 0)
            self.assertTrue(row["skewness_exact"])
            self.assertEqual(row["absolute_gap"], 0)
            self.assertEqual(row["inflation_upper_bound"], 0)
            self.assertEqual(row["solver_upper_bound"], 12)

    def test_five_pieces_keep_four_bridges_even_with_nonmaximum_planarizations(self):
        graph = nx.Graph()
        pieces = []
        for i in range(5):
            nodes = range(5 * i, 5 * i + 5)
            piece = nx.complete_graph(nodes)
            # A path is deliberately not a maximum planar subgraph of K5.
            planar = nx.path_graph(nodes)
            graph = nx.compose(graph, piece)
            pieces.append(CertifiedPlanarPiece(piece, planar, 1))
            if i:
                graph.add_edges_from((5 * (i - 1) + j, 5 * i + j) for j in range(3))
        candidate = direct_solution(graph, tuple(pieces))
        count, upper = _piece_forest_bound(graph, pieces)
        self.assertEqual(count, 4)
        self.assertEqual(upper, 5 * 6 + 12 - 4)
        self.assertTrue(nx.is_connected(candidate))
        self.assertTrue(nx.check_planarity(candidate)[0])
        self.assertLessEqual(
            graph.number_of_edges() - candidate.number_of_edges(), upper
        )

    def test_disconnected_composite_does_not_subtract_p_minus_one(self):
        graph = nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3))
        pieces = tuple(
            CertifiedPlanarPiece(
                graph.subgraph(nodes).copy(), graph.subgraph(nodes).copy(), 0
            )
            for nodes in (range(3), range(3, 6))
        )
        self.assertEqual(_piece_forest_bound(graph, pieces), (0, 0))
        self.assertEqual(direct_solution(graph, pieces).number_of_edges(), 6)

    def test_reporting_tightens_total_and_added_bounds_without_renormalizing(self):
        raw = SkewnessBounds(0, 30, "TIME_LIMIT", 300)
        observation = SkewnessObservation(raw, 5, 8, 12, source_piece_count=5)
        bounded = propagate_composition_bounds(raw, observation)
        self.assertEqual((bounded.lower_bound, bounded.upper_bound), (5, 16))
        row = {}
        _add_inflation(row, bounded, observation)
        self.assertEqual(row["inflation_upper_bound"], 8)
        self.assertEqual(row["normalized_inflation_upper_bound"], 8 / 12)
        self.assertEqual(bounded.runtime_seconds, 300)
        with self.assertRaises(ValueError):
            SkewnessObservation(raw, 0, 0, 2, source_piece_count=5)

    def test_report_refresh_is_idempotent_and_keeps_solver_bounds(self):
        row = dict(
            skewness_lower_bound=0,
            skewness_upper_bound=30,
            status="TIME_LIMIT",
            source_floor_lower_bound=5,
            source_floor_upper_bound=8,
            coupling_edge_count=12,
            runtime_seconds=300,
        )
        result = refresh_row(row, 5)
        self.assertEqual(result["solver_upper_bound"], 30)
        self.assertEqual(result["skewness_upper_bound"], 16)
        self.assertEqual(result["runtime_seconds"], 300)
        self.assertEqual(refresh_row(result, 5), result)

    def test_solver_receives_a_feasible_start_within_the_forest_bound(self):
        graph = nx.complete_graph(6)
        pieces = tuple(
            CertifiedPlanarPiece(
                graph.subgraph(nodes).copy(), graph.subgraph(nodes).copy(), 0
            )
            for nodes in (range(3), range(3, 6))
        )
        result = maximum_planar_subgraph(
            graph, certified_pieces=pieces, time_limit_seconds=10, threads=1
        )
        self.assertTrue(nx.check_planarity(result.planar_subgraph)[0])
        self.assertTrue(nx.is_connected(result.planar_subgraph))
        self.assertLessEqual(result.omitted_upper_bound, result.piece_start_upper_bound)
        self.assertLessEqual(
            result.piece_start_upper_bound, result.composition_upper_bound
        )
        self.assertEqual(result.coupling_forest_edge_count, 1)
        self.assertEqual(result.omitted_upper_bound, 3)
