from __future__ import annotations

import unittest
from unittest import mock

import networkx as nx
import numpy as np

from planar_instance_generation.core import CouplingBranchConfig
from planar_instance_generation.evaluation import power_generation
from planar_instance_generation.evaluation import (
    FeasibleSource,
    extract_power_pieces,
    merge_power_pieces,
    visual_merge_stages,
)
from planar_instance_generation.gates import DcOpfResult, DcOpfSolution
from planar_instance_generation.io import PglibCase


def _case(name: str) -> PglibCase:
    bus = np.asarray(
        [
            [1, 3, 0, 0, 0, 0, 1, 1, 0, 100, 1, 1.1, 0.9],
            [2, 1, 5, 0, 0, 0, 1, 1, 0, 100, 1, 1.1, 0.9],
            [3, 1, 5, 0, 0, 0, 1, 1, 0, 100, 1, 1.1, 0.9],
        ],
        dtype=float,
    )
    gen = np.asarray([[1, 10, 0, 10, -10, 1, 100, 1, 20, 0]], dtype=float)
    branch = np.asarray(
        [
            [1, 2, 0, 0.1, 0, 10, 10, 10, 0, 0, 1, -360, 360],
            [2, 3, 0, 0.1, 0, 10, 10, 10, 0, 0, 1, -360, 360],
        ],
        dtype=float,
    )
    cost = np.asarray([[2, 0, 0, 2, 1, 0]], dtype=float)
    return PglibCase(name, "2", 100.0, bus, gen, branch, cost)


def _source(name: str) -> FeasibleSource:
    case = _case(name)
    solution = DcOpfSolution(
        bus_angles=np.zeros(3),
        branch_from_flows=np.asarray([10.0, 5.0]),
        branch_to_flows=np.asarray([-10.0, -5.0]),
        generator_dispatch=np.asarray([10.0]),
    )
    return FeasibleSource(case, solution, 10.0, 0.1, 0.05)


class _FeasibleGate:
    def solve(self, case: PglibCase, /) -> DcOpfResult:
        flows = np.arange(1, case.branch.shape[0] + 1, dtype=float)
        return DcOpfResult(
            True,
            "FEASIBLE",
            12.0,
            0.2,
            DcOpfSolution(
                np.zeros(case.bus.shape[0]),
                flows,
                -flows,
                np.zeros(case.gen.shape[0]),
            ),
            0.1,
        )


class PowerGenerationTests(unittest.TestCase):
    def test_extraction_diagnostics_keep_original_demand_denominator(self):
        sources = (_source("diagnostic"), _source("other"))
        piece = extract_power_pieces(sources, target_nodes=(2, 2), seed=7)[0]
        stats = piece.statistics()
        expected_demand = sum(max(0, row[2]) for row in sources[0].case.bus
                              if int(row[0]) in piece.subcase.selected_buses)
        expected_compensation = sum(abs(v) for v in piece.subcase.supply_compensation_mw.values())
        self.assertEqual(stats["retained_original_positive_demand_mw"], expected_demand)
        self.assertEqual(stats["absolute_compensation_mw"], expected_compensation)
        self.assertEqual(stats["compensation_generator_capacity_mw"], expected_compensation)
        self.assertEqual(stats["compensation_to_original_demand_ratio"], expected_compensation / expected_demand)

    def setUp(self) -> None:
        self.sources = tuple(_source(f"source_{index}") for index in range(1, 4))

    def test_extracts_once_for_reuse_across_strategies(self) -> None:
        first = extract_power_pieces(self.sources, 1.0, seed=7)
        second = extract_power_pieces(self.sources, [1.0, 1.0, 1.0], seed=7)

        self.assertEqual([piece.seed for piece in first], [piece.seed for piece in second])
        self.assertEqual(
            [piece.subcase.selected_buses for piece in first],
            [piece.subcase.selected_buses for piece in second],
        )
        self.assertEqual([piece.subcase.case.name for piece in first], ["piece_001", "piece_002", "piece_003"])

    def test_sequential_merge_records_timings_gate_and_coupling_flows(self) -> None:
        pieces = extract_power_pieces(self.sources, 1.0, seed=4)
        candidate = merge_power_pieces(
            pieces,
            _FeasibleGate(),
            strategy="high_degree",
            seed=9,
            coupling=CouplingBranchConfig(rate_mva=100.0),
        )

        self.assertTrue(candidate.feasible)
        self.assertEqual(len(candidate.stages), 2)
        self.assertEqual(candidate.case.bus.shape[0], 9)
        self.assertEqual(
            candidate.final_merge.coupling_branch_rows,
            (4, 5, 6, 9, 10, 11),
        )
        self.assertEqual(candidate.coupling_flows.coupling_branch_count, 6)
        self.assertEqual(candidate.statistics()["branch_count"], 12)
        self.assertEqual(candidate.coupling_flows.finite_rating_count, 6)
        self.assertGreater(candidate.coupling_flows.utilization_max or 0.0, 0.0)
        self.assertGreaterEqual(candidate.statistics()["merge_node_selection_seconds"], 0.0)
        self.assertEqual(candidate.stages[1].statistics()["left_bus_count"], 6)
        self.assertEqual(candidate.stages[0].statistics()["requested_pair_count"], 3)
        self.assertEqual(candidate.stages[0].statistics()["pair_count"], 3)

    def test_visual_sequence_preserves_original_source_colors(self) -> None:
        pieces = extract_power_pieces(self.sources, 1.0, seed=2)
        candidate = merge_power_pieces(
            pieces,
            _FeasibleGate(),
            strategy="random",
            seed=3,
        )

        stages = visual_merge_stages(candidate)

        self.assertEqual(len(stages), 2)
        sources = {data["source_graph"] for _, data in stages[-1].final_graph.nodes(data=True)}
        self.assertEqual(sources, {"G_1", "G_2", "G_3"})
        edge_sources = {
            data.get("source_graph")
            for _, _, data in stages[-1].final_graph.edges(data=True)
            if not data.get("merge_edge")
        }
        self.assertEqual(edge_sources, {"G_1", "G_2", "G_3"})

    def test_planar_sources_use_exact_strategy_without_gurobi(self) -> None:
        pieces = extract_power_pieces(self.sources[:2], 1.0, seed=1)
        candidate = merge_power_pieces(
            pieces,
            _FeasibleGate(),
            strategy="exact_outer_face",
            seed=1,
            outer_face_sampling="consecutive",
            outer_face_prefer_non_leaf=True,
        )

        stage = candidate.stages[0]
        self.assertEqual(stage.selection.strategy, "exact_outer_face")
        self.assertTrue(stage.selection.left_planarization.exact)
        self.assertTrue(stage.selection.right_planarization.exact)
        statistics = stage.statistics()
        self.assertEqual(statistics["outer_face_sampling"], "consecutive")
        self.assertTrue(statistics["outer_face_prefer_non_leaf"])
        self.assertEqual(statistics["left_face_non_leaf_candidates"], 1)
        self.assertEqual(statistics["right_face_non_leaf_candidates"], 1)
        self.assertEqual(statistics["left_selected_leaf_count"], 2)
        self.assertEqual(statistics["right_selected_leaf_count"], 2)

    def test_both_face_strategies_reuse_certificates_across_five_power_pieces(self) -> None:
        pieces = extract_power_pieces(tuple(_source(f"source_{i}") for i in range(5)), 1.0, seed=4)
        for strategy in ("heuristic_outer_face", "exact_outer_face"):
            with self.subTest(strategy=strategy):
                candidate = merge_power_pieces(
                    pieces, _FeasibleGate(), strategy=strategy, seed=9,

                )
                self.assertEqual(len(candidate.stages), 4)
                for index, stage in enumerate(candidate.stages):
                    left = stage.selection.left_planarization
                    right = stage.selection.right_planarization
                    self.assertEqual(left.shared_piece_count, 0 if index == 0 else 2)
                    self.assertEqual(right.shared_piece_count, 0)
                    self.assertEqual((left.omitted_lower_bound, left.omitted_upper_bound), (0, 0))
                    self.assertGreaterEqual(stage.selection.selection_seconds,
                                            left.total_runtime_seconds + right.total_runtime_seconds)
                    if index:
                        self.assertEqual(set(left.planar_subgraph),
                                         set(candidate.stages[index - 1].merge.case.topology()))
                        key = "left_planarization_shared_piece_count" if strategy == "exact_outer_face" else "left_shared_piece_count"
                        self.assertEqual(stage.statistics()[key], 2)
                self.assertEqual(candidate.case.bus.shape[0], 15)

    def test_exact_strategy_reuses_heuristic_iteration_count(self) -> None:
        sentinel = object()
        with mock.patch.object(
            __import__("planar_instance_generation.core.selection", fromlist=["select_exact_outer_face_pairs"]),
            "select_exact_outer_face_pairs",
            return_value=sentinel,
        ) as selector:
            selected = power_generation._select_pairs(
                nx.path_graph(5),
                nx.path_graph(4),
                3,
                strategy="exact_outer_face",
                seed=19,
                exact_time_limit_seconds=2.0,
                exact_threads=1,

                log_directory=None,
                outer_face_prefer_non_leaf=True,
            )

        self.assertIs(selected, sentinel)
        self.assertNotIn("warm_start_iterations", selector.call_args.kwargs)
        self.assertTrue(
            selector.call_args.kwargs["prefer_non_leaf"],
        )


    def test_heuristic_strategy_records_both_timed_planarizations(self) -> None:
        pieces = extract_power_pieces(self.sources[:2], 1.0, seed=1)
        candidate = merge_power_pieces(
            pieces,
            _FeasibleGate(),
            strategy="heuristic_outer_face",
            seed=1,

        )

        stage = candidate.stages[0]
        statistics = stage.statistics()
        self.assertEqual(stage.selection.strategy, "heuristic_outer_face")
        self.assertEqual(statistics["iterations_per_graph"], 10)
        self.assertIn("left_total_runtime_seconds", statistics)
        self.assertIn("right_total_runtime_seconds", statistics)
        self.assertIn("left_omitted_lower_bound", statistics)
        self.assertIn("right_omitted_upper_bound", statistics)
        self.assertGreaterEqual(statistics["left_face_candidates"], 2)

    def test_validates_shapes_and_strategy(self) -> None:
        pieces = extract_power_pieces(self.sources, 1.0, seed=0)
        with self.assertRaisesRegex(ValueError, "unknown"):
            merge_power_pieces(
                pieces,
                _FeasibleGate(),
                strategy="bad",  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "at least two"):
            extract_power_pieces(self.sources[:1], 1.0)


if __name__ == "__main__":
    unittest.main()
