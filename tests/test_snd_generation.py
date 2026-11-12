from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest

import networkx as nx

from planar_instance_generation.core import SndSolution
from planar_instance_generation.evaluation.snd_generation import (
    extract_snd_pieces,
    merge_snd_pieces,
    visual_snd_merge_stages,
)
from planar_instance_generation.evaluation.snd_experiment import run_snd_experiment
from planar_instance_generation.gates import McmcfModelEstimate, McmcfResult
from planar_instance_generation.io import (
    SndlibCapacityModule,
    SndlibDemand,
    SndlibLink,
    SndlibMetadata,
    SndlibNetwork,
    SndlibNode,
)


@dataclass(frozen=True)
class _Source:
    source_name: str
    network: SndlibNetwork
    solution: SndSolution


def _source(
    name: str,
    *,
    unit: str | None = "MBITPERSEC",
    coordinates_type: str = "pixel",
) -> _Source:
    metadata = (
        None
        if unit is None
        else SndlibMetadata("daily", "now", unit, name)
    )
    network = SndlibNetwork(
        version="1.0",
        coordinates_type=coordinates_type,
        nodes=(
            SndlibNode("a", 0, 0),
            SndlibNode("b", 1, 2),
            SndlibNode("c", 3, 1),
        ),
        links=(
            SndlibLink(
                "ab",
                "a",
                "b",
                additional_modules=(SndlibCapacityModule(10, 1),),
                routing_cost=1,
            ),
            SndlibLink(
                "bc",
                "b",
                "c",
                additional_modules=(SndlibCapacityModule(10, 1),),
                routing_cost=1,
            ),
        ),
        demands=(SndlibDemand("d", "a", "c", 2),),
        metadata=metadata,
    )
    return _Source(
        name,
        network,
        SndSolution(
            {"ab": 10, "bc": 10},
            {("d", "ab"): 2, ("d", "bc"): 2},
        ),
    )


class _FeasibleGate:
    def estimate(self, network: SndlibNetwork, /) -> McmcfModelEstimate:
        return McmcfModelEstimate.for_network(network)

    def solve(
        self,
        network: SndlibNetwork,
        installed_capacities: dict[str, float],
        /,
        *,
        source_name: str | None = None,
    ) -> McmcfResult:
        graph = network.topology()
        links_by_endpoints: dict[frozenset[str], str] = {}
        for link in network.links:
            links_by_endpoints[frozenset((link.source, link.target))] = link.id
        link_by_id = {link.id: link for link in network.links}
        flows: dict[tuple[str, str], float] = {}
        for demand in network.demands:
            path = nx.shortest_path(graph, demand.source, demand.target)
            for first, second in zip(path, path[1:]):
                link_id = links_by_endpoints[frozenset((first, second))]
                link = link_by_id[link_id]
                sign = 1 if (link.source, link.target) == (first, second) else -1
                flows[(demand.id, link_id)] = sign * demand.value
        solution = SndSolution(installed_capacities, flows)
        return McmcfResult(
            feasible=True,
            status="OPTIMAL",
            objective=sum(abs(value) for value in flows.values()),
            runtime_seconds=0.01,
            optimizer_seconds=0.005,
            solution=solution,
            estimate=self.estimate(network),
        )


class _ResourceGate(_FeasibleGate):
    def solve(
        self,
        network: SndlibNetwork,
        installed_capacities: dict[str, float],
        /,
        *,
        source_name: str | None = None,
    ) -> McmcfResult:
        return McmcfResult(
            feasible=None,
            status="RESOURCE_SKIPPED_VARIABLE_LIMIT",
            objective=None,
            runtime_seconds=0.0,
            optimizer_seconds=None,
            solution=None,
            estimate=self.estimate(network),
            resource_skipped=True,
            reason="test limit",
        )


class SndGenerationTests(unittest.TestCase):
    def test_both_face_strategies_reuse_certificates_across_five_snd_pieces(self) -> None:
        pieces = extract_snd_pieces(tuple(_source(str(i)) for i in range(5)), 1.0, seed=4)
        for strategy in ("heuristic_outer_face", "exact_outer_face"):
            with self.subTest(strategy=strategy):
                candidate = merge_snd_pieces(
                    pieces, _FeasibleGate(), strategy=strategy, seed=9,
                    source_stratum_size=5,
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
                                         set(candidate.stages[index - 1].merge.instance.topology()))
                        key = "left_planarization_shared_piece_count" if strategy == "exact_outer_face" else "left_shared_piece_count"
                        self.assertEqual(stage.statistics()[key], 2)
                self.assertEqual(len(candidate.instance.nodes), 15)

    def test_sequential_merge_records_units_timings_and_visual_provenance(self) -> None:
        sources = (
            _source("one", coordinates_type="pixel"),
            _source("two", coordinates_type="geographical"),
            _source("three", coordinates_type="pixel"),
        )
        pieces = extract_snd_pieces(sources, 1.0, seed=3)
        candidate = merge_snd_pieces(
            pieces,
            _FeasibleGate(),
            strategy="random",
            seed=4,
            source_stratum_size=3,
        )

        self.assertTrue(candidate.feasible)
        self.assertEqual(len(candidate.stages), 2)
        self.assertEqual(candidate.routing_unit, "MBITPERSEC")
        self.assertEqual(candidate.instance.coordinates_type, "pixel")
        self.assertTrue(candidate.coordinate_type_overridden)
        self.assertEqual(candidate.instance.metadata.unit, "MBITPERSEC")
        self.assertEqual(
            candidate.coupling_flows.positive_capacity_count,
            0,
        )
        self.assertIn("gate_estimate_total_variable_count", candidate.statistics())

        visual = visual_snd_merge_stages(candidate)
        self.assertEqual(len(visual), 2)
        self.assertEqual(
            {
                attributes["source_graph"]
                for _, attributes in visual[-1].final_graph.nodes(data=True)
            },
            {"G_1", "G_2", "G_3"},
        )
        self.assertEqual(
            {
                attributes.get("source_graph")
                for _, _, attributes in visual[-1].final_graph.edges(data=True)
                if not attributes.get("merge_edge")
            },
            {"G_1", "G_2", "G_3"},
        )

    def test_heuristic_stage_exposes_full_runtime_and_bounds(self) -> None:
        pieces = extract_snd_pieces((_source("one"), _source("two")), 1.0)
        candidate = merge_snd_pieces(
            pieces,
            _FeasibleGate(),
            strategy="heuristic_outer_face",

        )
        statistics = candidate.stages[0].statistics()
        self.assertIn("left_total_runtime_seconds", statistics)
        self.assertIn("right_omitted_lower_bound", statistics)
        self.assertIn("right_omitted_upper_bound", statistics)

    def test_shared_extraction_unit_strata_and_tristate_status(self) -> None:
        sources = (
            _source("one", unit="MBITPERSEC"),
            _source("two", unit="MBITPERSEC"),
            _source("bits", unit="BITSPERSEC"),
        )
        result = run_snd_experiment(
            sources,
            lambda *_: _ResourceGate(),
            strategies=("random", "high_degree"),
            requested_per_strategy=1,
            num_subinstances=(3, 3),
            target_size=(9, 9),
            size_range=(1.0, 1.0),
            max_attempts_per_output=2,
            seed=7,
        )

        self.assertFalse(result.accepted)
        self.assertEqual(
            {record.status for record in result.attempts},
            {"gate_resource_skipped", "paired_retry"},
        )
        for attempt_index in {record.attempt_index for record in result.attempts}:
            records = [
                record
                for record in result.attempts
                if record.attempt_index == attempt_index
            ]
            self.assertEqual(len({record.draw_seed for record in records}), 1)
            skipped = next(record for record in records if record.status == "paired_retry")
            self.assertEqual(skipped.individual_status, "not_run")
            self.assertEqual(skipped.retry_trigger_strategy, "random")
            self.assertEqual(skipped.retry_trigger_status, "gate_resource_skipped")
            self.assertEqual(len({record.source_names for record in records}), 1)
            unit = records[0].routing_unit
            self.assertTrue(
                all(
                    next(source for source in sources if source.source_name == name)
                    .network.routing_unit
                    == unit
                    for name in records[0].source_names
                )
            )
        self.assertEqual(
            {stratum.source_count for stratum in result.routing_unit_strata},
            {1, 2},
        )

    def test_unspecified_unit_joins_the_only_declared_unit_group(self) -> None:
        sources = (
            _source("declared", unit="MBITPERSEC"),
            _source("unknown-one", unit=None),
            _source("unknown-two", unit=None),
        )
        result = run_snd_experiment(
            sources,
            lambda *_: _ResourceGate(),
            strategies=("random",),
            requested_per_strategy=1,
            num_subinstances=(3, 3),
            target_size=(9, 9),
            size_range=(1.0, 1.0),
            max_attempts_per_output=1,
            seed=11,
        )

        self.assertEqual(len(result.routing_unit_strata), 1)
        self.assertEqual(result.routing_unit_strata[0].routing_unit, "MBITPERSEC")
        self.assertEqual(result.routing_unit_strata[0].source_count, 3)
        self.assertTrue(result.attempts)
        self.assertTrue(
            all(record.source_stratum_size == 3 for record in result.attempts)
        )

    def test_infeasible_target_is_a_recorded_source_rejection(self) -> None:
        result = run_snd_experiment(
            (_source("one"),),
            lambda *_: _FeasibleGate(),
            strategies=("random",),
            requested_per_strategy=1,
            num_subinstances=(2, 2),
            target_size=(5, 5),
            size_range=(1.0, 1.0),
            max_attempts_per_output=1,
        )
        self.assertEqual(result.attempts[0].status, "size_limit")
        self.assertEqual(result.attempts[0].target_size, 5)
        self.assertEqual(result.failures, (("random", 1),))

    def test_exact_target_is_inclusive_and_validated(self) -> None:
        source = _source("one")
        boundary = run_snd_experiment(
            (source,),
            lambda *_: _FeasibleGate(),
            strategies=("random",),
            requested_per_strategy=1,
            num_subinstances=(2, 2),
            target_size=(6, 6),
            size_range=(1.0, 1.0),
            max_attempts_per_output=1,
        )
        self.assertEqual(len(boundary.accepted), 1)
        self.assertEqual(len(boundary.accepted[0].candidate.instance.nodes), 6)

        with self.assertRaisesRegex(ValueError, "must not exceed"):
            run_snd_experiment(
                (source,),
                lambda *_: _FeasibleGate(),
                strategies=("random",),
                requested_per_strategy=1,
                num_subinstances=(2, 2),
                target_size=(7, 6),
                size_range=(1.0, 1.0),
                max_attempts_per_output=1,
            )

    def test_source_count_is_limited_by_persistent_plot_palette(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most seven"):
            run_snd_experiment(
                (_source("one"),),
                lambda *_: _FeasibleGate(),
                strategies=("random",),
                requested_per_strategy=1,
                num_subinstances=(8, 8),
                target_size=(6, 6),
                size_range=(1.0, 1.0),
                max_attempts_per_output=1,
            )


if __name__ == "__main__":
    unittest.main()
