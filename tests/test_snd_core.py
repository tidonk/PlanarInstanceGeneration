from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import networkx as nx

from planar_instance_generation.core.snd import (
    SndCommodity,
    SndInstance,
    SndSolution,
    build_snd_instance,
    validate_snd_solution,
)
from planar_instance_generation.core.snd_serialization import (
    decompose_commodity_flow,
    serialize_snd_instance,
    sndlib_serialization_capability,
    write_snd_instance,
)
from planar_instance_generation.core.snd_extraction import (
    build_snd_subinstance,
    extract_random_snd_subinstance,
)
from planar_instance_generation.core.snd_merge import (
    SndCouplingLinkConfig,
    SndInterNetworkDemandConfig,
    SndMergeConfig,
    compose_snd_instances,
)
from planar_instance_generation.io.sndlib import (
    SndlibCapacityModule,
    SndlibDemand,
    SndlibLink,
    SndlibMetadata,
    SndlibNetwork,
    SndlibNode,
    read_sndlib,
)


def _module(capacity: float) -> SndlibCapacityModule:
    return SndlibCapacityModule(capacity=capacity, cost=1.0)


def _extraction_source() -> tuple[SndlibNetwork, SndSolution]:
    nodes = tuple(SndlibNode(node, index, 0.0) for index, node in enumerate("ABCD"))
    links = (
        SndlibLink("l_ab", "A", "B", additional_modules=(_module(3),)),
        # Reversed storage orientation exercises signed-flow incidence.
        SndlibLink("l_cb", "C", "B", additional_modules=(_module(3),)),
        SndlibLink("l_cd", "C", "D", additional_modules=(_module(9),)),
        SndlibLink("l_ac", "A", "C", additional_modules=(_module(6),)),
    )
    demands = (
        SndlibDemand("d1", "A", "D", 5.0, max_path_length=5),
        SndlibDemand("d2", "D", "A", 4.0),
    )
    network = SndlibNetwork("1.0", "pixel", nodes, links, demands)
    solution = SndSolution(
        {"l_ab": 3, "l_cb": 3, "l_cd": 9, "l_ac": 6},
        {
            ("d1", "l_ab"): 3,
            ("d1", "l_cb"): -3,
            ("d1", "l_cd"): 5,
            ("d1", "l_ac"): 2,
            ("d2", "l_cd"): -4,
            ("d2", "l_ac"): -4,
        },
    )
    return network, solution


def _two_node_instance(
    name: str,
    *,
    reverse: bool = False,
    value: float = 2.0,
    coordinates_type: str = "pixel",
    metadata: SndlibMetadata | None = None,
):
    nodes = (SndlibNode("A", 0, 0), SndlibNode("B", 1, 0))
    source, target = ("B", "A") if reverse else ("A", "B")
    link = SndlibLink(
        "link",
        source,
        target,
        preinstalled_module=SndlibCapacityModule(value, 11),
        additional_modules=(SndlibCapacityModule(value * 2, 17),),
        routing_cost=3,
        setup_cost=5,
    )
    demand = SndlibDemand("demand", source, target, value, max_path_length=4)
    network = SndlibNetwork(
        "1.0", coordinates_type, nodes, (link,), (demand,), metadata
    )
    return build_snd_instance(
        network,
        SndSolution({"link": value}, {("demand", "link"): value}),
        name=name,
    )


def _multi_terminal_instance(*, with_cycle: bool = False) -> SndInstance:
    nodes = tuple(
        SndlibNode(node, index, 0) for index, node in enumerate("ABCD")
    )
    links = [
        SndlibLink("l_ca", "C", "A", additional_modules=(_module(2),)),
        SndlibLink("l_bd", "B", "D", additional_modules=(_module(3),)),
        SndlibLink("l_ab", "A", "B", additional_modules=(_module(1),)),
    ]
    capacities = {"l_ca": 2.0, "l_bd": 3.0, "l_ab": 0.0}
    flows = {("commodity", "l_ca"): -2.0, ("commodity", "l_bd"): 3.0}
    if with_cycle:
        links.extend(
            (
                SndlibLink("l_bc", "B", "C", additional_modules=(_module(1),)),
                SndlibLink("l_cb", "C", "B", additional_modules=(_module(1),)),
            )
        )
        capacities.update({"l_bc": 1.0, "l_cb": 1.0})
        flows.update(
            {("commodity", "l_bc"): 1.0, ("commodity", "l_cb"): 1.0}
        )
    return SndInstance(
        name="multi",
        version="1.0",
        coordinates_type="pixel",
        nodes=nodes,
        links=tuple(links),
        commodities=(
            SndCommodity(
                "commodity",
                {"A": 2, "B": 3, "C": -2, "D": -3},
                "source",
                None,
            ),
        ),
        reference_solution=SndSolution(capacities, flows),
    )


class SndSolutionAndExtractionTests(unittest.TestCase):
    def test_solution_uses_literal_orientation_and_shared_capacity(self) -> None:
        network, solution = _extraction_source()
        validate_snd_solution(network, solution)
        self.assertEqual(solution.flow("d1", "l_cb"), -3)
        self.assertEqual(solution.flow("d2", "l_ab"), 0)

        too_small = SndSolution(
            {**solution.installed_capacities, "l_cd": 8.5}, solution.flows
        )
        with self.assertRaisesRegex(ValueError, "shared flow"):
            validate_snd_solution(network, too_small)

    def test_exact_cut_compensation_restricts_a_feasible_flow(self) -> None:
        network, solution = _extraction_source()
        result = build_snd_subinstance(
            network, ["A", "B"], solution, source_name="source"
        )

        self.assertEqual(result.retained_node_indices, (0, 1))
        self.assertEqual(result.internal_link_indices, (0,))
        self.assertEqual(result.cut_link_indices, (1, 3))
        self.assertEqual(
            dict(result.compensation_by_demand["d1"]), {"A": -2, "B": -3}
        )
        self.assertEqual(dict(result.compensation_by_demand["d2"]), {"A": 4})
        commodities = {commodity.id: commodity for commodity in result.instance.commodities}
        self.assertEqual(dict(commodities["d1"].node_supplies), {"A": 3, "B": -3})
        self.assertEqual(dict(commodities["d2"].node_supplies), {})
        self.assertIs(commodities["d1"].source_demand, network.demands[0])
        self.assertEqual(commodities["d1"].source_demand.max_path_length, 5)
        self.assertEqual(result.instance.reference_solution.flow("d1", "l_ab"), 3)
        self.assertEqual(result.instance.reference_solution.installed_capacities, {"l_ab": 3})

    def test_random_wrapper_selects_ceiling_and_connected_nodes(self) -> None:
        network, solution = _extraction_source()
        result = extract_random_snd_subinstance(
            network, 0.5, solution, source_name="source", seed=19
        )
        self.assertEqual(len(result.selected_nodes), 2)
        self.assertTrue(nx.is_connected(result.instance.topology()))

    def test_rejects_disconnected_manual_selection(self) -> None:
        network, solution = _extraction_source()
        with self.assertRaisesRegex(ValueError, "induce a connected"):
            build_snd_subinstance(
                network, ["B", "D"], solution, source_name="source"
            )


class SndCompositionTests(unittest.TestCase):
    def test_composition_preserves_parameters_orientation_and_provenance(self) -> None:
        left = _two_node_instance("left", value=2)
        right = _two_node_instance("right", reverse=True, value=3)
        original_right_link = right.links[0]

        result = compose_snd_instances(left, right, [("B", "A")], name="merged")

        self.assertEqual(tuple(node.id for node in result.instance.nodes), (
            "A", "B", "right1::A", "right1::B"
        ))
        self.assertEqual(result.merged_endpoint_pairs, (("B", "right1::A"),))
        self.assertEqual(result.right.node_ids, {
            "A": "right1::A", "B": "right1::B"
        })
        mapped_right = result.instance.links[1]
        self.assertEqual((mapped_right.source, mapped_right.target), (
            "right1::B", "right1::A"
        ))
        self.assertEqual(mapped_right.preinstalled_module, original_right_link.preinstalled_module)
        self.assertEqual(mapped_right.additional_modules, original_right_link.additional_modules)
        self.assertEqual(mapped_right.routing_cost, 3)
        self.assertEqual(mapped_right.setup_cost, 5)

        coupling = result.instance.links[2]
        self.assertEqual((coupling.source, coupling.target), ("B", "right1::A"))
        self.assertIsNone(coupling.preinstalled_module)
        self.assertEqual(coupling.additional_modules[0].capacity, 5)
        self.assertEqual(coupling.additional_modules[0].cost, 0)
        self.assertEqual(result.instance.reference_solution.installed_capacities[coupling.id], 0)
        self.assertFalse(
            any(
                link_id == coupling.id
                for _, link_id in result.instance.reference_solution.flows
            )
        )
        self.assertEqual(result.coupling_link_indices, (2,))
        self.assertEqual(result.synthetic_demand_indices, ())
        self.assertTrue(nx.is_connected(result.instance.topology()))

        mapped_commodity = result.instance.commodities[1]
        self.assertEqual(mapped_commodity.id, "right1::demand")
        self.assertEqual(mapped_commodity.source_demand, right.commodities[0].source_demand)
        self.assertEqual(tuple(mapping.source_name for mapping in result.source_mappings), (
            "left", "right"
        ))

    def test_sequential_composition_propagates_all_source_mappings(self) -> None:
        first = _two_node_instance("first")
        second = _two_node_instance("second", reverse=True)
        third = _two_node_instance("third")
        stage_one = compose_snd_instances(first, second, [("B", "A")])
        stage_two = compose_snd_instances(
            stage_one, third, [("right1::A", "A")]
        )

        self.assertEqual(stage_two.coupling_link_indices, (2, 4))
        self.assertEqual(tuple(mapping.source_name for mapping in stage_two.source_mappings), (
            "first", "second", "third"
        ))
        self.assertEqual(stage_two.source_mappings[1].node_ids["A"], "right1::A")
        self.assertEqual(stage_two.source_mappings[2].node_ids["A"], "right2::A")
        self.assertEqual(stage_two.source_mappings[2].link_indices, (3,))

    def test_sequential_composition_retains_common_unit_across_metadata(self) -> None:
        sources = tuple(
            _two_node_instance(
                f"source_{index}",
                metadata=SndlibMetadata(
                    f"granularity_{index}",
                    f"time_{index}",
                    "MBIT",
                    f"origin_{index}",
                ),
            )
            for index in range(3)
        )

        first = compose_snd_instances(sources[0], sources[1], [("B", "A")])
        second = compose_snd_instances(first, sources[2], [("B", "A")])

        self.assertIsNotNone(second.instance.metadata)
        assert second.instance.metadata is not None
        self.assertEqual(second.instance.metadata.unit, "MBIT")
        self.assertEqual(second.instance.metadata.origin, "planar-instance-generation")
        self.assertEqual(
            [mapping.metadata.origin for mapping in second.source_mappings],
            ["origin_0", "origin_1", "origin_2"],
        )

    def test_inter_network_demands_are_explicit_and_reference_feasible(self) -> None:
        result = compose_snd_instances(
            _two_node_instance("left", value=2),
            _two_node_instance("right", value=3),
            [("B", "A")],
            config=SndMergeConfig(
                coupling=SndCouplingLinkConfig(
                    module_capacity=1.5,
                    module_cost=7,
                    routing_cost=2,
                    setup_cost=3,
                ),
                inter_network_demands=SndInterNetworkDemandConfig(value=4),
            ),
        )

        coupling = result.instance.links[result.new_coupling_link_indices[0]]
        synthetic = result.instance.commodities[result.synthetic_demand_indices[0]]
        self.assertEqual(coupling.additional_modules[0], SndlibCapacityModule(1.5, 7))
        self.assertEqual((coupling.routing_cost, coupling.setup_cost), (2, 3))
        self.assertIsNone(synthetic.source_demand)
        self.assertEqual(dict(synthetic.node_supplies), {
            "B": 4, "right1::A": -4
        })
        self.assertEqual(result.instance.reference_solution.flow(synthetic.id, coupling.id), 4)
        self.assertEqual(result.instance.reference_solution.installed_capacities[coupling.id], 4.5)

    def test_rejects_reused_endpoints_and_implicit_scale_mixing(self) -> None:
        left = _two_node_instance(
            "left",
            metadata=SndlibMetadata("day", "now", "MBIT", "left"),
        )
        right = _two_node_instance(
            "right",
            coordinates_type="geographical",
            metadata=SndlibMetadata("day", "now", "GBIT", "right"),
        )
        with self.assertRaisesRegex(ValueError, "right endpoint"):
            compose_snd_instances(left, right, [("A", "A"), ("B", "A")])
        with self.assertRaisesRegex(ValueError, "coordinate types differ"):
            compose_snd_instances(left, right, [("A", "A")])
        with self.assertRaisesRegex(ValueError, "routing units differ"):
            compose_snd_instances(
                left,
                right,
                [("A", "A")],
                config=SndMergeConfig(coordinates_type="pixel"),
            )

        unspecified = _two_node_instance("unspecified")
        compatible = compose_snd_instances(left, unspecified, [("A", "A")])
        self.assertIsNone(compatible.instance.metadata)

        result = compose_snd_instances(
            left,
            right,
            [("A", "A")],
            config=SndMergeConfig(
                coordinates_type="pixel", allow_mixed_units=True
            ),
        )
        self.assertIsNone(result.instance.metadata)
        self.assertEqual(result.source_mappings[0].metadata.unit, "MBIT")
        self.assertEqual(result.source_mappings[1].metadata.unit, "GBIT")


class SndSerializationTests(unittest.TestCase):
    def test_decomposes_multi_terminal_balance_without_changing_link_flow(self) -> None:
        instance = _multi_terminal_instance()
        walks = decompose_commodity_flow(instance, "commodity")
        self.assertEqual(
            tuple((walk.node_ids, walk.link_ids, walk.link_signs, walk.value) for walk in walks),
            (
                (("A", "C"), ("l_ca",), (-1,), 2),
                (("B", "D"), ("l_bd",), (1,), 3),
            ),
        )

        serialized = serialize_snd_instance(instance)
        self.assertEqual(
            tuple(
                (demand.source, demand.target, demand.value)
                for demand in serialized.network.demands
            ),
            (("A", "C", 2), ("B", "D", 3)),
        )
        self.assertTrue(
            all(not demand.admissible_paths for demand in serialized.network.demands)
        )
        self.assertEqual(
            sum(
                serialized.reference_solution.flow(demand.id, "l_ca")
                for demand in serialized.network.demands
            ),
            -2,
        )
        self.assertEqual(
            sum(
                serialized.reference_solution.flow(demand.id, "l_bd")
                for demand in serialized.network.demands
            ),
            3,
        )

    def test_reference_paths_are_explicit_and_cycle_safe(self) -> None:
        serialized = serialize_snd_instance(
            _multi_terminal_instance(), include_reference_paths=True
        )
        for demand in serialized.network.demands:
            self.assertEqual(len(demand.admissible_paths), 1)
            self.assertEqual(demand.max_path_length, 1)

        cyclic = _multi_terminal_instance(with_cycle=True)
        default = serialize_snd_instance(cyclic)
        self.assertTrue(default.demand_provenance[0].carries_circulation)
        original = cyclic.reference_solution.flow("commodity", "l_bc")
        total = sum(
            default.reference_solution.flow(demand.id, "l_bc")
            for demand in default.network.demands
        )
        self.assertEqual(total, original)
        capability = sndlib_serialization_capability(
            cyclic, include_reference_paths=True
        )
        self.assertFalse(capability.supported)
        self.assertIn("circulation", capability.reasons[0])

    def test_rejects_inherited_routing_restrictions_instead_of_dropping_them(self) -> None:
        capability = sndlib_serialization_capability(_two_node_instance("restricted"))
        self.assertFalse(capability.supported)
        self.assertIn("maxPathLength", capability.reasons[0])
        with self.assertRaisesRegex(ValueError, "unsupported maxPathLength"):
            serialize_snd_instance(_two_node_instance("restricted"))

    def test_writes_round_trip_gate_ready_xml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generated.xml"
            serialized = write_snd_instance(_multi_terminal_instance(), path)
            self.assertEqual(read_sndlib(path), serialized.network)


if __name__ == "__main__":
    unittest.main()
