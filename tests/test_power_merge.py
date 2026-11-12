from __future__ import annotations

from dataclasses import asdict
import unittest

import networkx as nx
import numpy as np

from planar_instance_generation.core.power_merge import (
    CouplingBranchConfig,
    compose_power_cases,
)
from planar_instance_generation.io import PglibCase


def _two_bus_case(
    name: str,
    bus_ids: tuple[int, int],
    *,
    reactance: float,
    extra_columns: bool = False,
    reactive_cost: bool = False,
) -> PglibCase:
    bus_width = 14 if extra_columns else 13
    bus = np.zeros((2, bus_width), dtype=float)
    bus[:, 0] = bus_ids
    bus[:, 1] = [3, 1]
    bus[:, 2] = [0, 40]
    bus[:, 6] = 1
    bus[:, 7] = 1
    bus[:, 9] = 110
    bus[:, 11:13] = [1.1, 0.9]
    if extra_columns:
        bus[:, 13] = [17, 19]

    gen_width = 11 if extra_columns else 10
    gen = np.zeros((1, gen_width), dtype=float)
    gen[0, 0] = bus_ids[0]
    gen[0, 1] = 40
    gen[0, 5:8] = [1, 100, 1]
    gen[0, 8:10] = [80, 0]
    if extra_columns:
        gen[0, 10] = 23

    branch_width = 14 if extra_columns else 13
    branch = np.zeros((1, branch_width), dtype=float)
    branch[0, :2] = bus_ids
    branch[0, 3] = reactance
    branch[0, 5:8] = 100
    branch[0, 10:13] = [1, -45, 45]
    if extra_columns:
        branch[0, 13] = 29

    active_cost = np.array([[2, 0, 0, 3, 0.01, 1, 0]], dtype=float)
    gencost = (
        np.vstack((active_cost, [[2, 0, 0, 3, 0.02, 2, 0]]))
        if reactive_cost
        else active_cost
    )
    areas = np.array([[1, bus_ids[0], 31 if extra_columns else 0]], dtype=float)
    if not extra_columns:
        areas = areas[:, :2]
    return PglibCase(
        name=name,
        version="2",
        base_mva=100,
        bus=bus,
        gen=gen,
        branch=branch,
        gencost=gencost,
        areas=areas,
    )


class PowerMergeTests(unittest.TestCase):
    def test_composes_tables_and_records_complete_provenance(self) -> None:
        left = _two_bus_case("left_case", (1, 2), reactance=0.2)
        right = _two_bus_case(
            "right_case", (1, 3), reactance=0.1, extra_columns=True
        )
        right_bus_before = right.bus.copy()

        result = compose_power_cases(
            left,
            right,
            [(2, 1)],
            name="case4",
            config=CouplingBranchConfig(
                reactance_pu=0.005,
                rate_mva=999,
                angle_min_degrees=-180,
                angle_max_degrees=180,
            ),
        )

        self.assertEqual(result.case.name, "case4")
        np.testing.assert_array_equal(result.case.bus[:, 0], [1, 2, 3, 5])
        self.assertEqual(dict(result.left.bus_ids), {1: 1, 2: 2})
        self.assertEqual(dict(result.right.bus_ids), {1: 3, 3: 5})
        self.assertEqual(asdict(result.right)["bus_ids"], {1: 3, 3: 5})
        self.assertEqual(result.left.bus_rows, (0, 1))
        self.assertEqual(result.right.bus_rows, (2, 3))
        self.assertEqual(result.left.generator_rows, (0,))
        self.assertEqual(result.right.generator_rows, (1,))
        self.assertEqual(result.left.branch_rows, (0,))
        self.assertEqual(result.right.branch_rows, (1,))
        self.assertEqual(result.coupling_branch_rows, (2,))
        self.assertEqual(result.new_coupling_branch_rows, (2,))
        self.assertEqual(
            tuple(mapping.source_name for mapping in result.source_mappings),
            ("left_case", "right_case"),
        )
        self.assertEqual(result.source_endpoint_pairs, ((2, 1),))
        self.assertEqual(result.merged_endpoint_pairs, ((2, 3),))

        # Wider source rows are padded or copied, never truncated.
        self.assertEqual(result.case.bus[2, 13], 17)
        self.assertEqual(result.case.gen[1, 10], 23)
        self.assertEqual(result.case.branch[1, 13], 29)
        np.testing.assert_array_equal(right.bus, right_bus_before)

        coupling = result.case.branch[2]
        np.testing.assert_array_equal(coupling[:2], [2, 3])
        self.assertEqual(coupling[2], 0)
        self.assertEqual(coupling[3], 0.005)
        self.assertEqual(coupling[4], 0)
        np.testing.assert_array_equal(coupling[5:8], [999, 999, 999])
        self.assertEqual(coupling[8], 1)
        self.assertEqual(coupling[9], 0)
        self.assertEqual(coupling[10], 1)
        np.testing.assert_array_equal(coupling[11:13], [-180, 180])

        self.assertEqual(np.count_nonzero(result.case.bus[:, 1] == 3), 1)
        self.assertEqual(result.right.demoted_reference_buses, (3,))
        self.assertEqual(result.case.bus[2, 1], 2)
        self.assertEqual(dict(result.left.area_ids), {1: 1})
        self.assertEqual(dict(result.right.area_ids), {1: 2})
        np.testing.assert_array_equal(result.case.areas[:, :2], [[1, 1], [2, 3]])
        self.assertEqual(result.case.areas[1, 2], 31)
        self.assertTrue(nx.is_connected(result.case.topology()))

    def test_preserves_already_disjoint_right_bus_identifiers(self) -> None:
        result = compose_power_cases(
            _two_bus_case("left_case", (1, 2), reactance=0.2),
            _two_bus_case("right_case", (10, 11), reactance=0.1),
            [(2, 10)],
        )

        self.assertEqual(dict(result.right.bus_ids), {10: 10, 11: 11})
        self.assertEqual(result.merged_endpoint_pairs, ((2, 10),))
        self.assertAlmostEqual(result.coupling_parameters.reactance_pu, 1e-4)
        self.assertEqual(result.coupling_parameters.rate_mva, 0)
        self.assertEqual(result.coupling_parameters.angle_min_degrees, -360)
        self.assertEqual(result.coupling_parameters.angle_max_degrees, 360)

    def test_orders_reactive_costs_and_synthesizes_only_missing_rows(self) -> None:
        left = _two_bus_case(
            "left_case", (1, 2), reactance=0.2, reactive_cost=True
        )
        right = _two_bus_case("right_case", (3, 4), reactance=0.1)

        result = compose_power_cases(left, right, [(2, 3)])

        self.assertEqual(result.case.gencost.shape, (4, 7))
        self.assertEqual(result.left.gencost_rows, (0, 2))
        self.assertEqual(result.right.gencost_rows, (1,))
        np.testing.assert_array_equal(result.case.gencost[2], left.gencost[1])
        np.testing.assert_array_equal(result.case.gencost[3], [2, 0, 0, 1, 0, 0, 0])
        self.assertEqual(result.synthetic_gencost_rows, (3,))

    def test_sequential_merge_keeps_parameters_and_original_source_mappings(self) -> None:
        first = _two_bus_case(
            "first", (1, 2), reactance=0.2, reactive_cost=True
        )
        second = _two_bus_case("second", (3, 4), reactance=0.1)
        third = _two_bus_case("third", (5, 6), reactance=0.05)

        stage_one = compose_power_cases(first, second, [(2, 3)], name="case4")
        stage_two = compose_power_cases(stage_one, third, [(4, 5)], name="case6")

        self.assertEqual(stage_one.coupling_parameters.reactance_pu, 1e-4)
        self.assertEqual(stage_two.coupling_parameters.reactance_pu, 1e-4)
        self.assertEqual(stage_two.coupling_branch_rows, (2, 4))
        self.assertEqual(stage_two.new_coupling_branch_rows, (4,))
        self.assertEqual(stage_two.synthetic_gencost_rows, (4, 5))
        mappings = {mapping.source_name: mapping for mapping in stage_two.source_mappings}
        self.assertEqual(tuple(mappings), ("first", "second", "third"))
        self.assertEqual(mappings["first"].gencost_rows, (0, 3))
        self.assertEqual(mappings["second"].gencost_rows, (1,))
        self.assertEqual(mappings["third"].gencost_rows, (2,))
        self.assertEqual(mappings["first"].branch_rows, (0,))
        self.assertEqual(mappings["second"].branch_rows, (1,))
        self.assertEqual(mappings["third"].branch_rows, (3,))
        self.assertEqual(mappings["first"].demoted_reference_buses, ())
        self.assertEqual(mappings["second"].demoted_reference_buses, (3,))
        self.assertEqual(mappings["third"].demoted_reference_buses, (5,))
        self.assertEqual(np.count_nonzero(stage_two.case.bus[:, 1] == 3), 1)

    def test_rejects_duplicate_reused_and_unknown_endpoints(self) -> None:
        left = _two_bus_case("left_case", (1, 2), reactance=0.2)
        right = _two_bus_case("right_case", (1, 3), reactance=0.1)
        invalid = (
            ([], "at least one"),
            ([(1, 1), (1, 1)], "duplicate pairs"),
            ([(1, 1), (1, 3)], "left endpoint"),
            ([(1, 1), (2, 1)], "right endpoint"),
            ([(99, 1)], "unknown left"),
            ([(1, 99)], "unknown right"),
        )
        for pairs, message in invalid:
            with self.subTest(pairs=pairs), self.assertRaisesRegex(ValueError, message):
                compose_power_cases(left, right, pairs)

    def test_rejects_disconnected_source_and_incompatible_base(self) -> None:
        left = _two_bus_case("left_case", (1, 2), reactance=0.2)
        right = _two_bus_case("right_case", (3, 4), reactance=0.1)
        disconnected = _two_bus_case("disconnected", (5, 6), reactance=0.3)
        extra = disconnected.bus[1].copy()
        extra[0] = 7
        extra[1] = 1
        disconnected.bus = np.vstack((disconnected.bus, extra))

        with self.assertRaisesRegex(ValueError, "connected active topology"):
            compose_power_cases(left, disconnected, [(2, 5)])

        right.base_mva = 50
        with self.assertRaisesRegex(ValueError, "same baseMVA"):
            compose_power_cases(left, right, [(2, 3)])

    def test_rejects_nonphysical_coupling_parameters(self) -> None:
        left = _two_bus_case("left_case", (1, 2), reactance=0.2)
        right = _two_bus_case("right_case", (3, 4), reactance=0.1)
        invalid = (
            CouplingBranchConfig(reactance_pu=0),
            CouplingBranchConfig(rate_mva=-1),
            CouplingBranchConfig(angle_min_degrees=10, angle_max_degrees=10),
        )
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                compose_power_cases(left, right, [(2, 3)], config=config)


if __name__ == "__main__":
    unittest.main()
