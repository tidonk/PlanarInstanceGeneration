from __future__ import annotations

import unittest

import numpy as np

from planar_instance_generation.core.power_isomorphism import pglib_isomorphic
from planar_instance_generation.io import PglibCase, PglibFormatError


def _case(name: str = "case_three") -> PglibCase:
    bus = np.zeros((3, 13), dtype=float)
    bus[:, 0] = [10, 20, 30]
    bus[:, 1] = [3, 1, 2]
    bus[:, 2] = [0, 40, 15]
    bus[:, 3] = [0, 5, 2]
    bus[:, 6] = [1, 1, 2]
    bus[:, 7] = [1.00, 0.99, 1.01]
    bus[:, 8] = [0, -1, 2]
    bus[:, 9] = [220, 220, 110]
    bus[:, 10] = [4, 4, 7]
    bus[:, 11] = 1.1
    bus[:, 12] = 0.9

    gen = np.zeros((3, 10), dtype=float)
    gen[:, 0] = [10, 10, 30]
    gen[:, 1] = [35, 20, 5]
    gen[:, 5] = 1
    gen[:, 6] = 100
    gen[:, 7] = [1, 0, 1]
    gen[:, 8] = [80, 40, 30]
    gen[:, 9] = [0, -5, 0]

    branch = np.zeros((4, 13), dtype=float)
    branch[:, :2] = [[10, 20], [10, 20], [20, 30], [30, 10]]
    branch[:, 2] = [0.01, 0.02, 0.03, 0.04]
    branch[:, 3] = [0.10, 0.20, 0.30, 0.40]
    branch[:, 5:8] = [100, 100, 100]
    branch[:, 10] = [1, 1, 1, 0]
    branch[:, 11] = -360
    branch[:, 12] = 360

    active_cost = np.array(
        [
            [2, 1, 2, 3, 0.01, 1.0, 0],
            [2, 3, 4, 3, 0.02, 2.0, 1],
            [2, 5, 6, 3, 0.03, 3.0, 2],
        ],
        dtype=float,
    )
    reactive_cost = active_cost.copy()
    reactive_cost[:, 4:] += 10
    areas = np.array([[1, 10, 11], [2, 30, 22]], dtype=float)
    return PglibCase(
        name=name,
        version="2",
        base_mva=100,
        bus=bus,
        gen=gen,
        branch=branch,
        gencost=np.vstack((active_cost, reactive_cost)),
        areas=areas,
    )


def _relabel_and_reorder(source: PglibCase) -> PglibCase:
    bus_map = {10: 103, 20: 101, 30: 102}
    area_map = {1: 9, 2: 4}

    bus = source.bus.copy()
    for row in bus:
        row[0] = bus_map[int(row[0])]
        row[6] = area_map[int(row[6])]
    bus = bus[[2, 0, 1]]

    generator_order = np.array([2, 1, 0])
    gen = source.gen.copy()
    for row in gen:
        row[0] = bus_map[int(row[0])]
    gen = gen[generator_order]
    count = source.gen.shape[0]
    gencost = np.vstack(
        (
            source.gencost[:count][generator_order],
            source.gencost[count:][generator_order],
        )
    )

    branch = source.branch.copy()
    for row in branch:
        row[0] = bus_map[int(row[0])]
        row[1] = bus_map[int(row[1])]
    branch = branch[[3, 1, 2, 0]]

    areas = source.areas.copy()
    for row in areas:
        row[0] = area_map[int(row[0])]
        row[1] = bus_map[int(row[1])]
    areas = areas[[1, 0]]
    return PglibCase(
        name="different_name",
        version=source.version,
        base_mva=source.base_mva,
        bus=bus,
        gen=gen,
        branch=branch,
        gencost=gencost,
        areas=areas,
    )


class PowerIsomorphismTests(unittest.TestCase):
    def test_ignores_consistent_identifiers_names_and_row_order(self) -> None:
        original = _case()
        relabelled = _relabel_and_reorder(original)

        self.assertTrue(pglib_isomorphic(original, relabelled))
        self.assertTrue(pglib_isomorphic(relabelled, original))

    def test_compares_global_data_and_table_schemas(self) -> None:
        original = _case()
        changed_base = _case("changed_base")
        changed_base.base_mva = 50
        wider_bus = _case("wider_bus")
        wider_bus.bus = np.column_stack((wider_bus.bus, np.zeros(3)))
        without_areas = _case("without_areas")
        without_areas.areas = None

        self.assertFalse(pglib_isomorphic(original, changed_base))
        self.assertFalse(pglib_isomorphic(original, wider_bus))
        self.assertFalse(pglib_isomorphic(original, without_areas))

    def test_compares_bus_generator_and_paired_cost_attributes(self) -> None:
        original = _case()
        changed_bus = _case("changed_bus")
        changed_bus.bus[1, 2] += 1
        changed_generator = _case("changed_generator")
        changed_generator.gen[1, 8] += 1
        changed_active_cost = _case("changed_active_cost")
        changed_active_cost.gencost[0, 4] += 1
        changed_reactive_cost = _case("changed_reactive_cost")
        changed_reactive_cost.gencost[3, 4] += 1

        for changed in (
            changed_bus,
            changed_generator,
            changed_active_cost,
            changed_reactive_cost,
        ):
            with self.subTest(case=changed.name):
                self.assertFalse(pglib_isomorphic(original, changed))

        # Costs belong to generator rows; reordering only the generators changes
        # which physical generator receives which active/reactive costs.
        changed_pairing = _case("changed_pairing")
        changed_pairing.gen[[0, 1]] = changed_pairing.gen[[1, 0]]
        self.assertFalse(pglib_isomorphic(original, changed_pairing))

    def test_preserves_every_parallel_and_inactive_branch(self) -> None:
        original = _case()
        reordered = _case("reordered")
        reordered.branch = reordered.branch[[1, 0, 3, 2]]
        changed_parallel = _case("changed_parallel")
        changed_parallel.branch[1, 3] += 0.5
        changed_inactive = _case("changed_inactive")
        changed_inactive.branch[3, 2] += 0.5

        self.assertTrue(pglib_isomorphic(original, reordered))
        self.assertFalse(pglib_isomorphic(original, changed_parallel))
        self.assertFalse(pglib_isomorphic(original, changed_inactive))

    def test_preserves_branch_from_and_to_orientation(self) -> None:
        original = _case()
        reversed_branch = _case("reversed_branch")
        reversed_branch.branch[2, :2] = reversed_branch.branch[2, 1::-1]

        self.assertFalse(pglib_isomorphic(original, reversed_branch))

    def test_compares_area_membership_reference_and_attributes(self) -> None:
        original = _case()
        changed_membership = _case("changed_membership")
        changed_membership.bus[1, 6] = 2
        changed_reference = _case("changed_reference")
        changed_reference.areas[0, 1] = 20
        changed_attribute = _case("changed_attribute")
        changed_attribute.areas[0, 2] += 1

        self.assertFalse(pglib_isomorphic(original, changed_membership))
        self.assertFalse(pglib_isomorphic(original, changed_reference))
        self.assertFalse(pglib_isomorphic(original, changed_attribute))

    def test_fails_fast_on_malformed_area_data(self) -> None:
        original = _case()
        noninteger_bus_area = _case("noninteger_bus_area")
        noninteger_bus_area.bus[0, 6] = 1.5
        duplicate_area = _case("duplicate_area")
        duplicate_area.areas[1, 0] = 1
        unknown_reference = _case("unknown_reference")
        unknown_reference.areas[0, 1] = 999

        invalid = (
            (noninteger_bus_area, "area identifiers"),
            (duplicate_area, "duplicate area identifiers"),
            (unknown_reference, "unknown bus"),
        )
        for case, message in invalid:
            with self.subTest(case=case.name), self.assertRaisesRegex(
                PglibFormatError, message
            ):
                pglib_isomorphic(original, case)

    def test_rejects_non_cases(self) -> None:
        with self.assertRaisesRegex(TypeError, "PglibCase"):
            pglib_isomorphic(_case(), object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
