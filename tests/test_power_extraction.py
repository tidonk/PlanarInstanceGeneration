from __future__ import annotations

import unittest

import numpy as np

from planar_instance_generation.core import (
    build_power_subcase,
    extract_random_power_subcase,
)
from planar_instance_generation.io import PglibCase


def _line_case() -> PglibCase:
    bus = np.zeros((3, 13), dtype=float)
    bus[:, 0] = [1, 2, 3]
    bus[:, 1] = [3, 1, 1]
    bus[:, 2] = [0, 50, 30]
    bus[:, 7] = 1
    bus[:, 9] = 110
    bus[:, 11] = 1.1
    bus[:, 12] = 0.9

    gen = np.zeros((2, 10), dtype=float)
    gen[:, 0] = [1, 3]
    gen[:, 1] = [80, 0]
    gen[:, 5] = 1
    gen[:, 6] = 100
    gen[:, 7] = 1
    gen[:, 8] = [100, 20]

    branch = np.zeros((2, 13), dtype=float)
    branch[:, :2] = [[1, 2], [2, 3]]
    branch[:, 3] = 0.1
    branch[:, 5:8] = 100
    branch[:, 10] = 1
    branch[:, 11:13] = [-360, 360]

    return PglibCase(
        name="line_case",
        version="2",
        base_mva=100,
        bus=bus,
        gen=gen,
        branch=branch,
        gencost=np.array(
            [[2, 0, 0, 3, 0.01, 1, 0], [2, 0, 0, 3, 0.02, 2, 0]],
            dtype=float,
        ),
    )


class PowerExtractionTests(unittest.TestCase):
    def test_fixed_compensation_adds_removed_export_to_demand(self) -> None:
        source = _line_case()
        original_bus = source.bus.copy()

        result = build_power_subcase(
            source,
            [1, 2],
            np.array([80.0, 30.0]),
            np.array([-80.0, -30.0]),
            compensation="fixed",
        )

        self.assertEqual(result.internal_branch_indices, (0,))
        self.assertEqual(result.cut_branch_indices, (1,))
        self.assertEqual(result.retained_generator_indices, (0,))
        self.assertEqual(result.supply_compensation_mw, {2: -30.0})
        self.assertEqual(result.case.bus[result.case.bus[:, 0] == 2, 2].item(), 80.0)
        self.assertEqual(result.case.gen.shape[0], 1)
        self.assertEqual(result.compensation_generator_indices, ())
        np.testing.assert_array_equal(source.bus, original_bus)

    def test_flexible_import_uses_zero_cost_bounded_supply(self) -> None:
        result = build_power_subcase(
            _line_case(),
            [2, 3],
            np.array([80.0, 30.0]),
            # OptiMods' strict DC adapter derives these values as -Pf.
            compensation="flexible",
        )

        self.assertEqual(result.supply_compensation_mw, {2: 80.0})
        self.assertEqual(result.retained_generator_indices, (1,))
        self.assertEqual(result.compensation_generator_indices, (1,))
        compensation_generator = result.case.gen[1]
        self.assertEqual(int(compensation_generator[0]), 2)
        self.assertEqual(compensation_generator[1], 80.0)
        self.assertEqual(compensation_generator[8], 80.0)
        self.assertEqual(compensation_generator[9], 0.0)
        np.testing.assert_array_equal(result.case.gencost[1], [2, 0, 0, 3, 0, 0, 0])
        self.assertEqual(result.promoted_reference_bus, 2)

    def test_flexible_export_spans_exact_paper_interval(self) -> None:
        result = build_power_subcase(
            _line_case(),
            [1, 2],
            np.array([80.0, 30.0]),
            compensation="flexible",
        )

        bus_two_demand = result.case.bus[result.case.bus[:, 0] == 2, 2].item()
        slack = result.case.gen[result.compensation_generator_indices[0]]
        self.assertEqual(bus_two_demand, 80.0)
        self.assertEqual(slack[1], 0.0)
        self.assertEqual(slack[8], 30.0)
        self.assertEqual(slack[9], 0.0)
        # Fixed added demand -30 plus generation in [0, 30] gives [-30, 0].
        self.assertEqual(result.supply_compensation_mw, {2: -30.0})

    def test_inactive_cut_branch_is_recorded_but_not_compensated(self) -> None:
        source = _line_case()
        source.branch[1, 10] = 0

        result = build_power_subcase(source, [1, 2], np.array([80.0, 999.0]))

        self.assertEqual(result.cut_branch_indices, (1,))
        self.assertEqual(result.supply_compensation_mw, {})
        self.assertEqual(result.case.bus[result.case.bus[:, 0] == 2, 2].item(), 50.0)

    def test_random_wrapper_selects_the_exact_ceiling(self) -> None:
        result = extract_random_power_subcase(
            _line_case(),
            0.5,
            np.array([80.0, 30.0]),
            seed=7,
        )

        self.assertEqual(len(result.selected_buses), 2)
        self.assertEqual(result.case.bus.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
