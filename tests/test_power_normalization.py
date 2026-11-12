from __future__ import annotations

import unittest

import numpy as np

from planar_instance_generation.core.power_merge import compose_power_cases
from planar_instance_generation.evaluation.power_normalization import (
    normalize_component_voltage_aware_load,
)
from tests.test_power_merge import _two_bus_case


class PowerNormalizationTests(unittest.TestCase):
    def test_voltage_difference_scales_mva_quantities_by_relative_voltage(self) -> None:
        left = _two_bus_case("left", (1, 2), reactance=0.2)
        right = _two_bus_case("right", (1, 3), reactance=0.1)
        right.bus[:, 9] = 220
        merged = compose_power_cases(left, right, ((1, 1),))
        groups = {
            mapping.source_name: tuple(mapping.bus_ids.values())
            for mapping in merged.source_mappings
        }

        result = normalize_component_voltage_aware_load(
            merged.case,
            groups,
            merged.coupling_branch_rows,
        )

        factors = {
            component.source: component.factor for component in result.components
        }
        self.assertAlmostEqual(factors["left"], 0.75)
        self.assertAlmostEqual(factors["right"], 1.5)
        self.assertAlmostEqual(factors["right"] / factors["left"], 2.0)
        self.assertAlmostEqual(result.case.branch[0, 5], 75.0)
        self.assertAlmostEqual(result.case.branch[1, 5], 150.0)
        np.testing.assert_array_equal(result.case.bus[:, 9], merged.case.bus[:, 9])
        np.testing.assert_array_equal(
            result.case.branch[-1], merged.case.branch[-1]
        )

    def test_equalizes_voltage_aware_load_and_preserves_electrical_data(self) -> None:
        left = _two_bus_case("left", (1, 2), reactance=0.2)
        right = _two_bus_case("right", (1, 3), reactance=0.1)
        left.branch[0, 2] = 0.02
        left.branch[0, 4] = 0.03
        right.branch[0, 2] = 0.01
        right.branch[0, 4] = 0.04
        right.bus[:, 2] *= 4
        right.gen[:, [1, 8]] *= 4
        right.bus[:, 9] = 220
        merged = compose_power_cases(left, right, ((1, 1),))
        groups = {
            mapping.source_name: tuple(mapping.bus_ids.values())
            for mapping in merged.source_mappings
        }

        result = normalize_component_voltage_aware_load(
            merged.case,
            groups,
            merged.coupling_branch_rows,
        )

        factors = {component.source: component.factor for component in result.components}
        self.assertAlmostEqual(factors["left"], 1.5)
        self.assertAlmostEqual(factors["right"], 0.75)
        self.assertTrue(np.array_equal(result.case.branch[:, :2], merged.case.branch[:, :2]))
        self.assertAlmostEqual(result.case.branch[0, 2], 0.02 / 1.5)
        self.assertAlmostEqual(result.case.branch[1, 2], 0.01 / 0.75)
        self.assertAlmostEqual(result.case.branch[0, 3], 0.2 / 1.5)
        self.assertAlmostEqual(result.case.branch[1, 3], 0.1 / 0.75)
        self.assertAlmostEqual(result.case.branch[0, 4], 0.03 * 1.5)
        self.assertAlmostEqual(result.case.branch[1, 4], 0.04 * 0.75)
        self.assertAlmostEqual(result.case.branch[0, 5], 150.0)
        self.assertAlmostEqual(result.case.branch[1, 5], 75.0)
        self.assertEqual(result.case.branch[2, 5], 0.0)
        self.assertEqual(result.case.branch[2, 3], 1e-4)
        self.assertEqual(result.case.branch[2, 8], 1.0)
        self.assertAlmostEqual(result.case.gen[0, 8], 120.0)
        self.assertAlmostEqual(result.case.gen[1, 8], 240.0)
        self.assertAlmostEqual(result.case.gen[0, 6], 150.0)
        self.assertAlmostEqual(result.case.gen[1, 6], 75.0)
        np.testing.assert_array_equal(result.case.bus[:, 9], merged.case.bus[:, 9])
        self.assertAlmostEqual(result.case.gencost[0, 4], 0.01 / 1.5**2)
        self.assertAlmostEqual(result.case.gencost[1, 4], 0.01 / 0.75**2)
        self.assertEqual(result.voltage.coupling_branch_count, 1)
        self.assertEqual(result.voltage.median_voltage_ratio, 2.0)
        self.assertEqual(result.voltage.same_voltage_fraction, 0.0)

    def test_requires_a_complete_partition_and_crossing_certification(self) -> None:
        left = _two_bus_case("left", (1, 2), reactance=0.2)
        right = _two_bus_case("right", (3, 4), reactance=0.1)
        merged = compose_power_cases(left, right, ((1, 3),))
        with self.assertRaisesRegex(ValueError, "partition"):
            normalize_component_voltage_aware_load(
                merged.case,
                {"left": (1, 2)},
                merged.coupling_branch_rows,
            )
        with self.assertRaisesRegex(ValueError, "uncertified"):
            normalize_component_voltage_aware_load(
                merged.case,
                {"left": (1, 2), "right": (3, 4)},
                (),
            )


if __name__ == "__main__":
    unittest.main()
