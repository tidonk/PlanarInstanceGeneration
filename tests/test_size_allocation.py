from __future__ import annotations

import math
import random
import unittest

from planar_instance_generation.core import (
    TargetSizeInfeasibleError,
    allocate_target_size,
)


class TargetSizeAllocationTests(unittest.TestCase):
    def test_allocates_exact_target_inside_every_source_bound(self) -> None:
        counts = (100, 50, 25)

        first = allocate_target_size(
            counts, 100, (0.15, 0.85), rng=random.Random(7)
        )
        second = allocate_target_size(
            counts, 100, (0.15, 0.85), rng=random.Random(7)
        )

        self.assertEqual(first, second)
        self.assertEqual(sum(first.target_nodes), 100)
        self.assertEqual(set(first.allocation_order), {0, 1, 2})
        for count, target in zip(counts, first.target_nodes, strict=True):
            self.assertGreaterEqual(target, math.ceil(0.15 * count))
            self.assertLessEqual(target, math.floor(0.85 * count))
        self.assertEqual(
            first.realized_fractions,
            tuple(target / count for target, count in zip(first.target_nodes, counts)),
        )

    def test_enforces_three_node_floor_per_source(self) -> None:
        allocation = allocate_target_size(
            (10, 10), 6, (0.15, 0.85), rng=random.Random(3)
        )

        self.assertEqual(allocation.lower_bounds, (3, 3))
        self.assertEqual(allocation.target_nodes, (3, 3))

    def test_rejects_source_selection_that_cannot_reach_target(self) -> None:
        with self.assertRaisesRegex(TargetSizeInfeasibleError, "does not contain"):
            allocate_target_size(
                (20, 20), 35, (0.15, 0.85), rng=random.Random(1)
            )

    def test_rejects_empty_integer_source_interval(self) -> None:
        with self.assertRaisesRegex(TargetSizeInfeasibleError, "no integer size"):
            allocate_target_size(
                (3, 3), 2, (0.3, 0.3), rng=random.Random(1)
            )

    def test_validates_ranges_and_random_source(self) -> None:
        for size_range in ((0.0, 0.5), (0.9, 0.2), (0.2, 1.1)):
            with self.subTest(size_range=size_range), self.assertRaises(ValueError):
                allocate_target_size(
                    (10, 10), 10, size_range, rng=random.Random(1)
                )
        with self.assertRaises(TypeError):
            allocate_target_size((10, 10), 10, (0.2, 0.8), rng=object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
