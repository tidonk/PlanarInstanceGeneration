"""Exact random allocation of a composite node target across source graphs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
import random
from typing import Sequence


class TargetSizeInfeasibleError(ValueError):
    """Raised when selected sources cannot realize the requested target size."""


@dataclass(frozen=True, slots=True)
class TargetSizeAllocation:
    """A feasible exact allocation in the original source order."""

    target_size: int
    source_node_counts: tuple[int, ...]
    lower_bounds: tuple[int, ...]
    upper_bounds: tuple[int, ...]
    target_nodes: tuple[int, ...]
    allocation_order: tuple[int, ...]

    @property
    def realized_fractions(self) -> tuple[float, ...]:
        return tuple(
            target / source
            for target, source in zip(
                self.target_nodes, self.source_node_counts, strict=True
            )
        )


def allocate_target_size(
    source_node_counts: Sequence[int],
    target_size: int,
    size_range: Sequence[Real],
    *,
    rng: random.Random,
) -> TargetSizeAllocation:
    """Randomly allocate exactly ``target_size`` within per-source bounds."""

    counts = tuple(_positive_integer("source node count", value) for value in source_node_counts)
    if len(counts) < 2:
        raise ValueError("at least two source node counts are required")
    target = _positive_integer("target_size", target_size)
    lower_fraction, upper_fraction = _validate_size_range(size_range)
    if not isinstance(rng, random.Random):
        raise TypeError("rng must be an instance of random.Random")

    lower = tuple(max(3, math.ceil(lower_fraction * count)) for count in counts)
    upper = tuple(math.floor(upper_fraction * count) for count in counts)
    lower_sum = sum(lower)
    upper_sum = sum(upper)
    if any(minimum > maximum for minimum, maximum in zip(lower, upper, strict=True)):
        raise TargetSizeInfeasibleError(
            "selected source has no integer size inside its fraction bounds"
        )
    if lower_sum > target or upper_sum < target:
        raise TargetSizeInfeasibleError(
            f"selected sources allow [{lower_sum}, {upper_sum}] nodes, "
            f"which does not contain target {target}"
        )

    order = list(range(len(counts)))
    rng.shuffle(order)
    allocations = [0] * len(counts)
    remaining = target
    for offset, source_index in enumerate(order[:-1]):
        later = order[offset + 1 :]
        feasible_lower = max(
            lower[source_index],
            remaining - sum(upper[index] for index in later),
        )
        feasible_upper = min(
            upper[source_index],
            remaining - sum(lower[index] for index in later),
        )
        allocations[source_index] = rng.randint(feasible_lower, feasible_upper)
        remaining -= allocations[source_index]
    allocations[order[-1]] = remaining

    resolved = tuple(allocations)
    if sum(resolved) != target:
        raise RuntimeError("target-size allocation did not preserve the exact total")
    if any(
        not minimum <= value <= maximum
        for value, minimum, maximum in zip(resolved, lower, upper, strict=True)
    ):
        raise RuntimeError("target-size allocation violated a source bound")
    return TargetSizeAllocation(
        target,
        counts,
        lower,
        upper,
        resolved,
        tuple(order),
    )


def _validate_size_range(values: Sequence[Real]) -> tuple[float, float]:
    raw = tuple(values)
    if len(raw) != 2:
        raise ValueError("size_range must contain exactly two fractions")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in raw):
        raise TypeError("size_range values must be real numbers")
    lower, upper = (float(value) for value in raw)
    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower <= 0.0
        or upper > 1.0
        or lower > upper
    ):
        raise ValueError("size_range must satisfy 0 < f_min <= f_max <= 1")
    return lower, upper


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)
