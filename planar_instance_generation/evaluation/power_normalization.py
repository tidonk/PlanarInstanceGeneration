"""Component-wise scaling for PGLib composition sensitivity studies."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Iterable, Mapping

import numpy as np

from planar_instance_generation.io.pglib import PglibCase


@dataclass(frozen=True, slots=True)
class ComponentScale:
    """Scale applied to one source piece in a composite."""

    source: str
    bus_count: int
    absolute_pd_mw: float
    absolute_pd_per_bus_mw: float
    apparent_demand_mva: float
    apparent_demand_per_kv: float
    voltage_aware_density_mva_per_kv_bus: float
    target_voltage_aware_density_mva_per_kv_bus: float
    factor: float


@dataclass(frozen=True, slots=True)
class CouplingVoltageMetrics:
    """Nominal-voltage mismatch across certified coupling branches."""

    coupling_branch_count: int
    same_voltage_fraction: float
    within_ten_percent_fraction: float
    median_voltage_ratio: float
    maximum_voltage_ratio: float


@dataclass(frozen=True, slots=True)
class ComponentNormalizationResult:
    """A normalized case and an audit trail for every source piece."""

    case: PglibCase
    components: tuple[ComponentScale, ...]
    voltage: CouplingVoltageMetrics


def normalize_component_voltage_aware_load(
    case: PglibCase,
    source_bus_groups: Mapping[str, Iterable[int]],
    coupling_branch_rows: Iterable[int],
    *,
    name: str | None = None,
) -> ComponentNormalizationResult:
    """Equalize voltage-aware apparent-demand density across source pieces.

    For each source component, all bus injections, generator power quantities,
    branch charging, and source-branch thermal ratings are multiplied by the
    factor that brings its sum of apparent demand divided by local bus base-kV,
    per retained bus, to the within-composite median. This is proportional to
    current density; the common inverse-square-root-of-three factor cancels.
    MATPOWER branch ratings are apparent-power limits in MVA, so this already
    base-kV-derived factor is applied to RATE_A/B/C exactly once: multiplying
    the ratings by a second endpoint-voltage ratio would double-count voltage.
    Source resistance and reactance are divided by the same factor. Topology,
    voltage bases, voltage limits, transformer taps, and coupling branches are
    unchanged.
    Generator cost curves are reparameterized so that the cost of a scaled
    dispatch equals the cost of its unscaled counterpart.
    """

    case.validate()
    groups = {
        str(source): tuple(int(bus) for bus in buses)
        for source, buses in source_bus_groups.items()
    }
    if not groups or any(not source or not buses for source, buses in groups.items()):
        raise ValueError("source_bus_groups must contain nonempty named groups")
    known_buses = {int(row[0]) for row in case.bus}
    flattened = tuple(bus for buses in groups.values() for bus in buses)
    if len(flattened) != len(set(flattened)):
        raise ValueError("source bus groups must be disjoint")
    if set(flattened) != known_buses:
        raise ValueError("source bus groups must partition every case bus")

    coupling = tuple(int(row) for row in coupling_branch_rows)
    if len(coupling) != len(set(coupling)):
        raise ValueError("coupling branch rows must be distinct")
    if any(row < 0 or row >= case.branch.shape[0] for row in coupling):
        raise ValueError("coupling branch row is outside the case")
    coupling_set = set(coupling)

    bus_rows = {int(row[0]): index for index, row in enumerate(case.bus)}
    source_by_bus = {
        bus: source for source, buses in groups.items() for bus in buses
    }
    density_by_source: dict[str, float] = {}
    demand_by_source: dict[str, float] = {}
    apparent_demand_by_source: dict[str, float] = {}
    apparent_demand_per_kv_by_source: dict[str, float] = {}
    for source, buses in groups.items():
        demand = 0.0
        apparent_demand = 0.0
        apparent_demand_per_kv = 0.0
        for bus in buses:
            bus_row = case.bus[bus_rows[bus]]
            voltage = float(bus_row[9])
            if not math.isfinite(voltage) or voltage <= 0.0:
                raise ValueError(
                    f"source component {source!r} has a nonpositive base-kV value"
                )
            active = float(bus_row[2])
            reactive = float(bus_row[3])
            apparent = math.hypot(active, reactive)
            demand += abs(active)
            apparent_demand += apparent
            apparent_demand_per_kv += apparent / voltage
        density = apparent_demand_per_kv / len(buses)
        if not math.isfinite(density) or density <= 0.0:
            raise ValueError(
                f"source component {source!r} has nonpositive "
                "voltage-aware demand density"
            )
        demand_by_source[source] = demand
        apparent_demand_by_source[source] = apparent_demand
        apparent_demand_per_kv_by_source[source] = apparent_demand_per_kv
        density_by_source[source] = density
    target = float(statistics.median(density_by_source.values()))
    factor_by_source = {
        source: target / density
        for source, density in density_by_source.items()
    }

    bus = case.bus.copy()
    gen = case.gen.copy()
    branch = case.branch.copy()
    gencost = case.gencost.copy()
    for source, buses in groups.items():
        factor = factor_by_source[source]
        indices = [bus_rows[value] for value in buses]
        bus[np.asarray(indices, dtype=np.int64), 2:6] *= factor

    generator_source: list[str] = []
    generator_power_columns = (
        1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
    )
    for row in range(gen.shape[0]):
        source = source_by_bus[int(gen[row, 0])]
        generator_source.append(source)
        factor = factor_by_source[source]
        for column in generator_power_columns:
            if column < gen.shape[1]:
                gen[row, column] *= factor

    for row in range(branch.shape[0]):
        left = source_by_bus[int(branch[row, 0])]
        right = source_by_bus[int(branch[row, 1])]
        if row in coupling_set:
            if left == right:
                raise ValueError("certified coupling branch lies within one source group")
            continue
        if left != right:
            raise ValueError("uncertified branch crosses source groups")
        factor = factor_by_source[left]
        branch[row, 2:4] /= factor
        branch[row, 4] *= factor
        branch[row, 5:8] *= factor

    _reparameterize_gencost(gencost, generator_source, factor_by_source)
    normalized = PglibCase(
        name=name or f"{case.name}_voltage_aware_normalized",
        version=case.version,
        base_mva=case.base_mva,
        bus=bus,
        gen=gen,
        branch=branch,
        gencost=gencost,
        areas=None if case.areas is None else case.areas.copy(),
    )
    normalized.validate()
    components = tuple(
        ComponentScale(
            source=source,
            bus_count=len(groups[source]),
            absolute_pd_mw=demand_by_source[source],
            absolute_pd_per_bus_mw=demand_by_source[source] / len(groups[source]),
            apparent_demand_mva=apparent_demand_by_source[source],
            apparent_demand_per_kv=apparent_demand_per_kv_by_source[source],
            voltage_aware_density_mva_per_kv_bus=density_by_source[source],
            target_voltage_aware_density_mva_per_kv_bus=target,
            factor=factor_by_source[source],
        )
        for source in groups
    )
    return ComponentNormalizationResult(
        case=normalized,
        components=components,
        voltage=coupling_voltage_metrics(case, coupling),
    )


def coupling_voltage_metrics(
    case: PglibCase, coupling_branch_rows: Iterable[int]
) -> CouplingVoltageMetrics:
    """Measure endpoint base-kV ratios on certified coupling branches."""

    rows = tuple(int(row) for row in coupling_branch_rows)
    if not rows:
        raise ValueError("at least one coupling branch row is required")
    if len(rows) != len(set(rows)):
        raise ValueError("coupling branch rows must be distinct")
    if any(row < 0 or row >= case.branch.shape[0] for row in rows):
        raise ValueError("coupling branch row is outside the case")
    voltage_by_bus = {int(row[0]): float(row[9]) for row in case.bus}
    ratios = []
    same = 0
    within = 0
    for row in rows:
        left, right = (int(value) for value in case.branch[row, :2])
        low, high = sorted((voltage_by_bus[left], voltage_by_bus[right]))
        if low <= 0 or not math.isfinite(low) or not math.isfinite(high):
            raise ValueError("coupling endpoint voltage bases must be positive and finite")
        ratio = high / low
        ratios.append(ratio)
        same += int(math.isclose(ratio, 1.0, rel_tol=0.0, abs_tol=1e-12))
        within += int(ratio <= 1.1)
    return CouplingVoltageMetrics(
        coupling_branch_count=len(rows),
        same_voltage_fraction=same / len(rows),
        within_ten_percent_fraction=within / len(rows),
        median_voltage_ratio=float(statistics.median(ratios)),
        maximum_voltage_ratio=max(ratios),
    )


def _reparameterize_gencost(
    gencost: np.ndarray,
    generator_sources: list[str],
    factors: Mapping[str, float],
) -> None:
    generator_count = len(generator_sources)
    for row in range(gencost.shape[0]):
        generator = row if row < generator_count else row - generator_count
        if generator >= generator_count:
            continue
        factor = factors[generator_sources[generator]]
        model = int(gencost[row, 0])
        count = int(gencost[row, 3])
        if model == 1:
            for point in range(count):
                column = 4 + 2 * point
                if column < gencost.shape[1]:
                    gencost[row, column] *= factor
        elif model == 2:
            start = 4
            for offset in range(count):
                column = start + offset
                if column < gencost.shape[1]:
                    power = count - offset - 1
                    gencost[row, column] /= factor**power
        else:
            raise ValueError(f"unsupported MATPOWER generator-cost model {model}")
