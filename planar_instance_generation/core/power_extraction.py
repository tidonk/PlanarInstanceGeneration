"""Induced MATPOWER cases with DC cut-flow compensation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Literal, Mapping

import numpy as np
from numpy.typing import NDArray

from planar_instance_generation.core.extraction import randomized_bfs_extraction
from planar_instance_generation.io.pglib import PglibCase


CompensationMode = Literal["fixed", "flexible"]
FloatVector = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PowerSubcase:
    """An induced case and its trace back to physical source rows.

    ``supply_compensation_mw`` uses the paper's sign convention: positive is
    added supply and negative is added demand. Branch and generator indices
    always refer to rows in the source case.
    """

    case: PglibCase
    selected_buses: tuple[int, ...]
    internal_branch_indices: tuple[int, ...]
    cut_branch_indices: tuple[int, ...]
    retained_generator_indices: tuple[int, ...]
    supply_compensation_mw: Mapping[int, float]
    compensation_generator_indices: tuple[int, ...]
    promoted_reference_bus: int | None


def extract_random_power_subcase(
    source: PglibCase,
    fraction: float | None,
    branch_from_flows_mw: FloatVector,
    branch_to_flows_mw: FloatVector | None = None,
    *,
    target_nodes: int | None = None,
    compensation: CompensationMode = "flexible",
    seed: int | None = None,
    rng: random.Random | None = None,
    name: str | None = None,
) -> PowerSubcase:
    """Select buses by the paper's randomized BFS and build their subcase."""

    selection = randomized_bfs_extraction(
        source.topology(active_only=True),
        fraction,
        target_nodes=target_nodes,
        seed=seed,
        rng=rng,
    )
    return build_power_subcase(
        source,
        (int(bus) for bus in selection.selected_nodes),
        branch_from_flows_mw,
        branch_to_flows_mw,
        compensation=compensation,
        name=name,
    )


def build_power_subcase(
    source: PglibCase,
    selected_buses: Iterable[int],
    branch_from_flows_mw: FloatVector,
    branch_to_flows_mw: FloatVector | None = None,
    *,
    compensation: CompensationMode = "flexible",
    name: str | None = None,
) -> PowerSubcase:
    """Build the bus-induced subcase and compensate removed active branches.

    For a retained terminal, the required supply change is the negative of
    that terminal's source-solution injection into the removed branch. Fixed
    compensation changes ``Pd`` directly. Flexible compensation realizes the
    interval ``[min(0, delta), max(0, delta)]`` with a zero-cost dispatchable
    term, using an added load plus generator when ``delta < 0`` so generated
    MATPOWER cases never need negative generator output.
    """

    mode = _validate_compensation_mode(compensation)
    selected = _validate_selected_buses(source, selected_buses)
    selected_set = set(selected)
    from_flows, to_flows = _validate_branch_flows(
        source, branch_from_flows_mw, branch_to_flows_mw
    )

    bus_mask = np.fromiter(
        (int(row[0]) in selected_set for row in source.bus),
        dtype=np.bool_,
        count=source.bus.shape[0],
    )
    generator_mask = np.fromiter(
        (int(row[0]) in selected_set for row in source.gen),
        dtype=np.bool_,
        count=source.gen.shape[0],
    )

    internal_branches: list[int] = []
    cut_branches: list[int] = []
    delta_by_bus = {bus: 0.0 for bus in selected}
    for index, row in enumerate(source.branch):
        from_bus, to_bus = int(row[0]), int(row[1])
        from_inside = from_bus in selected_set
        to_inside = to_bus in selected_set
        if from_inside and to_inside:
            internal_branches.append(index)
        elif from_inside != to_inside:
            cut_branches.append(index)
            if row[10] > 0:
                inside_bus = from_bus if from_inside else to_bus
                terminal_flow = from_flows[index] if from_inside else to_flows[index]
                delta_by_bus[inside_bus] -= float(terminal_flow)

    buses = source.bus[bus_mask].copy()
    generators = source.gen[generator_mask].copy()
    retained_generator_indices = tuple(np.flatnonzero(generator_mask).tolist())
    costs = _select_generator_costs(source, generator_mask)

    nonzero_delta = {
        bus: delta
        for bus, delta in delta_by_bus.items()
        if not math.isclose(delta, 0.0, abs_tol=1e-9)
    }
    compensation_indices: list[int] = []
    if mode == "fixed":
        _apply_fixed_compensation(buses, nonzero_delta)
    else:
        generators, costs, compensation_indices = _apply_flexible_compensation(
            source,
            buses,
            generators,
            costs,
            nonzero_delta,
        )

    promoted_reference = _ensure_reference_bus(buses, generators)
    subcase = PglibCase(
        name=name or f"{source.name}_sub",
        version=source.version,
        base_mva=source.base_mva,
        bus=buses,
        gen=generators,
        branch=source.branch[internal_branches].copy(),
        gencost=costs,
        areas=_select_areas(source, buses),
    )
    return PowerSubcase(
        case=subcase,
        selected_buses=selected,
        internal_branch_indices=tuple(internal_branches),
        cut_branch_indices=tuple(cut_branches),
        retained_generator_indices=retained_generator_indices,
        supply_compensation_mw=nonzero_delta,
        compensation_generator_indices=tuple(compensation_indices),
        promoted_reference_bus=promoted_reference,
    )


def _validate_compensation_mode(value: str) -> CompensationMode:
    if value not in {"fixed", "flexible"}:
        raise ValueError("compensation must be 'fixed' or 'flexible'")
    return value  # type: ignore[return-value]


def _validate_selected_buses(
    source: PglibCase, selected_buses: Iterable[int]
) -> tuple[int, ...]:
    selected = tuple(selected_buses)
    if not selected:
        raise ValueError("selected_buses must not be empty")
    if any(isinstance(bus, bool) or not isinstance(bus, (int, np.integer)) for bus in selected):
        raise TypeError("selected_buses must contain integer bus identifiers")
    normalized = tuple(int(bus) for bus in selected)
    if len(set(normalized)) != len(normalized):
        raise ValueError("selected_buses must not contain duplicates")
    known = {int(row[0]) for row in source.bus}
    unknown = set(normalized) - known
    if unknown:
        raise ValueError(f"selected_buses contains unknown bus(es): {sorted(unknown)}")
    return normalized


def _validate_branch_flows(
    source: PglibCase,
    branch_from_flows_mw: FloatVector,
    branch_to_flows_mw: FloatVector | None,
) -> tuple[FloatVector, FloatVector]:
    from_flows = np.asarray(branch_from_flows_mw, dtype=np.float64)
    to_flows = -from_flows if branch_to_flows_mw is None else np.asarray(
        branch_to_flows_mw, dtype=np.float64
    )
    expected = source.branch.shape[0]
    for label, vector in (("branch_from_flows_mw", from_flows), ("branch_to_flows_mw", to_flows)):
        if vector.ndim != 1 or vector.shape[0] != expected:
            raise ValueError(f"{label} must be a vector of length {expected}")
        if not np.isfinite(vector).all():
            raise ValueError(f"{label} must contain only finite values")
    return from_flows, to_flows


def _select_generator_costs(source: PglibCase, mask: NDArray[np.bool_]) -> NDArray[np.float64]:
    generator_count = source.gen.shape[0]
    active_costs = source.gencost[:generator_count][mask].copy()
    if source.gencost.shape[0] == generator_count:
        return active_costs
    reactive_costs = source.gencost[generator_count:][mask].copy()
    return np.vstack((active_costs, reactive_costs))


def _apply_fixed_compensation(
    buses: NDArray[np.float64], delta_by_bus: Mapping[int, float]
) -> None:
    row_by_bus = {int(row[0]): index for index, row in enumerate(buses)}
    for bus, delta_supply in delta_by_bus.items():
        buses[row_by_bus[bus], 2] -= delta_supply


def _apply_flexible_compensation(
    source: PglibCase,
    buses: NDArray[np.float64],
    generators: NDArray[np.float64],
    costs: NDArray[np.float64],
    delta_by_bus: Mapping[int, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[int]]:
    if not delta_by_bus:
        return generators, costs, []
    if source.gencost.shape[1] < 5:
        raise ValueError("flexible compensation needs at least five gencost columns")

    row_by_bus = {int(row[0]): index for index, row in enumerate(buses)}
    added_generators = []
    for bus, delta_supply in delta_by_bus.items():
        if delta_supply < 0:
            buses[row_by_bus[bus], 2] -= delta_supply
        added_generators.append(
            _compensation_generator(
                source,
                buses[row_by_bus[bus]],
                bus,
                delta_supply,
            )
        )

    original_count = generators.shape[0]
    added = np.vstack(added_generators)
    combined_generators = np.vstack((generators, added))
    added_indices = list(range(original_count, combined_generators.shape[0]))
    combined_costs = _append_zero_costs(
        costs,
        source_generator_count=sum(int(row[0]) in row_by_bus for row in source.gen),
        added_count=len(added_generators),
        has_reactive_costs=source.gencost.shape[0] == 2 * source.gen.shape[0],
    )
    return combined_generators, combined_costs, added_indices


def _compensation_generator(
    source: PglibCase,
    bus_row: NDArray[np.float64],
    bus: int,
    delta_supply: float,
) -> NDArray[np.float64]:
    row = np.zeros(source.gen.shape[1], dtype=np.float64)
    row[0] = bus
    row[1] = max(0.0, delta_supply)
    row[5] = bus_row[7] if bus_row[7] > 0 else 1.0
    row[6] = source.base_mva
    row[7] = 1.0
    row[8] = abs(delta_supply)
    row[9] = 0.0
    return row


def _append_zero_costs(
    selected_costs: NDArray[np.float64],
    *,
    source_generator_count: int,
    added_count: int,
    has_reactive_costs: bool,
) -> NDArray[np.float64]:
    zero_costs = np.zeros((added_count, selected_costs.shape[1]), dtype=np.float64)
    zero_costs[:, 0] = 2.0
    zero_costs[:, 3] = selected_costs.shape[1] - 4
    if not has_reactive_costs:
        return np.vstack((selected_costs, zero_costs))

    active = selected_costs[:source_generator_count]
    reactive = selected_costs[source_generator_count:]
    return np.vstack((active, zero_costs, reactive, zero_costs.copy()))


def _ensure_reference_bus(
    buses: NDArray[np.float64], generators: NDArray[np.float64]
) -> int | None:
    if np.any(buses[:, 1] == 3):
        return None
    online = generators[generators[:, 7] != 0]
    if online.shape[0]:
        candidate = online[np.argmax(online[:, 8]), 0]
    else:
        candidate = buses[0, 0]
    bus = int(candidate)
    buses[np.flatnonzero(buses[:, 0] == bus)[0], 1] = 3
    return bus


def _select_areas(
    source: PglibCase, selected_bus_rows: NDArray[np.float64]
) -> NDArray[np.float64] | None:
    if source.areas is None:
        return None
    selected_ids = {int(row[0]) for row in selected_bus_rows}
    buses_by_area: dict[int, list[int]] = {}
    for row in selected_bus_rows:
        buses_by_area.setdefault(int(row[6]), []).append(int(row[0]))

    retained = []
    for area_row in source.areas:
        area = int(area_row[0])
        if area not in buses_by_area:
            continue
        row = area_row.copy()
        if int(row[1]) not in selected_ids:
            row[1] = min(buses_by_area[area])
        retained.append(row)
    return np.vstack(retained) if retained else None
