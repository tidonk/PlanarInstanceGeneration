"""Compose two connected MATPOWER cases with explicit DC coupling branches."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from planar_instance_generation.io.pglib import PglibCase


FloatMatrix = NDArray[np.float64]
FloatVector = NDArray[np.float64]
EndpointPair = tuple[int, int]

__all__ = [
    "CouplingBranchConfig",
    "CouplingBranchParameters",
    "EndpointPair",
    "PowerMergeResult",
    "SourceCaseMapping",
    "compose_power_cases",
]


@dataclass(frozen=True, slots=True)
class CouplingBranchConfig:
    """Optional overrides for the deliberately simple DC transformer model.

    The stable ``1e-4`` default avoids progressively shrinking the reactance
    in sequential joins. ``rate_mva=None`` writes MATPOWER's exact unlimited
    rating value (zero). Its conventional unconstrained angle interval is the
    default as well. Explicit overrides are useful for controlled experiments.
    """

    reactance_pu: float = 1e-4
    rate_mva: float | None = None
    angle_min_degrees: float = -360.0
    angle_max_degrees: float = 360.0


@dataclass(frozen=True, slots=True)
class CouplingBranchParameters:
    """Numerical parameters actually written to every added branch row."""

    reactance_pu: float
    rate_mva: float
    angle_min_degrees: float
    angle_max_degrees: float


@dataclass(frozen=True, slots=True)
class SourceCaseMapping:
    """Map source identifiers and physical rows to merged counterparts.

    Demoted reference buses are expressed in the current merged bus IDs.
    """

    source_name: str
    bus_ids: Mapping[int, int]
    area_ids: Mapping[int, int]
    bus_rows: tuple[int, ...]
    generator_rows: tuple[int, ...]
    branch_rows: tuple[int, ...]
    gencost_rows: tuple[int, ...]
    area_rows: tuple[int, ...]
    demoted_reference_buses: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class PowerMergeResult:
    """A composed case plus complete source and coupling provenance.

    ``source_mappings`` is flattened across sequential compositions, while
    ``left`` and ``right`` describe the two immediate inputs.
    """

    case: PglibCase
    left: SourceCaseMapping
    right: SourceCaseMapping
    source_endpoint_pairs: tuple[EndpointPair, ...]
    merged_endpoint_pairs: tuple[EndpointPair, ...]
    source_mappings: tuple[SourceCaseMapping, ...]
    coupling_branch_rows: tuple[int, ...]
    new_coupling_branch_rows: tuple[int, ...]
    synthetic_gencost_rows: tuple[int, ...]
    coupling_parameters: CouplingBranchParameters


def compose_power_cases(
    left: PglibCase | PowerMergeResult,
    right: PglibCase | PowerMergeResult,
    endpoint_pairs: Iterable[EndpointPair],
    *,
    name: str | None = None,
    config: CouplingBranchConfig | None = None,
) -> PowerMergeResult:
    """Join two connected cases at a one-to-one set of supplied bus pairs.

    Left bus identifiers are always retained. Right identifiers are retained
    when already disjoint and otherwise translated as one block above the
    largest left identifier. Exactly one reference bus is retained: the
    right-hand reference is demoted to PV (or PQ without an online generator).
    Pass a prior :class:`PowerMergeResult` directly in a sequential merge to
    retain and reindex its original-source provenance.
    """

    if not isinstance(left, (PglibCase, PowerMergeResult)) or not isinstance(
        right, (PglibCase, PowerMergeResult)
    ):
        raise TypeError("left and right must be PglibCase or PowerMergeResult instances")
    if config is not None and not isinstance(config, CouplingBranchConfig):
        raise TypeError("config must be a CouplingBranchConfig or None")
    left_case = left.case if isinstance(left, PowerMergeResult) else left
    right_case = right.case if isinstance(right, PowerMergeResult) else right
    left_case.validate()
    right_case.validate()
    _validate_source_case("left", left_case)
    _validate_source_case("right", right_case)
    if not math.isclose(
        left_case.base_mva, right_case.base_mva, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("source cases must use the same baseMVA")

    pairs = _validate_endpoint_pairs(left_case, right_case, endpoint_pairs)
    requested = CouplingBranchConfig() if config is None else config
    parameters = _resolve_coupling_parameters(requested)

    left_bus_ids = tuple(int(value) for value in left_case.bus[:, 0])
    right_bus_ids = tuple(int(value) for value in right_case.bus[:, 0])
    left_bus_map = {bus: bus for bus in left_bus_ids}
    right_bus_map = _disjoint_translation(left_bus_ids, right_bus_ids)

    left_area_map, right_area_map = _area_mappings(left_case, right_case)
    left_bus = _mapped_bus_table(left_case, left_bus_map, left_area_map)
    right_bus = _mapped_bus_table(right_case, right_bus_map, right_area_map)
    demoted_references = _demote_reference_bus(
        right_bus, right_case.gen, right_bus_map
    )

    bus_width = max(left_case.bus.shape[1], right_case.bus.shape[1])
    gen_width = max(left_case.gen.shape[1], right_case.gen.shape[1])
    branch_width = max(left_case.branch.shape[1], right_case.branch.shape[1], 13)
    buses = np.vstack((_pad_columns(left_bus, bus_width), _pad_columns(right_bus, bus_width)))
    generators = np.vstack(
        (
            _pad_columns(_mapped_generator_table(left_case, left_bus_map), gen_width),
            _pad_columns(_mapped_generator_table(right_case, right_bus_map), gen_width),
        )
    )
    source_branches = np.vstack(
        (
            _pad_columns(_mapped_branch_table(left_case, left_bus_map), branch_width),
            _pad_columns(_mapped_branch_table(right_case, right_bus_map), branch_width),
        )
    )

    merged_pairs = tuple((left_bus_map[a], right_bus_map[b]) for a, b in pairs)
    new_coupling_rows = tuple(
        range(source_branches.shape[0], source_branches.shape[0] + len(merged_pairs))
    )
    coupling_branches = np.vstack(
        [_coupling_branch(branch_width, pair, parameters) for pair in merged_pairs]
    )
    branches = np.vstack((source_branches, coupling_branches))

    costs, left_cost_rows, right_cost_rows, new_synthetic_cost_rows = _combine_gencost(
        left_case, right_case
    )
    areas, left_area_rows, right_area_rows = _combine_areas(
        left_case,
        right_case,
        left_bus_map,
        right_bus_map,
        left_area_map,
        right_area_map,
    )

    left_mapping = SourceCaseMapping(
        source_name=left_case.name,
        bus_ids=left_bus_map.copy(),
        area_ids=left_area_map.copy(),
        bus_rows=tuple(range(left_case.bus.shape[0])),
        generator_rows=tuple(range(left_case.gen.shape[0])),
        branch_rows=tuple(range(left_case.branch.shape[0])),
        gencost_rows=left_cost_rows,
        area_rows=left_area_rows,
    )
    right_mapping = SourceCaseMapping(
        source_name=right_case.name,
        bus_ids=right_bus_map.copy(),
        area_ids=right_area_map.copy(),
        bus_rows=tuple(range(left_case.bus.shape[0], buses.shape[0])),
        generator_rows=tuple(range(left_case.gen.shape[0], generators.shape[0])),
        branch_rows=tuple(
            range(
                left_case.branch.shape[0],
                left_case.branch.shape[0] + right_case.branch.shape[0],
            )
        ),
        gencost_rows=right_cost_rows,
        area_rows=right_area_rows,
        demoted_reference_buses=demoted_references,
    )
    source_mappings = _propagate_source_mappings(left, left_mapping) + (
        _propagate_source_mappings(right, right_mapping)
    )
    prior_coupling_rows = _propagate_coupling_rows(left, left_mapping) + (
        _propagate_coupling_rows(right, right_mapping)
    )
    prior_synthetic_cost_rows = _propagate_synthetic_cost_rows(
        left, left_cost_rows
    ) + _propagate_synthetic_cost_rows(right, right_cost_rows)

    merged = PglibCase(
        name=name or f"{left_case.name}_{right_case.name}",
        version="2",
        base_mva=left_case.base_mva,
        bus=buses,
        gen=generators,
        branch=branches,
        gencost=costs,
        areas=areas,
    )
    if not nx.is_connected(merged.topology(active_only=True)):
        raise RuntimeError("composition produced a disconnected active topology")
    return PowerMergeResult(
        case=merged,
        left=left_mapping,
        right=right_mapping,
        source_endpoint_pairs=pairs,
        merged_endpoint_pairs=merged_pairs,
        source_mappings=source_mappings,
        coupling_branch_rows=prior_coupling_rows + new_coupling_rows,
        new_coupling_branch_rows=new_coupling_rows,
        synthetic_gencost_rows=prior_synthetic_cost_rows + new_synthetic_cost_rows,
        coupling_parameters=parameters,
    )


def _validate_source_case(label: str, case: PglibCase) -> None:
    graph = case.topology(active_only=True)
    if graph.number_of_nodes() == 0 or not nx.is_connected(graph):
        raise ValueError(f"{label} case must have a connected active topology")
    references = np.flatnonzero(case.bus[:, 1] == 3)
    if len(references) != 1:
        raise ValueError(f"{label} case must contain exactly one reference bus")


def _validate_endpoint_pairs(
    left: PglibCase,
    right: PglibCase,
    endpoint_pairs: Iterable[EndpointPair],
) -> tuple[EndpointPair, ...]:
    left_buses = {int(value) for value in left.bus[:, 0]}
    right_buses = {int(value) for value in right.bus[:, 0]}
    normalized: list[EndpointPair] = []
    for index, raw_pair in enumerate(endpoint_pairs):
        try:
            pair = tuple(raw_pair)
        except TypeError as error:
            raise TypeError(f"endpoint_pairs[{index}] must be a pair") from error
        if len(pair) != 2:
            raise ValueError(f"endpoint_pairs[{index}] must contain two bus identifiers")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in pair
        ):
            raise TypeError(f"endpoint_pairs[{index}] must contain integer bus identifiers")
        normalized.append((int(pair[0]), int(pair[1])))
    if not normalized:
        raise ValueError("endpoint_pairs must contain at least one pair")
    if len(set(normalized)) != len(normalized):
        raise ValueError("endpoint_pairs must not contain duplicate pairs")
    left_endpoints = [pair[0] for pair in normalized]
    right_endpoints = [pair[1] for pair in normalized]
    if len(set(left_endpoints)) != len(left_endpoints):
        raise ValueError("a left endpoint must not occur in more than one pair")
    if len(set(right_endpoints)) != len(right_endpoints):
        raise ValueError("a right endpoint must not occur in more than one pair")
    unknown_left = set(left_endpoints) - left_buses
    unknown_right = set(right_endpoints) - right_buses
    if unknown_left or unknown_right:
        details = []
        if unknown_left:
            details.append(f"unknown left buses {sorted(unknown_left)}")
        if unknown_right:
            details.append(f"unknown right buses {sorted(unknown_right)}")
        raise ValueError("; ".join(details))
    return tuple(normalized)


def _propagate_source_mappings(
    source: PglibCase | PowerMergeResult,
    immediate: SourceCaseMapping,
) -> tuple[SourceCaseMapping, ...]:
    if isinstance(source, PglibCase):
        return (immediate,)

    propagated = []
    for original in source.source_mappings:
        bus_ids = {
            source_bus: immediate.bus_ids[current_bus]
            for source_bus, current_bus in original.bus_ids.items()
        }
        area_ids = {
            source_area: immediate.area_ids[current_area]
            for source_area, current_area in original.area_ids.items()
        }
        demoted = tuple(
            immediate.bus_ids[current_bus]
            for current_bus in original.demoted_reference_buses
        )
        for newly_demoted in immediate.demoted_reference_buses:
            if newly_demoted in bus_ids.values() and newly_demoted not in demoted:
                demoted += (newly_demoted,)
        propagated.append(
            SourceCaseMapping(
                source_name=original.source_name,
                bus_ids=bus_ids,
                area_ids=area_ids,
                bus_rows=_map_rows(original.bus_rows, immediate.bus_rows),
                generator_rows=_map_rows(
                    original.generator_rows, immediate.generator_rows
                ),
                branch_rows=_map_rows(original.branch_rows, immediate.branch_rows),
                gencost_rows=_map_rows(original.gencost_rows, immediate.gencost_rows),
                area_rows=_map_rows(original.area_rows, immediate.area_rows),
                demoted_reference_buses=demoted,
            )
        )
    return tuple(propagated)


def _propagate_coupling_rows(
    source: PglibCase | PowerMergeResult,
    immediate: SourceCaseMapping,
) -> tuple[int, ...]:
    if isinstance(source, PglibCase):
        return ()
    return _map_rows(source.coupling_branch_rows, immediate.branch_rows)


def _propagate_synthetic_cost_rows(
    source: PglibCase | PowerMergeResult,
    row_map: tuple[int, ...],
) -> tuple[int, ...]:
    if isinstance(source, PglibCase):
        return ()
    return _map_rows(source.synthetic_gencost_rows, row_map)


def _map_rows(rows: tuple[int, ...], row_map: tuple[int, ...]) -> tuple[int, ...]:
    try:
        return tuple(row_map[row] for row in rows)
    except IndexError as error:
        raise ValueError("stored source-row provenance does not match its case") from error


def _disjoint_translation(left_ids: tuple[int, ...], right_ids: tuple[int, ...]) -> dict[int, int]:
    if set(left_ids).isdisjoint(right_ids):
        return {bus: bus for bus in right_ids}
    offset = max(left_ids) - min(right_ids) + 1
    return {bus: bus + offset for bus in right_ids}


def _area_mappings(
    left: PglibCase, right: PglibCase
) -> tuple[dict[int, int], dict[int, int]]:
    left_ids = _area_ids(left)
    right_ids = _area_ids(right)
    left_map = {area: area for area in left_ids}
    right_map = _disjoint_translation(left_ids, right_ids)
    return left_map, right_map


def _area_ids(case: PglibCase) -> tuple[int, ...]:
    values = list(case.bus[:, 6])
    if case.areas is not None:
        values.extend(case.areas[:, 0])
    rounded = np.rint(values)
    if not np.array_equal(values, rounded):
        raise ValueError(f"{case.name} contains non-integer area identifiers")
    return tuple(sorted({int(value) for value in rounded}))


def _mapped_bus_table(
    case: PglibCase,
    bus_map: Mapping[int, int],
    area_map: Mapping[int, int],
) -> FloatMatrix:
    table = case.bus.copy()
    for row in table:
        row[0] = bus_map[int(row[0])]
        row[6] = area_map[int(row[6])]
    return table


def _mapped_generator_table(case: PglibCase, bus_map: Mapping[int, int]) -> FloatMatrix:
    table = case.gen.copy()
    for row in table:
        row[0] = bus_map[int(row[0])]
    return table


def _mapped_branch_table(case: PglibCase, bus_map: Mapping[int, int]) -> FloatMatrix:
    table = case.branch.copy()
    for row in table:
        row[0] = bus_map[int(row[0])]
        row[1] = bus_map[int(row[1])]
    return table


def _demote_reference_bus(
    mapped_buses: FloatMatrix,
    source_generators: FloatMatrix,
    bus_map: Mapping[int, int],
) -> tuple[int, ...]:
    source_references = tuple(int(row[0]) for row in mapped_buses if row[1] == 3)
    online_generator_buses = {
        bus_map[int(row[0])] for row in source_generators if row[7] > 0
    }
    for output_bus in source_references:
        row_index = int(np.flatnonzero(mapped_buses[:, 0] == output_bus)[0])
        mapped_buses[row_index, 1] = 2 if output_bus in online_generator_buses else 1
    return source_references


def _resolve_coupling_parameters(
    config: CouplingBranchConfig,
) -> CouplingBranchParameters:
    for label, value in (
        ("angle_min_degrees", config.angle_min_degrees),
        ("angle_max_degrees", config.angle_max_degrees),
    ):
        if isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"{label} must be finite")
    if config.angle_min_degrees >= config.angle_max_degrees:
        raise ValueError("angle_min_degrees must be smaller than angle_max_degrees")

    reactance = config.reactance_pu
    if isinstance(reactance, bool) or not math.isfinite(reactance) or reactance <= 0:
        raise ValueError("reactance_pu must be finite and positive")

    rate = 0.0 if config.rate_mva is None else config.rate_mva
    if isinstance(rate, bool) or not math.isfinite(rate) or rate < 0:
        raise ValueError("rate_mva must be finite and nonnegative")
    return CouplingBranchParameters(
        reactance_pu=float(reactance),
        rate_mva=float(rate),
        angle_min_degrees=float(config.angle_min_degrees),
        angle_max_degrees=float(config.angle_max_degrees),
    )


def _coupling_branch(
    width: int,
    pair: EndpointPair,
    parameters: CouplingBranchParameters,
) -> FloatVector:
    row = np.zeros(width, dtype=np.float64)
    row[0:2] = pair
    row[3] = parameters.reactance_pu
    row[5:8] = parameters.rate_mva
    row[8] = 1.0  # nonzero tap ratio makes the coupling an explicit transformer
    row[9] = 0.0  # phase-shift compensation is deliberately forbidden
    row[10] = 1.0
    row[11] = parameters.angle_min_degrees
    row[12] = parameters.angle_max_degrees
    return row


def _combine_gencost(
    left: PglibCase, right: PglibCase
) -> tuple[FloatMatrix, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    width = max(left.gencost.shape[1], right.gencost.shape[1])
    left_costs = _pad_columns(left.gencost, width)
    right_costs = _pad_columns(right.gencost, width)
    left_count = left.gen.shape[0]
    right_count = right.gen.shape[0]
    left_has_reactive = left_count > 0 and left_costs.shape[0] == 2 * left_count
    right_has_reactive = right_count > 0 and right_costs.shape[0] == 2 * right_count

    left_active = left_costs[:left_count]
    right_active = right_costs[:right_count]
    total_generators = left_count + right_count
    if not left_has_reactive and not right_has_reactive:
        combined = np.vstack((left_active, right_active))
        return (
            combined,
            tuple(range(left_count)),
            tuple(range(left_count, total_generators)),
            (),
        )

    left_reactive = (
        left_costs[left_count:] if left_has_reactive else _zero_costs(left_count, width)
    )
    right_reactive = (
        right_costs[right_count:] if right_has_reactive else _zero_costs(right_count, width)
    )
    combined = np.vstack((left_active, right_active, left_reactive, right_reactive))
    left_rows = tuple(range(left_count))
    right_rows = tuple(range(left_count, total_generators))
    if left_has_reactive:
        left_rows += tuple(range(total_generators, total_generators + left_count))
    if right_has_reactive:
        right_rows += tuple(
            range(total_generators + left_count, 2 * total_generators)
        )
    synthetic_rows = ()
    if not left_has_reactive:
        synthetic_rows += tuple(range(total_generators, total_generators + left_count))
    if not right_has_reactive:
        synthetic_rows += tuple(
            range(total_generators + left_count, 2 * total_generators)
        )
    return combined, left_rows, right_rows, synthetic_rows


def _zero_costs(row_count: int, width: int) -> FloatMatrix:
    if row_count and width < 5:
        raise ValueError("cannot synthesize reactive costs with fewer than five columns")
    rows = np.zeros((row_count, width), dtype=np.float64)
    if row_count:
        rows[:, 0] = 2.0
        rows[:, 3] = 1.0
    return rows


def _combine_areas(
    left: PglibCase,
    right: PglibCase,
    left_bus_map: Mapping[int, int],
    right_bus_map: Mapping[int, int],
    left_area_map: Mapping[int, int],
    right_area_map: Mapping[int, int],
) -> tuple[FloatMatrix | None, tuple[int, ...], tuple[int, ...]]:
    if left.areas is None and right.areas is None:
        return None, (), ()
    width = max(
        left.areas.shape[1] if left.areas is not None else 2,
        right.areas.shape[1] if right.areas is not None else 2,
    )
    left_table = _mapped_area_table(left, left_bus_map, left_area_map, width)
    right_table = _mapped_area_table(right, right_bus_map, right_area_map, width)
    combined = np.vstack((left_table, right_table))
    left_rows = tuple(range(left_table.shape[0]))
    right_rows = tuple(range(left_table.shape[0], combined.shape[0]))
    return combined, left_rows, right_rows


def _mapped_area_table(
    case: PglibCase,
    bus_map: Mapping[int, int],
    area_map: Mapping[int, int],
    width: int,
) -> FloatMatrix:
    if case.areas is None:
        return np.empty((0, width), dtype=np.float64)
    table = _pad_columns(case.areas, width)
    for row in table:
        area = int(row[0])
        reference_bus = int(row[1])
        if reference_bus not in bus_map:
            raise ValueError(f"{case.name} area {area} references unknown bus {reference_bus}")
        row[0] = area_map[area]
        row[1] = bus_map[reference_bus]
    return table


def _pad_columns(table: FloatMatrix, width: int) -> FloatMatrix:
    if table.shape[1] == width:
        return table.copy()
    padded = np.zeros((table.shape[0], width), dtype=np.float64)
    padded[:, : table.shape[1]] = table
    return padded
