"""Lossless reference-flow decomposition into point-to-point SNDlib demands."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from math import isclose, isfinite
from pathlib import Path
from typing import TypeAlias

from planar_instance_generation.core.snd import (
    SndCommodity,
    SndInstance,
    SndSolution,
    validate_snd_solution,
)
from planar_instance_generation.io.sndlib import (
    SndlibAdmissiblePath,
    SndlibDemand,
    SndlibLink,
    SndlibNetwork,
    write_sndlib,
)


class SndSerializationError(ValueError):
    """Raised when a core SND instance cannot be serialized without guessing."""


@dataclass(frozen=True, slots=True)
class SndFlowWalk:
    """One positive path or circulation in a signed-flow decomposition."""

    node_ids: tuple[str, ...]
    link_ids: tuple[str, ...]
    link_signs: tuple[int, ...]
    value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_ids", tuple(self.node_ids))
        object.__setattr__(self, "link_ids", tuple(self.link_ids))
        object.__setattr__(self, "link_signs", tuple(self.link_signs))
        if len(self.node_ids) != len(self.link_ids) + 1 or not self.link_ids:
            raise ValueError("a flow walk needs one more node than links")
        if len(self.link_signs) != len(self.link_ids) or any(
            sign not in {-1, 1} for sign in self.link_signs
        ):
            raise ValueError("flow-walk link signs must be +1 or -1")
        if not isfinite(self.value) or self.value <= 0.0:
            raise ValueError("flow-walk value must be finite and positive")

    @property
    def is_cycle(self) -> bool:
        return self.node_ids[0] == self.node_ids[-1]


@dataclass(frozen=True, slots=True)
class SerializedDemandProvenance:
    serialized_demand_id: str
    source_instance: str
    source_commodity_id: str
    node_ids: tuple[str, ...]
    link_ids: tuple[str, ...]
    value: float
    carries_circulation: bool = False


@dataclass(frozen=True, slots=True)
class SndlibSerializationCapability:
    supported: bool
    reasons: tuple[str, ...]

    def require(self) -> None:
        if not self.supported:
            raise SndSerializationError("; ".join(self.reasons))


@dataclass(frozen=True, slots=True)
class SndlibSerialization:
    """A serializable network and its exactly decomposed reference solution."""

    network: SndlibNetwork
    reference_solution: SndSolution
    demand_provenance: tuple[SerializedDemandProvenance, ...]
    reference_paths_in_xml: bool


_Arc: TypeAlias = tuple[str, str, int]


def sndlib_serialization_capability(
    instance: SndInstance,
    *,
    include_reference_paths: bool = False,
) -> SndlibSerializationCapability:
    """Report whether serialization is safe for the requested XML semantics.

    The supported subset excludes inherited ``maxPathLength`` and
    ``admissiblePaths`` restrictions. Per-link aggregate flow does not retain
    path identity, so translating such a commodity after a cut could silently
    weaken or strengthen its original routing rules. The current Julia Benders
    application also reads but does not enforce those fields.
    """

    if not isinstance(instance, SndInstance):
        raise TypeError("instance must be an SndInstance")
    reasons: list[str] = []
    for commodity in instance.commodities:
        demand = commodity.source_demand
        if demand is None:
            continue
        restrictions = []
        if demand.max_path_length is not None:
            restrictions.append("maxPathLength")
        if demand.admissible_paths:
            restrictions.append("admissiblePaths")
        if restrictions:
            reasons.append(
                f"commodity {commodity.id!r} inherits unsupported "
                f"{' and '.join(restrictions)} routing restrictions"
            )

    flows_by_commodity = _flows_by_commodity(instance.reference_solution)
    link_by_id = {link.id: link for link in instance.links}
    for commodity in instance.commodities:
        try:
            walks = _decompose_commodity_flow(
                commodity,
                flows_by_commodity.get(commodity.id, {}),
                link_by_id,
            )
        except SndSerializationError as error:
            reasons.append(str(error))
            continue
        if include_reference_paths and any(walk.is_cycle for walk in walks):
            reasons.append(
                f"commodity {commodity.id!r} contains circulation that cannot "
                "be represented by point-to-point admissible paths exactly"
            )
    return SndlibSerializationCapability(not reasons, tuple(reasons))


def decompose_commodity_flow(
    instance: SndInstance,
    commodity_id: str,
    *,
    tolerance: float = 1e-7,
) -> tuple[SndFlowWalk, ...]:
    """Decompose one signed reference flow into terminal paths and cycles.

    The returned walks reproduce every signed per-link flow exactly within
    ``tolerance``. Paths reproduce the commodity's compensated node supplies;
    residual circulations have no effect on those supplies.
    """

    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    commodity = next(
        (value for value in instance.commodities if value.id == commodity_id),
        None,
    )
    if commodity is None:
        raise ValueError(f"unknown commodity {commodity_id!r}")
    link_by_id = {link.id: link for link in instance.links}
    flows = {
        link_id: value
        for (demand_id, link_id), value in instance.reference_solution.flows.items()
        if demand_id == commodity_id
    }
    return _decompose_commodity_flow(
        commodity, flows, link_by_id, tolerance=tolerance
    )


def _decompose_commodity_flow(
    commodity: SndCommodity,
    commodity_flows: dict[str, float],
    link_by_id: dict[str, SndlibLink],
    *,
    tolerance: float = 1e-7,
) -> tuple[SndFlowWalk, ...]:
    arcs: dict[str, _Arc] = {}
    residual: dict[str, float] = {}
    for link_id, flow in commodity_flows.items():
        if abs(flow) <= tolerance:
            continue
        link = link_by_id[link_id]
        if flow > 0.0:
            arcs[link_id] = (link.source, link.target, 1)
            residual[link_id] = flow
        else:
            arcs[link_id] = (link.target, link.source, -1)
            residual[link_id] = -flow

    adjacency = _adjacency(arcs)
    remaining_supply = {
        node: value
        for node, value in commodity.node_supplies.items()
        if value > tolerance
    }
    remaining_demand = {
        node: -value
        for node, value in commodity.node_supplies.items()
        if value < -tolerance
    }
    walks: list[SndFlowWalk] = []
    while remaining_supply:
        source = min(remaining_supply)
        path_nodes, path_links = _residual_path(
            source, set(remaining_demand), adjacency, residual, tolerance
        )
        target = path_nodes[-1]
        amount = min(
            remaining_supply[source],
            remaining_demand[target],
            *(residual[link_id] for link_id in path_links),
        )
        walks.append(
            SndFlowWalk(
                node_ids=path_nodes,
                link_ids=path_links,
                link_signs=tuple(arcs[link_id][2] for link_id in path_links),
                value=amount,
            )
        )
        _subtract_residual(residual, path_links, amount, tolerance)
        _subtract_terminal(remaining_supply, source, amount, tolerance)
        _subtract_terminal(remaining_demand, target, amount, tolerance)

    if remaining_demand:
        raise SndSerializationError(
            f"commodity {commodity.id!r} has unmatched demand after flow decomposition"
        )
    while residual:
        cycle_nodes, cycle_links = _residual_cycle(
            adjacency, arcs, residual, tolerance
        )
        amount = min(residual[link_id] for link_id in cycle_links)
        walks.append(
            SndFlowWalk(
                node_ids=cycle_nodes,
                link_ids=cycle_links,
                link_signs=tuple(arcs[link_id][2] for link_id in cycle_links),
                value=amount,
            )
        )
        _subtract_residual(residual, cycle_links, amount, tolerance)

    _validate_walk_decomposition(
        commodity,
        walks,
        commodity_flows,
        tolerance,
    )
    return tuple(walks)


def serialize_snd_instance(
    instance: SndInstance,
    *,
    include_reference_paths: bool = False,
) -> SndlibSerialization:
    """Convert compensated commodities to point-to-point SNDlib demands.

    Each positive path in the reference-flow decomposition becomes one demand.
    Consequently, the serialized reference solution has exactly the same node
    balances and aggregate signed link flow as ``instance``. Splitting fixes a
    source-to-sink pairing and therefore preserves reference feasibility, not
    the entire feasible region of a multi-terminal commodity.

    ``include_reference_paths=True`` records each derived route as its sole
    admissible path. The default omits these restrictions because the external
    Benders application currently models unrestricted point-to-point demands.
    """

    capability = sndlib_serialization_capability(
        instance, include_reference_paths=include_reference_paths
    )
    capability.require()
    demands: list[SndlibDemand] = []
    flows: dict[tuple[str, str], float] = {}
    provenance: list[SerializedDemandProvenance] = []
    occupied_ids: set[str] = set()
    flows_by_commodity = _flows_by_commodity(instance.reference_solution)
    link_by_id = {link.id: link for link in instance.links}

    for commodity in instance.commodities:
        walks = _decompose_commodity_flow(
            commodity,
            flows_by_commodity.get(commodity.id, {}),
            link_by_id,
        )
        paths = [walk for walk in walks if not walk.is_cycle]
        cycles = [walk for walk in walks if walk.is_cycle]
        serialized_ids: list[str] = []
        first_provenance_index = len(provenance)
        for index, walk in enumerate(paths, start=1):
            preferred = commodity.id if len(paths) == 1 else f"{commodity.id}__part_{index}"
            demand_id = _fresh_identifier(preferred, occupied_ids)
            serialized_ids.append(demand_id)
            path_id = f"{demand_id}__reference"
            admissible_paths = (
                (SndlibAdmissiblePath(path_id, walk.link_ids),)
                if include_reference_paths
                else ()
            )
            demands.append(
                SndlibDemand(
                    id=demand_id,
                    source=walk.node_ids[0],
                    target=walk.node_ids[-1],
                    value=walk.value,
                    max_path_length=len(walk.link_ids) if include_reference_paths else None,
                    admissible_paths=admissible_paths,
                )
            )
            _add_walk_flow(flows, demand_id, walk)
            provenance.append(
                SerializedDemandProvenance(
                    serialized_demand_id=demand_id,
                    source_instance=commodity.source_instance,
                    source_commodity_id=commodity.id,
                    node_ids=walk.node_ids,
                    link_ids=walk.link_ids,
                    value=walk.value,
                )
            )

        if cycles:
            if include_reference_paths:
                raise AssertionError("cycle capability check was bypassed")
            if serialized_ids:
                carrier_id = serialized_ids[0]
                provenance[first_provenance_index] = replace(
                    provenance[first_provenance_index], carries_circulation=True
                )
            else:
                if len(instance.nodes) < 2:
                    raise SndSerializationError(
                        "a pure circulation needs two distinct nodes for a zero-value carrier"
                    )
                carrier_id = _fresh_identifier(
                    f"{commodity.id}__circulation", occupied_ids
                )
                demands.append(
                    SndlibDemand(
                        carrier_id,
                        instance.nodes[0].id,
                        instance.nodes[1].id,
                        0.0,
                    )
                )
                provenance.append(
                    SerializedDemandProvenance(
                        serialized_demand_id=carrier_id,
                        source_instance=commodity.source_instance,
                        source_commodity_id=commodity.id,
                        node_ids=(),
                        link_ids=(),
                        value=0.0,
                        carries_circulation=True,
                    )
                )
            for cycle in cycles:
                _add_walk_flow(flows, carrier_id, cycle)

    network = SndlibNetwork(
        version=instance.version,
        coordinates_type=instance.coordinates_type,
        nodes=instance.nodes,
        links=instance.links,
        demands=tuple(demands),
        metadata=instance.metadata,
    )
    solution = SndSolution(instance.reference_solution.installed_capacities, flows)
    validate_snd_solution(network, solution)
    _validate_aggregate_flow(instance, solution, provenance)
    return SndlibSerialization(
        network=network,
        reference_solution=solution,
        demand_provenance=tuple(provenance),
        reference_paths_in_xml=include_reference_paths,
    )


def write_snd_instance(
    instance: SndInstance,
    path: str | Path,
    *,
    include_reference_paths: bool = False,
) -> SndlibSerialization:
    """Serialize and write a gate-ready SNDlib XML file."""

    serialization = serialize_snd_instance(
        instance, include_reference_paths=include_reference_paths
    )
    write_sndlib(serialization.network, path)
    return serialization


def _adjacency(arcs: dict[str, _Arc]) -> dict[str, tuple[tuple[str, str], ...]]:
    values: dict[str, list[tuple[str, str]]] = {}
    for link_id, (source, target, _) in arcs.items():
        values.setdefault(source, []).append((target, link_id))
    return {
        source: tuple(sorted(targets, key=lambda item: (item[0], item[1])))
        for source, targets in values.items()
    }


def _residual_path(
    source: str,
    demand_nodes: set[str],
    adjacency: dict[str, tuple[tuple[str, str], ...]],
    residual: dict[str, float],
    tolerance: float,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    queue = deque([source])
    parent: dict[str, tuple[str, str] | None] = {source: None}
    target: str | None = None
    while queue and target is None:
        node = queue.popleft()
        for neighbor, link_id in adjacency.get(node, ()):
            if residual.get(link_id, 0.0) <= tolerance or neighbor in parent:
                continue
            parent[neighbor] = (node, link_id)
            if neighbor in demand_nodes:
                target = neighbor
                break
            queue.append(neighbor)
    if target is None:
        raise SndSerializationError(
            f"positive-supply node {source!r} cannot reach residual demand"
        )

    nodes = [target]
    links: list[str] = []
    current = target
    while current != source:
        previous = parent[current]
        assert previous is not None
        parent_node, link_id = previous
        links.append(link_id)
        nodes.append(parent_node)
        current = parent_node
    nodes.reverse()
    links.reverse()
    return tuple(nodes), tuple(links)


def _residual_cycle(
    adjacency: dict[str, tuple[tuple[str, str], ...]],
    arcs: dict[str, _Arc],
    residual: dict[str, float],
    tolerance: float,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    start_link = min(residual)
    current = arcs[start_link][0]
    nodes: list[str] = []
    links: list[str] = []
    position: dict[str, int] = {}
    while current not in position:
        position[current] = len(nodes)
        nodes.append(current)
        options = [
            (neighbor, link_id)
            for neighbor, link_id in adjacency.get(current, ())
            if residual.get(link_id, 0.0) > tolerance
        ]
        if not options:
            raise SndSerializationError("residual flow is not a circulation")
        neighbor, link_id = options[0]
        links.append(link_id)
        current = neighbor
    start = position[current]
    return tuple(nodes[start:] + [current]), tuple(links[start:])


def _subtract_residual(
    residual: dict[str, float],
    link_ids: tuple[str, ...],
    amount: float,
    tolerance: float,
) -> None:
    for link_id in link_ids:
        remaining = residual[link_id] - amount
        if remaining <= tolerance:
            residual.pop(link_id)
        else:
            residual[link_id] = remaining


def _subtract_terminal(
    values: dict[str, float], node: str, amount: float, tolerance: float
) -> None:
    remaining = values[node] - amount
    if remaining <= tolerance:
        values.pop(node)
    else:
        values[node] = remaining


def _validate_walk_decomposition(
    commodity: SndCommodity,
    walks: list[SndFlowWalk],
    expected_flows: dict[str, float],
    tolerance: float,
) -> None:
    supplies: dict[str, float] = {}
    flows: dict[str, float] = {}
    for walk in walks:
        if not walk.is_cycle:
            supplies[walk.node_ids[0]] = supplies.get(walk.node_ids[0], 0.0) + walk.value
            supplies[walk.node_ids[-1]] = supplies.get(walk.node_ids[-1], 0.0) - walk.value
        for link_id, sign in zip(walk.link_ids, walk.link_signs, strict=True):
            flows[link_id] = flows.get(link_id, 0.0) + sign * walk.value
    for node in set(supplies) | set(commodity.node_supplies):
        if not isclose(
            supplies.get(node, 0.0),
            commodity.node_supplies.get(node, 0.0),
            rel_tol=1e-9,
            abs_tol=tolerance,
        ):
            raise SndSerializationError(
                f"flow decomposition changed supply at node {node!r}"
            )
    for link_id in set(flows) | set(expected_flows):
        value = flows.get(link_id, 0.0)
        if not isclose(
            value,
            expected_flows.get(link_id, 0.0),
            rel_tol=1e-9,
            abs_tol=tolerance,
        ):
            raise SndSerializationError(
                f"flow decomposition changed flow on link {link_id!r}"
            )


def _fresh_identifier(preferred: str, occupied: set[str]) -> str:
    candidate = preferred
    suffix = 2
    while candidate in occupied:
        candidate = f"{preferred}_{suffix}"
        suffix += 1
    occupied.add(candidate)
    return candidate


def _add_walk_flow(
    flows: dict[tuple[str, str], float], demand_id: str, walk: SndFlowWalk
) -> None:
    for link_id, sign in zip(walk.link_ids, walk.link_signs, strict=True):
        key = (demand_id, link_id)
        flows[key] = flows.get(key, 0.0) + sign * walk.value
        if flows[key] == 0.0:
            flows.pop(key)


def _validate_aggregate_flow(
    source: SndInstance,
    serialized: SndSolution,
    provenance: list[SerializedDemandProvenance],
) -> None:
    source_by_derived: dict[str, str] = {}
    for item in provenance:
        source_by_derived[item.serialized_demand_id] = item.source_commodity_id
    aggregate: dict[tuple[str, str], float] = {}
    for (demand_id, link_id), flow in serialized.flows.items():
        key = (source_by_derived[demand_id], link_id)
        aggregate[key] = aggregate.get(key, 0.0) + flow
    expected = source.reference_solution.flows
    for key in set(aggregate) | set(expected):
        if not isclose(
            aggregate.get(key, 0.0),
            expected.get(key, 0.0),
            rel_tol=1e-9,
            abs_tol=1e-7,
        ):
            raise SndSerializationError(
                f"serialization changed aggregate flow of commodity "
                f"{key[0]!r} on link {key[1]!r}"
            )


def _flows_by_commodity(solution: SndSolution) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = {}
    for (demand_id, link_id), flow in solution.flows.items():
        grouped.setdefault(demand_id, {})[link_id] = flow
    return grouped
