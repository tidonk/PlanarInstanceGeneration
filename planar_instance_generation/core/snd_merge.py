"""Sequential composition of compensated SND instances."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from math import ceil, isfinite
from numbers import Real
from typing import TypeAlias

import networkx as nx

from planar_instance_generation.core.snd import SndCommodity, SndInstance, SndSolution
from planar_instance_generation.core.snd_extraction import SndSubinstance
from planar_instance_generation.io.sndlib import (
    SndlibCapacityModule,
    SndlibLink,
    SndlibMetadata,
)


SndEndpointPair: TypeAlias = tuple[str, str]


@dataclass(frozen=True, slots=True)
class SndCouplingLinkConfig:
    """Parameters for every synthetic coupling link.

    With ``module_capacity=None``, one module can carry the total positive
    demand of both immediate inputs (or one unit when both have zero demand).
    The reference solution installs zero coupling capacity and assigns zero
    coupling flow unless explicit inter-network demands are requested. Costs
    default to zero rather than being inferred from unrelated source scales.
    """

    module_capacity: float | None = None
    module_cost: float = 0.0
    routing_cost: float = 0.0
    setup_cost: float = 0.0
    link_id_prefix: str = "coupling"

    def __post_init__(self) -> None:
        if self.module_capacity is not None:
            _positive("module_capacity", self.module_capacity)
        _nonnegative("module_cost", self.module_cost)
        _nonnegative("routing_cost", self.routing_cost)
        _nonnegative("setup_cost", self.setup_cost)
        _identifier("link_id_prefix", self.link_id_prefix)


@dataclass(frozen=True, slots=True)
class SndInterNetworkDemandConfig:
    """Add one direct synthetic demand along every new coupling link."""

    value: float
    demand_id_prefix: str = "inter"

    def __post_init__(self) -> None:
        _positive("inter-network demand value", self.value)
        _identifier("demand_id_prefix", self.demand_id_prefix)


@dataclass(frozen=True, slots=True)
class SndMergeConfig:
    coupling: SndCouplingLinkConfig = field(default_factory=SndCouplingLinkConfig)
    inter_network_demands: SndInterNetworkDemandConfig | None = None
    coordinates_type: str | None = None
    allow_mixed_units: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.coupling, SndCouplingLinkConfig):
            raise TypeError("coupling must be an SndCouplingLinkConfig")
        if self.inter_network_demands is not None and not isinstance(
            self.inter_network_demands, SndInterNetworkDemandConfig
        ):
            raise TypeError(
                "inter_network_demands must be an SndInterNetworkDemandConfig or None"
            )
        if self.coordinates_type is not None:
            _identifier("coordinates_type", self.coordinates_type)
        if not isinstance(self.allow_mixed_units, bool):
            raise TypeError("allow_mixed_units must be bool")


@dataclass(frozen=True, slots=True)
class SndSourceMapping:
    """Trace an original source occurrence into a composed instance."""

    source_name: str
    node_ids: Mapping[str, str]
    link_ids: Mapping[str, str]
    demand_ids: Mapping[str, str]
    node_indices: tuple[int, ...]
    link_indices: tuple[int, ...]
    demand_indices: tuple[int, ...]
    coordinates_type: str
    metadata: SndlibMetadata | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_ids", dict(self.node_ids))
        object.__setattr__(self, "link_ids", dict(self.link_ids))
        object.__setattr__(self, "demand_ids", dict(self.demand_ids))


@dataclass(frozen=True, slots=True)
class SndMergeResult:
    """A composed SND instance with sequential source and coupling provenance."""

    instance: SndInstance
    left: SndSourceMapping
    right: SndSourceMapping
    source_endpoint_pairs: tuple[SndEndpointPair, ...]
    merged_endpoint_pairs: tuple[SndEndpointPair, ...]
    source_mappings: tuple[SndSourceMapping, ...]
    coupling_link_indices: tuple[int, ...]
    new_coupling_link_indices: tuple[int, ...]
    synthetic_demand_indices: tuple[int, ...]
    module_capacity: float


SndMergeInput: TypeAlias = SndInstance | SndSubinstance | SndMergeResult


def compose_snd_instances(
    left: SndMergeInput,
    right: SndMergeInput,
    endpoint_pairs: Iterable[SndEndpointPair],
    *,
    name: str | None = None,
    config: SndMergeConfig | None = None,
) -> SndMergeResult:
    """Join two connected instances using endpoint-disjoint supplied pairs.

    The right input is namespaced only when an identifier collision occurs.
    Existing nodes, links, commodities, installed capacities, signed flows and
    source provenance are otherwise copied exactly. New links are oriented
    from the left endpoint to the right endpoint.
    """

    left_instance = _instance(left)
    right_instance = _instance(right)
    _validate_connected("left", left_instance)
    _validate_connected("right", right_instance)
    pairs = _validate_endpoint_pairs(left_instance, right_instance, endpoint_pairs)
    settings = SndMergeConfig() if config is None else config
    if not isinstance(settings, SndMergeConfig):
        raise TypeError("config must be an SndMergeConfig or None")
    coordinates_type = _coordinates_type(left_instance, right_instance, settings)
    _validate_units(left_instance, right_instance, settings)

    right_node_map, right_link_map, right_demand_map = _right_id_maps(
        left_instance, right_instance
    )
    left_node_map = {node.id: node.id for node in left_instance.nodes}
    left_link_map = {link.id: link.id for link in left_instance.links}
    left_demand_map = {
        commodity.id: commodity.id for commodity in left_instance.commodities
    }

    left_nodes = tuple(left_instance.nodes)
    right_nodes = tuple(
        replace(node, id=right_node_map[node.id]) for node in right_instance.nodes
    )
    left_links = tuple(left_instance.links)
    right_links = tuple(
        replace(
            link,
            id=right_link_map[link.id],
            source=right_node_map[link.source],
            target=right_node_map[link.target],
        )
        for link in right_instance.links
    )
    left_commodities = tuple(left_instance.commodities)
    right_commodities = tuple(
        _map_commodity(commodity, right_node_map, right_demand_map)
        for commodity in right_instance.commodities
    )

    merged_pairs = tuple(
        (left_node_map[left_node], right_node_map[right_node])
        for left_node, right_node in pairs
    )
    occupied_link_ids = set(left_link_map.values()) | set(right_link_map.values())
    coupling_link_ids = _fresh_ids(
        settings.coupling.link_id_prefix, len(merged_pairs), occupied_link_ids
    )
    module_capacity = _module_capacity(left_instance, right_instance, settings)
    module = SndlibCapacityModule(
        capacity=module_capacity,
        cost=settings.coupling.module_cost,
    )
    coupling_links = tuple(
        SndlibLink(
            id=link_id,
            source=left_node,
            target=right_node,
            additional_modules=(module,),
            routing_cost=settings.coupling.routing_cost,
            setup_cost=settings.coupling.setup_cost,
        )
        for link_id, (left_node, right_node) in zip(
            coupling_link_ids, merged_pairs, strict=True
        )
    )

    capacities = dict(left_instance.reference_solution.installed_capacities)
    capacities.update(
        {
            right_link_map[link_id]: capacity
            for link_id, capacity in right_instance.reference_solution.installed_capacities.items()
        }
    )
    capacities.update({link_id: 0.0 for link_id in coupling_link_ids})
    flows = dict(left_instance.reference_solution.flows)
    flows.update(
        {
            (right_demand_map[demand_id], right_link_map[link_id]): flow
            for (demand_id, link_id), flow in right_instance.reference_solution.flows.items()
        }
    )

    commodities = list(left_commodities + right_commodities)
    synthetic_indices = list(_propagate_synthetic_indices(left, 0))
    synthetic_indices.extend(
        _propagate_synthetic_indices(right, len(left_commodities))
    )
    if settings.inter_network_demands is not None:
        occupied_demands = {commodity.id for commodity in commodities}
        demand_ids = _fresh_ids(
            settings.inter_network_demands.demand_id_prefix,
            len(merged_pairs),
            occupied_demands,
        )
        demand_value = settings.inter_network_demands.value
        for demand_id, link_id, (source, target) in zip(
            demand_ids, coupling_link_ids, merged_pairs, strict=True
        ):
            synthetic_indices.append(len(commodities))
            commodities.append(
                SndCommodity(
                    id=demand_id,
                    node_supplies={source: demand_value, target: -demand_value},
                    source_instance=name or f"{left_instance.name}_{right_instance.name}",
                    source_demand=None,
                )
            )
            flows[(demand_id, link_id)] = demand_value
            capacities[link_id] = ceil(demand_value / module_capacity) * module_capacity

    nodes = left_nodes + right_nodes
    links = left_links + right_links + coupling_links
    metadata = _merged_metadata(left_instance, right_instance)
    merged = SndInstance(
        name=name or f"{left_instance.name}_{right_instance.name}",
        version="1.0",
        coordinates_type=coordinates_type,
        nodes=nodes,
        links=links,
        commodities=tuple(commodities),
        reference_solution=SndSolution(capacities, flows),
        metadata=metadata,
    )

    left_mapping = _immediate_mapping(
        left_instance,
        left_node_map,
        left_link_map,
        left_demand_map,
        node_offset=0,
        link_offset=0,
        demand_offset=0,
    )
    right_mapping = _immediate_mapping(
        right_instance,
        right_node_map,
        right_link_map,
        right_demand_map,
        node_offset=len(left_nodes),
        link_offset=len(left_links),
        demand_offset=len(left_commodities),
    )
    source_mappings = _propagate_source_mappings(left, left_mapping) + (
        _propagate_source_mappings(right, right_mapping)
    )
    new_coupling_indices = tuple(
        range(len(left_links) + len(right_links), len(links))
    )
    prior_coupling_indices = _propagate_coupling_indices(left, 0) + (
        _propagate_coupling_indices(right, len(left_links))
    )
    return SndMergeResult(
        instance=merged,
        left=left_mapping,
        right=right_mapping,
        source_endpoint_pairs=pairs,
        merged_endpoint_pairs=merged_pairs,
        source_mappings=source_mappings,
        coupling_link_indices=prior_coupling_indices + new_coupling_indices,
        new_coupling_link_indices=new_coupling_indices,
        synthetic_demand_indices=tuple(synthetic_indices),
        module_capacity=module_capacity,
    )


def _instance(value: SndMergeInput) -> SndInstance:
    if isinstance(value, SndInstance):
        return value
    if isinstance(value, (SndSubinstance, SndMergeResult)):
        return value.instance
    raise TypeError("merge inputs must be SndInstance, SndSubinstance, or SndMergeResult")


def _validate_connected(label: str, instance: SndInstance) -> None:
    topology = instance.topology()
    if topology.number_of_nodes() == 0 or not nx.is_connected(topology):
        raise ValueError(f"{label} instance must have a connected topology")


def _validate_endpoint_pairs(
    left: SndInstance,
    right: SndInstance,
    endpoint_pairs: Iterable[SndEndpointPair],
) -> tuple[SndEndpointPair, ...]:
    normalized: list[SndEndpointPair] = []
    for index, raw_pair in enumerate(endpoint_pairs):
        if isinstance(raw_pair, (str, bytes)):
            raise TypeError(f"endpoint_pairs[{index}] must be a pair")
        try:
            pair = tuple(raw_pair)
        except TypeError as error:
            raise TypeError(f"endpoint_pairs[{index}] must be a pair") from error
        if len(pair) != 2:
            raise ValueError(f"endpoint_pairs[{index}] must contain two node identifiers")
        if any(not isinstance(value, str) for value in pair):
            raise TypeError(f"endpoint_pairs[{index}] must contain string identifiers")
        normalized.append((pair[0], pair[1]))
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
    unknown_left = set(left_endpoints) - {node.id for node in left.nodes}
    unknown_right = set(right_endpoints) - {node.id for node in right.nodes}
    if unknown_left or unknown_right:
        details = []
        if unknown_left:
            details.append(f"unknown left nodes {sorted(unknown_left)}")
        if unknown_right:
            details.append(f"unknown right nodes {sorted(unknown_right)}")
        raise ValueError("; ".join(details))
    return tuple(normalized)


def _right_id_maps(
    left: SndInstance, right: SndInstance
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    left_sets = (
        {node.id for node in left.nodes},
        {link.id for link in left.links},
        {commodity.id for commodity in left.commodities},
    )
    right_values = (
        tuple(node.id for node in right.nodes),
        tuple(link.id for link in right.links),
        tuple(commodity.id for commodity in right.commodities),
    )
    if all(not occupied.intersection(values) for occupied, values in zip(left_sets, right_values)):
        return (
            {value: value for value in right_values[0]},
            {value: value for value in right_values[1]},
            {value: value for value in right_values[2]},
        )

    suffix = 1
    while True:
        prefix = f"right{suffix}::"
        candidates = tuple(
            {value: f"{prefix}{value}" for value in values} for values in right_values
        )
        if all(
            not occupied.intersection(mapping.values())
            for occupied, mapping in zip(left_sets, candidates)
        ):
            return candidates[0], candidates[1], candidates[2]
        suffix += 1


def _map_commodity(
    commodity: SndCommodity,
    node_map: Mapping[str, str],
    demand_map: Mapping[str, str],
) -> SndCommodity:
    return SndCommodity(
        id=demand_map[commodity.id],
        node_supplies={node_map[node]: value for node, value in commodity.node_supplies.items()},
        source_instance=commodity.source_instance,
        source_demand=commodity.source_demand,
    )


def _fresh_ids(prefix: str, count: int, occupied: set[str]) -> tuple[str, ...]:
    identifiers: list[str] = []
    candidate_number = 1
    while len(identifiers) < count:
        candidate = f"{prefix}_{candidate_number}"
        candidate_number += 1
        if candidate in occupied:
            continue
        occupied.add(candidate)
        identifiers.append(candidate)
    return tuple(identifiers)


def _module_capacity(
    left: SndInstance, right: SndInstance, config: SndMergeConfig
) -> float:
    explicit = config.coupling.module_capacity
    if explicit is not None:
        return float(explicit)
    positive_supply = sum(
        max(value, 0.0)
        for instance in (left, right)
        for commodity in instance.commodities
        for value in commodity.node_supplies.values()
    )
    return max(1.0, positive_supply)


def _coordinates_type(
    left: SndInstance, right: SndInstance, config: SndMergeConfig
) -> str:
    if config.coordinates_type is not None:
        return config.coordinates_type
    if left.coordinates_type != right.coordinates_type:
        raise ValueError(
            "source coordinate types differ; set SndMergeConfig.coordinates_type "
            "explicitly for the composed instance"
        )
    return left.coordinates_type


def _merged_metadata(
    left: SndInstance,
    right: SndInstance,
) -> SndlibMetadata | None:
    """Retain a common routing unit without inventing a numeric conversion."""

    if left.metadata == right.metadata:
        return left.metadata
    if (
        left.metadata is not None
        and right.metadata is not None
        and left.metadata.unit == right.metadata.unit
    ):
        return SndlibMetadata(
            granularity="composite",
            time="composite",
            unit=left.metadata.unit,
            origin="planar-instance-generation",
        )
    return None


def _validate_units(
    left: SndInstance, right: SndInstance, config: SndMergeConfig
) -> None:
    left_unit = None if left.metadata is None else left.metadata.unit
    right_unit = None if right.metadata is None else right.metadata.unit
    if (
        left_unit is not None
        and right_unit is not None
        and left_unit != right_unit
        and not config.allow_mixed_units
    ):
        raise ValueError(
            f"source routing units differ ({left_unit!r} and {right_unit!r}); "
            "scale them first or explicitly allow mixed units"
        )


def _immediate_mapping(
    instance: SndInstance,
    node_ids: Mapping[str, str],
    link_ids: Mapping[str, str],
    demand_ids: Mapping[str, str],
    *,
    node_offset: int,
    link_offset: int,
    demand_offset: int,
) -> SndSourceMapping:
    return SndSourceMapping(
        source_name=instance.name,
        node_ids=node_ids,
        link_ids=link_ids,
        demand_ids=demand_ids,
        node_indices=tuple(range(node_offset, node_offset + len(instance.nodes))),
        link_indices=tuple(range(link_offset, link_offset + len(instance.links))),
        demand_indices=tuple(
            range(demand_offset, demand_offset + len(instance.commodities))
        ),
        coordinates_type=instance.coordinates_type,
        metadata=instance.metadata,
    )


def _propagate_source_mappings(
    source: SndMergeInput, immediate: SndSourceMapping
) -> tuple[SndSourceMapping, ...]:
    if not isinstance(source, SndMergeResult):
        source_name = (
            source.source_name
            if isinstance(source, SndSubinstance)
            else immediate.source_name
        )
        return (replace(immediate, source_name=source_name),)

    propagated = []
    for original in source.source_mappings:
        propagated.append(
            SndSourceMapping(
                source_name=original.source_name,
                node_ids={
                    original_id: immediate.node_ids[current_id]
                    for original_id, current_id in original.node_ids.items()
                },
                link_ids={
                    original_id: immediate.link_ids[current_id]
                    for original_id, current_id in original.link_ids.items()
                },
                demand_ids={
                    original_id: immediate.demand_ids[current_id]
                    for original_id, current_id in original.demand_ids.items()
                },
                node_indices=tuple(
                    immediate.node_indices[index] for index in original.node_indices
                ),
                link_indices=tuple(
                    immediate.link_indices[index] for index in original.link_indices
                ),
                demand_indices=tuple(
                    immediate.demand_indices[index] for index in original.demand_indices
                ),
                coordinates_type=original.coordinates_type,
                metadata=original.metadata,
            )
        )
    return tuple(propagated)


def _propagate_coupling_indices(
    source: SndMergeInput, link_offset: int
) -> tuple[int, ...]:
    if not isinstance(source, SndMergeResult):
        return ()
    return tuple(link_offset + index for index in source.coupling_link_indices)


def _propagate_synthetic_indices(
    source: SndMergeInput, demand_offset: int
) -> tuple[int, ...]:
    if not isinstance(source, SndMergeResult):
        return ()
    return tuple(demand_offset + index for index in source.synthetic_demand_indices)


def _identifier(context: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{context} must be nonempty text without surrounding space")
    return value


def _nonnegative(context: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{context} must be a real number")
    numeric = float(value)
    if not isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{context} must be finite and nonnegative")
    return numeric


def _positive(context: str, value: object) -> float:
    numeric = _nonnegative(context, value)
    if numeric == 0.0:
        raise ValueError(f"{context} must be positive")
    return numeric
