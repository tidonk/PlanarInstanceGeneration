"""Solver-neutral data used by SND extraction and composition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isclose, isfinite
from numbers import Real
from typing import TypeAlias

import networkx as nx

from planar_instance_generation.io.sndlib import (
    SndlibDemand,
    SndlibLink,
    SndlibMetadata,
    SndlibNetwork,
    SndlibNode,
)


FlowKey: TypeAlias = tuple[str, str]


@dataclass(frozen=True, slots=True)
class SndSolution:
    """Installed capacities and sparse signed commodity flows.

    Installed capacity is total capacity, including any preinstalled module.
    A flow key is ``(demand_id, link_id)``. Missing keys mean zero. Positive
    flow follows the XML link's stored ``source -> target`` orientation and
    negative flow goes the other way. This is a representation convention;
    an adapter for a directed model remains responsible for rejecting reverse
    flow where its model does not allow it.
    """

    installed_capacities: Mapping[str, float]
    flows: Mapping[FlowKey, float]

    def __post_init__(self) -> None:
        if not isinstance(self.installed_capacities, Mapping):
            raise TypeError("installed_capacities must be a mapping")
        if not isinstance(self.flows, Mapping):
            raise TypeError("flows must be a mapping")
        capacities: dict[str, float] = {}
        for raw_link_id, raw_capacity in self.installed_capacities.items():
            link_id = _identifier("installed-capacity link", raw_link_id)
            capacity = _nonnegative("installed capacity", raw_capacity)
            if link_id in capacities:
                raise ValueError(f"duplicate installed capacity for link {link_id!r}")
            capacities[link_id] = capacity

        flows: dict[FlowKey, float] = {}
        for raw_key, raw_flow in self.flows.items():
            if not isinstance(raw_key, tuple) or len(raw_key) != 2:
                raise TypeError("flow keys must be (demand_id, link_id) tuples")
            demand_id = _identifier("flow demand", raw_key[0])
            link_id = _identifier("flow link", raw_key[1])
            value = _finite("flow", raw_flow)
            key = (demand_id, link_id)
            if key in flows:
                raise ValueError(f"duplicate flow value for {key!r}")
            if value != 0.0:
                flows[key] = value

        object.__setattr__(self, "installed_capacities", capacities)
        object.__setattr__(self, "flows", flows)

    def flow(self, demand_id: str, link_id: str) -> float:
        """Return a signed flow, interpreting an omitted sparse entry as zero."""

        return self.flows.get((demand_id, link_id), 0.0)


@dataclass(frozen=True, slots=True)
class SndCommodity:
    """One effective commodity and its unmodified source-demand record.

    ``node_supplies`` follows the paper: positive values supply flow and
    negative values consume it. Cut compensation can create more than two
    nonzero terminals, which SNDlib's point-to-point XML demand cannot express
    directly. ``source_demand`` therefore remains provenance rather than an
    assertion that its original routing restrictions still describe the
    compensated commodity.
    """

    id: str
    node_supplies: Mapping[str, float]
    source_instance: str
    source_demand: SndlibDemand | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _identifier("commodity", self.id))
        object.__setattr__(
            self, "source_instance", _identifier("source instance", self.source_instance)
        )
        if self.source_demand is not None and not isinstance(
            self.source_demand, SndlibDemand
        ):
            raise TypeError("source_demand must be an SndlibDemand or None")
        if not isinstance(self.node_supplies, Mapping):
            raise TypeError("node_supplies must be a mapping")
        supplies: dict[str, float] = {}
        for raw_node, raw_supply in self.node_supplies.items():
            node = _identifier("commodity-supply node", raw_node)
            supply = _finite("commodity supply", raw_supply)
            if node in supplies:
                raise ValueError(f"duplicate supply for node {node!r}")
            if supply != 0.0:
                supplies[node] = supply
        if not isclose(sum(supplies.values()), 0.0, rel_tol=1e-9, abs_tol=1e-7):
            raise ValueError(f"commodity {self.id!r} supplies must sum to zero")
        object.__setattr__(self, "node_supplies", supplies)


@dataclass(frozen=True, slots=True)
class SndInstance:
    """SND design data together with a feasible fixed-capacity reference flow."""

    name: str
    version: str
    coordinates_type: str
    nodes: tuple[SndlibNode, ...]
    links: tuple[SndlibLink, ...]
    commodities: tuple[SndCommodity, ...]
    reference_solution: SndSolution
    metadata: SndlibMetadata | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier("instance", self.name))
        if self.version != "1.0":
            raise ValueError("only SNDlib version '1.0' is supported")
        object.__setattr__(
            self, "coordinates_type", _identifier("coordinates type", self.coordinates_type)
        )
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "links", tuple(self.links))
        object.__setattr__(self, "commodities", tuple(self.commodities))
        self.validate()

    def validate(self) -> None:
        """Validate identifiers, references, balances, and the reference flow."""

        shell = self.topology_network()
        node_ids = {node.id for node in self.nodes}
        commodity_ids = [commodity.id for commodity in self.commodities]
        if len(set(commodity_ids)) != len(commodity_ids):
            raise ValueError("commodity identifiers must be unique")
        for commodity in self.commodities:
            unknown = set(commodity.node_supplies) - node_ids
            if unknown:
                raise ValueError(
                    f"commodity {commodity.id!r} references unknown node(s): "
                    f"{sorted(unknown)}"
                )
        validate_snd_solution(
            shell,
            self.reference_solution,
            supplies={
                commodity.id: commodity.node_supplies for commodity in self.commodities
            },
        )

    def topology_network(self) -> SndlibNetwork:
        """Return the node/link portion as a validated SNDlib network shell."""

        return SndlibNetwork(
            version=self.version,
            coordinates_type=self.coordinates_type,
            nodes=self.nodes,
            links=self.links,
            demands=(),
            metadata=self.metadata,
        )

    def topology(self) -> nx.Graph:
        """Return the simple undirected physical projection used for BFS."""

        return self.topology_network().topology()

    def directed_link_graph(self) -> nx.MultiDiGraph:
        """Return every stored link in its literal XML orientation."""

        return self.topology_network().directed_link_graph()


def build_snd_instance(
    network: SndlibNetwork,
    solution: SndSolution,
    *,
    name: str,
) -> SndInstance:
    """Lift a point-to-point SNDlib source and its feasible solution."""

    validate_snd_solution(network, solution)
    commodities = tuple(
        SndCommodity(
            id=demand.id,
            node_supplies={demand.source: demand.value, demand.target: -demand.value},
            source_instance=name,
            source_demand=demand,
        )
        for demand in network.demands
    )
    return SndInstance(
        name=name,
        version=network.version,
        coordinates_type=network.coordinates_type,
        nodes=network.nodes,
        links=network.links,
        commodities=commodities,
        reference_solution=solution,
        metadata=network.metadata,
    )


def validate_snd_solution(
    network: SndlibNetwork,
    solution: SndSolution,
    *,
    supplies: Mapping[str, Mapping[str, float]] | None = None,
    tolerance: float = 1e-7,
) -> None:
    """Validate a fixed-capacity multicommodity solution.

    Shared capacity is checked as the sum of absolute signed flows. Aggregate
    link flows cannot certify SNDlib admissible-path or maximum-path-length
    restrictions, so those restrictions deliberately remain outside this
    validator and must be handled by the solver adapter.
    """

    if not isinstance(solution, SndSolution):
        raise TypeError("solution must be an SndSolution")
    tolerance = _nonnegative("tolerance", tolerance)
    link_by_id = {link.id: link for link in network.links}
    node_ids = {node.id for node in network.nodes}
    if supplies is None:
        supplies = {
            demand.id: {demand.source: demand.value, demand.target: -demand.value}
            for demand in network.demands
        }
    commodity_ids = set(supplies)

    missing_capacities = link_by_id.keys() - solution.installed_capacities.keys()
    extra_capacities = solution.installed_capacities.keys() - link_by_id.keys()
    if missing_capacities or extra_capacities:
        details = []
        if missing_capacities:
            details.append(f"missing links {sorted(missing_capacities)}")
        if extra_capacities:
            details.append(f"unknown links {sorted(extra_capacities)}")
        raise ValueError("installed capacities do not match network links: " + "; ".join(details))

    for link in network.links:
        preinstalled = 0.0
        if link.preinstalled_module is not None:
            preinstalled = link.preinstalled_module.capacity
        if solution.installed_capacities[link.id] + tolerance < preinstalled:
            raise ValueError(
                f"installed capacity on link {link.id!r} is below its preinstalled capacity"
            )

    for demand_id, node_supply in supplies.items():
        _identifier("supply commodity", demand_id)
        unknown_nodes = set(node_supply) - node_ids
        if unknown_nodes:
            raise ValueError(
                f"commodity {demand_id!r} references unknown node(s): {sorted(unknown_nodes)}"
            )
        if not isclose(
            sum(float(value) for value in node_supply.values()),
            0.0,
            rel_tol=1e-9,
            abs_tol=tolerance,
        ):
            raise ValueError(f"commodity {demand_id!r} supplies must sum to zero")

    for demand_id, link_id in solution.flows:
        if demand_id not in commodity_ids:
            raise ValueError(f"flow references unknown commodity {demand_id!r}")
        if link_id not in link_by_id:
            raise ValueError(f"flow references unknown link {link_id!r}")

    net_out: dict[str, dict[str, float]] = {
        demand_id: {} for demand_id in commodity_ids
    }
    used_capacity = {link_id: 0.0 for link_id in link_by_id}
    for (demand_id, link_id), flow in solution.flows.items():
        link = link_by_id[link_id]
        balance = net_out[demand_id]
        balance[link.source] = balance.get(link.source, 0.0) + flow
        balance[link.target] = balance.get(link.target, 0.0) - flow
        used_capacity[link.id] += abs(flow)

    for demand_id, node_supply in supplies.items():
        actual_supply = net_out[demand_id]
        for node_id in set(node_supply) | set(actual_supply):
            expected = _finite("commodity supply", node_supply.get(node_id, 0.0))
            actual = actual_supply.get(node_id, 0.0)
            if not isclose(actual, expected, rel_tol=1e-9, abs_tol=tolerance):
                raise ValueError(
                    f"flow balance violation for commodity {demand_id!r} at node "
                    f"{node_id!r}: expected {expected}, got {actual}"
                )

    for link_id, used in used_capacity.items():
        installed = solution.installed_capacities[link_id]
        if used > installed + tolerance:
            raise ValueError(
                f"shared flow {used} exceeds installed capacity {installed} "
                f"on link {link_id!r}"
            )


def _identifier(context: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{context} identifier must be nonempty text without surrounding space")
    return value


def _finite(context: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{context} must be a real number")
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{context} must be finite")
    return numeric


def _nonnegative(context: str, value: object) -> float:
    numeric = _finite(context, value)
    if numeric < 0.0:
        raise ValueError(f"{context} must be nonnegative")
    return numeric
