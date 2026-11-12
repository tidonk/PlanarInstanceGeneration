"""Randomized induced SND subinstances with exact cut-flow compensation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import random

import networkx as nx

from planar_instance_generation.core.extraction import randomized_bfs_extraction
from planar_instance_generation.core.snd import (
    SndCommodity,
    SndInstance,
    SndSolution,
    validate_snd_solution,
)
from planar_instance_generation.io.sndlib import SndlibNetwork


@dataclass(frozen=True, slots=True)
class SndSubinstance:
    """A compensated subinstance and row-level source provenance."""

    instance: SndInstance
    source_name: str
    selected_nodes: tuple[str, ...]
    retained_node_indices: tuple[int, ...]
    internal_link_indices: tuple[int, ...]
    cut_link_indices: tuple[int, ...]
    compensation_by_demand: Mapping[str, Mapping[str, float]]

    def __post_init__(self) -> None:
        copied = {
            demand_id: dict(values)
            for demand_id, values in self.compensation_by_demand.items()
        }
        object.__setattr__(self, "compensation_by_demand", copied)


def extract_random_snd_subinstance(
    source: SndlibNetwork,
    fraction: float | None,
    solution: SndSolution,
    *,
    source_name: str,
    target_nodes: int | None = None,
    seed: int | None = None,
    rng: random.Random | None = None,
    name: str | None = None,
) -> SndSubinstance:
    """Select nodes by randomized BFS and compensate every source demand."""

    selection = randomized_bfs_extraction(
        source.topology(),
        fraction,
        target_nodes=target_nodes,
        seed=seed,
        rng=rng,
    )
    return build_snd_subinstance(
        source,
        (str(node) for node in selection.selected_nodes),
        solution,
        source_name=source_name,
        name=name,
    )


def build_snd_subinstance(
    source: SndlibNetwork,
    selected_nodes: Iterable[str],
    solution: SndSolution,
    *,
    source_name: str,
    name: str | None = None,
) -> SndSubinstance:
    """Build a node-induced subinstance using the paper's compensation sign.

    For an internal endpoint of a removed link, a positive outward signed flow
    is subtracted from supply and a positive inward signed flow is added. The
    restricted source flow then satisfies every compensated commodity balance
    exactly, up to the source solution's numerical tolerance.
    """

    validate_snd_solution(source, solution)
    selected = _validate_selected_nodes(source, selected_nodes)
    selected_set = set(selected)
    topology = source.topology().subgraph(selected_set)
    if not nx.is_connected(topology):
        raise ValueError("selected_nodes must induce a connected topology")

    retained_node_indices = tuple(
        index for index, node in enumerate(source.nodes) if node.id in selected_set
    )
    internal_link_indices: list[int] = []
    cut_link_indices: list[int] = []
    for index, link in enumerate(source.links):
        source_inside = link.source in selected_set
        target_inside = link.target in selected_set
        if source_inside and target_inside:
            internal_link_indices.append(index)
        elif source_inside != target_inside:
            cut_link_indices.append(index)

    internal_link_ids = {source.links[index].id for index in internal_link_indices}
    cut_links = {
        source.links[index].id: source.links[index] for index in cut_link_indices
    }
    compensation = {demand.id: {} for demand in source.demands}
    for (demand_id, link_id), flow in solution.flows.items():
        link = cut_links.get(link_id)
        if link is None:
            continue
        node = link.source if link.source in selected_set else link.target
        delta = -flow if node == link.source else flow
        values = compensation[demand_id]
        values[node] = values.get(node, 0.0) + delta

    commodities: list[SndCommodity] = []
    clean_compensation: dict[str, dict[str, float]] = {}
    for demand in source.demands:
        base: dict[str, float] = {}
        if demand.source in selected_set:
            base[demand.source] = base.get(demand.source, 0.0) + demand.value
        if demand.target in selected_set:
            base[demand.target] = base.get(demand.target, 0.0) - demand.value

        clean_delta = {
            node: value
            for node, value in compensation[demand.id].items()
            if value != 0.0
        }
        effective = base.copy()
        for node, delta in clean_delta.items():
            effective[node] = effective.get(node, 0.0) + delta
        effective = {
            node: value
            for node, value in effective.items()
            if value != 0.0
        }
        commodities.append(
            SndCommodity(
                id=demand.id,
                node_supplies=effective,
                source_instance=source_name,
                source_demand=demand,
            )
        )
        clean_compensation[demand.id] = clean_delta

    internal_capacities = {
        link_id: solution.installed_capacities[link_id]
        for link_id in internal_link_ids
    }
    internal_flows = {
        (demand_id, link_id): flow
        for (demand_id, link_id), flow in solution.flows.items()
        if link_id in internal_link_ids
    }
    instance = SndInstance(
        name=name or f"{source_name}_sub",
        version=source.version,
        coordinates_type=source.coordinates_type,
        nodes=tuple(source.nodes[index] for index in retained_node_indices),
        links=tuple(source.links[index] for index in internal_link_indices),
        commodities=tuple(commodities),
        reference_solution=SndSolution(internal_capacities, internal_flows),
        metadata=source.metadata,
    )
    return SndSubinstance(
        instance=instance,
        source_name=source_name,
        selected_nodes=selected,
        retained_node_indices=retained_node_indices,
        internal_link_indices=tuple(internal_link_indices),
        cut_link_indices=tuple(cut_link_indices),
        compensation_by_demand=clean_compensation,
    )


def _validate_selected_nodes(
    source: SndlibNetwork, selected_nodes: Iterable[str]
) -> tuple[str, ...]:
    selected = tuple(selected_nodes)
    if not selected:
        raise ValueError("selected_nodes must not be empty")
    if any(not isinstance(node, str) for node in selected):
        raise TypeError("selected_nodes must contain string node identifiers")
    if len(set(selected)) != len(selected):
        raise ValueError("selected_nodes must not contain duplicates")
    known = {node.id for node in source.nodes}
    unknown = set(selected) - known
    if unknown:
        raise ValueError(f"selected_nodes contains unknown node(s): {sorted(unknown)}")
    return selected
