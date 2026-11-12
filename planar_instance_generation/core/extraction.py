"""Randomized breadth-first extraction for simple undirected graphs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from numbers import Integral, Real
import random
from typing import Hashable, TypeAlias

import networkx as nx


CutEdge: TypeAlias = tuple[Hashable, Hashable]


class ComponentTooSmallError(ValueError):
    """Raised when the sampled start component cannot fill the extraction."""

    def __init__(self, start_node: Hashable, component_size: int, target_size: int) -> None:
        self.start_node = start_node
        self.component_size = component_size
        self.target_size = target_size
        super().__init__(
            f"sampled start node {start_node!r} lies in a component of size "
            f"{component_size}, smaller than target size {target_size}"
        )


@dataclass(frozen=True, slots=True)
class BfsExtraction:
    """Node and boundary-edge selection produced by randomized BFS."""

    start_node: Hashable
    target_size: int
    selected_nodes: tuple[Hashable, ...]
    cut_edges: tuple[CutEdge, ...]


def randomized_bfs_extraction(
    graph: nx.Graph,
    fraction: float | None = None,
    *,
    target_nodes: int | None = None,
    seed: int | None = None,
    rng: random.Random | None = None,
) -> BfsExtraction:
    """Select an exact number of nodes by randomized BFS.

    Pass exactly one of ``fraction`` or ``target_nodes``. A fraction selects
    ``ceil(fraction * |V|)`` nodes; the integer path avoids any rounding when
    a composite-level target has already been allocated across sources.

    The start node is sampled uniformly. Neighbors are shuffled before they
    enter the BFS queue, randomizing ties without changing breadth-first
    distances. A supplied ``rng`` is advanced in place; otherwise a local
    generator is initialized from ``seed``.

    Runtime is ``O(|V| + |E|)`` and auxiliary memory is ``O(|V|)``.
    """

    _validate_graph(graph)
    target_size = _resolve_target_size(
        graph.number_of_nodes(), fraction=fraction, target_nodes=target_nodes
    )
    random_source = _random_source(seed, rng)

    nodes = tuple(graph.nodes)
    start_node = random_source.choice(nodes)
    queue = deque([start_node])
    discovered = {start_node}
    selected: list[Hashable] = []

    while queue and len(selected) < target_size:
        node = queue.popleft()
        selected.append(node)
        neighbors = [neighbor for neighbor in graph.neighbors(node) if neighbor not in discovered]
        random_source.shuffle(neighbors)
        discovered.update(neighbors)
        queue.extend(neighbors)

    if len(selected) < target_size:
        raise ComponentTooSmallError(start_node, len(selected), target_size)

    selected_set = set(selected)
    cut_edges = tuple(
        (first, second)
        for first, second in graph.edges
        if (first in selected_set) != (second in selected_set)
    )
    return BfsExtraction(
        start_node=start_node,
        target_size=target_size,
        selected_nodes=tuple(selected),
        cut_edges=cut_edges,
    )


def _validate_graph(graph: nx.Graph) -> None:
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a NetworkX graph")
    if graph.is_directed():
        raise TypeError("graph must be undirected")
    if graph.is_multigraph():
        raise TypeError("graph must be simple")
    if graph.number_of_nodes() == 0:
        raise ValueError("graph must not be empty")


def _target_size(node_count: int, fraction: float) -> int:
    if isinstance(fraction, bool) or not isinstance(fraction, Real):
        raise TypeError("fraction must be a real number")
    numeric_fraction = float(fraction)
    if not math.isfinite(numeric_fraction) or not 0.0 < numeric_fraction <= 1.0:
        raise ValueError("fraction must be finite and in (0, 1]")
    return math.ceil(numeric_fraction * node_count)


def _resolve_target_size(
    node_count: int,
    *,
    fraction: float | None,
    target_nodes: int | None,
) -> int:
    if (fraction is None) == (target_nodes is None):
        raise ValueError("pass exactly one of fraction or target_nodes")
    if target_nodes is None:
        assert fraction is not None
        return _target_size(node_count, fraction)
    if (
        isinstance(target_nodes, bool)
        or not isinstance(target_nodes, Integral)
        or not 1 <= target_nodes <= node_count
    ):
        raise ValueError(
            f"target_nodes must be an integer in [1, {node_count}]"
        )
    return int(target_nodes)


def _random_source(seed: int | None, rng: random.Random | None) -> random.Random:
    if rng is not None and seed is not None:
        raise ValueError("pass either seed or rng, not both")
    if rng is not None and not isinstance(rng, random.Random):
        raise TypeError("rng must be an instance of random.Random")
    return rng if rng is not None else random.Random(seed)
