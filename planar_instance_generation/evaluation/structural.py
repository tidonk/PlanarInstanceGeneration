"""Structural measurements used by the paper's computational study."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from time import perf_counter
from typing import Hashable, Iterable, Mapping
import warnings

import networkx as nx
import numpy as np
from networkx.algorithms.approximation import treewidth_min_degree


@dataclass(frozen=True, slots=True)
class StructuralStatistics:
    """Local and global simple-graph properties from the paper."""

    node_count: int
    edge_count: int
    density: float
    degree_distribution: Mapping[int, float]
    mean_degree: float
    maximum_degree: int
    average_clustering: float
    degeneracy: int
    treewidth_upper_bound: int
    connected: bool
    diameter: int | None
    algebraic_connectivity: float
    mean_betweenness: float
    betweenness_p95: float
    betweenness_sample_count: int | None
    coupling_betweenness: Mapping[Hashable, float]
    degree_assortativity: float | None
    measurement_seconds: float


def measure_structure(
    graph: nx.Graph,
    *,
    coupling_nodes: Iterable[Hashable] = (),
    betweenness_samples: int | None = None,
    seed: int = 0,
) -> StructuralStatistics:
    """Measure the full property set on a nonempty simple undirected graph.

    Betweenness is exact unless ``betweenness_samples`` is explicitly set.
    Treewidth is reported honestly as the min-degree heuristic's upper bound.
    Diameter is undefined (``None``) for disconnected graphs.
    """

    _validate_graph(graph)
    if betweenness_samples is not None and (
        isinstance(betweenness_samples, bool)
        or not isinstance(betweenness_samples, int)
        or not 1 <= betweenness_samples <= graph.number_of_nodes()
    ):
        raise ValueError("betweenness_samples must be between 1 and the node count")
    requested_coupling_nodes = tuple(coupling_nodes)
    missing = set(requested_coupling_nodes) - set(graph)
    if missing:
        raise ValueError(f"coupling_nodes contains unknown node(s): {sorted(map(repr, missing))}")

    started = perf_counter()
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    degrees = [degree for _, degree in graph.degree]
    degree_counts = Counter(degrees)
    degree_distribution = {
        degree: degree_counts[degree] / node_count for degree in sorted(degree_counts)
    }
    connected = nx.is_connected(graph)
    core_numbers = nx.core_number(graph)
    treewidth_upper, _ = treewidth_min_degree(graph)

    if node_count == 1:
        algebraic_connectivity = 0.0
    elif connected:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            algebraic_connectivity = float(
                nx.algebraic_connectivity(graph, method="tracemin_lu")
            )
    else:
        algebraic_connectivity = 0.0

    betweenness = nx.betweenness_centrality(
        graph,
        k=betweenness_samples,
        normalized=True,
        seed=seed if betweenness_samples is not None else None,
    )
    values = np.fromiter(betweenness.values(), dtype=np.float64, count=node_count)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assortativity = (
            nx.degree_assortativity_coefficient(graph) if edge_count else math.nan
        )
    elapsed = perf_counter() - started
    return StructuralStatistics(
        node_count=node_count,
        edge_count=edge_count,
        density=float(nx.density(graph)),
        degree_distribution=degree_distribution,
        mean_degree=float(sum(degrees) / node_count),
        maximum_degree=max(degrees),
        average_clustering=float(nx.average_clustering(graph)),
        degeneracy=max(core_numbers.values()),
        treewidth_upper_bound=int(treewidth_upper),
        connected=connected,
        diameter=nx.diameter(graph) if connected else None,
        algebraic_connectivity=algebraic_connectivity,
        mean_betweenness=float(values.mean()),
        betweenness_p95=float(np.quantile(values, 0.95)),
        betweenness_sample_count=betweenness_samples,
        coupling_betweenness={
            node: float(betweenness[node]) for node in requested_coupling_nodes
        },
        degree_assortativity=(
            float(assortativity) if math.isfinite(float(assortativity)) else None
        ),
        measurement_seconds=elapsed,
    )


def _validate_graph(graph: nx.Graph) -> None:
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a NetworkX graph")
    if graph.is_directed() or graph.is_multigraph():
        raise TypeError("graph must be simple and undirected")
    if graph.number_of_nodes() == 0:
        raise ValueError("graph must not be empty")
    if nx.number_of_selfloops(graph):
        raise ValueError("graph must not contain self-loops")
