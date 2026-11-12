"""Merge-node selection strategies for pairs of undirected graphs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
import random
from time import perf_counter
from typing import Hashable, Literal, TypeAlias

import networkx as nx
from pathlib import Path
from .planarization import (
    CertifiedPlanarPiece,
    ExactMergeNodeSelection,
    HeuristicMergeNodeSelection,
    OuterFaceSampling,
    select_exact_outer_face_pairs,
    select_heuristic_outer_face_pairs,
)

import numpy as np

NodePair: TypeAlias = tuple[Hashable, Hashable]
SelectionStrategy: TypeAlias = Literal["random", "high_degree", "kamada_kawai"]
CouplingCountRule: TypeAlias = Literal["log2", "sqrt"]
COUPLING_PAIR_RULES = {
    "log2": "min(n_min, max(3, ceil(log2(n_min))))",
    "sqrt": "min(n_min, max(3, ceil(sqrt(n_min))))",
}
# Retained for callers that refer to the historical default formula.
COUPLING_PAIR_RULE = COUPLING_PAIR_RULES["log2"]


def coupling_pair_rule(rule: CouplingCountRule = "log2") -> str:
    """Validate a merge-count rule and return its formula for provenance."""
    try:
        return COUPLING_PAIR_RULES[rule]
    except (KeyError, TypeError) as error:
        raise ValueError("coupling count rule must be 'log2' or 'sqrt'") from error


@dataclass(frozen=True, slots=True)
class MergeNodeSelection:
    """The endpoint-disjoint node pairs selected for one graph merge."""

    pairs: tuple[NodePair, ...]
    strategy: SelectionStrategy
    seed: int | None
    selection_seconds: float
    left_candidate_count: int
    right_candidate_count: int

    @property
    def left_nodes(self) -> tuple[Hashable, ...]:
        return tuple(left for left, _ in self.pairs)

    @property
    def right_nodes(self) -> tuple[Hashable, ...]:
        return tuple(right for _, right in self.pairs)


def coupling_pair_count(
    left: nx.Graph,
    right: nx.Graph,
    *,
    rule: CouplingCountRule = "log2",
) -> int:
    """Return ``min(n_min, max(3, ceil(f(n_min))))`` for log2 or sqrt.

    Here ``n_min`` is the smaller node count of the two graphs being joined.
    Outer-face capacity is a feasibility condition, never an alternative count.
    """

    _validate_graph("left", left)
    _validate_graph("right", right)
    n_min = min(left.number_of_nodes(), right.number_of_nodes())
    coupling_pair_rule(rule)
    if rule == "sqrt":
        # Integer arithmetic avoids rounding errors at perfect squares.
        requested = math.isqrt(n_min - 1) + 1
    else:
        requested = math.ceil(math.log2(n_min)) if n_min > 1 else 0
    return min(n_min, max(3, requested))


def select_random_pairs(
    left: nx.Graph,
    right: nx.Graph,
    pair_count: int,
    *,
    seed: int | None = None,
) -> MergeNodeSelection:
    """Select uniformly sampled, endpoint-disjoint node pairs.

    Each graph contributes a uniform ordered sample without replacement; the
    two sample orders define the pairing. Runtime and auxiliary memory are
    ``O(|V_left| + |V_right|)`` for NetworkX's sampling implementation.
    """

    _validate_inputs(left, right, pair_count)
    started = perf_counter()
    random_source = random.Random(seed)
    left_sample = random_source.sample(tuple(left), pair_count)
    right_sample = random_source.sample(tuple(right), pair_count)
    pairs = tuple(zip(left_sample, right_sample, strict=True))
    elapsed = perf_counter() - started
    return MergeNodeSelection(
        pairs,
        "random",
        seed,
        elapsed,
        left.number_of_nodes(),
        right.number_of_nodes(),
    )


def select_high_degree_pairs(
    left: nx.Graph,
    right: nx.Graph,
    pair_count: int,
    *,
    seed: int | None = None,
) -> MergeNodeSelection:
    """Pair each graph's highest-degree nodes in descending degree order.

    Nodes are shuffled with a local seeded generator before the stable degree
    bucket scan, making degree ties reproducible without systematically
    favoring insertion order. Runtime is ``O(|V| + |E|)`` over both graphs
    and auxiliary memory is ``O(|V_left| + |V_right|)``.
    """

    _validate_inputs(left, right, pair_count)
    started = perf_counter()
    random_source = random.Random(seed)
    left_nodes = _nodes_by_degree(left, random_source)
    right_nodes = _nodes_by_degree(right, random_source)
    pairs = tuple(zip(left_nodes[:pair_count], right_nodes[:pair_count], strict=True))
    elapsed = perf_counter() - started
    return MergeNodeSelection(
        pairs,
        "high_degree",
        seed,
        elapsed,
        left.number_of_nodes(),
        right.number_of_nodes(),
    )


def select_kamada_kawai_pairs(
    left: nx.Graph,
    right: nx.Graph,
    pair_count: int,
    *,
    seed: int | None = None,
) -> MergeNodeSelection:
    """Greedily pair nearby nodes after separate Kamada--Kawai embeddings.

    Both layouts are recomputed for every call, isotropically normalized to a
    unit maximum span, and placed side by side with a unit gap. Cross pairs
    are considered by increasing Euclidean distance and accepted only while
    both endpoints remain unused. The candidate sets are the convex-hull
    vertices of the two embeddings. A request larger than the smaller boundary
    is a capacity error; the request is never silently capped.

    Apart from the layout computation, the hulls take ``O(|V| log |V|)`` time.
    For boundary sets ``B_left`` and ``B_right``, forming and ordering the
    cross-distance matrix takes ``O(|B_left| |B_right|
    log(|B_left| |B_right|))`` time and ``O(|B_left| |B_right|)`` memory.
    ``selection_seconds`` includes both layouts and all subsequent work.
    """

    _validate_inputs(left, right, pair_count)
    started = perf_counter()
    left_nodes = tuple(left)
    right_nodes = tuple(right)
    left_coordinates = _normalized_kamada_kawai(left, left_nodes)
    right_coordinates = _normalized_kamada_kawai(right, right_nodes)
    left_boundary = _convex_hull_indices(left_coordinates)
    right_boundary = _convex_hull_indices(right_coordinates)
    available = min(len(left_boundary), len(right_boundary))
    if available < pair_count:
        raise ValueError(
            f"Kamada--Kawai selection requested {pair_count} distinct boundary "
            "nodes per graph, but found "
            f"{len(left_boundary)} and {len(right_boundary)}"
        )
    left_coordinates, right_coordinates = _place_side_by_side(
        left_coordinates, right_coordinates
    )

    left_candidates = left_coordinates[np.asarray(left_boundary, dtype=int)]
    right_candidates = right_coordinates[np.asarray(right_boundary, dtype=int)]
    offsets = left_candidates[:, np.newaxis, :] - right_candidates[np.newaxis, :, :]
    squared_distances = np.einsum("ijk,ijk->ij", offsets, offsets)
    candidate_indices = list(range(squared_distances.size))
    random.Random(seed).shuffle(candidate_indices)
    candidate_indices.sort(key=squared_distances.flat.__getitem__)

    selected_left: set[int] = set()
    selected_right: set[int] = set()
    pairs: list[NodePair] = []
    right_count = len(right_boundary)
    for flat_index in candidate_indices:
        left_offset, right_offset = divmod(int(flat_index), right_count)
        if left_offset in selected_left or right_offset in selected_right:
            continue
        pairs.append(
            (
                left_nodes[left_boundary[left_offset]],
                right_nodes[right_boundary[right_offset]],
            )
        )
        selected_left.add(left_offset)
        selected_right.add(right_offset)
        if len(pairs) == pair_count:
            break

    elapsed = perf_counter() - started
    return MergeNodeSelection(
        tuple(pairs),
        "kamada_kawai",
        seed,
        elapsed,
        len(left_boundary),
        len(right_boundary),
    )


def select_merge_node_pairs(
    left: nx.Graph,
    right: nx.Graph,
    pair_count: int,
    *,
    strategy: SelectionStrategy,
    seed: int | None = None,
) -> MergeNodeSelection:
    """Dispatch to a named merge-node selection strategy."""

    selectors = {
        "random": select_random_pairs,
        "high_degree": select_high_degree_pairs,
        "kamada_kawai": select_kamada_kawai_pairs,
    }
    try:
        selector = selectors[strategy]
    except (KeyError, TypeError) as error:
        choices = ", ".join(selectors)
        raise ValueError(
            f"unknown selection strategy {strategy!r}; choose from {choices}"
        ) from error
    return selector(left, right, pair_count, seed=seed)


def _nodes_by_degree(graph: nx.Graph, random_source: random.Random) -> list[Hashable]:
    nodes = list(graph)
    random_source.shuffle(nodes)
    buckets: list[list[Hashable]] = [[] for _ in range(graph.number_of_nodes())]
    for node in nodes:
        buckets[graph.degree[node]].append(node)
    return [node for bucket in reversed(buckets) for node in bucket]


def _normalized_kamada_kawai(
    graph: nx.Graph,
    nodes: tuple[Hashable, ...],
) -> np.ndarray:
    positions = nx.kamada_kawai_layout(graph, weight=None, scale=1.0, center=(0.0, 0.0))
    coordinates = np.asarray([positions[node] for node in nodes], dtype=float)
    coordinates -= coordinates.mean(axis=0)
    maximum_span = float(np.ptp(coordinates, axis=0).max())
    if maximum_span > 0.0:
        coordinates /= maximum_span
    return coordinates


def _convex_hull_indices(coordinates: np.ndarray) -> tuple[int, ...]:
    """Return stable indices of convex-hull vertices in a 2-D layout."""

    points: dict[tuple[float, float], list[int]] = {}
    for index, (x_coordinate, y_coordinate) in enumerate(coordinates):
        points.setdefault((float(x_coordinate), float(y_coordinate)), []).append(index)
    ordered = sorted(points)
    if len(ordered) <= 2:
        return tuple(index for point in ordered for index in points[point])

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 1e-12:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 1e-12:
            upper.pop()
        upper.append(point)
    boundary_points = set(lower[:-1] + upper[:-1])
    return tuple(
        index
        for index in range(len(coordinates))
        if tuple(float(value) for value in coordinates[index]) in boundary_points
    )


def _place_side_by_side(
    left_coordinates: np.ndarray,
    right_coordinates: np.ndarray,
    *,
    gap: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    left_placed = left_coordinates.copy()
    right_placed = right_coordinates.copy()
    left_placed[:, 0] += -gap / 2.0 - float(left_placed[:, 0].max())
    right_placed[:, 0] += gap / 2.0 - float(right_placed[:, 0].min())
    return left_placed, right_placed


def _validate_inputs(left: nx.Graph, right: nx.Graph, pair_count: int) -> None:
    _validate_graph("left", left)
    _validate_graph("right", right)
    _validate_pair_count(pair_count)
    maximum = min(left.number_of_nodes(), right.number_of_nodes())
    if pair_count > maximum:
        raise ValueError(
            f"cannot select {pair_count} endpoint-disjoint pairs; at most {maximum} are possible"
        )


def _validate_pair_count(pair_count: int) -> None:
    if not isinstance(pair_count, Integral) or isinstance(pair_count, bool):
        raise TypeError("pair_count must be an integer")
    if pair_count < 1:
        raise ValueError("pair_count must be positive")


def _validate_graph(label: str, graph: nx.Graph) -> None:
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"{label} must be a NetworkX graph")
    if graph.is_directed():
        raise TypeError(f"{label} must be undirected")
    if graph.is_multigraph():
        raise TypeError(f"{label} must be simple")
    if graph.number_of_nodes() == 0:
        raise ValueError(f"{label} graph must not be empty")


def select_composition_pairs(
    left: nx.Graph,
    right: nx.Graph,
    count: int,
    *,
    strategy: str,
    seed: int,
    exact_time_limit_seconds: float,
    exact_threads: int,
    log_directory: Path | None,
    outer_face_sampling: OuterFaceSampling = "consecutive",
    outer_face_prefer_non_leaf: bool = False,
    left_certified_pieces: tuple[CertifiedPlanarPiece, ...] = (),
    left_prior_certificates: tuple[CertifiedPlanarPiece, ...] = (),
) -> MergeNodeSelection | ExactMergeNodeSelection | HeuristicMergeNodeSelection:
    if strategy == "exact_outer_face":
        return select_exact_outer_face_pairs(
            left,
            right,
            count,
            time_limit_seconds=exact_time_limit_seconds,
            seed=seed,
            threads=exact_threads,
            log_directory=log_directory,
            sampling=outer_face_sampling,
            prefer_non_leaf=outer_face_prefer_non_leaf,
            left_certified_pieces=left_certified_pieces,
            left_prior_certificates=left_prior_certificates,
        )
    if strategy == "heuristic_outer_face":
        return select_heuristic_outer_face_pairs(
            left,
            right,
            count,
            threads=exact_threads,
            log_directory=log_directory,
            seed=seed,
            sampling=outer_face_sampling,
            prefer_non_leaf=outer_face_prefer_non_leaf,
            left_certified_pieces=left_certified_pieces,
            left_prior_certificates=left_prior_certificates,
        )
    return select_merge_node_pairs(
        left,
        right,
        count,
        strategy=strategy,
        seed=seed,
    )
