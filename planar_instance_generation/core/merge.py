"""Graph merge primitives with separately measured edge-selection time."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
import random
from time import perf_counter
from typing import Hashable, Iterator, Sequence, TypeAlias

import networkx as nx


SourceId: TypeAlias = Hashable
NamespacedNode: TypeAlias = tuple[SourceId, Hashable]
MergeEdge: TypeAlias = tuple[NamespacedNode, NamespacedNode]


@dataclass(frozen=True, slots=True)
class MergeResult:
    """The immutable record of one random graph merge.

    Graph objects are independent copies. ``selection_seconds`` measures only
    cross-edge selection; graph copying and node namespacing are deliberately
    outside both reported algorithm timings.
    """

    source_graphs: tuple[nx.Graph, nx.Graph]
    initial_graph: nx.Graph
    final_graph: nx.Graph
    added_edges: tuple[MergeEdge, ...]
    namespaces: tuple[SourceId, SourceId]
    seed: int | None
    stage_index: int | None
    selection_seconds: float
    addition_seconds: float

    @property
    def total_algorithm_seconds(self) -> float:
        return self.selection_seconds + self.addition_seconds

    def graph_at_step(self, step: int) -> nx.Graph:
        """Return a copy after exactly ``step`` merge edges have been added."""

        if not isinstance(step, Integral) or isinstance(step, bool):
            raise TypeError("step must be an integer")
        if not 0 <= step <= len(self.added_edges):
            raise ValueError(f"step must be between 0 and {len(self.added_edges)}")
        graph = self.initial_graph.copy()
        for index, edge in enumerate(self.added_edges[:step], start=1):
            graph.add_edge(*edge, **_merge_edge_attributes(index, self.stage_index))
        return graph

    def iter_steps(self) -> Iterator[tuple[int, nx.Graph, MergeEdge]]:
        """Yield each one-based step, its graph copy, and newest edge."""

        graph = self.initial_graph.copy()
        for index, edge in enumerate(self.added_edges, start=1):
            graph.add_edge(*edge, **_merge_edge_attributes(index, self.stage_index))
            yield index, graph.copy(), edge


@dataclass(frozen=True, slots=True)
class MergeSequenceResult:
    """A left-associative sequence of pairwise merge stages."""

    source_graphs: tuple[nx.Graph, ...]
    source_ids: tuple[SourceId, ...]
    stages: tuple[MergeResult, ...]
    seed: int | None

    @property
    def final_graph(self) -> nx.Graph:
        return self.stages[-1].final_graph

    @property
    def selection_seconds(self) -> float:
        return sum(stage.selection_seconds for stage in self.stages)

    @property
    def addition_seconds(self) -> float:
        return sum(stage.addition_seconds for stage in self.stages)


def random_merge(
    left: nx.Graph,
    right: nx.Graph,
    edge_count: int,
    *,
    seed: int | None = None,
    left_namespace: SourceId = "left",
    right_namespace: SourceId = "right",
    stage_index: int | None = None,
) -> MergeResult:
    """Merge two simple undirected graphs with uniformly sampled cross edges.

    Sampling uses integer indices into the Cartesian product, so memory is
    ``O(|V_left| + |V_right| + edge_count)`` rather than
    ``O(|V_left| * |V_right|)``. Passing the same graph twice intentionally
    creates two independently namespaced copies.
    """

    _validate_source("left", left)
    _validate_source("right", right)
    _validate_namespace("left_namespace", left_namespace)
    _validate_namespace("right_namespace", right_namespace)
    if left_namespace == right_namespace:
        raise ValueError("left_namespace and right_namespace must be distinct")
    if stage_index is not None and (not isinstance(stage_index, Integral) or stage_index < 1):
        raise ValueError("stage_index must be a positive integer or None")
    if not isinstance(edge_count, Integral) or isinstance(edge_count, bool):
        raise TypeError("edge_count must be an integer")
    if edge_count < 1:
        raise ValueError("edge_count must be positive")

    left_nodes = tuple(left.nodes)
    right_nodes = tuple(right.nodes)
    pair_count = len(left_nodes) * len(right_nodes)
    if edge_count > pair_count:
        raise ValueError(
            f"cannot select {edge_count} distinct cross edges from {pair_count} node pairs"
        )

    left_copy = left.copy()
    right_copy = right.copy()
    left_names = {node: (left_namespace, node) for node in left_nodes}
    right_names = {node: (right_namespace, node) for node in right_nodes}
    namespaced_left = nx.relabel_nodes(left_copy, left_names, copy=True)
    namespaced_right = nx.relabel_nodes(right_copy, right_names, copy=True)
    _annotate_source(namespaced_left, left_namespace)
    _annotate_source(namespaced_right, right_namespace)
    initial = nx.compose(namespaced_left, namespaced_right)
    initial.graph.update(merge_edge_count=edge_count, merge_seed=seed)

    selection_started = perf_counter()
    sampled_indices = random.Random(seed).sample(range(pair_count), edge_count)
    added_edges = tuple(
        (
            left_names[left_nodes[index // len(right_nodes)]],
            right_names[right_nodes[index % len(right_nodes)]],
        )
        for index in sampled_indices
    )
    selection_seconds = perf_counter() - selection_started

    final = initial.copy()
    addition_started = perf_counter()
    for step, edge in enumerate(added_edges, start=1):
        final.add_edge(*edge, **_merge_edge_attributes(step, stage_index))
    addition_seconds = perf_counter() - addition_started

    return MergeResult(
        source_graphs=(left_copy, right_copy),
        initial_graph=initial,
        final_graph=final,
        added_edges=added_edges,
        namespaces=(left_namespace, right_namespace),
        seed=seed,
        stage_index=stage_index,
        selection_seconds=selection_seconds,
        addition_seconds=addition_seconds,
    )


def merge_with_pairs(
    left: nx.Graph,
    right: nx.Graph,
    pairs: Sequence[tuple[Hashable, Hashable]],
    *,
    selection_seconds: float,
    seed: int | None = None,
    left_namespace: SourceId = "left",
    right_namespace: SourceId = "right",
    stage_index: int | None = None,
) -> MergeResult:
    """Merge graphs using preselected endpoint-disjoint pairs.

    This bridges the timed selection strategies to the staged plotting model;
    namespacing and graph copying are intentionally excluded from both the
    supplied selection time and measured edge-addition time.
    """

    _validate_source("left", left)
    _validate_source("right", right)
    _validate_namespace("left_namespace", left_namespace)
    _validate_namespace("right_namespace", right_namespace)
    if left_namespace == right_namespace:
        raise ValueError("left_namespace and right_namespace must be distinct")
    if stage_index is not None and (
        not isinstance(stage_index, Integral) or isinstance(stage_index, bool) or stage_index < 1
    ):
        raise ValueError("stage_index must be a positive integer or None")
    if not math.isfinite(selection_seconds) or selection_seconds < 0:
        raise ValueError("selection_seconds must be finite and nonnegative")

    normalized = tuple(tuple(pair) for pair in pairs)
    if not normalized:
        raise ValueError("pairs must not be empty")
    if any(len(pair) != 2 for pair in normalized):
        raise ValueError("every entry in pairs must contain two endpoints")
    left_nodes = {pair[0] for pair in normalized}
    right_nodes = {pair[1] for pair in normalized}
    if len(left_nodes) != len(normalized) or len(right_nodes) != len(normalized):
        raise ValueError("pairs must not reuse an endpoint")
    unknown_left = left_nodes - set(left)
    unknown_right = right_nodes - set(right)
    if unknown_left or unknown_right:
        raise ValueError("pairs contain an endpoint outside its source graph")

    left_names = {node: (left_namespace, node) for node in left}
    right_names = {node: (right_namespace, node) for node in right}
    namespaced_left = nx.relabel_nodes(left.copy(), left_names, copy=True)
    namespaced_right = nx.relabel_nodes(right.copy(), right_names, copy=True)
    _annotate_source(namespaced_left, left_namespace)
    _annotate_source(namespaced_right, right_namespace)
    initial = nx.compose(namespaced_left, namespaced_right)
    initial.graph.update(merge_edge_count=len(normalized), merge_seed=seed)
    added_edges = tuple(
        (left_names[left_node], right_names[right_node])
        for left_node, right_node in normalized
    )

    final = initial.copy()
    addition_started = perf_counter()
    for step, edge in enumerate(added_edges, start=1):
        final.add_edge(*edge, **_merge_edge_attributes(step, stage_index))
    addition_seconds = perf_counter() - addition_started
    return MergeResult(
        source_graphs=(left.copy(), right.copy()),
        initial_graph=initial,
        final_graph=final,
        added_edges=added_edges,
        namespaces=(left_namespace, right_namespace),
        seed=seed,
        stage_index=stage_index,
        selection_seconds=float(selection_seconds),
        addition_seconds=addition_seconds,
    )


def random_merge_sequence(
    graphs: Sequence[nx.Graph],
    edge_counts: int | Sequence[int],
    *,
    seed: int | None = None,
    source_ids: Sequence[SourceId] | None = None,
) -> MergeSequenceResult:
    """Merge ``G1`` with ``G2``, then that composite with ``G3``, and so on."""

    if len(graphs) < 2:
        raise ValueError("a merge sequence requires at least two source graphs")
    for index, graph in enumerate(graphs, start=1):
        _validate_source(f"graphs[{index - 1}]", graph)

    ids = tuple(source_ids) if source_ids is not None else tuple(f"G_{i}" for i in range(1, len(graphs) + 1))
    if len(ids) != len(graphs):
        raise ValueError("source_ids must contain one identifier per graph")
    for index, source_id in enumerate(ids):
        _validate_namespace(f"source_ids[{index}]", source_id)
    if len(set(ids)) != len(ids):
        raise ValueError("source_ids must be distinct")

    if isinstance(edge_counts, Integral) and not isinstance(edge_counts, bool):
        counts = (int(edge_counts),) * (len(graphs) - 1)
    else:
        counts = tuple(edge_counts)
        if len(counts) != len(graphs) - 1:
            raise ValueError("edge_counts must contain one value per merge stage")

    random_source = random.Random(seed)
    stage_seeds = [random_source.randrange(2**63) for _ in counts]
    stages = []
    composite = graphs[0]
    for stage_index, (new_graph, edge_count, stage_seed) in enumerate(
        zip(graphs[1:], counts, stage_seeds, strict=True), start=1
    ):
        left_namespace: SourceId = ids[0] if stage_index == 1 else ("composite", stage_index - 1)
        stage = random_merge(
            composite,
            new_graph,
            edge_count,
            seed=stage_seed,
            left_namespace=left_namespace,
            right_namespace=ids[stage_index],
            stage_index=stage_index,
        )
        stages.append(stage)
        composite = stage.final_graph

    return MergeSequenceResult(
        source_graphs=tuple(graph.copy() for graph in graphs),
        source_ids=ids,
        stages=tuple(stages),
        seed=seed,
    )


def _validate_source(label: str, graph: nx.Graph) -> None:
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"{label} must be a NetworkX graph")
    if graph.is_directed():
        raise TypeError(f"{label} must be undirected")
    if graph.is_multigraph():
        raise TypeError(f"{label} must be simple; project parallel edges before merging")
    if graph.number_of_nodes() == 0:
        raise ValueError(f"{label} graph must not be empty")


def _annotate_source(graph: nx.Graph, source: SourceId) -> None:
    for namespaced_node in graph:
        attributes = graph.nodes[namespaced_node]
        attributes.setdefault("source_graph", source)
        attributes.setdefault("original_node", namespaced_node[1])
    for first, second, attributes in graph.edges(data=True):
        first_source = graph.nodes[first].get("source_graph")
        second_source = graph.nodes[second].get("source_graph")
        if first_source == second_source and first_source is not None:
            attributes.setdefault("source_graph", first_source)


def _validate_namespace(label: str, value: SourceId) -> None:
    try:
        hash(value)
    except TypeError as error:
        raise TypeError(f"{label} must be hashable") from error


def _merge_edge_attributes(step: int, stage_index: int | None) -> dict[str, object]:
    attributes: dict[str, object] = {"merge_edge": True, "merge_step": step}
    if stage_index is not None:
        attributes["merge_stage"] = stage_index
    return attributes
