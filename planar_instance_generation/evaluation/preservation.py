"""Instancewise empirical audit of the paper's preservation proposition."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Hashable, Mapping, Sequence

import networkx as nx
from networkx.algorithms.approximation import treewidth_min_degree


@dataclass(frozen=True, slots=True)
class EmbeddedSourceGraph:
    """A source graph and its injective node map into one composite."""

    name: str
    graph: nx.Graph
    node_map: Mapping[Hashable, Hashable]


@dataclass(frozen=True, slots=True)
class PreservationPropertyAudit:
    property: str
    paper_bound: str
    paper_lower_bound: float | None
    paper_upper_bound: float | None
    observed_value: float | None
    observed_lower_bound: float | None
    observed_upper_bound: float | None
    empirically_within_bound: bool | None
    lower_slack: float | None
    upper_slack: float | None
    note: str = ""
    paper_hypothesis: str = "unconditional"
    paper_hypothesis_satisfied: bool | None = True
    numerically_within_bound: bool | None = None
    paper_rhs_lower_estimate: float | None = None
    paper_rhs_upper_estimate: float | None = None


@dataclass(frozen=True, slots=True)
class DegreeDistributionAudit:
    degree: int
    paper_bound: str
    source_probability: float
    composite_probability: float
    absolute_change: float
    paper_upper_bound: float
    slack: float
    within_bound: bool


@dataclass(frozen=True, slots=True)
class NodeClusteringAudit:
    source: str
    source_node: str
    composite_node: str
    paper_bound: str
    coupling_node: bool
    source_clustering: float
    composite_clustering: float
    absolute_change: float
    equality_required: bool
    equality_holds: bool | None


@dataclass(frozen=True, slots=True)
class PreservationAudit:
    properties: tuple[PreservationPropertyAudit, ...]
    degree_distribution: tuple[DegreeDistributionAudit, ...]
    node_clustering: tuple[NodeClusteringAudit, ...]
    source_node_count: int
    source_edge_count: int
    coupling_edge_count: int
    coupling_node_count: int
    maximum_coupling_degree: int
    sequential_matching: bool
    noncrossing_matching_certified: bool | None
    composition_strategy: str | None


@dataclass(frozen=True, slots=True)
class PreservationReportArtifacts:
    properties_csv: Path
    degree_distribution_csv: Path
    node_clustering_csv: Path
    json: Path


def audit_preserved_properties(
    sources: Sequence[EmbeddedSourceGraph],
    composite: nx.Graph,
    coupling_edges: Sequence[tuple[Hashable, Hashable]],
    *,
    noncrossing_matching_certified: bool | None = None,
    composition_strategy: str | None = None,
    tolerance: float = 1e-9,
) -> PreservationAudit:
    """Compare one composite to its exact constituents and paper bounds.

    Source order is the sequential join order.  Set the noncrossing flag to
    ``True`` only when the construction certifies it, ``False`` for a known
    violation, and ``None`` when it was not checked.
    """

    if not sources:
        raise ValueError("sources must not be empty")
    _validate_graph("composite", composite)
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and nonnegative")
    if (
        noncrossing_matching_certified is not None
        and not isinstance(noncrossing_matching_certified, bool)
    ):
        raise TypeError("noncrossing_matching_certified must be bool or None")
    if composition_strategy is not None and not composition_strategy:
        raise ValueError("composition_strategy must be nonempty when supplied")
    mapped_sources, owner = _mapped_sources(sources, composite)
    coupling = tuple(_canonical_edge(edge) for edge in coupling_edges)
    if len(set(coupling)) != len(coupling):
        raise ValueError("coupling_edges must be distinct")
    for edge in coupling:
        first, second = tuple(edge)
        if not composite.has_edge(first, second):
            raise ValueError(f"coupling edge {edge!r} is absent from the composite")
        if owner[first] == owner[second]:
            raise ValueError("coupling edges must join different original sources")

    source_edges = {
        _canonical_edge((first, second))
        for graph in mapped_sources
        for first, second in graph.edges
    }
    coupling_set = set(coupling)
    actual_edges = {_canonical_edge(edge) for edge in composite.edges}
    if source_edges & coupling_set:
        raise ValueError("a coupling edge duplicates a source edge")
    if actual_edges != source_edges | coupling_set:
        raise ValueError("composite edges are not exactly source plus coupling edges")

    node_count = composite.number_of_nodes()
    source_node_count = sum(graph.number_of_nodes() for graph in mapped_sources)
    source_edge_count = sum(graph.number_of_edges() for graph in mapped_sources)
    coupling_degrees = Counter(node for edge in coupling for node in edge)
    coupling_nodes = set(coupling_degrees)
    coupling_node_count = len(coupling_nodes)
    maximum_coupling_degree = max(coupling_degrees.values(), default=0)
    affected_fraction = coupling_node_count / node_count
    sequential_matching = _is_sequential_matching(coupling, owner)
    conditional_hypothesis = (
        False
        if not sequential_matching or noncrossing_matching_certified is False
        else True
        if noncrossing_matching_certified is True
        else None
    )

    source_degrees = Counter(
        degree for graph in mapped_sources for _, degree in graph.degree
    )
    composite_degrees = Counter(dict(composite.degree).values())
    degree_rows = []
    for degree in sorted(set(source_degrees) | set(composite_degrees)):
        source_probability = source_degrees[degree] / node_count
        composite_probability = composite_degrees[degree] / node_count
        change = abs(composite_probability - source_probability)
        degree_rows.append(
            DegreeDistributionAudit(
                degree=degree,
                paper_bound="|p_C(q) - p_src(q)| <= |B| / |V(C)| for every q",
                source_probability=source_probability,
                composite_probability=composite_probability,
                absolute_change=change,
                paper_upper_bound=affected_fraction,
                slack=affected_fraction - change,
                within_bound=change <= affected_fraction + tolerance,
            )
        )

    source_clustering_by_node = {}
    clustering_rows = []
    for source, graph in zip(sources, mapped_sources, strict=True):
        original_values = nx.clustering(source.graph)
        mapped_values = nx.clustering(composite, nodes=graph.nodes)
        for original_node, composite_node in source.node_map.items():
            source_value = float(original_values[original_node])
            composite_value = float(mapped_values[composite_node])
            source_clustering_by_node[composite_node] = source_value
            required = composite_node not in coupling_nodes
            change = abs(composite_value - source_value)
            clustering_rows.append(
                NodeClusteringAudit(
                    source=source.name,
                    source_node=repr(original_node),
                    composite_node=repr(composite_node),
                    paper_bound="c_C(v) = c_Gi(v) for every v not in B",
                    coupling_node=not required,
                    source_clustering=source_value,
                    composite_clustering=composite_value,
                    absolute_change=change,
                    equality_required=required,
                    equality_holds=change <= tolerance if required else None,
                )
            )

    maximum_source_degree = max(
        max(dict(graph.degree).values(), default=0) for graph in mapped_sources
    )
    source_average_clustering = sum(source_clustering_by_node.values()) / node_count
    composite_average_clustering = float(nx.average_clustering(composite))
    average_clustering_difference = (
        source_average_clustering - composite_average_clustering
    )
    source_degeneracies = [max(nx.core_number(graph).values()) for graph in mapped_sources]
    composite_degeneracy = max(nx.core_number(composite).values())
    source_treewidth_upper = [int(treewidth_min_degree(graph)[0]) for graph in mapped_sources]
    composite_treewidth_upper = int(treewidth_min_degree(composite)[0])
    composite_maximum_degree = max(dict(composite.degree).values())
    degree_identity_error = max(
        abs(
            composite.degree[node]
            - next(graph.degree[node] for graph in mapped_sources if node in graph)
            - coupling_degrees[node]
        )
        for node in composite
    )
    noncoupling_clustering_error = max(
        (row.absolute_change for row in clustering_rows if row.equality_required),
        default=0.0,
    )

    properties = (
        _exact("node_count", "|V(C)| = sum_i |V(G_i)|", source_node_count, node_count, tolerance),
        _exact(
            "edge_count",
            "|E(C)| = sum_i |E(G_i)| + |M|",
            source_edge_count + len(coupling),
            composite.number_of_edges(),
            tolerance,
        ),
        _exact(
            "node_degree_identity_max_error",
            "d_C(v) = d_Gi(v) + d_M(v) for every v",
            0.0,
            float(degree_identity_error),
            tolerance,
        ),
        _interval(
            "maximum_degree",
            "max{max_i Delta(G_i), Delta(M)} <= Delta(C) <= max_i Delta(G_i) + Delta(M)",
            float(max(maximum_source_degree, maximum_coupling_degree)),
            float(maximum_source_degree + maximum_coupling_degree),
            float(composite_maximum_degree),
            tolerance,
        ),
        _exact(
            "noncoupling_node_clustering_max_error",
            "c_C(v) = c_Gi(v) for every v not in B",
            0.0,
            noncoupling_clustering_error,
            tolerance,
        ),
        _interval(
            "average_clustering_difference",
            "0 <= cbar_src - cbar(C) <= |B|/|V(C)|",
            0.0,
            affected_fraction,
            average_clustering_difference,
            tolerance,
            paper_hypothesis=(
                "sequential composition using the noncrossing matchings of "
                "Lemma lem:floor"
            ),
            paper_hypothesis_satisfied=conditional_hypothesis,
            note=(
                f"cbar_src={source_average_clustering:.17g}; "
                f"cbar(C)={composite_average_clustering:.17g}."
            ),
        ),
        _upper(
            "degeneracy",
            "degen(C) <= max_i degen(G_i) + 1",
            float(max(source_degeneracies) + 1),
            float(composite_degeneracy),
            tolerance,
            paper_hypothesis=(
                "sequential composition using the matchings of Lemma lem:floor"
            ),
            paper_hypothesis_satisfied=conditional_hypothesis,
            note=f"max_i degen(G_i)={max(source_degeneracies)}.",
        ),
        _treewidth_audit(
            max(source_degeneracies),
            max(source_treewidth_upper),
            coupling_node_count,
            composite_degeneracy,
            composite_treewidth_upper,
        ),
    )
    return PreservationAudit(
        properties=properties,
        degree_distribution=tuple(degree_rows),
        node_clustering=tuple(clustering_rows),
        source_node_count=source_node_count,
        source_edge_count=source_edge_count,
        coupling_edge_count=len(coupling),
        coupling_node_count=coupling_node_count,
        maximum_coupling_degree=maximum_coupling_degree,
        sequential_matching=sequential_matching,
        noncrossing_matching_certified=noncrossing_matching_certified,
        composition_strategy=composition_strategy,
    )


def write_preservation_report(
    audits: Mapping[str, PreservationAudit],
    output_directory: str | Path,
    *,
    overwrite: bool = False,
) -> PreservationReportArtifacts:
    """Write tidy per-instance property, degree, and node-clustering audits."""

    if not audits:
        raise ValueError("audits must not be empty")
    output = Path(output_directory)
    artifacts = PreservationReportArtifacts(
        properties_csv=output / "preservation_properties.csv",
        degree_distribution_csv=output / "preservation_degree_distribution.csv",
        node_clustering_csv=output / "preservation_node_clustering.csv",
        json=output / "preservation.json",
    )
    existing = [path for path in asdict(artifacts).values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"preservation artifact already exists: {existing[0]}")
    output.mkdir(parents=True, exist_ok=True)
    property_rows = [
        {**_audit_context(instance, audit), **asdict(row)}
        for instance, audit in audits.items()
        for row in audit.properties
    ]
    degree_rows = [
        {**_audit_context(instance, audit), **asdict(row)}
        for instance, audit in audits.items()
        for row in audit.degree_distribution
    ]
    clustering_rows = [
        {**_audit_context(instance, audit), **asdict(row)}
        for instance, audit in audits.items()
        for row in audit.node_clustering
    ]
    _write_csv(artifacts.properties_csv, property_rows)
    _write_csv(artifacts.degree_distribution_csv, degree_rows)
    _write_csv(artifacts.node_clustering_csv, clustering_rows)
    artifacts.json.write_text(
        json.dumps(
            {
                instance: asdict(audit) for instance, audit in audits.items()
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return artifacts


def _mapped_sources(
    sources: Sequence[EmbeddedSourceGraph], composite: nx.Graph
) -> tuple[tuple[nx.Graph, ...], dict[Hashable, int]]:
    mapped = []
    owner = {}
    for source_index, source in enumerate(sources):
        if not isinstance(source, EmbeddedSourceGraph) or not source.name:
            raise TypeError("sources must contain named EmbeddedSourceGraph records")
        _validate_graph(source.name, source.graph)
        if set(source.node_map) != set(source.graph):
            raise ValueError(f"{source.name}: node_map must cover exactly the source nodes")
        values = tuple(source.node_map.values())
        if len(set(values)) != len(values):
            raise ValueError(f"{source.name}: node_map must be injective")
        for value in values:
            if value not in composite:
                raise ValueError(f"{source.name}: mapped node {value!r} is absent")
            if value in owner:
                raise ValueError("source node maps must have disjoint images")
            owner[value] = source_index
        mapped.append(nx.relabel_nodes(source.graph, source.node_map, copy=True))
    if set(owner) != set(composite):
        raise ValueError("source node maps must partition the composite nodes")
    return tuple(mapped), owner


def _exact(
    name: str, statement: str, expected: float, observed: float, tolerance: float
) -> PreservationPropertyAudit:
    return _interval(name, statement, float(expected), float(expected), float(observed), tolerance)


def _interval(
    name: str,
    statement: str,
    lower: float,
    upper: float,
    observed: float,
    tolerance: float,
    *,
    paper_hypothesis: str = "unconditional",
    paper_hypothesis_satisfied: bool | None = True,
    note: str = "",
) -> PreservationPropertyAudit:
    numerical = lower - tolerance <= observed <= upper + tolerance
    return PreservationPropertyAudit(
        property=name,
        paper_bound=statement,
        paper_lower_bound=lower,
        paper_upper_bound=upper,
        observed_value=observed,
        observed_lower_bound=observed,
        observed_upper_bound=observed,
        empirically_within_bound=(
            numerical if paper_hypothesis_satisfied is True else None
        ),
        lower_slack=observed - lower,
        upper_slack=upper - observed,
        note=_condition_note(note, paper_hypothesis_satisfied),
        paper_hypothesis=paper_hypothesis,
        paper_hypothesis_satisfied=paper_hypothesis_satisfied,
        numerically_within_bound=numerical,
    )


def _upper(
    name: str,
    statement: str,
    upper: float,
    observed: float,
    tolerance: float,
    *,
    paper_hypothesis: str = "unconditional",
    paper_hypothesis_satisfied: bool | None = True,
    note: str = "",
) -> PreservationPropertyAudit:
    numerical = observed <= upper + tolerance
    return PreservationPropertyAudit(
        property=name,
        paper_bound=statement,
        paper_lower_bound=None,
        paper_upper_bound=upper,
        observed_value=observed,
        observed_lower_bound=observed,
        observed_upper_bound=observed,
        empirically_within_bound=(
            numerical if paper_hypothesis_satisfied is True else None
        ),
        lower_slack=None,
        upper_slack=upper - observed,
        note=_condition_note(note, paper_hypothesis_satisfied),
        paper_hypothesis=paper_hypothesis,
        paper_hypothesis_satisfied=paper_hypothesis_satisfied,
        numerically_within_bound=numerical,
    )


def _treewidth_audit(
    maximum_source_lower: int,
    maximum_source_upper: int,
    coupling_node_count: int,
    observed_lower: int,
    observed_upper: int,
) -> PreservationPropertyAudit:
    rhs_lower = maximum_source_lower + coupling_node_count
    rhs_upper = maximum_source_upper + coupling_node_count
    empirical = True if observed_upper <= rhs_lower else (
        False if observed_lower > rhs_upper else None
    )
    exact_rhs = rhs_lower == rhs_upper
    exact_observation = observed_lower == observed_upper
    return PreservationPropertyAudit(
        property="treewidth",
        paper_bound="tw(C) <= max_i tw(G_i) + |B|",
        paper_lower_bound=None,
        paper_upper_bound=float(rhs_lower) if exact_rhs else None,
        observed_value=float(observed_lower) if exact_observation else None,
        observed_lower_bound=float(observed_lower),
        observed_upper_bound=float(observed_upper),
        empirically_within_bound=empirical,
        lower_slack=None,
        upper_slack=(
            float(rhs_lower - observed_lower)
            if exact_rhs and exact_observation
            else None
        ),
        note=(
            "Treewidth is bracketed by degeneracy and the min-degree heuristic: "
            f"max_i tw(G_i) in [{maximum_source_lower}, {maximum_source_upper}], "
            f"paper RHS in [{rhs_lower}, {rhs_upper}], and tw(C) in "
            f"[{observed_lower}, {observed_upper}]."
        ),
        numerically_within_bound=empirical,
        paper_rhs_lower_estimate=float(rhs_lower),
        paper_rhs_upper_estimate=float(rhs_upper),
    )


def _is_sequential_matching(
    coupling: Sequence[frozenset[Hashable]], owner: Mapping[Hashable, int]
) -> bool:
    """Check the matching used when each source joins all earlier sources."""

    stage_endpoints: dict[int, tuple[set[Hashable], set[Hashable]]] = {}
    for edge in coupling:
        first, second = tuple(edge)
        first_owner = owner[first]
        second_owner = owner[second]
        later = max(first_owner, second_owner)
        new_node, old_node = (
            (first, second) if first_owner == later else (second, first)
        )
        old_nodes, new_nodes = stage_endpoints.setdefault(later, (set(), set()))
        if old_node in old_nodes or new_node in new_nodes:
            return False
        old_nodes.add(old_node)
        new_nodes.add(new_node)
    return True


def _condition_note(note: str, status: bool | None) -> str:
    parts = [note] if note else []
    if status is False:
        parts.append(
            "The paper hypothesis is not satisfied; the numerical comparison is descriptive only."
        )
    elif status is None:
        parts.append(
            "The noncrossing hypothesis is not certified; the numerical comparison "
            "is descriptive only."
        )
    return " ".join(parts)


def _audit_context(instance: str, audit: PreservationAudit) -> dict[str, object]:
    return {
        "instance": instance,
        "composition_strategy": audit.composition_strategy,
        "sequential_matching": audit.sequential_matching,
        "noncrossing_matching_certified": audit.noncrossing_matching_certified,
    }


def _canonical_edge(edge: tuple[Hashable, Hashable]) -> frozenset[Hashable]:
    if len(edge) != 2 or edge[0] == edge[1]:
        raise ValueError("edges must contain two distinct endpoints")
    return frozenset(edge)


def _validate_graph(label: str, graph: nx.Graph) -> None:
    if not isinstance(graph, nx.Graph) or graph.is_directed() or graph.is_multigraph():
        raise TypeError(f"{label} must be a simple undirected graph")
    if not graph or nx.number_of_selfloops(graph):
        raise ValueError(f"{label} must be nonempty and loop-free")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
