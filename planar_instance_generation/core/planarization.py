"""OGDF planarization results, certificate transfer, and outer-face selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from numbers import Integral
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Hashable, Literal, Mapping, TypeAlias, TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from .ogdf import OgdfStatistics

Edge: TypeAlias = tuple[Hashable, Hashable]
OuterFaceSampling: TypeAlias = Literal[
    "equidistant", "consecutive", "adaptive_skip_one"
]


class OuterFaceCapacityError(ValueError):
    """Raised when an outer-face matching cannot supply its requested endpoints."""


@dataclass(frozen=True, slots=True)
class CertifiedPlanarPiece:
    """A vertex-disjoint piece relabeled into the composite.

    Numeric lower bounds are trusted certificates supplied by the caller.
    The planar subgraph is a constructive certificate checked here.
    """

    graph: nx.Graph
    planar_subgraph: nx.Graph
    lower_bound: int
    bound_rows: tuple = ()


def _piece_forest_bound(graph, pieces):
    """Bound after a validated piece partition; handles disconnected graphs."""
    forest_edges = sum(
        nx.number_connected_components(piece.graph) for piece in pieces
    ) - nx.number_connected_components(graph)
    coupling_edges = graph.number_of_edges() - sum(
        piece.graph.number_of_edges() for piece in pieces
    )
    piece_deletions = sum(
        piece.graph.number_of_edges() - piece.planar_subgraph.number_of_edges()
        for piece in pieces
    )
    return forest_edges, piece_deletions + coupling_edges - forest_edges


@dataclass(frozen=True, slots=True)
class HeuristicPlanarizationResult:
    """A planar-subgraph heuristic with an explicit certificate.

    The returned subgraph realizes ``omitted_upper_bound``.  The lower bound
    is the componentwise Euler bound (and one for any non-planar graph), so
    equality of the two bounds is already an exact skewness proof even though
    no optimizer was used.
    """

    planar_subgraph: nx.Graph
    omitted_edges: tuple[Edge, ...]
    omitted_lower_bound: int
    omitted_upper_bound: int
    exact: bool
    termination_status: str
    original_node_count: int
    original_edge_count: int
    seed: int
    time_limit_seconds: float | None
    total_runtime_seconds: float
    restarts_completed: int
    largest_face_candidates: int
    iteration_limit: int | None = None
    shared_piece_count: int = 0
    shared_piece_lower_bound: int = 0
    greedy_start_upper_bound: int | None = None
    piece_start_upper_bound: int | None = None
    reuse_construction_seconds: float = 0.0
    prior_certificate_count: int = 0
    coupling_forest_edge_count: int = 0
    composition_upper_bound: int | None = None
    bound_rows: tuple = ()
    solver_metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.omitted_lower_bound <= self.omitted_upper_bound:
            raise ValueError("invalid omitted-edge bounds")
        if len(self.omitted_edges) != self.omitted_upper_bound:
            raise ValueError("the omitted edges must certify the upper bound")
        if self.exact != (self.omitted_lower_bound == self.omitted_upper_bound):
            raise ValueError("exact must agree with the omitted-edge bounds")

    def statistics(self) -> dict[str, Any]:
        """Return every heuristic measurement as a flat evaluation row."""

        return {
            "method": "ogdf_default_fast",
            "nodes": self.original_node_count,
            "edges": self.original_edge_count,
            "omitted_lower_bound": self.omitted_lower_bound,
            "omitted_upper_bound": self.omitted_upper_bound,
            "absolute_gap": self.omitted_upper_bound - self.omitted_lower_bound,
            "exact": self.exact,
            "status": self.termination_status,
            "seed": self.seed,
            "time_limit_seconds": self.time_limit_seconds,
            "iteration_limit": self.iteration_limit,
            "shared_piece_count": self.shared_piece_count,
            "shared_piece_lower_bound": self.shared_piece_lower_bound,
            "greedy_start_upper_bound": self.greedy_start_upper_bound,
            "piece_start_upper_bound": self.piece_start_upper_bound,
            "reuse_construction_seconds": self.reuse_construction_seconds,
            "prior_certificate_count": self.prior_certificate_count,
            "coupling_forest_edge_count": self.coupling_forest_edge_count,
            "composition_upper_bound": self.composition_upper_bound,
            "total_runtime_seconds": self.total_runtime_seconds,
            "budget_overrun_seconds": (
                None
                if self.time_limit_seconds is None
                else max(0.0, self.total_runtime_seconds - self.time_limit_seconds)
            ),
            "restarts_completed": self.restarts_completed,
            "largest_face_candidates": self.largest_face_candidates,
        }


@dataclass(frozen=True, slots=True)
class PlanarizationResult:
    """An incumbent MPS and a certified interval for omitted edges.

    ``omitted_lower_bound <= skewness <= omitted_upper_bound`` always holds.
    The returned planar subgraph realizes ``omitted_upper_bound``.  Thus a
    time-limited result remains both usable and auditable.
    """

    planar_subgraph: nx.Graph
    omitted_edges: tuple[Edge, ...]
    omitted_lower_bound: int
    omitted_upper_bound: int
    exact: bool
    termination_status: str
    original_node_count: int
    original_edge_count: int
    seed: int
    threads: int
    time_limit_seconds: float
    total_runtime_seconds: float
    incumbent_construction_seconds: float
    optimizer: OgdfStatistics | None
    log_path: str | None

    shared_piece_count: int = 0
    prior_certificate_count: int = 0
    shared_piece_lower_bound: int = 0
    greedy_start_upper_bound: int | None = None
    piece_start_upper_bound: int | None = None
    coupling_forest_edge_count: int = 0
    composition_upper_bound: int | None = None
    bound_rows: tuple = ()
    solver_metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.omitted_lower_bound <= self.omitted_upper_bound:
            raise ValueError("invalid omitted-edge bounds")
        if len(self.omitted_edges) != self.omitted_upper_bound:
            raise ValueError("the omitted edges must certify the upper bound")
        if self.exact != (self.omitted_lower_bound == self.omitted_upper_bound):
            raise ValueError("exact must agree with the omitted-edge bounds")

    @property
    def skewness(self) -> int | None:
        """Return exact skewness, or ``None`` while the bracket is open."""

        return self.omitted_upper_bound if self.exact else None

    @property
    def maximum_planar_edges_lower_bound(self) -> int:
        return self.original_edge_count - self.omitted_upper_bound

    @property
    def maximum_planar_edges_upper_bound(self) -> int:
        return self.original_edge_count - self.omitted_lower_bound

    def statistics(self) -> dict[str, Any]:
        """Return a flat, JSON/CSV-friendly evaluation record."""

        row: dict[str, Any] = {
            "nodes": self.original_node_count,
            "edges": self.original_edge_count,
            "omitted_lower_bound": self.omitted_lower_bound,
            "omitted_upper_bound": self.omitted_upper_bound,
            "absolute_gap": self.omitted_upper_bound - self.omitted_lower_bound,
            "relative_gap": (
                (self.omitted_upper_bound - self.omitted_lower_bound)
                / self.omitted_upper_bound
                if self.omitted_upper_bound
                else 0.0
            ),
            "exact": self.exact,
            "status": self.termination_status,
            "total_runtime_seconds": self.total_runtime_seconds,
            "incumbent_construction_seconds": self.incumbent_construction_seconds,
            "seed": self.seed,
            "threads": self.threads,
            "time_limit_seconds": self.time_limit_seconds,
            "log_path": self.log_path,
            "prior_certificate_count": self.prior_certificate_count,
            "shared_piece_count": self.shared_piece_count,
            "shared_piece_lower_bound": self.shared_piece_lower_bound,
            "greedy_start_upper_bound": self.greedy_start_upper_bound,
            "piece_start_upper_bound": self.piece_start_upper_bound,
            "coupling_forest_edge_count": self.coupling_forest_edge_count,
            "composition_upper_bound": self.composition_upper_bound,
        }
        if self.optimizer is not None:
            row.update(
                {
                    f"optimizer_{key}": value
                    for key, value in asdict(self.optimizer).items()
                }
            )
        return row


@dataclass(frozen=True, slots=True)
class OuterFace:
    """One face boundary and its distinct candidate nodes in cyclic order."""

    boundary_walk: tuple[Hashable, ...]
    candidates: tuple[Hashable, ...]
    all_face_candidate_counts: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ExactOuterFaceSelection:
    """A noncrossing matching between two planar-subgraph outer faces."""

    pairs: tuple[tuple[Hashable, Hashable], ...]
    left_face: OuterFace
    right_face: OuterFace
    selection_seconds: float
    seed: int
    sampling: OuterFaceSampling
    prefer_non_leaf: bool
    left_non_leaf_candidate_count: int
    right_non_leaf_candidate_count: int
    left_selected_leaf_count: int
    right_selected_leaf_count: int


@dataclass(frozen=True, slots=True)
class ExactMergeNodeSelection:
    """End-to-end exact selector, including both MPS computations."""

    pairs: tuple[tuple[Hashable, Hashable], ...]
    strategy: str
    seed: int
    selection_seconds: float
    left_planarization: PlanarizationResult
    right_planarization: PlanarizationResult
    faces: ExactOuterFaceSelection


@dataclass(frozen=True, slots=True)
class HeuristicMergeNodeSelection:
    """Default OGDF heuristic and outer-face selection for two graphs."""

    pairs: tuple[tuple[Hashable, Hashable], ...]
    strategy: str
    seed: int
    selection_seconds: float
    iterations_per_graph: int
    left_planarization: HeuristicPlanarizationResult
    right_planarization: HeuristicPlanarizationResult
    faces: ExactOuterFaceSelection

    def statistics(self) -> dict[str, Any]:
        """Return selection and both planarization measurements in one row."""

        row: dict[str, Any] = {
            "strategy": self.strategy,
            "seed": self.seed,
            "pair_count": len(self.pairs),
            "selection_seconds": self.selection_seconds,
            "iterations_per_graph": self.iterations_per_graph,
            "planarity_certificate": True,
        }
        for side, result in (
            ("left", self.left_planarization),
            ("right", self.right_planarization),
        ):
            row.update(
                {f"{side}_{key}": value for key, value in result.statistics().items()}
            )
        return row


def relabel_planar_piece(
    graph: nx.Graph,
    result: HeuristicPlanarizationResult | PlanarizationResult,
    node_mapping: Mapping[Hashable, Hashable],
) -> CertifiedPlanarPiece:
    """Transfer a constituent certificate into the next composite's node IDs.

    Copies are local to one generation attempt. Old solve times are not carried
    forward; only the constructive certificate and certified bound are reused.
    """
    if set(node_mapping) != set(graph) or len(set(node_mapping.values())) != len(graph):
        raise ValueError(
            "certificate node mapping must be a bijection on the constituent"
        )
    _certify_planar_subgraph(graph, result.planar_subgraph, result.omitted_edges)
    return CertifiedPlanarPiece(
        nx.relabel_nodes(graph, node_mapping, copy=True),
        nx.relabel_nodes(result.planar_subgraph, node_mapping, copy=True),
        result.omitted_lower_bound,
        tuple(
            (tuple((node_mapping[u], node_mapping[v]) for u, v in edges), lower)
            for edges, lower in result.bound_rows
        ),
    )


def maximum_planar_subgraph(
    graph: nx.Graph,
    *,
    time_limit_seconds: float = 600.0,
    seed: int = 0,
    threads: int = 16,
    log_path: str | Path | None = None,
    certified_pieces: tuple[CertifiedPlanarPiece, ...] = (),
    prior_certificates: tuple[CertifiedPlanarPiece, ...] = (),
) -> PlanarizationResult:
    """Run OGDF exact planarization with static rows and direct solution reuse.

    The native time budget includes native initialization and is shared across
    blocks. Fresh Fast10 and static preprocessing are reported in total time.
    The fresh heuristic uses OGDF's constructor default of ten runs.
    """
    from .ogdf import planarize

    return planarize(
        graph,
        exact=True,
        time_limit_seconds=time_limit_seconds,
        seed=seed,
        threads=threads,
        log_path=log_path,
        certified_pieces=certified_pieces,
        prior_certificates=prior_certificates,
    )


def ogdf_planar_subgraph(
    graph: nx.Graph,
    *,
    seed: int = 0,
    threads: int = 16,
    certified_pieces=(),
    prior_certificates=(),
    log_path=None,
):
    """Default OGDF Fast10, compared with directly retained planar pieces."""
    from .ogdf import planarize

    return planarize(
        graph,
        seed=seed,
        threads=threads,
        log_path=log_path,
        certified_pieces=certified_pieces,
        prior_certificates=prior_certificates,
    )


def largest_outer_face(graph: nx.Graph) -> OuterFace:
    """Return a face whose boundary walk has the most distinct vertices.

    Every face of a planar embedding may be designated as its outer face.
    Boundary walks can repeat articulation vertices, whereas merge endpoints
    must be distinct, so both representations are retained.
    """

    _validate_simple_graph(graph)
    planar, embedding = nx.check_planarity(graph, counterexample=False)
    if not planar:
        raise ValueError("outer-face extraction requires a planar graph")
    if graph.number_of_nodes() == 0:
        return OuterFace((), (), ())
    if graph.number_of_edges() == 0:
        candidates = tuple(graph.nodes)
        return OuterFace(candidates, candidates, (len(candidates),))
    if not nx.is_connected(graph):
        raise ValueError("outer-face extraction currently requires a connected graph")

    seen_half_edges: set[tuple[Hashable, Hashable]] = set()
    faces: list[tuple[Hashable, ...]] = []
    for first, second in embedding.edges():
        if (first, second) in seen_half_edges:
            continue
        faces.append(
            tuple(
                embedding.traverse_face(first, second, mark_half_edges=seen_half_edges)
            )
        )

    if not faces:  # A single isolated vertex was handled above.
        candidates = tuple(graph.nodes)
        return OuterFace(candidates, candidates, (len(candidates),))
    unique_faces = [tuple(dict.fromkeys(face)) for face in faces]
    best_index = max(range(len(faces)), key=lambda index: len(unique_faces[index]))
    counts = tuple(sorted((len(face) for face in unique_faces), reverse=True))
    return OuterFace(faces[best_index], unique_faces[best_index], counts)


def exact_outer_face_pairs(
    left: nx.Graph | PlanarizationResult,
    right: nx.Graph | PlanarizationResult,
    count: int,
    *,
    seed: int = 0,
    sampling: OuterFaceSampling = "consecutive",
    prefer_non_leaf: bool = False,
    left_degree_graph: nx.Graph | None = None,
    right_degree_graph: nx.Graph | None = None,
) -> ExactOuterFaceSelection:
    """Select a noncrossing matching in reversed cyclic boundary order.

    The requested count is strict. If either face has fewer distinct candidates,
    the selection fails instead of silently changing the coupling count. The
    selected nodes are spread evenly, form a consecutive cyclic arc, or form an
    adaptive arc with as many evenly distributed one-node gaps as the boundary
    capacity permits.  When 'prefer_non_leaf' is enabled, degree-one boundary
    vertices are used only after every available non-leaf candidate is needed.
    """

    if not isinstance(count, Integral) or isinstance(count, bool):
        raise TypeError("count must be an integer")
    if count < 0:
        raise ValueError("count must be non-negative")
    if sampling not in {"equidistant", "consecutive", "adaptive_skip_one"}:
        raise ValueError(
            "sampling must be 'equidistant', 'consecutive', or " "'adaptive_skip_one'"
        )
    left_graph = left.planar_subgraph if isinstance(left, PlanarizationResult) else left
    right_graph = (
        right.planar_subgraph if isinstance(right, PlanarizationResult) else right
    )
    if not isinstance(prefer_non_leaf, bool):
        raise TypeError("prefer_non_leaf must be a boolean")
    _validate_simple_graph(left_graph)
    _validate_simple_graph(right_graph)
    started = perf_counter()
    left_face = largest_outer_face(left_graph)
    right_face = largest_outer_face(right_graph)
    left_degree_source = left_graph if left_degree_graph is None else left_degree_graph
    right_degree_source = (
        right_graph if right_degree_graph is None else right_degree_graph
    )
    left_non_leaf = _non_leaf_face_candidates(left_face, left_degree_source)
    right_non_leaf = _non_leaf_face_candidates(right_face, right_degree_source)
    available = min(len(left_face.candidates), len(right_face.candidates))
    if available < count:
        raise OuterFaceCapacityError(
            f"outer-face selection requested {count} distinct boundary nodes "
            "per graph, but found "
            f"{len(left_face.candidates)} and {len(right_face.candidates)}"
        )

    rng = random.Random(seed)
    selected_left = _sample_cyclic_preferred(
        left_face.candidates,
        count,
        rng,
        sampling=sampling,
        preferred=left_non_leaf if prefer_non_leaf else None,
    )
    selected_right_forward = _sample_cyclic_preferred(
        right_face.candidates,
        count,
        rng,
        sampling=sampling,
        preferred=right_non_leaf if prefer_non_leaf else None,
    )
    selected_right = tuple(reversed(selected_right_forward))
    pairs = tuple(zip(selected_left, selected_right, strict=True))

    # This cheap certificate catches ordering or face-extraction regressions.
    certificate_left = nx.relabel_nodes(
        left_graph, {node: ("left", node) for node in left_graph}, copy=True
    )
    certificate_right = nx.relabel_nodes(
        right_graph, {node: ("right", node) for node in right_graph}, copy=True
    )
    combined = nx.compose(certificate_left, certificate_right)
    combined.add_edges_from(
        (("left", left_node), ("right", right_node)) for left_node, right_node in pairs
    )
    if not nx.check_planarity(combined, counterexample=False)[0]:
        raise RuntimeError(
            "reversed outer-face order did not produce a planar matching"
        )

    return ExactOuterFaceSelection(
        pairs=pairs,
        left_face=left_face,
        right_face=right_face,
        selection_seconds=perf_counter() - started,
        seed=int(seed),
        sampling=sampling,
        prefer_non_leaf=prefer_non_leaf,
        left_non_leaf_candidate_count=len(left_non_leaf),
        right_non_leaf_candidate_count=len(right_non_leaf),
        left_selected_leaf_count=sum(
            node not in left_non_leaf for node in selected_left
        ),
        right_selected_leaf_count=sum(
            node not in right_non_leaf for node in selected_right
        ),
    )


def select_exact_outer_face_pairs(
    left: nx.Graph,
    right: nx.Graph,
    count: int,
    *,
    time_limit_seconds: float = 600.0,
    seed: int = 0,
    threads: int = 16,
    log_directory: str | Path | None = None,
    prefer_non_leaf: bool = False,
    sampling: OuterFaceSampling = "consecutive",
    left_certified_pieces: tuple[CertifiedPlanarPiece, ...] = (),
    left_prior_certificates: tuple[CertifiedPlanarPiece, ...] = (),
    right_certified_pieces: tuple[CertifiedPlanarPiece, ...] = (),
) -> ExactMergeNodeSelection:
    """Run the complete exact strategy and time every selection operation."""

    if not isinstance(count, Integral) or isinstance(count, bool) or count < 1:
        raise ValueError("count must be a positive integer")
    started = perf_counter()
    logs = None if log_directory is None else Path(log_directory)
    left_result = maximum_planar_subgraph(
        left,
        time_limit_seconds=time_limit_seconds,
        seed=seed,
        threads=threads,
        log_path=None if logs is None else logs / "left_planarization.log",
        certified_pieces=left_certified_pieces,
        prior_certificates=left_prior_certificates,
    )
    right_result = maximum_planar_subgraph(
        right,
        time_limit_seconds=time_limit_seconds,
        seed=(int(seed) + 1) % 2**31,
        threads=threads,
        log_path=None if logs is None else logs / "right_planarization.log",
        certified_pieces=right_certified_pieces,
    )
    faces = exact_outer_face_pairs(
        left_result,
        right_result,
        count,
        seed=seed,
        sampling=sampling,
        prefer_non_leaf=prefer_non_leaf,
        left_degree_graph=left,
        right_degree_graph=right,
    )
    return ExactMergeNodeSelection(
        pairs=faces.pairs,
        strategy="exact_outer_face",
        seed=seed,
        selection_seconds=perf_counter() - started,
        left_planarization=left_result,
        right_planarization=right_result,
        faces=faces,
    )


def select_heuristic_outer_face_pairs(
    left: nx.Graph,
    right: nx.Graph,
    count: int,
    *,
    seed: int = 0,
    threads: int = 16,
    log_directory: str | Path | None = None,
    prefer_non_leaf: bool = False,
    sampling: OuterFaceSampling = "consecutive",
    left_certified_pieces: tuple[CertifiedPlanarPiece, ...] = (),
    left_prior_certificates: tuple[CertifiedPlanarPiece, ...] = (),
    right_certified_pieces: tuple[CertifiedPlanarPiece, ...] = (),
) -> HeuristicMergeNodeSelection:
    """Select endpoints using default OGDF Fast10 and direct reuse."""

    if not isinstance(count, Integral) or isinstance(count, bool) or count < 1:
        raise ValueError("count must be a positive integer")

    started = perf_counter()
    left_result = ogdf_planar_subgraph(
        left,
        log_path=(
            None
            if log_directory is None
            else Path(log_directory) / "left_planarization.log"
        ),
        threads=threads,
        seed=seed,
        certified_pieces=left_certified_pieces,
        prior_certificates=left_prior_certificates,
    )
    right_result = ogdf_planar_subgraph(
        right,
        log_path=(
            None
            if log_directory is None
            else Path(log_directory) / "right_planarization.log"
        ),
        threads=threads,
        seed=(int(seed) + 1) % 2**31,
        certified_pieces=right_certified_pieces,
    )
    faces = exact_outer_face_pairs(
        left_result.planar_subgraph,
        right_result.planar_subgraph,
        count,
        seed=seed,
        sampling=sampling,
        prefer_non_leaf=prefer_non_leaf,
        left_degree_graph=left,
        right_degree_graph=right,
    )
    return HeuristicMergeNodeSelection(
        pairs=faces.pairs,
        strategy="heuristic_outer_face",
        seed=int(seed),
        selection_seconds=perf_counter() - started,
        iterations_per_graph=10,
        left_planarization=left_result,
        right_planarization=right_result,
        faces=faces,
    )


def _largest_face_candidate_count(graph: nx.Graph) -> int:
    if graph.number_of_nodes() == 0:
        return 0
    return max(
        len(largest_outer_face(graph.subgraph(component).copy()).candidates)
        for component in nx.connected_components(graph)
    )


def _certify_planar_subgraph(
    original: nx.Graph,
    planar: nx.Graph,
    omitted_edges: tuple[Edge, ...],
) -> None:
    if set(planar) != set(original):
        raise RuntimeError("planar-subgraph certificate is not spanning")
    if not nx.check_planarity(planar, counterexample=False)[0]:
        raise RuntimeError("planar-subgraph certificate is non-planar")
    original_edges = {frozenset((first, second)) for first, second in original.edges()}
    certified_edges = {
        frozenset((first, second)) for first, second in planar.edges()
    } | {frozenset(edge) for edge in omitted_edges}
    if certified_edges != original_edges:
        raise RuntimeError("planar-subgraph certificate does not partition the edges")


def _validate_simple_graph(graph: nx.Graph) -> None:
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a NetworkX graph")
    if graph.is_directed():
        raise TypeError("graph must be undirected")
    if graph.is_multigraph():
        raise TypeError("graph must be simple; project parallel edges first")
    if nx.number_of_selfloops(graph):
        raise ValueError("graph must not contain self-loops")


def _non_leaf_face_candidates(
    face: OuterFace,
    degree_graph: nx.Graph,
) -> tuple[Hashable, ...]:
    missing = tuple(node for node in face.candidates if node not in degree_graph)
    if missing:
        raise ValueError("degree graph does not contain every outer-face candidate")
    return tuple(node for node in face.candidates if degree_graph.degree[node] != 1)


def _sample_cyclic_preferred(
    order: tuple[Hashable, ...],
    count: int,
    rng: random.Random,
    *,
    sampling: OuterFaceSampling,
    preferred: tuple[Hashable, ...] | None,
) -> tuple[Hashable, ...]:
    """Sample in cyclic order, exhausting preferred candidates before fallback.

    Each subset is selected with the requested spacing rule.  The union is then
    restored to face order, which preserves the noncrossing matching certificate.
    """

    if preferred is None or count == 0:
        return _sample_cyclic(order, count, rng, sampling=sampling)
    preferred_set = set(preferred)
    if not preferred_set <= set(order):
        raise ValueError("preferred candidates must belong to the cyclic order")
    preferred_order = tuple(node for node in order if node in preferred_set)
    preferred_count = min(count, len(preferred_order))
    primary = _sample_cyclic(
        preferred_order,
        preferred_count,
        rng,
        sampling=sampling,
    )
    fallback_count = count - preferred_count
    fallback_order = tuple(node for node in order if node not in preferred_set)
    fallback = _sample_cyclic(
        fallback_order,
        fallback_count,
        rng,
        sampling=sampling,
    )
    selected = (*primary, *fallback)
    if not selected:
        return ()
    selected_set = set(selected)
    anchor = order.index(selected[0])
    rotated = (*order[anchor:], *order[:anchor])
    result = tuple(node for node in rotated if node in selected_set)
    if len(result) != count:
        raise RuntimeError("preferred cyclic sampling returned the wrong count")
    return result


def _sample_cyclic(
    order: tuple[Hashable, ...],
    count: int,
    rng: random.Random,
    *,
    sampling: OuterFaceSampling,
) -> tuple[Hashable, ...]:
    if count == 0:
        return ()
    offset = rng.randrange(len(order))
    if sampling == "consecutive":
        return tuple(order[(offset + index) % len(order)] for index in range(count))
    if sampling == "adaptive_skip_one":
        skip_count = min(count, len(order) - count)
        phase = count // 2
        gap_skips = tuple(
            (((index + 1) * skip_count + phase) // count)
            - ((index * skip_count + phase) // count)
            for index in range(count)
        )
        positions = [offset]
        for skip in gap_skips[:-1]:
            positions.append((positions[-1] + 1 + skip) % len(order))
        return tuple(order[position] for position in positions)
    return tuple(
        order[(offset + (index * len(order)) // count) % len(order)]
        for index in range(count)
    )
