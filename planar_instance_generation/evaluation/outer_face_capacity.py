"""Capacity measurements for logarithmic exact outer-face joins."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Sequence

import networkx as nx

from planar_instance_generation.core.merge import merge_with_pairs
from planar_instance_generation.core.planarization import exact_outer_face_pairs


@dataclass(frozen=True, slots=True)
class OuterFaceJoinCapacity:
    """One measurement on the planar graphs actually joined at a stage."""

    stage_index: int
    left_node_count: int
    right_node_count: int
    n_min: int
    logarithmic_denominator: int
    proposed_pair_count: int
    left_eligible_distinct_boundary_nodes: int
    right_eligible_distinct_boundary_nodes: int
    available_distinct_boundary_nodes: int
    rho: float | None
    logarithmic_request_feasible: bool | None
    proposed_request_feasible: bool
    realized_pair_count: int
    request_was_capped: bool

    def statistics(self) -> dict[str, int | float | bool | None]:
        """Return a flat JSON/CSV-friendly row."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class RetainedPlanarJoinResult:
    """The counterfactual retained planar composite and all join capacities."""

    final_graph: nx.Graph
    joins: tuple[OuterFaceJoinCapacity, ...]

    @property
    def every_logarithmic_request_feasible(self) -> bool | None:
        outcomes = tuple(
            join.logarithmic_request_feasible for join in self.joins
        )
        if any(outcome is None for outcome in outcomes):
            return None
        return all(outcomes)

    @property
    def every_proposed_request_feasible(self) -> bool:
        return all(join.proposed_request_feasible for join in self.joins)


def logarithmic_pair_count(n_min: int) -> int:
    """Return ``min(n_min, max(3, ceil(log2(n_min))))``."""

    if not isinstance(n_min, int) or isinstance(n_min, bool) or n_min < 1:
        raise ValueError("n_min must be a positive integer")
    logarithmic = math.ceil(math.log2(n_min)) if n_min > 1 else 0
    return min(n_min, max(3, logarithmic))


def evaluate_retained_planar_joins(
    planar_pieces: Sequence[nx.Graph],
    *,
    seed: int = 0,
) -> RetainedPlanarJoinResult:
    """Measure and execute the proposed rule on retained planar subgraphs.

    Every input is an ``H_r`` incumbent returned by the time-limited maximum
    planar-subgraph solver.  ``J_{r-1}`` is retained from the preceding join,
    rather than recomputed from an original source or final accepted graph.
    The exact selector's own distinct face candidates are used, so the counts
    are measured after the selector's boundary de-duplication.  The current
    power selector has no additional bus-eligibility filter.

    A request that exceeds capacity is recorded and capped only so later joins
    in the same rejected attempt can still be observed.  Such an attempt is
    never counted as satisfying every join.
    """

    pieces = tuple(planar_pieces)
    if len(pieces) < 2:
        raise ValueError("at least two planar pieces are required")
    for piece in pieces:
        if not isinstance(piece, nx.Graph) or piece.is_directed() or piece.is_multigraph():
            raise TypeError("planar_pieces must contain simple undirected graphs")
        if piece.number_of_nodes() == 0:
            raise ValueError("planar pieces must not be empty")
        if not nx.check_planarity(piece, counterexample=False)[0]:
            raise ValueError("every input piece must be planar")

    stage_rng = random.Random(seed)
    stage_seeds = tuple(stage_rng.randrange(2**63) for _ in pieces[1:])
    current = pieces[0].copy()
    rows: list[OuterFaceJoinCapacity] = []
    for stage_index, (right, stage_seed) in enumerate(
        zip(pieces[1:], stage_seeds, strict=True), start=1
    ):
        faces = exact_outer_face_pairs(current, right, 0, seed=stage_seed)
        left_available = len(faces.left_face.candidates)
        right_available = len(faces.right_face.candidates)
        available = min(left_available, right_available)
        n_min = min(current.number_of_nodes(), right.number_of_nodes())
        denominator = math.ceil(math.log2(n_min)) if n_min > 1 else 0
        # The literal ratio is undefined for n_min=1.  Its logarithmic request
        # is zero, however, so it is trivially feasible and must remain in the
        # attempted-join denominator.  The proposed floor/cap below requests
        # the only node and evaluates that stricter rule separately.
        rho = available / denominator if denominator else None
        log_feasible = available >= denominator if denominator else True
        requested = logarithmic_pair_count(n_min)
        realized = min(requested, available)
        if realized < 1:
            raise RuntimeError("nonempty planar join inputs had no boundary candidate")
        selected = exact_outer_face_pairs(
            current,
            right,
            realized,
            seed=stage_seed,
        )
        merged = merge_with_pairs(
            current,
            right,
            selected.pairs,
            selection_seconds=selected.selection_seconds,
            seed=stage_seed,
            left_namespace=("retained", stage_index - 1),
            right_namespace=("piece", stage_index + 1),
            stage_index=stage_index,
        )
        current = merged.final_graph
        if not nx.check_planarity(current, counterexample=False)[0]:
            raise RuntimeError("outer-face join did not retain planarity")
        rows.append(
            OuterFaceJoinCapacity(
                stage_index=stage_index,
                left_node_count=merged.source_graphs[0].number_of_nodes(),
                right_node_count=right.number_of_nodes(),
                n_min=n_min,
                logarithmic_denominator=denominator,
                proposed_pair_count=requested,
                left_eligible_distinct_boundary_nodes=left_available,
                right_eligible_distinct_boundary_nodes=right_available,
                available_distinct_boundary_nodes=available,
                rho=rho,
                logarithmic_request_feasible=log_feasible,
                proposed_request_feasible=available >= requested,
                realized_pair_count=realized,
                request_was_capped=realized < requested,
            )
        )
    return RetainedPlanarJoinResult(current, tuple(rows))
