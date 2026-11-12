"""Pure orchestration for reproducible PGLib extraction and composition."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from numbers import Integral, Real
from pathlib import Path
import random
from time import perf_counter
from typing import Literal, Sequence, TypeAlias

from planar_instance_generation.core.selection import (
    select_composition_pairs as _select_pairs,
)


from planar_instance_generation.core.merge import MergeResult, merge_with_pairs
from planar_instance_generation.core.planarization import (
    CertifiedPlanarPiece,
    ExactMergeNodeSelection,
    HeuristicMergeNodeSelection,
    OuterFaceSampling,
    relabel_planar_piece,
)
from planar_instance_generation.core.power_extraction import (
    CompensationMode,
    PowerSubcase,
    extract_random_power_subcase,
)
from planar_instance_generation.core.power_merge import (
    CouplingBranchConfig,
    PowerMergeResult,
    compose_power_cases,
)
from planar_instance_generation.core.selection import (
    CouplingCountRule,
    coupling_pair_rule,
    MergeNodeSelection,
    coupling_pair_count,
)
from planar_instance_generation.evaluation.source_pool import FeasibleSource
from planar_instance_generation.gates.base import DcOpfGate, DcOpfResult
from planar_instance_generation.io.pglib import PglibCase

PowerSelectionStrategy: TypeAlias = Literal[
    "random",
    "high_degree",
    "kamada_kawai",
    "heuristic_outer_face",
    "exact_outer_face",
]
SelectionResult: TypeAlias = (
    MergeNodeSelection | HeuristicMergeNodeSelection | ExactMergeNodeSelection
)


class PowerMergeSelectionError(ValueError):
    """A candidate-specific merge-node selection failure."""


@dataclass(frozen=True, slots=True)
class ExtractedPowerPiece:
    """One compensated source extraction, reusable across all strategies."""

    source_position: int
    source_name: str
    source_node_count: int
    target_nodes: int
    realized_fraction: float
    seed: int
    extraction_seconds: float
    subcase: PowerSubcase
    retained_original_positive_demand_mw: float | None = None

    def statistics(self) -> dict[str, object]:
        case = self.subcase.case
        absolute_compensation = sum(
            abs(value) for value in self.subcase.supply_compensation_mw.values()
        )
        original_demand = self.retained_original_positive_demand_mw
        return {
            "source_position": self.source_position,
            "source_name": self.source_name,
            "source_node_count": self.source_node_count,
            "target_nodes": self.target_nodes,
            "realized_fraction": self.realized_fraction,
            "seed": self.seed,
            "extraction_seconds": self.extraction_seconds,
            "bus_count": case.bus.shape[0],
            "generator_count": case.gen.shape[0],
            "branch_count": case.branch.shape[0],
            "cut_branch_count": len(self.subcase.cut_branch_indices),
            "compensated_bus_count": len(self.subcase.supply_compensation_mw),
            "compensated_bus_fraction": len(self.subcase.supply_compensation_mw)
            / len(case.bus),
            "absolute_compensation_mw": absolute_compensation,
            "retained_original_positive_demand_mw": original_demand,
            "compensation_to_original_demand_ratio": (
                absolute_compensation / original_demand
                if original_demand is not None and original_demand > 0
                else None
            ),
            "compensation_generator_capacity_mw": sum(
                float(case.gen[i, 8])
                for i in self.subcase.compensation_generator_indices
            ),
            "compensation_generator_count": len(
                self.subcase.compensation_generator_indices
            ),
        }


@dataclass(frozen=True, slots=True)
class PowerMergeStage:
    """One left-associative join and its complete timing/provenance record."""

    stage_index: int
    selection_seed: int
    requested_pair_count: int
    pair_count: int
    selection: SelectionResult
    composition_seconds: float
    merge: PowerMergeResult

    def statistics(self) -> dict[str, object]:
        row: dict[str, object] = {
            "stage_index": self.stage_index,
            "strategy": self.selection.strategy,
            "selection_seed": self.selection_seed,
            "requested_pair_count": self.requested_pair_count,
            "pair_count": self.pair_count,
            "selection_seconds": self.selection.selection_seconds,
            "composition_seconds": self.composition_seconds,
            "left_bus_count": len(self.merge.left.bus_rows),
            "right_bus_count": len(self.merge.right.bus_rows),
            "merged_bus_count": self.merge.case.bus.shape[0],
            "merged_branch_count": self.merge.case.branch.shape[0],
        }
        if isinstance(self.selection, MergeNodeSelection):
            row.update(
                {
                    "left_candidate_count": self.selection.left_candidate_count,
                    "right_candidate_count": self.selection.right_candidate_count,
                }
            )
        if isinstance(self.selection, ExactMergeNodeSelection):
            row.update(
                {
                    "outer_face_sampling": self.selection.faces.sampling,
                    "outer_face_prefer_non_leaf": self.selection.faces.prefer_non_leaf,
                    "left_face_non_leaf_candidates": self.selection.faces.left_non_leaf_candidate_count,
                    "right_face_non_leaf_candidates": self.selection.faces.right_non_leaf_candidate_count,
                    "left_selected_leaf_count": self.selection.faces.left_selected_leaf_count,
                    "right_selected_leaf_count": self.selection.faces.right_selected_leaf_count,
                    "left_face_candidates": len(
                        self.selection.faces.left_face.candidates
                    ),
                    "right_face_candidates": len(
                        self.selection.faces.right_face.candidates
                    ),
                    "left_skewness_lower_bound": self.selection.left_planarization.omitted_lower_bound,
                    "left_skewness_upper_bound": self.selection.left_planarization.omitted_upper_bound,
                    "right_skewness_lower_bound": self.selection.right_planarization.omitted_lower_bound,
                    "right_skewness_upper_bound": self.selection.right_planarization.omitted_upper_bound,
                }
            )
            for side, result in (
                ("left", self.selection.left_planarization),
                ("right", self.selection.right_planarization),
            ):
                row.update(
                    {
                        f"{side}_planarization_{key}": value
                        for key, value in result.statistics().items()
                    }
                )
        elif isinstance(self.selection, HeuristicMergeNodeSelection):
            row.update(self.selection.statistics())
            row.update(
                {
                    "outer_face_sampling": self.selection.faces.sampling,
                    "outer_face_prefer_non_leaf": self.selection.faces.prefer_non_leaf,
                    "left_face_non_leaf_candidates": self.selection.faces.left_non_leaf_candidate_count,
                    "right_face_non_leaf_candidates": self.selection.faces.right_non_leaf_candidate_count,
                    "left_selected_leaf_count": self.selection.faces.left_selected_leaf_count,
                    "right_selected_leaf_count": self.selection.faces.right_selected_leaf_count,
                    "left_face_candidates": len(
                        self.selection.faces.left_face.candidates
                    ),
                    "right_face_candidates": len(
                        self.selection.faces.right_face.candidates
                    ),
                }
            )
        return row


@dataclass(frozen=True, slots=True)
class CouplingFlowStatistics:
    """Flow statistics for the synthetic coupling branches in a gate solution."""

    coupling_branch_count: int
    absolute_flow_mean_mw: float | None
    absolute_flow_max_mw: float | None
    finite_rating_count: int
    utilization_mean: float | None
    utilization_max: float | None


@dataclass(frozen=True, slots=True)
class PowerGenerationCandidate:
    """A fully composed candidate and the result of its base-case DC gate."""

    strategy: PowerSelectionStrategy
    seed: int
    pieces: tuple[ExtractedPowerPiece, ...]
    stages: tuple[PowerMergeStage, ...]
    gate_result: DcOpfResult
    merge_seconds: float
    total_seconds: float
    coupling_flows: CouplingFlowStatistics
    coupling_count_rule: CouplingCountRule = "log2"

    @property
    def final_merge(self) -> PowerMergeResult:
        return self.stages[-1].merge

    @property
    def case(self) -> PglibCase:
        return self.final_merge.case

    @property
    def feasible(self) -> bool:
        return self.gate_result.feasible

    def statistics(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "coupling_rule": coupling_pair_rule(self.coupling_count_rule),
            "coupling_count_rule": self.coupling_count_rule,
            "source_count": len(self.pieces),
            "target_size": sum(piece.target_nodes for piece in self.pieces),
            "target_nodes": tuple(piece.target_nodes for piece in self.pieces),
            "realized_fractions": tuple(
                piece.realized_fraction for piece in self.pieces
            ),
            "coupling_pair_counts": tuple(
                stage.requested_pair_count for stage in self.stages
            ),
            "bus_count": self.case.bus.shape[0],
            "branch_count": self.case.branch.shape[0],
            "merge_node_selection_seconds": sum(
                stage.selection.selection_seconds for stage in self.stages
            ),
            "composition_seconds": sum(
                stage.composition_seconds for stage in self.stages
            ),
            "merge_seconds": self.merge_seconds,
            "generation_and_gate_seconds": self.total_seconds,
            "gate_feasible": self.gate_result.feasible,
            "gate_status": self.gate_result.status,
            "objective": self.gate_result.objective,
            "gate_seconds": self.gate_result.runtime_seconds,
            "gate_optimizer_seconds": self.gate_result.optimizer_seconds,
            **asdict(self.coupling_flows),
        }


def extract_power_pieces(
    sources: Sequence[FeasibleSource],
    fractions: Real | Sequence[Real] | None = None,
    *,
    target_nodes: Sequence[int] | None = None,
    compensation: CompensationMode = "flexible",
    seed: int = 0,
) -> tuple[ExtractedPowerPiece, ...]:
    """Extract each sampled source once, before strategy comparison.

    Exact composite generation passes integer ``target_nodes``. ``fractions``
    remains supported for direct callers and legacy experiments.
    """

    if len(sources) < 2:
        raise ValueError("at least two feasible sources are required")
    if (fractions is None) == (target_nodes is None):
        raise ValueError("pass exactly one of fractions or target_nodes")
    resolved_fractions = (
        None if fractions is None else _expand_fractions(fractions, len(sources))
    )
    resolved_targets = (
        None
        if target_nodes is None
        else _expand_target_nodes(target_nodes, len(sources))
    )
    rng = random.Random(_validate_seed(seed))
    extraction_seeds = tuple(rng.randrange(2**63) for _ in sources)
    pieces = []
    for position, (source, extraction_seed) in enumerate(
        zip(sources, extraction_seeds, strict=True), start=1
    ):
        if not isinstance(source, FeasibleSource):
            raise TypeError("sources must contain FeasibleSource records")
        source.solution.validate_for(source.case)
        source_node_count = int(source.case.bus.shape[0])
        fraction = (
            None if resolved_fractions is None else resolved_fractions[position - 1]
        )
        target = (
            math.ceil(fraction * source_node_count)
            if resolved_targets is None
            else resolved_targets[position - 1]
        )
        started = perf_counter()
        subcase = extract_random_power_subcase(
            source.case,
            fraction,
            source.solution.branch_from_flows,
            source.solution.branch_to_flows,
            target_nodes=None if resolved_targets is None else target,
            compensation=compensation,
            seed=extraction_seed,
            name=f"piece_{position:03d}",
        )
        actual_target = int(subcase.case.bus.shape[0])
        if actual_target != target:
            raise RuntimeError("power extraction did not realize its exact target")
        selected_bus_ids = set(subcase.selected_buses)
        original_positive_demand = sum(
            max(0.0, float(row[2]))
            for row in source.case.bus
            if int(row[0]) in selected_bus_ids
        )
        pieces.append(
            ExtractedPowerPiece(
                source_position=position,
                source_name=source.case.name,
                source_node_count=source_node_count,
                target_nodes=actual_target,
                realized_fraction=actual_target / source_node_count,
                seed=extraction_seed,
                extraction_seconds=perf_counter() - started,
                subcase=subcase,
                retained_original_positive_demand_mw=original_positive_demand,
            )
        )
    return tuple(pieces)


def merge_power_pieces(
    pieces: Sequence[ExtractedPowerPiece],
    gate: DcOpfGate,
    *,
    strategy: PowerSelectionStrategy,
    seed: int = 0,
    coupling: CouplingBranchConfig | None = None,
    exact_time_limit_seconds: float = 600.0,
    exact_threads: int = 16,
    coupling_count_rule: CouplingCountRule = "log2",
    outer_face_sampling: OuterFaceSampling = "consecutive",
    outer_face_prefer_non_leaf: bool = False,
    log_directory: str | Path | None = None,
) -> PowerGenerationCandidate:
    """Join reusable pieces with one strategy, then run the DC feasibility gate."""

    coupling_pair_rule(coupling_count_rule)
    if len(pieces) < 2:
        raise ValueError("at least two extracted pieces are required")
    if strategy not in {
        "random",
        "high_degree",
        "kamada_kawai",
        "heuristic_outer_face",
        "exact_outer_face",
    }:
        raise ValueError(f"unknown power selection strategy {strategy!r}")
    if outer_face_sampling not in {
        "equidistant",
        "consecutive",
        "adaptive_skip_one",
    }:
        raise ValueError(
            "outer_face_sampling must be 'equidistant', 'consecutive', or "
            "'adaptive_skip_one'"
        )
    if not isinstance(outer_face_prefer_non_leaf, bool):
        raise TypeError("outer_face_prefer_non_leaf must be a boolean")
    root_seed = _validate_seed(seed)
    rng = random.Random(root_seed)
    stage_seeds = tuple(rng.randrange(2**31) for _ in range(len(pieces) - 1))
    logs = None if log_directory is None else Path(log_directory)

    started = perf_counter()
    stages: list[PowerMergeStage] = []
    left_certified_pieces: tuple[CertifiedPlanarPiece, ...] = ()
    left_prior_certificates: tuple[CertifiedPlanarPiece, ...] = ()
    current: PglibCase | PowerMergeResult = pieces[0].subcase.case
    for stage_index, (piece, stage_seed) in enumerate(
        zip(pieces[1:], stage_seeds, strict=True), start=1
    ):
        left_case = current.case if isinstance(current, PowerMergeResult) else current
        right_case = piece.subcase.case
        left_graph = left_case.topology(active_only=True)
        right_graph = right_case.topology(active_only=True)
        requested_pair_count = coupling_pair_count(
            left_graph, right_graph, rule=coupling_count_rule
        )
        stage_logs = None if logs is None else logs / f"stage_{stage_index:03d}"
        try:
            selection = _select_pairs(
                left_graph,
                right_graph,
                requested_pair_count,
                strategy=strategy,
                seed=stage_seed,
                exact_time_limit_seconds=exact_time_limit_seconds,
                exact_threads=exact_threads,
                outer_face_sampling=outer_face_sampling,
                outer_face_prefer_non_leaf=outer_face_prefer_non_leaf,
                log_directory=stage_logs,
                left_certified_pieces=left_certified_pieces,
                left_prior_certificates=left_prior_certificates,
            )
            if len(selection.pairs) != requested_pair_count:
                raise ValueError(
                    f"{strategy} returned {len(selection.pairs)} pairs for "
                    f"a strict request of {requested_pair_count}"
                )
        except ValueError as error:
            raise PowerMergeSelectionError(
                f"merge stage {stage_index}: {error}"
            ) from error
        composition_started = perf_counter()
        merged = compose_power_cases(
            current,
            right_case,
            selection.pairs,
            name=f"candidate_stage_{stage_index:03d}",
            config=coupling,
        )
        composition_seconds = perf_counter() - composition_started
        if (
            isinstance(
                selection, (ExactMergeNodeSelection, HeuristicMergeNodeSelection)
            )
            and stage_index < len(pieces) - 1
        ):
            reuse_started = perf_counter()
            left_prior_certificates = ()
            left_certified_pieces = (
                relabel_planar_piece(
                    left_graph, selection.left_planarization, merged.left.bus_ids
                ),
                relabel_planar_piece(
                    right_graph, selection.right_planarization, merged.right.bus_ids
                ),
            )
            # Count certificate transfer once as selection work, never count
            # the preceding planarization solve again at the following join.
            selection = replace(
                selection,
                selection_seconds=(
                    selection.selection_seconds + (perf_counter() - reuse_started)
                ),
            )
        stages.append(
            PowerMergeStage(
                stage_index=stage_index,
                selection_seed=stage_seed,
                requested_pair_count=requested_pair_count,
                pair_count=len(selection.pairs),
                selection=selection,
                composition_seconds=composition_seconds,
                merge=merged,
            )
        )
        current = merged

    merge_seconds = perf_counter() - started
    final_merge = stages[-1].merge
    gate_result = gate.solve(final_merge.case)
    if gate_result.feasible:
        assert gate_result.solution is not None
        gate_result.solution.validate_for(final_merge.case)
    coupling_flows = _coupling_flow_statistics(final_merge, gate_result)
    extraction_seconds = sum(piece.extraction_seconds for piece in pieces)
    return PowerGenerationCandidate(
        strategy=strategy,
        seed=root_seed,
        pieces=tuple(pieces),
        stages=tuple(stages),
        gate_result=gate_result,
        merge_seconds=merge_seconds,
        total_seconds=extraction_seconds + perf_counter() - started,
        coupling_flows=coupling_flows,
        coupling_count_rule=coupling_count_rule,
    )


def visual_merge_stages(candidate: PowerGenerationCandidate) -> tuple[MergeResult, ...]:
    """Rebuild the exact selected joins in the provenance-rich plotting model."""

    visual = candidate.pieces[0].subcase.case.topology(active_only=True)
    current_case = candidate.pieces[0].subcase.case
    bus_to_visual: dict[int, object] = {
        int(bus): int(bus) for bus in current_case.bus[:, 0]
    }
    results: list[MergeResult] = []

    for stage, piece in zip(candidate.stages, candidate.pieces[1:], strict=True):
        right_graph = piece.subcase.case.topology(active_only=True)
        visual_pairs = tuple(
            (bus_to_visual[int(left_bus)], int(right_bus))
            for left_bus, right_bus in stage.selection.pairs
        )
        left_namespace: object = (
            "G_1" if stage.stage_index == 1 else ("composite", stage.stage_index - 1)
        )
        right_namespace = f"G_{stage.stage_index + 1}"
        visual_result = merge_with_pairs(
            visual,
            right_graph,
            visual_pairs,
            selection_seconds=stage.selection.selection_seconds,
            seed=stage.selection_seed,
            left_namespace=left_namespace,
            right_namespace=right_namespace,
            stage_index=stage.stage_index,
        )
        next_bus_to_visual = {
            int(output_bus): (left_namespace, bus_to_visual[int(input_bus)])
            for input_bus, output_bus in stage.merge.left.bus_ids.items()
        }
        next_bus_to_visual.update(
            {
                int(output_bus): (right_namespace, int(input_bus))
                for input_bus, output_bus in stage.merge.right.bus_ids.items()
            }
        )
        visual = visual_result.final_graph
        current_case = stage.merge.case
        if set(next_bus_to_visual) != {int(bus) for bus in current_case.bus[:, 0]}:
            raise RuntimeError("visual and electrical bus provenance diverged")
        bus_to_visual = next_bus_to_visual
        results.append(visual_result)
    return tuple(results)


def _coupling_flow_statistics(
    merge: PowerMergeResult, result: DcOpfResult
) -> CouplingFlowStatistics:
    branch_count = len(merge.coupling_branch_rows)
    if not result.feasible:
        return CouplingFlowStatistics(branch_count, None, None, 0, None, None)
    assert result.solution is not None
    absolute_flows = [
        abs(float(result.solution.branch_from_flows[row]))
        for row in merge.coupling_branch_rows
    ]
    rated = [
        (flow, float(merge.case.branch[row, 5]))
        for flow, row in zip(absolute_flows, merge.coupling_branch_rows, strict=True)
        if merge.case.branch[row, 5] > 0
    ]
    utilizations = [flow / rate for flow, rate in rated]
    return CouplingFlowStatistics(
        coupling_branch_count=branch_count,
        absolute_flow_mean_mw=sum(absolute_flows) / branch_count,
        absolute_flow_max_mw=max(absolute_flows),
        finite_rating_count=len(rated),
        utilization_mean=(
            sum(utilizations) / len(utilizations) if utilizations else None
        ),
        utilization_max=max(utilizations) if utilizations else None,
    )


def _expand_fractions(values: Real | Sequence[Real], count: int) -> tuple[float, ...]:
    raw = (
        (values,) * count
        if isinstance(values, Real) and not isinstance(values, bool)
        else tuple(values)
    )
    if len(raw) != count:
        raise ValueError("fractions must contain one value per source")
    resolved = tuple(float(value) for value in raw)
    if any(not math.isfinite(value) or not 0 < value <= 1 for value in resolved):
        raise ValueError("every extraction fraction must be finite and in (0, 1]")
    return resolved


def _expand_target_nodes(values: Sequence[int], count: int) -> tuple[int, ...]:
    raw = tuple(values)
    if len(raw) != count:
        raise ValueError("target_nodes must contain one value per source")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or value < 1
        for value in raw
    ):
        raise ValueError("every target node count must be a positive integer")
    return tuple(int(value) for value in raw)


def _validate_seed(seed: int) -> int:
    if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int(seed)
