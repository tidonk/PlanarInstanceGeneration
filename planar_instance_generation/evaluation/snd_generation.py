"""Pure orchestration for reproducible SND extraction and composition."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
import math
from numbers import Integral, Real
from pathlib import Path
import random
from time import perf_counter
from typing import Literal, Protocol, Sequence, TypeAlias

from planar_instance_generation.core.selection import (
    select_composition_pairs as _select_pairs,
)


from planar_instance_generation.core.merge import MergeResult, merge_with_pairs
from planar_instance_generation.core.planarization import (
    CertifiedPlanarPiece,
    ExactMergeNodeSelection,
    HeuristicMergeNodeSelection,
    OuterFaceCapacityError,
    OuterFaceSampling,
    relabel_planar_piece,
)
from planar_instance_generation.core.selection import (
    CouplingCountRule,
    coupling_pair_rule,
    MergeNodeSelection,
    coupling_pair_count,
)
from planar_instance_generation.core.snd import (
    SndInstance,
    SndSolution,
    validate_snd_solution,
)
from planar_instance_generation.core.snd_extraction import (
    SndSubinstance,
    extract_random_snd_subinstance,
)
from planar_instance_generation.core.snd_merge import (
    SndMergeConfig,
    SndMergeResult,
    compose_snd_instances,
)
from planar_instance_generation.core.snd_serialization import (
    SndlibSerialization,
    serialize_snd_instance,
)
from planar_instance_generation.gates.snd_base import McmcfGate, McmcfResult
from planar_instance_generation.io.sndlib import SndlibNetwork

SndSelectionStrategy: TypeAlias = Literal[
    "random",
    "high_degree",
    "kamada_kawai",
    "heuristic_outer_face",
    "exact_outer_face",
]
SelectionResult: TypeAlias = (
    MergeNodeSelection | HeuristicMergeNodeSelection | ExactMergeNodeSelection
)


class SndSource(Protocol):
    """Minimal source-pool entry consumed by generation."""

    source_name: str
    network: SndlibNetwork
    solution: SndSolution


class SndMergeSelectionError(ValueError):
    """A candidate-specific merge-node selection failure."""


@dataclass(frozen=True, slots=True)
class ExtractedSndPiece:
    """One compensated source extraction, reusable across all strategies."""

    source_position: int
    source_name: str
    routing_unit: str | None
    source_node_count: int
    target_nodes: int
    realized_fraction: float
    seed: int
    extraction_seconds: float
    subinstance: SndSubinstance

    def statistics(self) -> dict[str, object]:
        instance = self.subinstance.instance
        compensated_nodes = {
            node
            for values in self.subinstance.compensation_by_demand.values()
            for node, delta in values.items()
            if delta != 0.0
        }
        compensated_demands = sum(
            any(delta != 0.0 for delta in values.values())
            for values in self.subinstance.compensation_by_demand.values()
        )
        return {
            "source_position": self.source_position,
            "source_name": self.source_name,
            "routing_unit": self.routing_unit,
            "source_coordinates_type": instance.coordinates_type,
            "source_node_count": self.source_node_count,
            "target_nodes": self.target_nodes,
            "realized_fraction": self.realized_fraction,
            "seed": self.seed,
            "extraction_seconds": self.extraction_seconds,
            "node_count": len(instance.nodes),
            "link_count": len(instance.links),
            "commodity_count": len(instance.commodities),
            "cut_link_count": len(self.subinstance.cut_link_indices),
            "compensated_node_count": len(compensated_nodes),
            "compensated_node_fraction": len(compensated_nodes) / len(instance.nodes),
            "absolute_compensation": sum(
                abs(delta)
                for values in self.subinstance.compensation_by_demand.values()
                for delta in values.values()
            ),
            "compensated_commodity_count": compensated_demands,
            "reference_flow_nonzero_count": len(instance.reference_solution.flows),
        }


@dataclass(frozen=True, slots=True)
class SndMergeStage:
    """One left-associative SND join with timings and certificates."""

    stage_index: int
    selection_seed: int
    requested_pair_count: int
    pair_count: int
    selection: SelectionResult
    composition_seconds: float
    merge: SndMergeResult

    def statistics(self) -> dict[str, object]:
        row: dict[str, object] = {
            "stage_index": self.stage_index,
            "strategy": self.selection.strategy,
            "selection_seed": self.selection_seed,
            "requested_pair_count": self.requested_pair_count,
            "pair_count": self.pair_count,
            "selection_seconds": self.selection.selection_seconds,
            "composition_seconds": self.composition_seconds,
            "left_node_count": len(self.merge.left.node_indices),
            "right_node_count": len(self.merge.right.node_indices),
            "merged_node_count": len(self.merge.instance.nodes),
            "merged_link_count": len(self.merge.instance.links),
            "new_coupling_link_count": len(self.merge.new_coupling_link_indices),
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
                    "left_skewness_lower_bound": (
                        self.selection.left_planarization.omitted_lower_bound
                    ),
                    "left_skewness_upper_bound": (
                        self.selection.left_planarization.omitted_upper_bound
                    ),
                    "right_skewness_lower_bound": (
                        self.selection.right_planarization.omitted_lower_bound
                    ),
                    "right_skewness_upper_bound": (
                        self.selection.right_planarization.omitted_upper_bound
                    ),
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
class SndCouplingFlowStatistics:
    """Gate-flow statistics on the synthetic coupling links."""

    coupling_link_count: int
    positive_capacity_count: int
    absolute_flow_mean: float | None
    absolute_flow_max: float | None
    utilization_mean: float | None
    utilization_max: float | None


@dataclass(frozen=True, slots=True)
class SndGenerationCandidate:
    """A composed candidate and its fixed-capacity MCMCF gate result."""

    strategy: SndSelectionStrategy
    seed: int
    routing_unit: str | None
    source_stratum_size: int
    pieces: tuple[ExtractedSndPiece, ...]
    stages: tuple[SndMergeStage, ...]
    merge_config: SndMergeConfig
    coordinate_type_overridden: bool
    serialization: SndlibSerialization
    serialization_seconds: float
    gate_result: McmcfResult
    merge_seconds: float
    total_seconds: float
    coupling_flows: SndCouplingFlowStatistics
    coupling_count_rule: CouplingCountRule = "log2"

    @property
    def final_merge(self) -> SndMergeResult:
        return self.stages[-1].merge

    @property
    def instance(self) -> SndInstance:
        return self.final_merge.instance

    @property
    def feasible(self) -> bool | None:
        return self.gate_result.feasible

    def statistics(self) -> dict[str, object]:
        row: dict[str, object] = {
            "strategy": self.strategy,
            "seed": self.seed,
            "coupling_rule": coupling_pair_rule(self.coupling_count_rule),
            "coupling_count_rule": self.coupling_count_rule,
            "routing_unit": self.routing_unit,
            "source_stratum_size": self.source_stratum_size,
            "source_count": len(self.pieces),
            "target_size": sum(piece.target_nodes for piece in self.pieces),
            "target_nodes": tuple(piece.target_nodes for piece in self.pieces),
            "realized_fractions": tuple(
                piece.realized_fraction for piece in self.pieces
            ),
            "coupling_pair_counts": tuple(
                stage.requested_pair_count for stage in self.stages
            ),
            "node_count": len(self.instance.nodes),
            "link_count": len(self.instance.links),
            "commodity_count": len(self.instance.commodities),
            "serialized_demand_count": len(self.serialization.network.demands),
            "coordinates_type": self.instance.coordinates_type,
            "coordinate_type_overridden": self.coordinate_type_overridden,
            "merge_node_selection_seconds": sum(
                stage.selection.selection_seconds for stage in self.stages
            ),
            "composition_seconds": sum(
                stage.composition_seconds for stage in self.stages
            ),
            "merge_seconds": self.merge_seconds,
            "serialization_seconds": self.serialization_seconds,
            "generation_and_gate_seconds": self.total_seconds,
            "gate_feasible": self.gate_result.feasible,
            "gate_status": self.gate_result.status,
            "gate_reason": self.gate_result.reason,
            "gate_resource_skipped": self.gate_result.resource_skipped,
            "objective": self.gate_result.objective,
            "gate_seconds": self.gate_result.runtime_seconds,
            "gate_optimizer_seconds": self.gate_result.optimizer_seconds,
            **asdict(self.coupling_flows),
        }
        estimate = self.gate_result.estimate
        if estimate is not None and is_dataclass(estimate):
            row.update(
                {
                    f"gate_estimate_{key}": value
                    for key, value in asdict(estimate).items()
                }
            )
        return row


def extract_snd_pieces(
    sources: Sequence[SndSource],
    fractions: Real | Sequence[Real] | None = None,
    *,
    target_nodes: Sequence[int] | None = None,
    seed: int = 0,
) -> tuple[ExtractedSndPiece, ...]:
    """Extract every sampled source exactly once before strategy comparison.

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
    pieces: list[ExtractedSndPiece] = []
    for position, (source, extraction_seed) in enumerate(
        zip(sources, extraction_seeds, strict=True), start=1
    ):
        _validate_source(source)
        source_node_count = len(source.network.nodes)
        fraction = (
            None if resolved_fractions is None else resolved_fractions[position - 1]
        )
        target = (
            math.ceil(fraction * source_node_count)
            if resolved_targets is None
            else resolved_targets[position - 1]
        )
        started = perf_counter()
        subinstance = extract_random_snd_subinstance(
            source.network,
            fraction,
            source.solution,
            source_name=source.source_name,
            target_nodes=None if resolved_targets is None else target,
            seed=extraction_seed,
            name=f"piece_{position:03d}",
        )
        actual_target = len(subinstance.instance.nodes)
        if actual_target != target:
            raise RuntimeError("SND extraction did not realize its exact target")
        pieces.append(
            ExtractedSndPiece(
                source_position=position,
                source_name=source.source_name,
                routing_unit=source.network.routing_unit,
                source_node_count=source_node_count,
                target_nodes=actual_target,
                realized_fraction=actual_target / source_node_count,
                seed=extraction_seed,
                extraction_seconds=perf_counter() - started,
                subinstance=subinstance,
            )
        )
    declared_units = {
        piece.routing_unit for piece in pieces if piece.routing_unit is not None
    }
    if len(declared_units) > 1:
        raise ValueError("SND pieces declare incompatible routing units")
    return tuple(pieces)


def merge_snd_pieces(
    pieces: Sequence[ExtractedSndPiece],
    gate: McmcfGate,
    *,
    strategy: SndSelectionStrategy,
    seed: int = 0,
    merge_config: SndMergeConfig | None = None,
    mixed_coordinates_type: str | None = "pixel",
    source_stratum_size: int = 1,
    exact_time_limit_seconds: float = 600.0,
    exact_threads: int = 16,
    coupling_count_rule: CouplingCountRule = "log2",
    outer_face_sampling: OuterFaceSampling = "consecutive",
    outer_face_prefer_non_leaf: bool = False,
    log_directory: str | Path | None = None,
) -> SndGenerationCandidate:
    """Join reusable pieces with one strategy and run the fixed-capacity gate."""

    coupling_pair_rule(coupling_count_rule)
    if len(pieces) < 2:
        raise ValueError("at least two extracted pieces are required")
    if strategy not in _STRATEGIES:
        raise ValueError(f"unknown SND selection strategy {strategy!r}")
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
    _validate_selection_settings(
        exact_time_limit_seconds=exact_time_limit_seconds,
        exact_threads=exact_threads,
    )
    declared_units = {
        piece.routing_unit for piece in pieces if piece.routing_unit is not None
    }
    if len(declared_units) > 1:
        raise ValueError("cannot compose SND pieces with incompatible routing units")
    _positive_integer("source_stratum_size", source_stratum_size)
    root_seed = _validate_seed(seed)
    rng = random.Random(root_seed)
    stage_seeds = tuple(rng.randrange(2**31) for _ in range(len(pieces) - 1))
    logs = None if log_directory is None else Path(log_directory)
    settings, coordinate_override = _resolve_merge_config(
        pieces,
        merge_config,
        mixed_coordinates_type=mixed_coordinates_type,
    )

    total_started = perf_counter()
    merge_started = perf_counter()
    stages: list[SndMergeStage] = []
    left_certified_pieces: tuple[CertifiedPlanarPiece, ...] = ()
    left_prior_certificates: tuple[CertifiedPlanarPiece, ...] = ()
    current: SndInstance | SndMergeResult = pieces[0].subinstance.instance
    for stage_index, (piece, stage_seed) in enumerate(
        zip(pieces[1:], stage_seeds, strict=True), start=1
    ):
        left_instance = (
            current.instance if isinstance(current, SndMergeResult) else current
        )
        right_instance = piece.subinstance.instance
        left_graph = left_instance.topology()
        right_graph = right_instance.topology()
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
        except (OuterFaceCapacityError, ValueError) as error:
            raise SndMergeSelectionError(
                f"merge stage {stage_index}: {error}"
            ) from error

        composition_started = perf_counter()
        merged = compose_snd_instances(
            current,
            right_instance,
            selection.pairs,
            name=f"candidate_stage_{stage_index:03d}",
            config=settings,
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
                    left_graph, selection.left_planarization, merged.left.node_ids
                ),
                relabel_planar_piece(
                    right_graph, selection.right_planarization, merged.right.node_ids
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
            SndMergeStage(
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
    merge_seconds = perf_counter() - merge_started

    final_merge = stages[-1].merge
    serialization_started = perf_counter()
    serialization = serialize_snd_instance(final_merge.instance)
    serialization_seconds = perf_counter() - serialization_started
    gate_result = gate.solve(
        serialization.network,
        serialization.reference_solution.installed_capacities,
        source_name=f"candidate_{strategy}_{root_seed}",
    )
    if gate_result.feasible is True:
        if gate_result.solution is None:
            raise ValueError("a feasible MCMCF result must contain a solution")
        validate_snd_solution(serialization.network, gate_result.solution)

    coupling_flows = _coupling_flow_statistics(
        final_merge,
        serialization,
        gate_result,
    )
    extraction_seconds = sum(piece.extraction_seconds for piece in pieces)
    return SndGenerationCandidate(
        strategy=strategy,
        seed=root_seed,
        routing_unit=next(iter(declared_units), None),
        source_stratum_size=source_stratum_size,
        pieces=tuple(pieces),
        stages=tuple(stages),
        merge_config=settings,
        coordinate_type_overridden=coordinate_override,
        serialization=serialization,
        serialization_seconds=serialization_seconds,
        gate_result=gate_result,
        merge_seconds=merge_seconds,
        total_seconds=extraction_seconds + perf_counter() - total_started,
        coupling_flows=coupling_flows,
        coupling_count_rule=coupling_count_rule,
    )


def visual_snd_merge_stages(
    candidate: SndGenerationCandidate,
) -> tuple[MergeResult, ...]:
    """Rebuild selected SND joins in the provenance-rich plotting model."""

    visual = candidate.pieces[0].subinstance.instance.topology()
    current_instance = candidate.pieces[0].subinstance.instance
    node_to_visual: dict[str, object] = {
        node.id: node.id for node in current_instance.nodes
    }
    results: list[MergeResult] = []

    for stage, piece in zip(candidate.stages, candidate.pieces[1:], strict=True):
        right_graph = piece.subinstance.instance.topology()
        visual_pairs = tuple(
            (node_to_visual[str(left_node)], str(right_node))
            for left_node, right_node in stage.selection.pairs
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
        next_node_to_visual = {
            output_node: (left_namespace, node_to_visual[input_node])
            for input_node, output_node in stage.merge.left.node_ids.items()
        }
        next_node_to_visual.update(
            {
                output_node: (right_namespace, input_node)
                for input_node, output_node in stage.merge.right.node_ids.items()
            }
        )
        visual = visual_result.final_graph
        current_instance = stage.merge.instance
        if set(next_node_to_visual) != {node.id for node in current_instance.nodes}:
            raise RuntimeError("visual and SND node provenance diverged")
        node_to_visual = next_node_to_visual
        results.append(visual_result)
    return tuple(results)


_STRATEGIES = {
    "random",
    "high_degree",
    "kamada_kawai",
    "heuristic_outer_face",
    "exact_outer_face",
}


def _resolve_merge_config(
    pieces: Sequence[ExtractedSndPiece],
    config: SndMergeConfig | None,
    *,
    mixed_coordinates_type: str | None,
) -> tuple[SndMergeConfig, bool]:
    settings = SndMergeConfig() if config is None else config
    if not isinstance(settings, SndMergeConfig):
        raise TypeError("merge_config must be an SndMergeConfig or None")
    source_types = {piece.subinstance.instance.coordinates_type for piece in pieces}
    if len(source_types) > 1 and settings.coordinates_type is None:
        if (
            not isinstance(mixed_coordinates_type, str)
            or not mixed_coordinates_type.strip()
        ):
            raise ValueError(
                "mixed source coordinate types require an explicit output type"
            )
        settings = replace(settings, coordinates_type=mixed_coordinates_type)
    output_type = settings.coordinates_type or next(iter(source_types))
    return settings, any(value != output_type for value in source_types)


def _coupling_flow_statistics(
    merge: SndMergeResult,
    serialization: SndlibSerialization,
    result: McmcfResult,
) -> SndCouplingFlowStatistics:
    link_ids = tuple(
        merge.instance.links[index].id for index in merge.coupling_link_indices
    )
    if result.feasible is not True:
        return SndCouplingFlowStatistics(len(link_ids), 0, None, None, None, None)
    assert result.solution is not None
    coupling_ids = set(link_ids)
    flow_by_link = {link_id: 0.0 for link_id in link_ids}
    for (_, link_id), flow in result.solution.flows.items():
        if link_id in coupling_ids:
            flow_by_link[link_id] += abs(flow)
    absolute_flows = [flow_by_link[link_id] for link_id in link_ids]
    capacities = serialization.reference_solution.installed_capacities
    positive = [
        (flow, capacities[link_id])
        for flow, link_id in zip(absolute_flows, link_ids, strict=True)
        if capacities[link_id] > 0.0
    ]
    utilizations = [flow / capacity for flow, capacity in positive]
    return SndCouplingFlowStatistics(
        coupling_link_count=len(link_ids),
        positive_capacity_count=len(positive),
        absolute_flow_mean=(
            sum(absolute_flows) / len(absolute_flows) if absolute_flows else None
        ),
        absolute_flow_max=max(absolute_flows) if absolute_flows else None,
        utilization_mean=(
            sum(utilizations) / len(utilizations) if utilizations else None
        ),
        utilization_max=max(utilizations) if utilizations else None,
    )


def _validate_source(source: SndSource) -> None:
    if not isinstance(getattr(source, "source_name", None), str):
        raise TypeError("SND sources must expose a source_name string")
    network = getattr(source, "network", None)
    solution = getattr(source, "solution", None)
    if not isinstance(network, SndlibNetwork) or not isinstance(solution, SndSolution):
        raise TypeError("SND sources must expose network and solution objects")
    validate_snd_solution(network, solution)


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


def _validate_selection_settings(
    *,
    exact_time_limit_seconds: float,
    exact_threads: int,
) -> None:
    if (
        not isinstance(exact_time_limit_seconds, Real)
        or isinstance(exact_time_limit_seconds, bool)
        or not math.isfinite(exact_time_limit_seconds)
        or exact_time_limit_seconds <= 0.0
    ):
        raise ValueError("exact_time_limit_seconds must be finite and positive")
    _positive_integer("exact_threads", exact_threads)


def _positive_integer(name: str, value: int) -> None:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
