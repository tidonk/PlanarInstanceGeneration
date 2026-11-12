"""Retry and acceptance control for SND experiments."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
import random
from typing import Callable, Literal, Sequence, TypeAlias

from planar_instance_generation.core.extraction import ComponentTooSmallError
from planar_instance_generation.core.size_allocation import (
    TargetSizeInfeasibleError,
    allocate_target_size,
)
from planar_instance_generation.core.snd_merge import SndMergeConfig
from planar_instance_generation.core.planarization import OuterFaceSampling
from planar_instance_generation.core.selection import (
    CouplingCountRule,
    coupling_pair_rule,
)
from planar_instance_generation.core.snd_serialization import SndSerializationError
from planar_instance_generation.evaluation.snd_generation import (
    SndGenerationCandidate,
    SndMergeSelectionError,
    SndSelectionStrategy,
    SndSource,
    extract_snd_pieces,
    merge_snd_pieces,
)
from planar_instance_generation.gates.snd_base import McmcfGate

SndAttemptStatus: TypeAlias = Literal[
    "accepted",
    "gate_infeasible",
    "gate_resource_skipped",
    "gate_unsupported",
    "gate_inconclusive",
    "extraction_failed",
    "selection_failed",
    "serialization_failed",
    "size_limit",
    "paired_retry",
]
SndGateFactory: TypeAlias = Callable[[SndSelectionStrategy, int, int], McmcfGate]


@dataclass(frozen=True, slots=True)
class SndRoutingUnitStratum:
    """One source-pool stratum used for unit-safe with-replacement draws."""

    routing_unit: str | None
    source_names: tuple[str, ...]

    @property
    def source_count(self) -> int:
        return len(self.source_names)


@dataclass(frozen=True, slots=True)
class SndAttemptRecord:
    """One strategy outcome on one shared SND extraction attempt."""

    output_index: int
    attempt_index: int
    strategy: SndSelectionStrategy
    random_seed: int
    target_size: int
    source_count: int
    draw_seed: int
    allocation_seed: int
    generation_seed: int
    routing_unit: str | None
    source_stratum_size: int
    source_names: tuple[str, ...]
    target_nodes: tuple[int, ...]
    realized_fractions: tuple[float, ...]
    status: SndAttemptStatus
    reason: str | None
    candidate_statistics: dict[str, object] | None
    individual_status: str | None = None
    retry_trigger_strategy: str | None = None
    retry_trigger_status: str | None = None


@dataclass(frozen=True, slots=True)
class AcceptedSndCandidate:
    """An accepted SND candidate and its output/attempt indices."""

    output_index: int
    attempt_index: int
    candidate: SndGenerationCandidate


@dataclass(frozen=True, slots=True)
class SndExperimentResult:
    """Accepted outputs, complete attempt history, and unmet requests."""

    accepted: tuple[AcceptedSndCandidate, ...]
    attempts: tuple[SndAttemptRecord, ...]
    failures: tuple[tuple[SndSelectionStrategy, int], ...]
    routing_unit_strata: tuple[SndRoutingUnitStratum, ...]

    def accepted_for(
        self, strategy: SndSelectionStrategy
    ) -> tuple[AcceptedSndCandidate, ...]:
        return tuple(
            result for result in self.accepted if result.candidate.strategy == strategy
        )

    def acceptance_rate(self, strategy: SndSelectionStrategy) -> float:
        relevant = tuple(
            record for record in self.attempts if record.strategy == strategy
        )
        if not relevant:
            return 0.0
        return sum(record.status == "accepted" for record in relevant) / len(relevant)


def run_snd_experiment(
    source_pool: Sequence[SndSource],
    gate_factory: SndGateFactory,
    *,
    strategies: Sequence[SndSelectionStrategy],
    requested_per_strategy: int,
    num_subinstances: Sequence[int],
    target_size: Sequence[int],
    size_range: Sequence[Real],
    max_attempts_per_output: int,
    seed: int = 0,
    merge_config: SndMergeConfig | None = None,
    mixed_coordinates_type: str | None = "pixel",
    exact_time_limit_seconds: float = 600.0,
    exact_threads: int = 16,
    coupling_count_rule: CouplingCountRule = "log2",
    outer_face_sampling: OuterFaceSampling = "consecutive",
    outer_face_prefer_non_leaf: bool = False,
    selection_log_directory: (
        Callable[[SndSelectionStrategy, int, int], str | None] | None
    ) = None,
) -> SndExperimentResult:
    """Generate exact-size paired SND strategy comparisons.

    One target is sampled per requested output and held fixed across retries.
    Every retry samples a source count and sources, allocates the
    target exactly, and reuses the extracted pieces for every strategy.
    """

    coupling_pair_rule(coupling_count_rule)
    if not source_pool:
        raise ValueError("source_pool must not be empty")
    normalized_strategies = tuple(strategies)
    allowed = {
        "random",
        "high_degree",
        "kamada_kawai",
        "heuristic_outer_face",
        "exact_outer_face",
    }
    if not normalized_strategies or len(set(normalized_strategies)) != len(
        normalized_strategies
    ):
        raise ValueError("strategies must be nonempty and distinct")
    if any(strategy not in allowed for strategy in normalized_strategies):
        raise ValueError("strategies contains an unknown strategy")
    _positive_integer("requested_per_strategy", requested_per_strategy)
    _positive_integer("max_attempts_per_output", max_attempts_per_output)
    source_min, source_max = _integer_range("num_subinstances", num_subinstances)
    target_min, target_max = _integer_range("target_size", target_size)
    if source_min < 2:
        raise ValueError("num_subinstances minimum must be at least two")
    if source_max > 7:
        raise ValueError("at most seven sources fit the persistent plot palette")
    if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    strata = _stratify(source_pool)
    rng = random.Random(int(seed))
    accepted: list[AcceptedSndCandidate] = []
    attempts: list[SndAttemptRecord] = []
    failures: list[tuple[SndSelectionStrategy, int]] = []

    for output_index in range(1, requested_per_strategy + 1):
        requested_target = rng.randint(target_min, target_max)
        cohort_accepted = False
        for attempt_index in range(1, max_attempts_per_output + 1):
            draw_seed = rng.randrange(2**63)
            allocation_seed = rng.randrange(2**63)
            generation_seed = rng.randrange(2**63)
            count_rng = random.Random(draw_seed)
            source_count = count_rng.randint(source_min, source_max)
            selected_sources, routing_unit, stratum_size = _draw_sources(
                source_pool,
                strata,
                source_count,
                seed=count_rng.randrange(2**63),
            )
            names = tuple(source.source_name for source in selected_sources)
            node_counts = tuple(
                len(source.network.nodes) for source in selected_sources
            )
            try:
                allocation = allocate_target_size(
                    node_counts,
                    requested_target,
                    size_range,
                    rng=random.Random(allocation_seed),
                )
            except TargetSizeInfeasibleError as error:
                attempts.extend(
                    _attempt(
                        output_index,
                        attempt_index,
                        strategy,
                        int(seed),
                        requested_target,
                        source_count,
                        draw_seed,
                        allocation_seed,
                        generation_seed,
                        routing_unit,
                        stratum_size,
                        names,
                        (),
                        (),
                        "size_limit",
                        str(error),
                    )
                    for strategy in normalized_strategies
                )
                continue

            try:
                pieces = extract_snd_pieces(
                    selected_sources,
                    target_nodes=allocation.target_nodes,
                    seed=generation_seed,
                )
            except ComponentTooSmallError as error:
                attempts.extend(
                    _attempt(
                        output_index,
                        attempt_index,
                        strategy,
                        int(seed),
                        requested_target,
                        source_count,
                        draw_seed,
                        allocation_seed,
                        generation_seed,
                        routing_unit,
                        stratum_size,
                        names,
                        allocation.target_nodes,
                        allocation.realized_fractions,
                        "extraction_failed",
                        str(error),
                    )
                    for strategy in normalized_strategies
                )
                continue

            if sum(piece.target_nodes for piece in pieces) != requested_target:
                raise RuntimeError("SND piece targets do not sum to requested target")

            provisional: list[tuple[SndSelectionStrategy, SndGenerationCandidate]] = []
            outcomes: dict[
                SndSelectionStrategy,
                tuple[SndAttemptStatus, str | None, dict[str, object] | None],
            ] = {}
            for strategy in normalized_strategies:
                logs = (
                    None
                    if selection_log_directory is None
                    else selection_log_directory(strategy, output_index, attempt_index)
                )
                try:
                    candidate = merge_snd_pieces(
                        pieces,
                        gate_factory(strategy, output_index, attempt_index),
                        strategy=strategy,
                        seed=generation_seed,
                        merge_config=merge_config,
                        mixed_coordinates_type=mixed_coordinates_type,
                        source_stratum_size=stratum_size,
                        exact_time_limit_seconds=exact_time_limit_seconds,
                        exact_threads=exact_threads,
                        coupling_count_rule=coupling_count_rule,
                        outer_face_sampling=outer_face_sampling,
                        outer_face_prefer_non_leaf=outer_face_prefer_non_leaf,
                        log_directory=logs,
                    )
                except SndMergeSelectionError as error:
                    outcomes[strategy] = ("selection_failed", str(error), None)
                    break
                except SndSerializationError as error:
                    outcomes[strategy] = ("serialization_failed", str(error), None)
                    break

                statistics = candidate.statistics()
                if len(candidate.instance.nodes) != requested_target:
                    raise RuntimeError(
                        "SND composition changed the allocated node total"
                    )
                if candidate.feasible is False:
                    outcomes[strategy] = (
                        "gate_infeasible",
                        candidate.gate_result.reason or candidate.gate_result.status,
                        statistics,
                    )
                    break
                if candidate.feasible is None:
                    outcomes[strategy] = (
                        _inconclusive_status(candidate.gate_result),
                        candidate.gate_result.reason or candidate.gate_result.status,
                        statistics,
                    )
                    break
                outcomes[strategy] = ("accepted", None, statistics)
                provisional.append((strategy, candidate))

            if len(outcomes) != len(normalized_strategies):
                failed = next(
                    strategy
                    for strategy in normalized_strategies
                    if strategy in outcomes and outcomes[strategy][0] != "accepted"
                )
                for skipped in normalized_strategies:
                    outcomes.setdefault(
                        skipped,
                        (
                            "paired_retry",
                            f"not run after paired failure in {failed}",
                            None,
                        ),
                    )

            failed_strategies = tuple(
                strategy
                for strategy in normalized_strategies
                if outcomes[strategy][0] != "accepted"
            )
            if failed_strategies:
                trigger = next(
                    s
                    for s in normalized_strategies
                    if outcomes[s][0] not in ("accepted", "paired_retry")
                )
                trigger_status = outcomes[trigger][0]
                reason = f"paired retry because {trigger}: {trigger_status}"
                for strategy in normalized_strategies:
                    status, strategy_reason, statistics = outcomes[strategy]
                    if status == "accepted":
                        status, strategy_reason = "paired_retry", reason
                    attempts.append(
                        _attempt(
                            output_index,
                            attempt_index,
                            strategy,
                            int(seed),
                            requested_target,
                            source_count,
                            draw_seed,
                            allocation_seed,
                            generation_seed,
                            routing_unit,
                            stratum_size,
                            names,
                            allocation.target_nodes,
                            allocation.realized_fractions,
                            status,
                            strategy_reason,
                            statistics,
                            individual_status=(
                                "not_run"
                                if outcomes[strategy][0] == "paired_retry"
                                else outcomes[strategy][0]
                            ),
                            retry_trigger_strategy=trigger,
                            retry_trigger_status=trigger_status,
                        )
                    )
                continue

            for strategy, candidate in provisional:
                accepted.append(
                    AcceptedSndCandidate(output_index, attempt_index, candidate)
                )
                status, reason, statistics = outcomes[strategy]
                attempts.append(
                    _attempt(
                        output_index,
                        attempt_index,
                        strategy,
                        int(seed),
                        requested_target,
                        source_count,
                        draw_seed,
                        allocation_seed,
                        generation_seed,
                        routing_unit,
                        stratum_size,
                        names,
                        allocation.target_nodes,
                        allocation.realized_fractions,
                        status,
                        reason,
                        statistics,
                    )
                )
            cohort_accepted = True
            break

        if not cohort_accepted:
            failures.extend(
                (strategy, output_index) for strategy in normalized_strategies
            )

    stratum_records = tuple(
        SndRoutingUnitStratum(
            routing_unit=unit,
            source_names=tuple(source.source_name for source in sources),
        )
        for unit, sources in sorted(strata.items(), key=_unit_sort_key)
    )
    return SndExperimentResult(
        tuple(accepted),
        tuple(attempts),
        tuple(failures),
        stratum_records,
    )


def _stratify(
    source_pool: Sequence[SndSource],
) -> dict[str | None, tuple[SndSource, ...]]:
    """Build compatibility groups, treating a missing unit as unspecified."""

    sources = tuple(source_pool)
    for source in sources:
        name = getattr(source, "source_name", None)
        network = getattr(source, "network", None)
        if not isinstance(name, str) or network is None:
            raise TypeError("source_pool entries must expose source_name and network")
    declared_units = sorted(
        {
            source.network.routing_unit
            for source in sources
            if source.network.routing_unit is not None
        }
    )
    if len(declared_units) <= 1:
        key = declared_units[0] if declared_units else None
        return {key: sources}
    unknown = tuple(source for source in sources if source.network.routing_unit is None)
    groups = {
        unit: tuple(
            source for source in sources if source.network.routing_unit in {None, unit}
        )
        for unit in declared_units
    }
    if unknown:
        groups[None] = unknown
    return groups


def _draw_sources(
    source_pool: Sequence[SndSource],
    strata: dict[str | None, tuple[SndSource, ...]],
    count: int,
    *,
    seed: int,
) -> tuple[tuple[SndSource, ...], str | None, int]:
    rng = random.Random(seed)
    first = rng.choice(source_pool)
    compatible = [
        (unit, stratum) for unit, stratum in strata.items() if first in stratum
    ]
    if not compatible:
        raise RuntimeError("selected SND source has no compatibility group")
    unit, stratum = rng.choice(compatible)
    selected = (first,) + tuple(rng.choice(stratum) for _ in range(count - 1))
    return selected, unit, len(stratum)


def _inconclusive_status(result: object) -> SndAttemptStatus:
    if getattr(result, "resource_skipped", False):
        return "gate_resource_skipped"
    if str(getattr(result, "status", "")).upper().startswith("UNSUPPORTED"):
        return "gate_unsupported"
    return "gate_inconclusive"


def _attempt(
    output_index: int,
    attempt_index: int,
    strategy: SndSelectionStrategy,
    random_seed: int,
    target_size: int,
    source_count: int,
    draw_seed: int,
    allocation_seed: int,
    generation_seed: int,
    routing_unit: str | None,
    source_stratum_size: int,
    source_names: tuple[str, ...],
    target_nodes: tuple[int, ...],
    realized_fractions: tuple[float, ...],
    status: SndAttemptStatus,
    reason: str | None,
    statistics: dict[str, object] | None = None,
    *,
    individual_status: str | None = None,
    retry_trigger_strategy: str | None = None,
    retry_trigger_status: str | None = None,
) -> SndAttemptRecord:
    return SndAttemptRecord(
        output_index,
        attempt_index,
        strategy,
        random_seed,
        target_size,
        source_count,
        draw_seed,
        allocation_seed,
        generation_seed,
        routing_unit,
        source_stratum_size,
        source_names,
        target_nodes,
        realized_fractions,
        status,
        reason,
        statistics,
        individual_status or status,
        retry_trigger_strategy,
        retry_trigger_status,
    )


def _integer_range(name: str, values: Sequence[int]) -> tuple[int, int]:
    raw = tuple(values)
    if len(raw) != 2:
        raise ValueError(f"{name} must contain exactly two integers")
    lower, upper = raw
    _positive_integer(f"{name} minimum", lower)
    _positive_integer(f"{name} maximum", upper)
    if lower > upper:
        raise ValueError(f"{name} minimum must not exceed its maximum")
    return int(lower), int(upper)


def _unit_sort_key(
    item: tuple[str | None, tuple[SndSource, ...]],
) -> tuple[bool, str]:
    unit = item[0]
    return unit is not None, "" if unit is None else unit


def _positive_integer(name: str, value: int) -> None:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
