"""Retry and acceptance control for power experiments."""

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
from planar_instance_generation.core.power_merge import CouplingBranchConfig
from planar_instance_generation.core.planarization import OuterFaceSampling
from planar_instance_generation.core.selection import (
    CouplingCountRule,
    coupling_pair_rule,
)
from planar_instance_generation.evaluation.power_generation import (
    PowerGenerationCandidate,
    PowerMergeSelectionError,
    PowerSelectionStrategy,
    extract_power_pieces,
    merge_power_pieces,
)
from planar_instance_generation.evaluation.source_pool import FeasibleSource
from planar_instance_generation.gates.base import DcOpfGate

AttemptStatus: TypeAlias = Literal[
    "accepted",
    "gate_infeasible",
    "extraction_failed",
    "selection_failed",
    "size_limit",
    "paired_retry",
]
GateFactory: TypeAlias = Callable[[PowerSelectionStrategy, int, int], DcOpfGate]


@dataclass(frozen=True, slots=True)
class PowerAttemptRecord:
    """One strategy outcome on one shared extraction attempt."""

    output_index: int
    attempt_index: int
    strategy: PowerSelectionStrategy
    random_seed: int
    target_size: int
    source_count: int
    draw_seed: int
    allocation_seed: int
    generation_seed: int
    source_names: tuple[str, ...]
    target_nodes: tuple[int, ...]
    realized_fractions: tuple[float, ...]
    status: AttemptStatus
    reason: str | None
    candidate_statistics: dict[str, object] | None
    individual_status: str | None = None
    retry_trigger_strategy: str | None = None
    retry_trigger_status: str | None = None


@dataclass(frozen=True, slots=True)
class AcceptedPowerCandidate:
    """An accepted candidate and its requested output/attempt indices."""

    output_index: int
    attempt_index: int
    candidate: PowerGenerationCandidate


@dataclass(frozen=True, slots=True)
class PowerExperimentResult:
    """Accepted outputs, all attempts, and unmet output requests."""

    accepted: tuple[AcceptedPowerCandidate, ...]
    attempts: tuple[PowerAttemptRecord, ...]
    failures: tuple[tuple[PowerSelectionStrategy, int], ...]

    def accepted_for(
        self, strategy: PowerSelectionStrategy
    ) -> tuple[AcceptedPowerCandidate, ...]:
        return tuple(
            result for result in self.accepted if result.candidate.strategy == strategy
        )

    def acceptance_rate(self, strategy: PowerSelectionStrategy) -> float:
        relevant = tuple(
            record for record in self.attempts if record.strategy == strategy
        )
        if not relevant:
            return 0.0
        return sum(record.status == "accepted" for record in relevant) / len(relevant)


def run_power_experiment(
    source_pool: Sequence[FeasibleSource],
    gate_factory: GateFactory,
    *,
    strategies: Sequence[PowerSelectionStrategy],
    requested_per_strategy: int,
    num_subinstances: Sequence[int],
    target_size: Sequence[int],
    size_range: Sequence[Real],
    max_attempts_per_output: int,
    seed: int = 0,
    coupling: CouplingBranchConfig | None = None,
    exact_time_limit_seconds: float = 600.0,
    exact_threads: int = 16,
    coupling_count_rule: CouplingCountRule = "log2",
    outer_face_sampling: OuterFaceSampling = "consecutive",
    outer_face_prefer_non_leaf: bool = False,
    selection_log_directory: (
        Callable[[PowerSelectionStrategy, int, int], str | None] | None
    ) = None,
) -> PowerExperimentResult:
    """Generate exact-size, paired strategy cohorts with bounded retries.

    One target is sampled for each requested output and remains fixed across
    retries. A retry samples its source count and sources, allocates the target
    exactly, extracts the pieces once, and reuses them for every strategy. A
    cohort is accepted only when every strategy succeeds on that same attempt.
    """

    coupling_pair_rule(coupling_count_rule)
    if not source_pool:
        raise ValueError("source_pool must not be empty")
    if any(not isinstance(source, FeasibleSource) for source in source_pool):
        raise TypeError("source_pool must contain FeasibleSource records")
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

    rng = random.Random(int(seed))
    accepted: list[AcceptedPowerCandidate] = []
    attempts: list[PowerAttemptRecord] = []
    failures: list[tuple[PowerSelectionStrategy, int]] = []

    for output_index in range(1, requested_per_strategy + 1):
        requested_target = rng.randint(target_min, target_max)
        cohort_accepted = False
        for attempt_index in range(1, max_attempts_per_output + 1):
            draw_seed = rng.randrange(2**63)
            allocation_seed = rng.randrange(2**63)
            generation_seed = rng.randrange(2**63)
            draw_rng = random.Random(draw_seed)
            source_count = draw_rng.randint(source_min, source_max)
            selected_sources = tuple(
                draw_rng.choice(source_pool) for _ in range(source_count)
            )
            names = tuple(source.case.name for source in selected_sources)
            node_counts = tuple(
                int(source.case.bus.shape[0]) for source in selected_sources
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
                pieces = extract_power_pieces(
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
                        names,
                        allocation.target_nodes,
                        allocation.realized_fractions,
                        "extraction_failed",
                        str(error),
                    )
                    for strategy in normalized_strategies
                )
                continue

            extracted_total = sum(piece.target_nodes for piece in pieces)
            if extracted_total != requested_target:
                raise RuntimeError("power piece targets do not sum to requested target")

            provisional: list[
                tuple[PowerSelectionStrategy, PowerGenerationCandidate]
            ] = []
            outcomes: dict[
                PowerSelectionStrategy,
                tuple[AttemptStatus, str | None, dict[str, object] | None],
            ] = {}
            for strategy in normalized_strategies:
                logs = (
                    None
                    if selection_log_directory is None
                    else selection_log_directory(strategy, output_index, attempt_index)
                )
                try:
                    candidate = merge_power_pieces(
                        pieces,
                        gate_factory(strategy, output_index, attempt_index),
                        strategy=strategy,
                        seed=generation_seed,
                        coupling=coupling,
                        exact_time_limit_seconds=exact_time_limit_seconds,
                        exact_threads=exact_threads,
                        coupling_count_rule=coupling_count_rule,
                        outer_face_sampling=outer_face_sampling,
                        outer_face_prefer_non_leaf=outer_face_prefer_non_leaf,
                        log_directory=logs,
                    )
                except PowerMergeSelectionError as error:
                    outcomes[strategy] = ("selection_failed", str(error), None)
                    break

                statistics = candidate.statistics()
                if candidate.case.bus.shape[0] != requested_target:
                    raise RuntimeError(
                        "power composition changed the allocated node total"
                    )
                if not candidate.feasible:
                    outcomes[strategy] = (
                        "gate_infeasible",
                        candidate.gate_result.status,
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
                    AcceptedPowerCandidate(output_index, attempt_index, candidate)
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

    return PowerExperimentResult(tuple(accepted), tuple(attempts), tuple(failures))


def _attempt(
    output_index: int,
    attempt_index: int,
    strategy: PowerSelectionStrategy,
    random_seed: int,
    target_size: int,
    source_count: int,
    draw_seed: int,
    allocation_seed: int,
    generation_seed: int,
    source_names: tuple[str, ...],
    target_nodes: tuple[int, ...],
    realized_fractions: tuple[float, ...],
    status: AttemptStatus,
    reason: str | None,
    statistics: dict[str, object] | None = None,
    *,
    individual_status: str | None = None,
    retry_trigger_strategy: str | None = None,
    retry_trigger_status: str | None = None,
) -> PowerAttemptRecord:
    return PowerAttemptRecord(
        output_index,
        attempt_index,
        strategy,
        random_seed,
        target_size,
        source_count,
        draw_seed,
        allocation_seed,
        generation_seed,
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


def _positive_integer(name: str, value: int) -> None:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
