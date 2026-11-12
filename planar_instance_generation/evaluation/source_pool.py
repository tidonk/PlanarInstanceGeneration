"""Screen source cases once and retain their feasible DC-OPF solutions."""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Iterable

from planar_instance_generation.gates.base import DcOpfGate, DcOpfResult, DcOpfSolution
from planar_instance_generation.io.pglib import PglibCase


class EulerPlanarityWarning(UserWarning):
    """An active simple topology violates the planar Euler edge bound."""


@dataclass(frozen=True, slots=True)
class BusLimitSelection:
    """Cases included by, and excluded from, an explicit bus-count cap."""

    included: tuple[PglibCase, ...]
    excluded: tuple[PglibCase, ...]


@dataclass(frozen=True, slots=True)
class FeasibleSource:
    """A retained source case and the solution used during extraction."""

    case: PglibCase
    solution: DcOpfSolution
    objective: float
    dc_opf_seconds: float
    optimizer_seconds: float | None


@dataclass(frozen=True, slots=True)
class SourceScreeningRecord:
    """One row of source-pool preprocessing statistics."""

    source_name: str
    bus_count: int
    active_branch_count: int
    active_simple_edge_count: int
    euler_edge_limit: int | None
    euler_bound_exceeded: bool
    excluded_by_euler_bound: bool
    retained: bool
    feasible: bool | None
    status: str
    objective: float | None
    dc_opf_seconds: float
    optimizer_seconds: float | None


@dataclass(frozen=True, slots=True)
class SourcePool:
    """Feasible sources plus a complete record of all attempted solves."""

    sources: tuple[FeasibleSource, ...]
    records: tuple[SourceScreeningRecord, ...]

    @property
    def attempted_count(self) -> int:
        return len(self.records)

    @property
    def feasible_count(self) -> int:
        return len(self.sources)

    @property
    def euler_excluded_count(self) -> int:
        return sum(record.excluded_by_euler_bound for record in self.records)

    @property
    def total_dc_opf_seconds(self) -> float:
        return sum(record.dc_opf_seconds for record in self.records)


def select_cases_by_bus_limit(
    cases: Iterable[PglibCase], *, max_buses: int
) -> BusLimitSelection:
    """Partition cases using an explicit inclusive bus-count limit."""

    if isinstance(max_buses, bool) or not isinstance(max_buses, int) or max_buses < 1:
        raise ValueError("max_buses must be a positive integer")
    included: list[PglibCase] = []
    excluded: list[PglibCase] = []
    for case in cases:
        (included if case.bus.shape[0] <= max_buses else excluded).append(case)
    return BusLimitSelection(tuple(included), tuple(excluded))


def screen_source_cases(
    cases: Iterable[PglibCase],
    gate: DcOpfGate,
    *,
    exclude_euler_bound_violations: bool = True,
    warn_on_euler_bound: bool = True,
) -> SourcePool:
    """Filter certified nonplanar cases, then retain DC-OPF-feasible cases.

    The Euler bound certifies nonplanarity when it is exceeded, but satisfying
    it does not certify planarity. Active parallel branches are collapsed
    before comparing ``|E|`` with ``3|V|-6``.
    """

    if not isinstance(exclude_euler_bound_violations, bool):
        raise TypeError("exclude_euler_bound_violations must be a bool")
    if not isinstance(warn_on_euler_bound, bool):
        raise TypeError("warn_on_euler_bound must be a bool")

    sources: list[FeasibleSource] = []
    records: list[SourceScreeningRecord] = []

    for case in cases:
        topology = case.topology(active_only=True)
        bus_count = topology.number_of_nodes()
        simple_edge_count = topology.number_of_edges()
        edge_limit = 3 * bus_count - 6 if bus_count >= 3 else None
        exceeds_bound = edge_limit is not None and simple_edge_count > edge_limit
        if exceeds_bound and warn_on_euler_bound:
            warnings.warn(
                f"{case.name}: active simple topology has {simple_edge_count} edges, "
                f"exceeding the planar Euler bound {edge_limit}",
                EulerPlanarityWarning,
                stacklevel=2,
            )

        active_branch_count = int(sum(row[10] > 0 for row in case.branch))
        if exceeds_bound and exclude_euler_bound_violations:
            records.append(
                SourceScreeningRecord(
                    source_name=case.name,
                    bus_count=bus_count,
                    active_branch_count=active_branch_count,
                    active_simple_edge_count=simple_edge_count,
                    euler_edge_limit=edge_limit,
                    euler_bound_exceeded=True,
                    excluded_by_euler_bound=True,
                    retained=False,
                    feasible=None,
                    status="EULER_BOUND_EXCEEDED",
                    objective=None,
                    dc_opf_seconds=0.0,
                    optimizer_seconds=None,
                )
            )
            continue

        result = gate.solve(case)
        _validate_result_for_case(result, case)
        records.append(
            SourceScreeningRecord(
                source_name=case.name,
                bus_count=bus_count,
                active_branch_count=active_branch_count,
                active_simple_edge_count=simple_edge_count,
                euler_edge_limit=edge_limit,
                euler_bound_exceeded=exceeds_bound,
                excluded_by_euler_bound=False,
                retained=result.feasible,
                feasible=result.feasible,
                status=result.status,
                objective=result.objective,
                dc_opf_seconds=result.runtime_seconds,
                optimizer_seconds=result.optimizer_seconds,
            )
        )
        if result.feasible:
            assert result.solution is not None and result.objective is not None
            sources.append(
                FeasibleSource(
                    case=case,
                    solution=result.solution,
                    objective=result.objective,
                    dc_opf_seconds=result.runtime_seconds,
                    optimizer_seconds=result.optimizer_seconds,
                )
            )

    return SourcePool(tuple(sources), tuple(records))


def _validate_result_for_case(result: DcOpfResult, case: PglibCase) -> None:
    if result.feasible:
        assert result.solution is not None
        result.solution.validate_for(case)
        if result.objective is None:
            raise ValueError(f"{case.name}: a feasible result must provide an objective")
