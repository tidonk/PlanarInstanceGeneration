"""Solver-neutral interfaces for fixed-capacity SND feasibility gates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Protocol, runtime_checkable

from planar_instance_generation.core.snd import SndSolution
from planar_instance_generation.io.sndlib import SndlibNetwork


@dataclass(frozen=True, slots=True)
class McmcfModelEstimate:
    """Exact size of the dense signed-flow/absolute-flow LP formulation."""

    node_count: int
    link_count: int
    demand_count: int
    active_demand_count: int
    flow_variable_count: int
    absolute_flow_variable_count: int
    total_variable_count: int
    balance_constraint_count: int
    absolute_value_constraint_count: int
    capacity_constraint_count: int
    total_constraint_count: int

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a nonnegative integer")
        if self.total_variable_count != (
            self.flow_variable_count + self.absolute_flow_variable_count
        ):
            raise ValueError("total_variable_count is inconsistent")
        if self.total_constraint_count != (
            self.balance_constraint_count
            + self.absolute_value_constraint_count
            + self.capacity_constraint_count
        ):
            raise ValueError("total_constraint_count is inconsistent")

    @classmethod
    def for_network(cls, network: SndlibNetwork) -> McmcfModelEstimate:
        """Count variables and constraints without importing a solver."""

        if not isinstance(network, SndlibNetwork):
            raise TypeError("network must be an SndlibNetwork")
        nodes = len(network.nodes)
        links = len(network.links)
        demands = len(network.demands)
        active_demands = sum(demand.value > 0.0 for demand in network.demands)
        commodity_links = active_demands * links
        return cls(
            node_count=nodes,
            link_count=links,
            demand_count=demands,
            active_demand_count=active_demands,
            flow_variable_count=commodity_links,
            absolute_flow_variable_count=commodity_links,
            total_variable_count=2 * commodity_links,
            balance_constraint_count=active_demands * nodes,
            absolute_value_constraint_count=2 * commodity_links,
            capacity_constraint_count=links,
            total_constraint_count=active_demands * nodes + 2 * commodity_links + links,
        )


@dataclass(frozen=True, slots=True)
class McmcfResult:
    """One exact fixed-capacity MCMCF gate outcome.

    ``feasible`` is tri-state: ``True`` means a validated feasible solution,
    ``False`` means the LP proved infeasible, and ``None`` means no feasibility
    conclusion (for example a resource skip or an unsupported input feature).
    ``runtime_seconds`` is end-to-end wall time; ``optimizer_seconds`` is
    Gurobi's model runtime when optimization was attempted.
    """

    feasible: bool | None
    status: str
    objective: float | None
    runtime_seconds: float
    optimizer_seconds: float | None
    solution: SndSolution | None
    estimate: McmcfModelEstimate
    resource_skipped: bool = False
    reason: str | None = None
    log_path: Path | None = None

    def __post_init__(self) -> None:
        if not (
            self.feasible is True
            or self.feasible is False
            or self.feasible is None
        ):
            raise TypeError("feasible must be True, False, or None")
        if not isinstance(self.status, str) or not self.status.strip():
            raise ValueError("status must not be empty")
        if not math.isfinite(self.runtime_seconds) or self.runtime_seconds < 0.0:
            raise ValueError("runtime_seconds must be finite and nonnegative")
        if self.optimizer_seconds is not None and (
            not math.isfinite(self.optimizer_seconds) or self.optimizer_seconds < 0.0
        ):
            raise ValueError("optimizer_seconds must be finite and nonnegative")
        if self.objective is not None and not math.isfinite(self.objective):
            raise ValueError("objective must be finite when provided")
        if self.feasible is True:
            if self.solution is None or self.objective is None:
                raise ValueError("a feasible result needs a solution and objective")
        elif self.solution is not None or self.objective is not None:
            raise ValueError("a result without proven feasibility cannot contain a solution")
        if self.resource_skipped and self.feasible is not None:
            raise ValueError("a resource-skipped result has no feasibility conclusion")
        if self.log_path is not None:
            object.__setattr__(self, "log_path", Path(self.log_path))

    @property
    def infeasible(self) -> bool:
        """Whether the optimizer proved the fixed-capacity LP infeasible."""

        return self.feasible is False


@runtime_checkable
class McmcfGate(Protocol):
    """Interface for fixed-capacity min-cost multicommodity-flow gates."""

    def estimate(self, network: SndlibNetwork, /) -> McmcfModelEstimate:
        """Return the solver-independent dense formulation size."""

    def solve(
        self,
        network: SndlibNetwork,
        installed_capacities: Mapping[str, float],
        /,
        *,
        source_name: str | None = None,
    ) -> McmcfResult:
        """Solve without mutating the network or capacities."""
