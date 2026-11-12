"""Exact fixed-capacity MCMCF feasibility gate backed by Gurobi."""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real
from pathlib import Path
import re
from time import perf_counter
from typing import Any

from planar_instance_generation.core.snd import SndSolution, validate_snd_solution
from planar_instance_generation.gates.snd_base import McmcfModelEstimate, McmcfResult
from planar_instance_generation.io.sndlib import SndlibNetwork


DEFAULT_MAX_VARIABLES = 2_000_000
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class GurobiMcmcfUnavailableError(ImportError):
    """Raised when the optional Gurobi Python API is unavailable."""


def restricted_demand_ids(network: SndlibNetwork) -> tuple[str, ...]:
    """Return demands whose inherited routing semantics are unsupported.

    This first gate supports unrestricted point-to-point commodities. Aggregate
    link-flow variables cannot certify SNDlib admissible paths or maximum path
    lengths, so a network containing either is rejected before optimization.
    """

    if not isinstance(network, SndlibNetwork):
        raise TypeError("network must be an SndlibNetwork")
    return tuple(
        demand.id
        for demand in network.demands
        if demand.max_path_length is not None or demand.admissible_paths
    )


class GurobiMcmcfGate:
    """Solve a continuous undirected fixed-capacity MCMCF LP.

    One free signed flow variable follows each XML link orientation, while a
    nonnegative companion variable linearizes its absolute value. Therefore a
    physical link's shared capacity is ``sum_k abs(flow[k, link])``. The dense
    formulation has ``2 * commodities * links`` variables; ``max_variables``
    prevents accidental construction of an oversized model and produces a
    resource-skip result rather than an infeasibility claim.
    """

    def __init__(
        self,
        *,
        max_variables: int = DEFAULT_MAX_VARIABLES,
        time_limit: float | None = None,
        threads: int = 1,
        seed: int = 0,
        default_routing_cost: float = 1.0,
        verbose: bool = False,
        log_directory: str | Path | None = None,
    ) -> None:
        if (
            isinstance(max_variables, bool)
            or not isinstance(max_variables, int)
            or max_variables < 1
        ):
            raise ValueError("max_variables must be a positive integer")
        if time_limit is not None:
            _positive_finite("time_limit", time_limit)
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError("threads must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a bool")
        self._max_variables = max_variables
        self._time_limit = None if time_limit is None else float(time_limit)
        self._threads = threads
        self._seed = seed
        self._default_routing_cost = _nonnegative_finite(
            "default_routing_cost", default_routing_cost
        )
        self._verbose = verbose
        self._log_directory = None if log_directory is None else Path(log_directory)

    @property
    def max_variables(self) -> int:
        return self._max_variables

    @property
    def default_routing_cost(self) -> float:
        return self._default_routing_cost

    def estimate(self, network: SndlibNetwork, /) -> McmcfModelEstimate:
        return McmcfModelEstimate.for_network(network)

    def solve(
        self,
        network: SndlibNetwork,
        installed_capacities: Mapping[str, float],
        /,
        *,
        source_name: str | None = None,
    ) -> McmcfResult:
        """Minimize routing cost at the supplied, fixed installed capacities."""

        started = perf_counter()
        if not isinstance(network, SndlibNetwork):
            raise TypeError("network must be an SndlibNetwork")
        capacities = _validated_capacities(network, installed_capacities)
        estimate = self.estimate(network)
        restricted = restricted_demand_ids(network)
        if restricted:
            reason = (
                "aggregate link-flow MCMCF does not implement maxPathLength or "
                "admissiblePaths; restricted demands: " + ", ".join(restricted)
            )
            return McmcfResult(
                feasible=None,
                status="UNSUPPORTED_ROUTING_RESTRICTIONS",
                objective=None,
                runtime_seconds=perf_counter() - started,
                optimizer_seconds=None,
                solution=None,
                estimate=estimate,
                reason=reason,
            )
        if estimate.total_variable_count > self._max_variables:
            return McmcfResult(
                feasible=None,
                status="RESOURCE_SKIPPED_VARIABLE_LIMIT",
                objective=None,
                runtime_seconds=perf_counter() - started,
                optimizer_seconds=None,
                solution=None,
                estimate=estimate,
                resource_skipped=True,
                reason=(
                    f"estimated {estimate.total_variable_count} variables exceed "
                    f"configured limit {self._max_variables}"
                ),
            )

        active_demands = tuple(
            demand for demand in network.demands if demand.value > 0.0
        )
        if not active_demands:
            solution = SndSolution(capacities, {})
            validate_snd_solution(network, solution)
            return McmcfResult(
                feasible=True,
                status="OPTIMAL_TRIVIAL",
                objective=0.0,
                runtime_seconds=perf_counter() - started,
                optimizer_seconds=0.0,
                solution=solution,
                estimate=estimate,
            )

        log_path = self._prepare_log_path(source_name)
        gp, GRB = _load_gurobi()
        environment_parameters: dict[str, Any] = {
            "OutputFlag": int(self._verbose or log_path is not None),
            "LogToConsole": int(self._verbose),
        }
        if log_path is not None:
            environment_parameters["LogFile"] = str(log_path)

        demand_indices = range(len(active_demands))
        link_indices = range(len(network.links))
        incidence: dict[str, list[tuple[int, float]]] = {
            node.id: [] for node in network.nodes
        }
        for link_index, link in enumerate(network.links):
            incidence[link.source].append((link_index, 1.0))
            incidence[link.target].append((link_index, -1.0))

        with gp.Env(params=environment_parameters) as environment:
            with gp.Model("fixed_capacity_mcmcf", env=environment) as model:
                model.Params.Threads = self._threads
                model.Params.Seed = self._seed
                model.Params.FeasibilityTol = 1e-8
                if self._time_limit is not None:
                    model.Params.TimeLimit = self._time_limit

                flow = model.addVars(
                    demand_indices,
                    link_indices,
                    lb=-GRB.INFINITY,
                    name="flow",
                )
                absolute = model.addVars(
                    demand_indices,
                    link_indices,
                    lb=0.0,
                    name="absolute_flow",
                )
                model.addConstrs(
                    (
                        absolute[demand_index, link_index]
                        >= flow[demand_index, link_index]
                        for demand_index in demand_indices
                        for link_index in link_indices
                    )
                )
                model.addConstrs(
                    (
                        absolute[demand_index, link_index]
                        >= -flow[demand_index, link_index]
                        for demand_index in demand_indices
                        for link_index in link_indices
                    )
                )
                model.addConstrs(
                    (
                        gp.quicksum(
                            sign * flow[demand_index, link_index]
                            for link_index, sign in incidence[node.id]
                        )
                        == (
                            demand.value
                            if node.id == demand.source
                            else -demand.value
                            if node.id == demand.target
                            else 0.0
                        )
                        for demand_index, demand in enumerate(active_demands)
                        for node in network.nodes
                    )
                )
                model.addConstrs(
                    (
                        gp.quicksum(absolute[demand_index, link_index]
                                    for demand_index in demand_indices)
                        <= capacities[link.id]
                        for link_index, link in enumerate(network.links)
                    )
                )
                routing_cost = tuple(
                    (
                        self._default_routing_cost
                        if link.routing_cost is None
                        else link.routing_cost
                    )
                    for link in network.links
                )
                model.setObjective(
                    gp.quicksum(
                        routing_cost[link_index]
                        * absolute[demand_index, link_index]
                        for demand_index in demand_indices
                        for link_index in link_indices
                    ),
                    GRB.MINIMIZE,
                )
                model.optimize()

                optimizer_seconds = float(model.Runtime)
                status_name = _gurobi_status_name(int(model.Status), GRB)
                solution_count = int(model.SolCount)
                if solution_count > 0:
                    sparse_flows = {
                        (demand.id, link.id): float(flow[demand_index, link_index].X)
                        for demand_index, demand in enumerate(active_demands)
                        for link_index, link in enumerate(network.links)
                        if abs(float(flow[demand_index, link_index].X)) > 1e-10
                    }
                    solution = SndSolution(capacities, sparse_flows)
                    validate_snd_solution(network, solution)
                    status = (
                        "OPTIMAL" if status_name == "OPTIMAL" else f"{status_name}_FEASIBLE"
                    )
                    return McmcfResult(
                        feasible=True,
                        status=status,
                        objective=float(model.ObjVal),
                        runtime_seconds=perf_counter() - started,
                        optimizer_seconds=optimizer_seconds,
                        solution=solution,
                        estimate=estimate,
                        log_path=log_path,
                    )
                if status_name == "INFEASIBLE":
                    feasible: bool | None = False
                else:
                    feasible = None
                return McmcfResult(
                    feasible=feasible,
                    status=status_name,
                    objective=None,
                    runtime_seconds=perf_counter() - started,
                    optimizer_seconds=optimizer_seconds,
                    solution=None,
                    estimate=estimate,
                    reason=(
                        "fixed installed capacities cannot route all demands"
                        if feasible is False
                        else (
                            "optimizer ended without a feasible solution or proof "
                            "of infeasibility"
                        )
                    ),
                    log_path=log_path,
                )

    def _prepare_log_path(self, source_name: str | None) -> Path | None:
        if self._log_directory is None:
            return None
        name = "mcmcf" if source_name is None else source_name
        if not isinstance(name, str) or not _SAFE_NAME.fullmatch(name):
            raise ValueError("source_name must be a safe filename component")
        path = self._log_directory.expanduser().resolve() / f"{name}.log"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite solver log: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def _validated_capacities(
    network: SndlibNetwork, capacities: Mapping[str, float]
) -> dict[str, float]:
    if not isinstance(capacities, Mapping):
        raise TypeError("installed_capacities must be a mapping")
    expected = {link.id for link in network.links}
    actual = set(capacities)
    if expected != actual:
        details = []
        if expected - actual:
            details.append(f"missing links {sorted(expected - actual)}")
        if actual - expected:
            details.append(f"unknown links {sorted(actual - expected)}")
        raise ValueError("installed capacities do not match links: " + "; ".join(details))
    normalized: dict[str, float] = {}
    for link in network.links:
        capacity = _nonnegative_finite(
            f"installed capacity for {link.id!r}", capacities[link.id]
        )
        preinstalled = (
            0.0
            if link.preinstalled_module is None
            else link.preinstalled_module.capacity
        )
        if capacity < preinstalled:
            raise ValueError(
                f"installed capacity for {link.id!r} is below preinstalled capacity"
            )
        normalized[link.id] = capacity
    return normalized


def _load_gurobi() -> tuple[Any, Any]:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as error:
        raise GurobiMcmcfUnavailableError(
            "the fixed-capacity MCMCF gate needs the optional 'gurobipy' package; "
            "install the 'snd' or 'evaluation' optional dependency"
        ) from error
    return gp, GRB


def _gurobi_status_name(status: int, GRB: Any) -> str:
    for name in (
        "LOADED",
        "OPTIMAL",
        "INFEASIBLE",
        "INF_OR_UNBD",
        "UNBOUNDED",
        "CUTOFF",
        "ITERATION_LIMIT",
        "NODE_LIMIT",
        "TIME_LIMIT",
        "SOLUTION_LIMIT",
        "INTERRUPTED",
        "NUMERIC",
        "SUBOPTIMAL",
        "USER_OBJ_LIMIT",
        "WORK_LIMIT",
        "MEM_LIMIT",
    ):
        if status == getattr(GRB, name, object()):
            return name
    return f"STATUS_{status}"


def _positive_finite(name: str, value: object) -> float:
    number = _nonnegative_finite(name, value)
    if number == 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _nonnegative_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number
