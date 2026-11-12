"""Deterministic construction and exact screening of SNDlib source cases."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import heapq
import math
from numbers import Real
from pathlib import Path
from time import perf_counter

from planar_instance_generation.core.snd import SndSolution, validate_snd_solution
from planar_instance_generation.gates.mcmcf import restricted_demand_ids
from planar_instance_generation.gates.snd_base import McmcfGate, McmcfModelEstimate
from planar_instance_generation.io.sndlib import SndlibLink, SndlibNetwork


CONSTRUCTIVE_METHOD = "deterministic_shortest_path_module_rounding"


class ConstructiveCapacityPlanError(RuntimeError):
    """Raised when the deterministic construction cannot produce a reference."""


@dataclass(frozen=True, slots=True)
class NamedSndSource:
    """An SNDlib network paired with its external source identifier."""

    source_name: str
    network: SndlibNetwork

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_name, str)
            or not self.source_name
            or self.source_name != self.source_name.strip()
        ):
            raise ValueError("source_name must be nonempty text without surrounding space")
        if not isinstance(self.network, SndlibNetwork):
            raise TypeError("network must be an SndlibNetwork")


@dataclass(frozen=True, slots=True)
class ConstructiveSndCapacityPlan:
    """A feasible, deterministic source reference; never a design optimum.

    Every positive demand is sent unsplit along one shortest path. Additional
    modules are available in unbounded integer copies, as in SNDlib, and one
    deterministic module type is rounded up per link. Preinstalled capacity is
    always retained. ``module_copies[link_id][i]`` corresponds to additional
    module ``i`` in the source XML.
    """

    solution: SndSolution
    demand_paths: Mapping[str, tuple[str, ...]]
    module_copies: Mapping[str, tuple[int, ...]]
    link_loads: Mapping[str, float]
    default_routing_cost: float
    routing_cost: float
    module_cost: float
    setup_cost: float
    total_design_cost: float
    method: str = CONSTRUCTIVE_METHOD
    design_optimal: bool = False

    def __post_init__(self) -> None:
        if self.method != CONSTRUCTIVE_METHOD:
            raise ValueError(f"unsupported constructive method {self.method!r}")
        if self.design_optimal is not False:
            raise ValueError("constructive design_optimal must be exactly False")
        for field_name in (
            "default_routing_cost",
            "routing_cost",
            "module_cost",
            "setup_cost",
            "total_design_cost",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and nonnegative")
        if not math.isclose(
            self.total_design_cost,
            self.module_cost + self.setup_cost,
            rel_tol=1e-10,
            abs_tol=1e-9,
        ):
            raise ValueError("total_design_cost must equal module_cost + setup_cost")
        object.__setattr__(
            self,
            "demand_paths",
            {key: tuple(value) for key, value in self.demand_paths.items()},
        )
        object.__setattr__(
            self,
            "module_copies",
            {key: tuple(value) for key, value in self.module_copies.items()},
        )
        object.__setattr__(self, "link_loads", dict(self.link_loads))

    @property
    def installed_capacity(self) -> float:
        return sum(self.solution.installed_capacities.values())

    @property
    def used_capacity(self) -> float:
        return sum(self.link_loads.values())

    @property
    def additional_module_count(self) -> int:
        return sum(sum(copies) for copies in self.module_copies.values())


@dataclass(frozen=True, slots=True)
class FeasibleSndSource:
    """A retained source with its exact LP flow and constructive capacity plan."""

    source_name: str
    network: SndlibNetwork
    solution: SndSolution
    constructive_plan: ConstructiveSndCapacityPlan
    objective: float
    constructive_seconds: float
    mcmcf_seconds: float
    preprocessing_seconds: float
    optimizer_seconds: float | None
    log_path: Path | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_name, str)
            or not self.source_name
            or self.source_name != self.source_name.strip()
        ):
            raise ValueError("source_name must be nonempty text without surrounding space")
        if not isinstance(self.network, SndlibNetwork):
            raise TypeError("network must be an SndlibNetwork")
        validate_constructive_snd_capacity_plan(self.network, self.constructive_plan)
        validate_snd_solution(self.network, self.solution)
        if dict(self.solution.installed_capacities) != dict(
            self.constructive_plan.solution.installed_capacities
        ):
            raise ValueError("exact and constructive installed capacities differ")
        for field_name in (
            "objective",
            "constructive_seconds",
            "mcmcf_seconds",
            "preprocessing_seconds",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and nonnegative")
        if self.optimizer_seconds is not None and (
            not math.isfinite(self.optimizer_seconds) or self.optimizer_seconds < 0.0
        ):
            raise ValueError("optimizer_seconds must be finite and nonnegative")
        if self.log_path is not None:
            object.__setattr__(self, "log_path", Path(self.log_path))


@dataclass(frozen=True, slots=True)
class SndSourceScreeningRecord:
    """One complete preprocessing outcome, including non-solver exclusions."""

    source_name: str
    routing_unit: str | None
    node_count: int
    link_count: int
    simple_edge_count: int
    euler_edge_limit: int | None
    euler_bound_exceeded: bool
    excluded_by_euler_bound: bool
    demand_count: int
    active_demand_count: int
    restricted_demand_count: int
    restricted_demand_ids: tuple[str, ...]
    has_max_path_length: bool
    has_admissible_paths: bool
    estimated_variables: int
    estimated_constraints: int
    max_variables: int | None
    retained: bool
    feasible: bool | None
    resource_skipped: bool
    status: str
    reason: str | None
    constructive_method: str | None
    constructive_design_optimal: bool | None
    constructive_installed_capacity: float | None
    constructive_module_count: int | None
    constructive_routing_cost: float | None
    constructive_design_cost: float | None
    constructive_seconds: float
    objective: float | None
    mcmcf_seconds: float
    preprocessing_seconds: float
    optimizer_seconds: float | None
    log_path: str | None


@dataclass(frozen=True, slots=True)
class SndSourcePool:
    """Retained feasible SND sources plus every attempted preprocessing row."""

    sources: tuple[FeasibleSndSource, ...]
    records: tuple[SndSourceScreeningRecord, ...]

    @property
    def attempted_count(self) -> int:
        return len(self.records)

    @property
    def feasible_count(self) -> int:
        return len(self.sources)

    @property
    def resource_skipped_count(self) -> int:
        return sum(record.resource_skipped for record in self.records)

    @property
    def euler_excluded_count(self) -> int:
        return sum(record.excluded_by_euler_bound for record in self.records)

    @property
    def unsupported_count(self) -> int:
        return sum(
            record.status == "UNSUPPORTED_ROUTING_RESTRICTIONS"
            for record in self.records
        )

    @property
    def total_mcmcf_seconds(self) -> float:
        return sum(record.mcmcf_seconds for record in self.records)

    @property
    def total_constructive_seconds(self) -> float:
        return sum(record.constructive_seconds for record in self.records)

    @property
    def total_preprocessing_seconds(self) -> float:
        return sum(record.preprocessing_seconds for record in self.records)


def construct_snd_capacity_plan(
    network: SndlibNetwork,
    *,
    source_name: str | None = None,
    default_routing_cost: float = 1.0,
) -> ConstructiveSndCapacityPlan:
    """Construct a deterministic feasible capacity/flow reference.

    This is deliberately a feasibility construction, not an SND design
    algorithm. Demands are processed by identifier and routed unsplit. Links
    with positive additional modules are expandable; preinstalled-only links
    are admitted only when their residual capacity can carry the full demand.
    """

    if not isinstance(network, SndlibNetwork):
        raise TypeError("network must be an SndlibNetwork")
    if (
        isinstance(default_routing_cost, bool)
        or not isinstance(default_routing_cost, Real)
        or not math.isfinite(float(default_routing_cost))
        or default_routing_cost < 0.0
    ):
        raise ValueError("default_routing_cost must be finite and nonnegative")
    restricted = restricted_demand_ids(network)
    if restricted:
        raise ConstructiveCapacityPlanError(
            "routing restrictions are unsupported for demands: "
            + ", ".join(restricted)
        )

    label = "SND source" if source_name is None else source_name
    costs = {
        link.id: (
            float(default_routing_cost)
            if link.routing_cost is None
            else link.routing_cost
        )
        for link in network.links
    }
    expandable = {
        link.id: any(module.capacity > 0.0 for module in link.additional_modules)
        for link in network.links
    }
    fixed_capacity = {
        link.id: (
            0.0
            if link.preinstalled_module is None
            else link.preinstalled_module.capacity
        )
        for link in network.links
    }
    adjacency: dict[str, list[tuple[str, str, float]]] = {
        node.id: [] for node in network.nodes
    }
    for link in network.links:
        adjacency[link.source].append((link.target, link.id, 1.0))
        adjacency[link.target].append((link.source, link.id, -1.0))
    for entries in adjacency.values():
        entries.sort(key=lambda item: (item[1], item[0], item[2]))

    loads = {link.id: 0.0 for link in network.links}
    flows: dict[tuple[str, str], float] = {}
    paths: dict[str, tuple[str, ...]] = {}
    for demand in sorted(network.demands, key=lambda item: item.id):
        if demand.value == 0.0:
            paths[demand.id] = ()
            continue
        allowed = {
            link.id
            for link in network.links
            if expandable[link.id]
            or fixed_capacity[link.id] - loads[link.id] >= demand.value - 1e-9
        }
        traversal = _deterministic_shortest_path(
            adjacency,
            costs,
            allowed,
            demand.source,
            demand.target,
        )
        if traversal is None:
            raise ConstructiveCapacityPlanError(
                f"{label}: deterministic residual-capacity routing failed for "
                f"demand {demand.id!r}; this is not an infeasibility proof"
            )
        paths[demand.id] = tuple(link_id for link_id, _ in traversal)
        for link_id, sign in traversal:
            flows[demand.id, link_id] = sign * demand.value
            loads[link_id] += demand.value

    capacities: dict[str, float] = {}
    module_copies: dict[str, tuple[int, ...]] = {}
    module_cost = 0.0
    setup_cost = 0.0
    for link in network.links:
        preinstalled = fixed_capacity[link.id]
        copies = [0] * len(link.additional_modules)
        installed = preinstalled
        if loads[link.id] > preinstalled:
            module_index = _select_module(link)
            if module_index is None:
                raise ConstructiveCapacityPlanError(
                    f"{label}: link {link.id!r} needs {loads[link.id]} capacity but "
                    "has no positive additional module; this is not an infeasibility proof"
                )
            module = link.additional_modules[module_index]
            count = math.ceil((loads[link.id] - preinstalled) / module.capacity)
            copies[module_index] = count
            installed += count * module.capacity
            module_cost += count * module.cost
            if link.setup_cost is not None:
                setup_cost += link.setup_cost
        capacities[link.id] = installed
        module_copies[link.id] = tuple(copies)

    routing_cost = sum(
        costs[link_id] * abs(flow) for (_, link_id), flow in flows.items()
    )
    solution = SndSolution(capacities, flows)
    validate_snd_solution(network, solution)
    plan = ConstructiveSndCapacityPlan(
        solution=solution,
        demand_paths=paths,
        module_copies=module_copies,
        link_loads=loads,
        default_routing_cost=float(default_routing_cost),
        routing_cost=routing_cost,
        module_cost=module_cost,
        setup_cost=setup_cost,
        total_design_cost=module_cost + setup_cost,
    )
    validate_constructive_snd_capacity_plan(network, plan)
    return plan


def validate_constructive_snd_capacity_plan(
    network: SndlibNetwork,
    plan: ConstructiveSndCapacityPlan,
    *,
    tolerance: float = 1e-7,
) -> None:
    """Recompute every path, load, module, capacity, and cost certificate."""

    if not isinstance(network, SndlibNetwork):
        raise TypeError("network must be an SndlibNetwork")
    if not isinstance(plan, ConstructiveSndCapacityPlan):
        raise TypeError("plan must be a ConstructiveSndCapacityPlan")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    validate_snd_solution(network, plan.solution, tolerance=tolerance)
    link_by_id = {link.id: link for link in network.links}
    demand_by_id = {demand.id: demand for demand in network.demands}
    if set(plan.demand_paths) != set(demand_by_id):
        raise ValueError("constructive demand_paths do not match network demands")
    if set(plan.module_copies) != set(link_by_id):
        raise ValueError("constructive module_copies do not match network links")
    if set(plan.link_loads) != set(link_by_id):
        raise ValueError("constructive link_loads do not match network links")

    expected_flows: dict[tuple[str, str], float] = {}
    expected_loads = {link_id: 0.0 for link_id in link_by_id}
    for demand in network.demands:
        current = demand.source
        path = plan.demand_paths[demand.id]
        if demand.value == 0.0 and path:
            raise ValueError(f"zero demand {demand.id!r} must have an empty path")
        for link_id in path:
            link = link_by_id.get(link_id)
            if link is None:
                raise ValueError(
                    f"constructive path for {demand.id!r} uses unknown link {link_id!r}"
                )
            if current == link.source:
                sign = 1.0
                current = link.target
            elif current == link.target:
                sign = -1.0
                current = link.source
            else:
                raise ValueError(
                    f"constructive path for {demand.id!r} is not a contiguous walk"
                )
            key = (demand.id, link_id)
            if key in expected_flows:
                raise ValueError(
                    f"constructive path for {demand.id!r} repeats link {link_id!r}"
                )
            if demand.value != 0.0:
                expected_flows[key] = sign * demand.value
                expected_loads[link_id] += demand.value
        if demand.value != 0.0 and current != demand.target:
            raise ValueError(
                f"constructive path for {demand.id!r} does not reach its target"
            )
    if set(plan.solution.flows) != set(expected_flows):
        raise ValueError("constructive flow support does not equal its routed paths")
    for key, expected in expected_flows.items():
        if not math.isclose(
            plan.solution.flows[key], expected, rel_tol=1e-10, abs_tol=tolerance
        ):
            raise ValueError(f"constructive flow {key!r} disagrees with its path")
    for link_id, expected in expected_loads.items():
        actual = plan.link_loads[link_id]
        if not math.isfinite(actual) or actual < 0.0 or not math.isclose(
            actual, expected, rel_tol=1e-10, abs_tol=tolerance
        ):
            raise ValueError(f"constructive load for {link_id!r} is inconsistent")

    module_cost = 0.0
    setup_cost = 0.0
    for link in network.links:
        copies = plan.module_copies[link.id]
        if len(copies) != len(link.additional_modules):
            raise ValueError(f"module copy vector for {link.id!r} has wrong length")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in copies
        ):
            raise ValueError(f"module copies for {link.id!r} must be nonnegative integers")
        if sum(count > 0 for count in copies) > 1:
            raise ValueError(
                f"constructive method permits only one additional module type on {link.id!r}"
            )
        preinstalled = (
            0.0
            if link.preinstalled_module is None
            else link.preinstalled_module.capacity
        )
        installed = preinstalled + sum(
            count * module.capacity
            for count, module in zip(copies, link.additional_modules, strict=True)
        )
        if not math.isclose(
            plan.solution.installed_capacities[link.id],
            installed,
            rel_tol=1e-10,
            abs_tol=tolerance,
        ):
            raise ValueError(f"installed capacity for {link.id!r} disagrees with modules")
        module_cost += sum(
            count * module.cost
            for count, module in zip(copies, link.additional_modules, strict=True)
        )
        if any(copies) and link.setup_cost is not None:
            setup_cost += link.setup_cost

    routing_cost = sum(
        (
            plan.default_routing_cost
            if link_by_id[link_id].routing_cost is None
            else link_by_id[link_id].routing_cost
        )
        * abs(flow)
        for (_, link_id), flow in plan.solution.flows.items()
    )
    for label, actual, expected in (
        ("routing_cost", plan.routing_cost, routing_cost),
        ("module_cost", plan.module_cost, module_cost),
        ("setup_cost", plan.setup_cost, setup_cost),
    ):
        if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=tolerance):
            raise ValueError(f"constructive {label} is inconsistent")


def screen_snd_source_cases(
    sources: Iterable[NamedSndSource],
    gate: McmcfGate,
    *,
    exclude_euler_bound_violations: bool = True,
) -> SndSourcePool:
    """Filter certified nonplanar cases and retain exact-LP-feasible sources."""

    source_tuple = tuple(sources)
    if not isinstance(gate, McmcfGate):
        raise TypeError("gate must implement McmcfGate")
    if not all(isinstance(source, NamedSndSource) for source in source_tuple):
        raise TypeError("sources must contain only NamedSndSource entries")
    if not isinstance(exclude_euler_bound_violations, bool):
        raise TypeError("exclude_euler_bound_violations must be a bool")
    names = [source.source_name for source in source_tuple]
    if len(set(names)) != len(names):
        raise ValueError("SND source names must be unique")

    retained: list[FeasibleSndSource] = []
    records: list[SndSourceScreeningRecord] = []
    max_variables = getattr(gate, "max_variables", None)
    default_routing_cost = float(getattr(gate, "default_routing_cost", 1.0))

    for source in source_tuple:
        preprocessing_started = perf_counter()
        network = source.network
        estimate = gate.estimate(network)
        restricted = restricted_demand_ids(network)
        has_max_path_length = any(
            demand.max_path_length is not None for demand in network.demands
        )
        has_admissible_paths = any(
            demand.admissible_paths for demand in network.demands
        )
        topology = network.topology()
        node_count = topology.number_of_nodes()
        edge_limit = 3 * node_count - 6 if node_count >= 3 else None
        exceeds_bound = (
            edge_limit is not None and topology.number_of_edges() > edge_limit
        )
        if restricted:
            records.append(
                _record(
                    source,
                    estimate,
                    max_variables=max_variables,
                    restricted=restricted,
                    has_max_path_length=has_max_path_length,
                    has_admissible_paths=has_admissible_paths,
                    status="UNSUPPORTED_ROUTING_RESTRICTIONS",
                    reason=(
                        "aggregate link-flow MCMCF does not implement maxPathLength "
                        "or admissiblePaths"
                    ),
                    preprocessing_seconds=perf_counter() - preprocessing_started,
                )
            )
            continue

        if exceeds_bound and exclude_euler_bound_violations:
            records.append(
                _record(
                    source,
                    estimate,
                    max_variables=max_variables,
                    restricted=(),
                    has_max_path_length=False,
                    has_admissible_paths=False,
                    status="EULER_BOUND_EXCEEDED",
                    reason=(
                        f"simple topology has {topology.number_of_edges()} edges, "
                        f"exceeding the planar Euler bound {edge_limit}"
                    ),
                    excluded_by_euler_bound=True,
                    preprocessing_seconds=perf_counter() - preprocessing_started,
                )
            )
            continue

        if max_variables is not None and estimate.total_variable_count > max_variables:
            records.append(
                _record(
                    source,
                    estimate,
                    max_variables=max_variables,
                    restricted=(),
                    has_max_path_length=False,
                    has_admissible_paths=False,
                    status="RESOURCE_SKIPPED_VARIABLE_LIMIT",
                    reason=(
                        f"estimated {estimate.total_variable_count} variables exceed "
                        f"configured limit {max_variables}"
                    ),
                    resource_skipped=True,
                    preprocessing_seconds=perf_counter() - preprocessing_started,
                )
            )
            continue

        constructive_started = perf_counter()
        try:
            plan = construct_snd_capacity_plan(
                network,
                source_name=source.source_name,
                default_routing_cost=default_routing_cost,
            )
        except ConstructiveCapacityPlanError as error:
            constructive_seconds = perf_counter() - constructive_started
            records.append(
                _record(
                    source,
                    estimate,
                    max_variables=max_variables,
                    restricted=(),
                    has_max_path_length=False,
                    has_admissible_paths=False,
                    status="CONSTRUCTIVE_PLAN_FAILED",
                    reason=str(error),
                    constructive_seconds=constructive_seconds,
                    preprocessing_seconds=perf_counter() - preprocessing_started,
                )
            )
            continue
        constructive_seconds = perf_counter() - constructive_started

        result = gate.solve(
            network,
            plan.solution.installed_capacities,
            source_name=source.source_name,
        )
        if result.estimate != estimate:
            raise ValueError(f"{source.source_name}: gate returned an inconsistent estimate")
        preprocessing_seconds = perf_counter() - preprocessing_started
        if result.feasible is True:
            assert result.solution is not None and result.objective is not None
            if dict(result.solution.installed_capacities) != dict(
                plan.solution.installed_capacities
            ):
                raise ValueError(
                    f"{source.source_name}: gate changed the fixed installed capacities"
                )
            validate_snd_solution(network, result.solution)
            retained.append(
                FeasibleSndSource(
                    source_name=source.source_name,
                    network=network,
                    solution=result.solution,
                    constructive_plan=plan,
                    objective=result.objective,
                    constructive_seconds=constructive_seconds,
                    mcmcf_seconds=result.runtime_seconds,
                    preprocessing_seconds=preprocessing_seconds,
                    optimizer_seconds=result.optimizer_seconds,
                    log_path=result.log_path,
                )
            )
        records.append(
            _record(
                source,
                estimate,
                max_variables=max_variables,
                restricted=(),
                has_max_path_length=False,
                has_admissible_paths=False,
                status=result.status,
                reason=result.reason,
                plan=plan,
                retained=result.feasible is True,
                feasible=result.feasible,
                resource_skipped=result.resource_skipped,
                objective=result.objective,
                constructive_seconds=constructive_seconds,
                mcmcf_seconds=result.runtime_seconds,
                preprocessing_seconds=preprocessing_seconds,
                optimizer_seconds=result.optimizer_seconds,
                log_path=result.log_path,
            )
        )

    return SndSourcePool(tuple(retained), tuple(records))


def _record(
    source: NamedSndSource,
    estimate: McmcfModelEstimate,
    *,
    max_variables: int | None,
    restricted: tuple[str, ...],
    has_max_path_length: bool,
    has_admissible_paths: bool,
    status: str,
    reason: str | None,
    plan: ConstructiveSndCapacityPlan | None = None,
    excluded_by_euler_bound: bool = False,
    retained: bool = False,
    feasible: bool | None = None,
    resource_skipped: bool = False,
    objective: float | None = None,
    constructive_seconds: float = 0.0,
    mcmcf_seconds: float = 0.0,
    preprocessing_seconds: float = 0.0,
    optimizer_seconds: float | None = None,
    log_path: Path | None = None,
) -> SndSourceScreeningRecord:
    network = source.network
    topology = network.topology()
    node_count = topology.number_of_nodes()
    edge_limit = 3 * node_count - 6 if node_count >= 3 else None
    exceeds_bound = (
        edge_limit is not None and topology.number_of_edges() > edge_limit
    )
    return SndSourceScreeningRecord(
        source_name=source.source_name,
        routing_unit=network.routing_unit,
        node_count=node_count,
        link_count=len(network.links),
        simple_edge_count=topology.number_of_edges(),
        euler_edge_limit=edge_limit,
        euler_bound_exceeded=exceeds_bound,
        excluded_by_euler_bound=excluded_by_euler_bound,
        demand_count=len(network.demands),
        active_demand_count=estimate.active_demand_count,
        restricted_demand_count=len(restricted),
        restricted_demand_ids=restricted,
        has_max_path_length=has_max_path_length,
        has_admissible_paths=has_admissible_paths,
        estimated_variables=estimate.total_variable_count,
        estimated_constraints=estimate.total_constraint_count,
        max_variables=max_variables,
        retained=retained,
        feasible=feasible,
        resource_skipped=resource_skipped,
        status=status,
        reason=reason,
        constructive_method=None if plan is None else plan.method,
        constructive_design_optimal=None if plan is None else plan.design_optimal,
        constructive_installed_capacity=(
            None if plan is None else plan.installed_capacity
        ),
        constructive_module_count=(
            None if plan is None else plan.additional_module_count
        ),
        constructive_routing_cost=None if plan is None else plan.routing_cost,
        constructive_design_cost=None if plan is None else plan.total_design_cost,
        constructive_seconds=constructive_seconds,
        objective=objective,
        mcmcf_seconds=mcmcf_seconds,
        preprocessing_seconds=preprocessing_seconds,
        optimizer_seconds=optimizer_seconds,
        log_path=None if log_path is None else str(log_path),
    )


def _deterministic_shortest_path(
    adjacency: Mapping[str, list[tuple[str, str, float]]],
    costs: Mapping[str, float],
    allowed_links: set[str],
    source: str,
    target: str,
) -> tuple[tuple[str, float], ...] | None:
    start_key: tuple[float, int, tuple[str, ...]] = (0.0, 0, ())
    best = {source: start_key}
    queue: list[
        tuple[
            float,
            int,
            tuple[str, ...],
            str,
            tuple[tuple[str, float], ...],
        ]
    ] = [(0.0, 0, (), source, ())]
    while queue:
        distance, hops, path_ids, node, path = heapq.heappop(queue)
        if best.get(node) != (distance, hops, path_ids):
            continue
        if node == target:
            return path
        for neighbor, link_id, sign in adjacency[node]:
            if link_id not in allowed_links:
                continue
            candidate_key = (
                distance + costs[link_id],
                hops + 1,
                path_ids + (link_id,),
            )
            if candidate_key < best.get(neighbor, (math.inf, 0, ())):
                best[neighbor] = candidate_key
                heapq.heappush(
                    queue,
                    (
                        candidate_key[0],
                        candidate_key[1],
                        candidate_key[2],
                        neighbor,
                        path + ((link_id, sign),),
                    ),
                )
    return None


def _select_module(link: SndlibLink) -> int | None:
    candidates = [
        (index, module)
        for index, module in enumerate(link.additional_modules)
        if module.capacity > 0.0
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            item[1].cost / item[1].capacity,
            item[1].cost,
            -item[1].capacity,
            item[0],
        ),
    )[0]
