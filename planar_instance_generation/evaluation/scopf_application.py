"""Structured hand-off to TopologyOptimizer's DC-SCOPF Benders application."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import tomllib
from typing import Any

import numpy as np

from planar_instance_generation.io.pglib import PglibCase, read_pglib, write_pglib

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_APPLICATION = "DC_SCOTS_TIME_SLACK_BENDERS"


@dataclass(frozen=True, slots=True)
class ScopfBendersConfiguration:
    """Resource settings for one DC-SCOPF/SCOTS solve."""

    time_limit_seconds: float = 10_800.0
    threads: int = 1
    seed: int = 42
    topological_action_limit: int = 0
    scopf_warmstart: bool = False
    warmstart_time_limit_seconds: float = 10800.0
    benders_settings: str | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.time_limit_seconds) or self.time_limit_seconds < 0:
            raise ValueError("time_limit_seconds must be finite and nonnegative (0 means unlimited)")
        if self.threads < 1:
            raise ValueError("threads must be positive")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if (
            not isfinite(self.warmstart_time_limit_seconds)
            or self.warmstart_time_limit_seconds <= 0
        ):
            raise ValueError("warmstart_time_limit_seconds must be finite and positive")
        if self.topological_action_limit < 0:
            raise ValueError("topological_action_limit must be nonnegative")


@dataclass(frozen=True, slots=True)
class ScopfBendersRunPaths:
    """All files belonging to one external application run."""

    log_directory: Path
    settings: Path
    invocation: Path
    process_log: Path
    innoptem_instance: Path
    normalization: Path
    adapter_result: Path
    summary: Path


@dataclass(frozen=True, slots=True)
class ScopfBendersRun:
    """Prepared TopologyOptimizer command and its output contract."""

    command: tuple[str, ...]
    paths: ScopfBendersRunPaths
    summary: dict[str, Any]


def render_scopf_benders_settings(config: ScopfBendersConfiguration) -> str:
    """Return the minimal TopologyOptimizer override for DC-SCOPF/SCOTS."""

    algorithm_settings = ""
    if config.benders_settings is not None:
        algorithm_settings = Path(config.benders_settings).read_text()
        if set(tomllib.loads(algorithm_settings)) != {"BENDERS"}:
            raise ValueError("algorithm overrides must contain only BENDERS settings")
    return f"""APPLICATION = "{_APPLICATION}"

[SOLVER]
SOLVING_TIME_LIMIT = {format(config.time_limit_seconds, ".17g") if config.time_limit_seconds else "inf"}
INCLUDE_WARMSTART_TIME_LIMIT = false

[SOLVER.MULTITHREADING]
N_THREADS_BENDERS = {config.threads}
N_THREADS_MIP_MP = {config.threads}
N_THREADS_MIP_SP = {config.threads}

[TECHNICAL]
SEED = {config.seed}

[LOSS_ESTIMATION]
ESTIMATE_LOSSES = false

[POSTPROCESSING.AC_FEAS]
CHECK = false

[WARMSTARTING]
ENABLED = {str(config.scopf_warmstart).lower()}
WARMSTART_METHOD = ":scopf"
SCOPF_TIME_LIMIT = {config.warmstart_time_limit_seconds:.17g}

[HEURISTIC_SOLVING]
ENABLED = false

[SUBSTATION_RECONF]
BRANCH_RECONF = false
COUPLER_RECONF = false

[LIMITS.TRANSFORMER]
TRANSFORMER_SHIFTING = false

[LIMITS.GENERATOR]
GEN_SWITCHING = false

[LIMITS.BRANCH]
BRANCH_SWITCHING = true
TAL_TIMESTEP = {config.topological_action_limit}

[PRINTING.SOLVER]
LOG_TO_CONSOLE = true
LOG_TO_CONSOLE_MP = true
LOG_TO_CONSOLE_IP = false
LOG_TO_CONSOLE_SP = false

{algorithm_settings}
"""


def prepare_scopf_benders_run(
    instance: str | Path,
    *,
    run_name: str,
    results_directory: str | Path,
    innoptem_repository: str | Path,
    adapter: str | Path,
    julia: str = "julia",
    use_manifest_julia: bool = True,
    julia_channel: str | None = None,
    config: ScopfBendersConfiguration = ScopfBendersConfiguration(),
) -> ScopfBendersRun:
    """Validate inputs and construct the isolated Julia invocation."""

    _validate_component("run_name", run_name)
    instance_path = Path(instance).expanduser().resolve()
    results = Path(results_directory).expanduser().resolve()
    innoptem = Path(innoptem_repository).expanduser().resolve()
    adapter_path = Path(adapter).resolve()
    if not instance_path.is_file() or instance_path.suffix.lower() != ".m":
        raise ValueError(f"generated PGLib input does not exist: {instance_path}")
    try:
        relative_input = instance_path.relative_to(results)
    except ValueError as error:
        raise ValueError(f"SCOPF inputs must be generated below {results}") from error
    if "instances" not in relative_input.parts[:-1]:
        raise ValueError("generated PGLib input must be in an instances directory")
    if (
        not (innoptem / "Project.toml").is_file()
        or not (innoptem / "src/TopologyOptimizer.jl").is_file()
    ):
        raise ValueError(f"not an TopologyOptimizer checkout: {innoptem}")
    if not adapter_path.is_file():
        raise ValueError(f"Julia adapter does not exist: {adapter_path}")

    instance_name = instance_path.stem
    _validate_component("instance filename stem", instance_name)
    log_directory = results / run_name / "logs" / instance_name
    paths = ScopfBendersRunPaths(
        log_directory=log_directory,
        settings=log_directory / "settings.toml",
        invocation=log_directory / "invocation.json",
        process_log=log_directory / "process.log",
        innoptem_instance=log_directory / "innoptem_input.m",
        normalization=log_directory / "normalization.json",
        adapter_result=log_directory / "adapter_result.json",
        summary=log_directory / "summary.json",
    )
    manifest_julia = _manifest_julia_version(innoptem)
    if julia_channel is not None and not re.fullmatch(
        r"\d+\.\d+(?:\.\d+)?", julia_channel
    ):
        raise ValueError("julia_channel must be a numeric Julia version")
    selected_julia = julia_channel or (manifest_julia if use_manifest_julia else None)
    julia_command = (
        (julia,) if selected_julia is None else (julia, f"+{selected_julia}")
    )
    if selected_julia is not None and Path(julia).resolve().name == "julialauncher":
        installed = (
            Path(julia).absolute().parents[2]
            / ".julia/juliaup"
            / f"julia-{selected_julia}+0.x64.linux.gnu/bin/julia"
        )
        if installed.is_file():
            julia_command = (str(installed),)
    command = julia_command + (
        f"--project={innoptem}",
        f"--threads={config.threads}",
        "--startup-file=no",
        "--compiled-modules=existing",
        "--history-file=no",
        str(adapter_path),
        str(paths.innoptem_instance),
        str(paths.settings),
        str(paths.adapter_result),
        str(log_directory),
    )
    summary = {
        "schema_version": 1,
        "application": _APPLICATION,
        "method": (
            "TopologyOptimizer DC-SCOTS with Benders decomposition"
            if config.topological_action_limit
            else "TopologyOptimizer DC-SCOPF with Benders decomposition"
        ),
        "instance": str(instance_path),
        "instance_sha256": sha256(instance_path.read_bytes()).hexdigest(),
        "run_name": run_name,
        "log_directory": str(log_directory),
        "configuration": asdict(config),
        "model_invariants": {
            "power_flow": "DC only",
            "topological_action_limit": config.topological_action_limit,
            "switching_actions_allowed": config.topological_action_limit > 0,
            "branch_switching_variable_layer": True,
            "transformer_switching": False,
            "generator_switching": False,
            "busbar_reconfiguration": False,
            "ac_loss_estimation": False,
            "ac_feasibility_check": False,
            "disconnecting_failures": "dropped by TopologyOptimizer preprocessing",
        },
        "innoptem_repository": str(innoptem),
        "innoptem_version": _project_version(innoptem),
        "innoptem_git_revision": _git_revision(innoptem),
        "manifest_julia_version": manifest_julia,
        "selected_julia_channel": selected_julia,
        "command": list(command),
    }
    return ScopfBendersRun(command, paths, summary)


def execute_scopf_benders_run(
    run: ScopfBendersRun,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Materialize a run and, unless dry, execute its Julia adapter."""

    paths = run.paths
    if paths.log_directory.exists():
        raise FileExistsError(
            f"application log directory already exists: {paths.log_directory}"
        )
    paths.log_directory.mkdir(parents=True)
    config = ScopfBendersConfiguration(**run.summary["configuration"])
    normalization = _materialize_innoptem_input(
        Path(run.summary["instance"]), paths.innoptem_instance
    )
    paths.normalization.write_text(
        json.dumps(normalization, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    base_summary = run.summary | {
        "innoptem_input": str(paths.innoptem_instance),
        "rating_normalization": normalization,
    }
    paths.settings.write_text(render_scopf_benders_settings(config), encoding="utf-8")
    paths.invocation.write_text(
        json.dumps(base_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    started = datetime.now(timezone.utc).isoformat()
    if dry_run:
        summary = base_summary | {"state": "dry_run", "started_at": started}
        _write_summary(paths.summary, summary)
        return summary

    started_clock = time.perf_counter()
    try:
        with paths.process_log.open("w", encoding="utf-8") as log:
            environment = os.environ.copy()
            environment["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
            exit_code, exit_recovery = _wait_for_julia(run, environment, log, config)
    except (OSError, TimeoutError) as error:
        summary = base_summary | {
            "state": "adapter_error",
            "started_at": started,
            "wall_seconds": time.perf_counter() - started_clock,
            "error": str(error),
            "process_log": str(paths.process_log),
            "gurobi_logs": _gurobi_logs(paths.log_directory),
        }
        _write_summary(paths.summary, summary)
        return summary
    wall_seconds = time.perf_counter() - started_clock
    recovered_result = None
    if exit_code != 0 and exit_recovery is None:
        recovered_result = _recover_no_incumbent_postprocessing_failure(
            base_summary,
            paths,
            normalization,
            wall_seconds,
        )
    summary = base_summary | {
        "state": (
            "complete"
            if exit_code == 0 or exit_recovery is not None or recovered_result is not None
            else "adapter_error"
        ),
        "started_at": started,
        "wall_seconds": wall_seconds,
        "adapter_exit_code": exit_code,
        "process_log": str(paths.process_log),
        "gurobi_logs": _gurobi_logs(paths.log_directory),
    }
    if recovered_result is not None:
        summary["result"] = recovered_result
        summary["adapter_recovery"] = (
            "TopologyOptimizer postprocessing failed after Gurobi reported zero solutions; "
            "the completed no-incumbent outcome was recovered from the process log"
        )
    elif exit_code == 0 or exit_recovery is not None:
        if exit_recovery is not None:
            summary["adapter_recovery"] = exit_recovery
        if not paths.adapter_result.is_file():
            summary["state"] = "adapter_error"
            summary["error"] = "Julia adapter returned success without a result file"
        else:
            try:
                summary["result"] = json.loads(
                    paths.adapter_result.read_text(encoding="utf-8")
                )
                validation = summary["result"].get(
                    "rating_normalization_validation", {}
                )
                empirically_nonbinding = validation.get("empirically_nonbinding")
                feasible_primal = summary["result"].get("scopf_feasible") is True
                validation_failed = empirically_nonbinding is False or (
                    feasible_primal and empirically_nonbinding is not True
                )
                if normalization["changed_branch_count"] and validation_failed:
                    summary["state"] = "adapter_error"
                    summary["error"] = (
                        "the normalized coupling rating was not verified as nonbinding"
                    )
            except (OSError, json.JSONDecodeError) as error:
                summary["state"] = "adapter_error"
                summary["error"] = f"cannot read adapter result: {error}"
    _write_summary(paths.summary, summary)
    return summary


def _wait_for_julia(run, environment, log, config):
    """Bound startup and shutdown waits without changing either solver budget."""
    process = subprocess.Popen(
        run.command, cwd=run.summary["innoptem_repository"], env=environment,
        stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True,
    )
    started = time.monotonic()
    last_cpu = None
    last_activity = started
    budget = (config.time_limit_seconds or float("inf")) + (
        config.warmstart_time_limit_seconds if config.scopf_warmstart else 0
    ) + 1800
    finished = False
    try:
        while True:
            code = process.poll()
            if code is not None:
                finished = True
                return code, None
            # Julia writes this only after solving and all result processing.
            # Ignore incomplete JSON and allow ten seconds for normal shutdown.
            path = run.paths.adapter_result
            if path.is_file() and time.time() - path.stat().st_mtime >= 10:
                try:
                    result = json.loads(path.read_text())
                except (OSError, ValueError):
                    result = {}
                if (
                    isinstance(result.get("scopf_feasible"), bool)
                    and isinstance(result.get("termination_status"), str)
                    and isinstance(result.get("time"), dict)
                    and isinstance(result.get("rating_normalization_validation"), dict)
                ):
                    return None, "Complete Julia result recovered after shutdown exceeded ten seconds; process termination requested."
            elapsed = time.monotonic() - started
            try:
                fields = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()
                cpu = fields[11:13]
                if cpu != last_cpu:
                    last_cpu, last_activity = cpu, started + elapsed
            except OSError:
                pass
            if started + elapsed - last_activity > 900 and run.paths.process_log.stat().st_size == 0:
                raise TimeoutError("Julia startup produced no output or CPU activity for 900 seconds")
            if elapsed > budget:
                raise TimeoutError("Julia exceeded the solver budgets plus 1800 seconds for startup and shutdown")
            time.sleep(2)
    finally:
        if not finished:
            # Do not wait indefinitely for a process in uninterruptible NFS I/O.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _recover_no_incumbent_postprocessing_failure(
    base_summary: dict[str, Any],
    paths: ScopfBendersRunPaths,
    normalization: dict[str, Any],
    wall_seconds: float,
) -> dict[str, Any] | None:
    """Recover TopologyOptimizer's known postprocessing crash after a no-incumbent solve.

    TopologyOptimizer 6.2 may label the callback state as FEASIBLE_POINT even when
    Gurobi reports Solution count 0. Its postprocessor then indexes the absent
    solution network. The optimization outcome is still usable for a
    found-feasibility study: no feasible incumbent was found. The strict marker
    conjunction keeps unrelated adapter failures fatal.
    """

    try:
        process_log = paths.process_log.read_text(encoding="utf-8")
    except OSError:
        return None
    markers = (
        "Solution count 0",
        'KeyError: key "nw" not found',
        "postprocess_results!",
        "calculate_power_flow!",
    )
    if not all(marker in process_log for marker in markers):
        return None

    case = read_pglib(Path(base_summary["instance"]))
    coupling_count = int(normalization["coupling_branch_count"])
    coupling_validation: dict[str, Any] = {
        "coupling_branch_count": coupling_count,
        "flow_unit": "MW",
        "observed_coupling_branch_count": 0,
        "observed_coupling_branch_fraction": (0.0 if coupling_count > 0 else None),
        "flow_observation_count": 0,
        "positive_flow_threshold_mw": 1.0e-6,
        "positive_flow_branch_count": None,
        "positive_flow_branch_fraction": None,
        "mean_absolute_flow_mw": None,
        "median_absolute_flow_mw": None,
        "p95_absolute_flow_mw": None,
        "maximum_absolute_flow_mw": None,
        "noncoupling_branch_count": max(0, int(case.branch.shape[0]) - coupling_count),
        "observed_noncoupling_branch_count": 0,
        "observed_noncoupling_branch_fraction": (
            0.0 if int(case.branch.shape[0]) > coupling_count else None
        ),
        "noncoupling_mean_absolute_flow_mw": None,
        "maximum_utilization": None,
        "normalized_branch_count": int(normalization["changed_branch_count"]),
        "observed_normalized_branch_count": 0,
        "maximum_absolute_flow_by_branch_mw": {},
        "nonbinding_threshold": 0.95,
        "empirically_nonbinding": None,
        "validation_status": "not_observable_without_primal_solution",
    }
    return {
        "schema_version": 5,
        "application": _APPLICATION,
        "formulation": "DC",
        "algorithm": "Benders",
        "termination_status": "NO_SOLUTION_FOUND",
        "primal_status": "NO_SOLUTION",
        "has_finite_objective": False,
        "has_primal_network": False,
        "scopf_feasible": False,
        "scopf_feasibility_definition": (
            "certified finite incumbent with a primal network; overload slack permitted"
        ),
        "scopf_feasibility_status": "no_feasible_incumbent_found",
        "scopf_feasible_without_overload": False,
        "overload_cost_all_contingencies": None,
        "overload_cost_base_case": None,
        "objective_components": {},
        "wall_seconds": wall_seconds,
        "n_switching_actions": None,
        "network": {
            "buses": int(case.bus.shape[0]),
            "branches": int(case.branch.shape[0]),
            "generators": int(case.gen.shape[0]),
            "base_mva": float(case.base_mva),
        },
        "contingencies": {},
        "model_invariants": base_summary["model_invariants"],
        "rating_normalization": normalization,
        "rating_normalization_validation": coupling_validation,
        "time": {},
        "callback": {},
        "info": {
            "recovered_from_innoptem_postprocessing_failure": True,
            "reason": "Gurobi reported Solution count 0",
        },
    }


def _validate_component(label: str, value: str) -> None:
    if not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(f"{label} must be a single safe filename component")


def _write_summary(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _gurobi_logs(directory: Path) -> list[str]:
    return [str(path) for path in sorted(directory.glob("GRB_*.out"))]


def _materialize_innoptem_input(source: Path, destination: Path) -> dict[str, Any]:
    """Replace zero coupling ratings in a logged copy, never in the accepted case."""

    case = read_pglib(source)
    run_root = source.parent.parent
    metadata_candidates = (
        run_root / "plots" / "composite" / source.stem / "metadata.json",
        run_root / "plots" / source.stem / "metadata.json",
    )
    metadata_path = next(
        (path for path in metadata_candidates if path.is_file()),
        metadata_candidates[0],
    )
    if not metadata_path.is_file():
        raise ValueError(f"missing generated-instance metadata: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read generated-instance metadata: {error}") from error
    if metadata.get("artifact_stem") != source.stem:
        raise ValueError(f"metadata artifact stem does not match {source.name}")
    rows_value = metadata.get("coupling_branch_rows_zero_based")
    if not isinstance(rows_value, list) or not all(
        isinstance(row, int) and not isinstance(row, bool) for row in rows_value
    ):
        raise ValueError(f"invalid coupling branch rows in {metadata_path}")
    coupling_rows = set(rows_value)
    if any(row < 0 or row >= case.branch.shape[0] for row in coupling_rows):
        raise ValueError(
            f"coupling branch row is outside mpc.branch in {metadata_path}"
        )

    branch = case.branch.copy()
    changed: list[dict[str, Any]] = []
    active_rows = np.flatnonzero(branch[:, 10] > 0)
    for row in active_rows:
        nonpositive_columns = [
            column for column in range(5, 8) if branch[row, column] <= 0
        ]
        if not nonpositive_columns:
            continue
        if row not in coupling_rows:
            raise ValueError(
                "active non-coupling branch has a nonpositive RATE_A/B/C value: "
                f"mpc.branch row {row + 1}"
            )
        if any(branch[row, column] < 0 for column in nonpositive_columns):
            raise ValueError(f"coupling branch row {row + 1} has a negative rating")
        changed.append(
            {
                "row_zero_based": int(row),
                "row_one_based": int(row + 1),
                "from_bus": int(branch[row, 0]),
                "to_bus": int(branch[row, 1]),
                "columns": [
                    ("RATE_A", "RATE_B", "RATE_C")[column - 5]
                    for column in nonpositive_columns
                ],
                "original_mva": [float(value) for value in branch[row, 5:8]],
            }
        )

    online = case.gen[:, 7] > 0
    online_pmax = float(np.maximum(case.gen[online, 8], 0).sum())
    absolute_demand = float(np.abs(case.bus[:, 2]).sum())
    positive_ratings = branch[:, 5:8][branch[:, 5:8] > 0]
    largest_existing_rating = (
        float(positive_ratings.max()) if positive_ratings.size else 0.0
    )
    system_scale = max(
        case.base_mva, online_pmax, absolute_demand, largest_existing_rating
    )
    safety_factor = 2.0
    replacement = safety_factor * system_scale
    for item in changed:
        row = item["row_zero_based"]
        for column in range(5, 8):
            if branch[row, column] == 0:
                branch[row, column] = replacement
        item["normalized_mva"] = [float(value) for value in branch[row, 5:8]]

    remaining_nonpositive = np.argwhere(branch[:, 5:8] <= 0)
    if remaining_nonpositive.size:
        row, relative_column = map(int, remaining_nonpositive[0])
        status = "active" if branch[row, 10] > 0 else "inactive"
        column = ("RATE_A", "RATE_B", "RATE_C")[relative_column]
        raise ValueError(
            f"{status} branch row {row + 1} still has nonpositive {column}; "
            "only active metadata-certified coupling ratings may be normalized"
        )

    normalized = PglibCase(
        name=f"{case.name}_innoptem",
        version=case.version,
        base_mva=case.base_mva,
        bus=case.bus.copy(),
        gen=case.gen.copy(),
        branch=branch,
        gencost=case.gencost.copy(),
        areas=None if case.areas is None else case.areas.copy(),
    )
    write_pglib(normalized, destination)
    return {
        "policy": "only active coupling branches with zero MATPOWER ratings are changed",
        "metadata_path": str(metadata_path),
        "source_path": str(source),
        "source_sha256": sha256(source.read_bytes()).hexdigest(),
        "normalized_path": str(destination),
        "normalized_sha256": sha256(destination.read_bytes()).hexdigest(),
        "system_scale_mva": system_scale,
        "system_scale_components_mva": {
            "total_online_pmax": online_pmax,
            "total_absolute_pd": absolute_demand,
            "largest_existing_rating": largest_existing_rating,
            "base_mva": case.base_mva,
        },
        "safety_factor": safety_factor,
        "replacement_rating_mva": replacement,
        "coupling_branch_count": len(coupling_rows),
        "coupling_branch_rows_zero_based": sorted(coupling_rows),
        "coupling_branch_rows_one_based": [row + 1 for row in sorted(coupling_rows)],
        "changed_branch_count": len(changed),
        "changed_branches": changed,
    }


def _manifest_julia_version(repository: Path) -> str | None:
    manifest = repository / "Manifest.toml"
    if not manifest.is_file():
        return None
    with manifest.open("rb") as stream:
        value = tomllib.load(stream).get("julia_version")
    if not isinstance(value, str) or not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", value):
        raise ValueError(f"invalid julia_version in {manifest}")
    return value


def _project_version(repository: Path) -> str | None:
    with (repository / "Project.toml").open("rb") as stream:
        value = tomllib.load(stream).get("version")
    return value if isinstance(value, str) else None


def _git_revision(repository: Path) -> str | None:
    exported = repository / "source.json"
    if exported.is_file():
        revision = json.loads(exported.read_text()).get("revision")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("invalid exported TopologyOptimizer revision")
        return revision
    # Do not accidentally report the enclosing generator repository's revision.
    if not (repository / ".git").exists():
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        )
    except OSError:
        return None
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else None
