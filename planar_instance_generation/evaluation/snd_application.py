"""Structured hand-off to the external Julia SND Benders application."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import isclose, isfinite
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import tomllib
from typing import Any, Literal


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True, slots=True)
class SndBendersConfiguration:
    """Settings for the full N-1 SND application solve."""

    time_limit_seconds: float = 10_800.0
    wall_clock_grace_seconds: float = 60.0
    threads: int = 1
    seed: int = 0
    method: Literal["standard", "adaptive_ml"] = "standard"

    def __post_init__(self) -> None:
        if not isfinite(self.time_limit_seconds) or self.time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be finite and positive")
        if (
            not isfinite(self.wall_clock_grace_seconds)
            or self.wall_clock_grace_seconds < 0
        ):
            raise ValueError("wall_clock_grace_seconds must be finite and nonnegative")
        if self.threads < 1:
            raise ValueError("threads must be positive")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.method not in {"standard", "adaptive_ml"}:
            raise ValueError("method must be 'standard' or 'adaptive_ml'")


@dataclass(frozen=True, slots=True)
class SndBendersRunPaths:
    """All files belonging to one external application run."""

    log_directory: Path
    settings: Path
    invocation: Path
    solver_log: Path
    adapter_result: Path
    summary: Path


@dataclass(frozen=True, slots=True)
class SndBendersRun:
    """Prepared external command and its output contract."""

    command: tuple[str, ...]
    paths: SndBendersRunPaths
    summary: dict[str, Any]


def render_snd_benders_settings(config: SndBendersConfiguration) -> str:
    """Return the minimal upstream settings override for a full Benders solve."""

    if config.method == "standard":
        method_settings = '''[BENDERS.SUBPROBLEM_SCORING]
ordering = "none"

[BENDERS.SUBPROBLEM_SELECTION]
strategy = "none"
score_initialization_enabled = false
stabilization_frequency = 0
root_node_stabilization = 0

[BENDERS.CUT_FILTERING]
strategy = "none"
'''
    else:
        # This is the ML-0.5-5C-20S-5R setting used with the generated
        # instances in the final donkiewicz2026adaptive code base.
        method_settings = '''[BENDERS.SUBPROBLEM_SCORING]
ordering = "score"

[BENDERS.SUBPROBLEM_SCORING.score]
weights = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
scale_score = true

[BENDERS.SUBPROBLEM_SELECTION]
strategy = "static"
score_initialization_enabled = true
stabilization_frequency = 20
root_node_stabilization = 5

[BENDERS.SUBPROBLEM_SELECTION.static]
min_score_threshold = 0.5

[BENDERS.SUBPROBLEM_SELECTION.static.cut_limit]
mode = "absolute"
absolute = 5

[BENDERS.CUT_FILTERING]
strategy = "none"

[BENDERS.ML]
'''
    return f'''[SOLVER]
SOLVER = "Gurobi"
time_limit = {config.time_limit_seconds:.17g}

[MODEL]
model_type = "benders"

[SCENARIOS]
num_outage_scenarios = -1
outage_sampling_seed = {config.seed}
contingency_k = 1

{method_settings}

[LOGGING]
statistics = true
ml_statistics = false
subproblem_log = false
subproblem_log_success = false
print_solution = false
'''


def prepare_snd_benders_run(
    instance: str | Path,
    *,
    run_name: str,
    results_directory: str | Path,
    benders_repository: str | Path,
    adapter: str | Path,
    julia: str = "julia",
    use_manifest_julia: bool = True,
    config: SndBendersConfiguration = SndBendersConfiguration(),
) -> SndBendersRun:
    """Validate inputs and construct the isolated Julia invocation."""

    _validate_component("run_name", run_name)
    instance_path = Path(instance).expanduser().resolve()
    results = Path(results_directory).expanduser().resolve()
    benders = Path(benders_repository).expanduser().resolve()
    adapter_path = Path(adapter).resolve()
    if not instance_path.is_file() or instance_path.suffix.lower() != ".xml":
        raise ValueError(f"generated SNDlib input does not exist: {instance_path}")
    try:
        relative_input = instance_path.relative_to(results)
    except ValueError as error:
        raise ValueError(
            f"SND application inputs must be generated below {results}"
        ) from error
    if "instances" not in relative_input.parts[:-1]:
        raise ValueError("generated SNDlib input must be in an instances directory")
    if not (benders / "Project.toml").is_file() or not (
        benders / "src/BendersNetworkDesign.jl"
    ).is_file():
        raise ValueError(f"not a BendersNetworkDesign.jl checkout: {benders}")
    if not adapter_path.is_file():
        raise ValueError(f"Julia adapter does not exist: {adapter_path}")

    instance_name = instance_path.stem
    _validate_component("instance filename stem", instance_name)
    log_directory = results / run_name / "logs" / instance_name
    paths = SndBendersRunPaths(
        log_directory=log_directory,
        settings=log_directory / "settings.toml",
        invocation=log_directory / "invocation.json",
        solver_log=log_directory / "gurobi.log",
        adapter_result=log_directory / "adapter_result.toml",
        summary=log_directory / "summary.json",
    )
    manifest_julia = _manifest_julia_version(benders) if use_manifest_julia else None
    julia_command = (julia,) if manifest_julia is None else (julia, f"+{manifest_julia}")
    command = julia_command + (
        f"--project={benders}",
        "--startup-file=no",
        "--compiled-modules=existing",
        "--history-file=no",
        str(adapter_path),
        str(instance_path),
        str(paths.settings),
        str(paths.adapter_result),
        str(config.threads),
    )
    summary = {
        "schema_version": 1,
        "application": "BendersNetworkDesign.jl full SND",
        "instance": str(instance_path),
        "instance_sha256": sha256(instance_path.read_bytes()).hexdigest(),
        "run_name": run_name,
        "log_directory": str(log_directory),
        "configuration": asdict(config),
        "wall_clock_limit_seconds": (
            config.time_limit_seconds + config.wall_clock_grace_seconds
        ),
        "contingencies": {
            "level": "N-1",
            "enumeration": "all single-link failures",
            "disconnecting_demand_failures": "dropped",
            "base_case": "represented in the Benders master only",
        },
        "benders_repository": str(benders),
        "benders_network_design_version": _project_version(benders),
        "benders_network_design_git_revision": _git_revision(benders),
        "manifest_julia_version": manifest_julia,
        "command": list(command),
    }
    return SndBendersRun(command, paths, summary)


def execute_snd_benders_run(
    run: SndBendersRun,
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
    config = SndBendersConfiguration(**run.summary["configuration"])
    paths.settings.write_text(render_snd_benders_settings(config), encoding="utf-8")
    paths.invocation.write_text(
        json.dumps(run.summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    started = datetime.now(timezone.utc).isoformat()
    if dry_run:
        summary = run.summary | {"state": "dry_run", "started_at": started}
        _write_summary(paths.summary, summary)
        return summary

    started_clock = time.perf_counter()
    wall_clock_limit = (
        config.time_limit_seconds + config.wall_clock_grace_seconds
    )
    try:
        with paths.solver_log.open("w", encoding="utf-8") as log:
            environment = os.environ.copy()
            environment["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
            returncode = _run_adapter_process(
                run.command,
                # Keep any incidental relative output from the external
                # application inside this run's result tree.
                cwd=paths.log_directory,
                env=environment,
                stdout=log,
                timeout=wall_clock_limit,
            )
    except subprocess.TimeoutExpired:
        summary = run.summary | {
            "state": "wall_time_limit",
            "started_at": started,
            "wall_seconds": time.perf_counter() - started_clock,
            "wall_clock_limit_seconds": wall_clock_limit,
            "error": (
                "Julia adapter exceeded the configured solver time limit plus "
                "wall-clock grace"
            ),
            "solver_log": str(paths.solver_log),
        }
        _write_summary(paths.summary, summary)
        return summary
    except OSError as error:
        summary = run.summary | {
            "state": "adapter_error",
            "started_at": started,
            "wall_seconds": time.perf_counter() - started_clock,
            "error": str(error),
            "solver_log": str(paths.solver_log),
        }
        _write_summary(paths.summary, summary)
        return summary
    wall_seconds = time.perf_counter() - started_clock
    summary = run.summary | {
        "state": "complete" if returncode in (0, None) else "adapter_error",
        "started_at": started,
        "wall_seconds": wall_seconds,
        "adapter_exit_code": returncode,
        "solver_log": str(paths.solver_log),
    }
    if returncode is None:
        summary["adapter_recovery"] = "Complete Julia result recovered after shutdown exceeded ten seconds; process termination requested."
    if returncode in (0, None):
        if not paths.adapter_result.is_file():
            summary["state"] = "adapter_error"
            summary["error"] = "Julia adapter returned success without a result file"
        else:
            try:
                with paths.adapter_result.open("rb") as stream:
                    result = tomllib.load(stream)
                _verify_adapter_time_limit(result, config.time_limit_seconds)
                summary["result"] = result
            except (OSError, tomllib.TOMLDecodeError, ValueError) as error:
                summary["state"] = "adapter_error"
                summary["error"] = f"cannot read adapter result: {error}"
    _write_summary(paths.summary, summary)
    return summary


def _validate_component(label: str, value: str) -> None:
    if not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(f"{label} must be a single safe filename component")


def _write_summary(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_adapter_process(
    command: tuple[str, ...],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout: Any,
    timeout: float,
) -> int | None:
    """Run Julia in its own process group and stop all descendants on timeout."""

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    started = time.monotonic()
    last_cpu = None
    last_activity = started
    try:
        while True:
            result_path = cwd / "adapter_result.toml"
            if result_path.is_file() and time.time() - result_path.stat().st_mtime >= 10:
                try:
                    result = tomllib.loads(result_path.read_text())
                except (OSError, ValueError):
                    result = {}
                if "termination_status" in result and result.get("time_limit_verified") is True:
                    code = process.poll()
                    if code is not None:
                        return code
                    _stop_process_group(process)
                    return None
            elapsed = time.monotonic() - started
            try:
                fields = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()
                cpu = fields[11:13]
                if cpu != last_cpu:
                    last_cpu, last_activity = cpu, started + elapsed
            except OSError:
                pass
            if started + elapsed - last_activity >= 900 and os.fstat(stdout.fileno()).st_size == 0:
                raise TimeoutError("Julia startup produced no output or CPU activity for 900 seconds")
            if elapsed >= timeout:
                raise subprocess.TimeoutExpired(command, elapsed)
            try:
                return process.wait(timeout=min(2, timeout - elapsed))
            except subprocess.TimeoutExpired:
                pass
    except (subprocess.TimeoutExpired, TimeoutError):
        _stop_process_group(process)
        raise


def _stop_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:  # pragma: no cover - the evaluation host is POSIX
        process.terminate()
    try:
        process.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    else:  # pragma: no cover - the evaluation host is POSIX
        process.kill()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        # A task blocked in kernel I/O may not exit even after SIGKILL.
        pass


def _verify_adapter_time_limit(
    result: dict[str, Any], requested_seconds: float
) -> None:
    for field in (
        "requested_time_limit_seconds",
        "effective_time_limit_seconds",
    ):
        value = result.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or not isclose(
                float(value),
                requested_seconds,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        ):
            raise ValueError(
                f"adapter {field}={value!r} does not match requested "
                f"{requested_seconds}"
            )
    if result.get("time_limit_verified") is not True:
        raise ValueError("adapter did not verify the master-model time limit")


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
