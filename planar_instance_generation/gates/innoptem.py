"""Base-case OPF through the existing TopologyOptimizer DC_OPF application entry."""

from __future__ import annotations

import atexit
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
from time import perf_counter
import tomllib

import numpy as np

from .base import DcOpfResult, DcOpfSolution
from ..io.pglib import write_pglib

ROOT = Path(__file__).resolve().parents[2]
BACKEND = "TopologyOptimizer DC_OPF"
_WORKERS = {}


class _Worker:
    def __init__(self, repository, julia, environment):
        self.directory = tempfile.TemporaryDirectory(prefix="pig-opf-")
        self.path = Path(self.directory.name)
        self.log = (self.path / "worker.log").open("w")
        version = tomllib.loads((repository / "Manifest.toml").read_text())[
            "julia_version"
        ]
        command = [julia]
        if Path(julia).resolve().name == "julialauncher":
            installed = (
                Path(julia).absolute().parents[2]
                / ".julia/juliaup"
                / f"julia-{version}+0.x64.linux.gnu/bin/julia"
            )
            command = (
                [str(installed)] if installed.is_file() else [julia, "+" + version]
            )
        command += [
            "--startup-file=no",
            "--compiled-modules=existing",
            "--threads=1",
            "--project=" + str(repository),
            str(Path(__file__).with_name("innoptem_opf.jl")),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
            env=environment,
        )
        self.messages = queue.Queue()
        self.lock = threading.Lock()

        def read():
            for line in self.process.stdout:
                if line.startswith("PIG_OPF "):
                    self.messages.put(json.loads(line[len("PIG_OPF ") :]))
                else:
                    self.log.write(line)
                    self.log.flush()
            self.messages.put({"error": "OPF process exited"})

        threading.Thread(target=read, daemon=True).start()
        self.versions = self.receive(1800)
        if not self.versions.get("ready"):
            raise RuntimeError(
                f'OPF worker failed to initialize: {self.path / "worker.log"}'
            )
        atexit.register(self.close)

    def receive(self, timeout):
        try:
            message = self.messages.get(timeout=timeout)
        except queue.Empty as error:
            self.process.kill()
            self.process.wait()
            raise RuntimeError(
                f'OPF worker exceeded its process allowance: {self.path / "worker.log"}'
            ) from error
        if "error" in message:
            self.log.flush()
            detail = (self.path / "worker.log").read_text()[-8000:]
            raise RuntimeError(f'{message["error"]}\n{detail}')
        return message

    def solve(self, case, options):
        with self.lock:
            source, output = self.path / "case.m", self.path / "result.json"
            write_pglib(case, source)
            output.unlink(missing_ok=True)
            self.process.stdin.write(
                json.dumps(
                    dict(input=str(source), output=str(output), solver_params=options)
                )
                + "\n"
            )
            self.process.stdin.flush()
            self.receive(float(options.get("TimeLimit", 86400)) + 300)
            return json.loads(output.read_text())

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"stop": true}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait()
        self.log.close()
        self.directory.cleanup()


class InnoptemDcOpfGate:
    """Strict fixed-topology OPF, with one Julia worker reused across cases."""

    def __init__(
        self,
        *,
        time_limit=None,
        solver_params=None,
        verbose=False,
        log_directory=None,
        log_path=None,
        repository=None,
        julia=None,
    ):
        if time_limit is not None and (
            isinstance(time_limit, bool)
            or not math.isfinite(time_limit)
            or time_limit <= 0
        ):
            raise ValueError("time_limit must be finite and positive")
        if log_directory is not None and log_path is not None:
            raise ValueError("provide log_directory or log_path, not both")
        self.repository = Path(
            repository or os.environ.get("PIG_OPF_REPOSITORY", ROOT / "build/TopologyOptimizer")
        ).resolve()
        self.julia = str(
            julia or os.environ.get("PIG_JULIA", "julia")
        )
        self.options = {
            "Threads": 1,
            "OutputFlag": int(verbose),
            **dict(solver_params or {}),
        }
        if time_limit is not None:
            self.options["TimeLimit"] = float(time_limit)
        self.log_directory = None if log_directory is None else Path(log_directory)
        self.log_path = None if log_path is None else Path(log_path)

    def identity(self):
        files = [
            self.repository / p
            for p in ("src/util/util.jl", "Manifest.toml", "Project.toml")
        ]
        files += [Path(__file__), Path(__file__).with_name("innoptem_opf.jl")]
        record = self.repository / "source.json"
        revision = json.loads(record.read_text()) if record.is_file() else {}
        return dict(
            name=BACKEND,
            model="base-case DC-OPF",
            branch_switching=False,
            source=revision,
            source_sha256={
                str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
            },
        )

    def solve(self, case, /):
        started = perf_counter()
        case.validate()
        if any(row[10] > 0 and row[3] == 0 for row in case.branch):
            return DcOpfResult(
                False, "UNSUPPORTED_DC_ZERO_REACTANCE", None, perf_counter() - started
            )
        environment = os.environ.copy()
        environment.update(
            JULIA_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"
        )
        if os.environ.get("PIG_OPF_JULIA_DEPOT"):
            environment["JULIA_DEPOT_PATH"] = os.environ["PIG_OPF_JULIA_DEPOT"]
        key = (
            os.getpid(),
            str(self.repository),
            self.julia,
            environment.get("JULIA_DEPOT_PATH"),
        )
        if key not in _WORKERS:
            _WORKERS[key] = _Worker(self.repository, self.julia, environment)
        worker = _WORKERS[key]
        options = dict(self.options)
        logfile = (
            self.log_path
            if self.log_directory is None
            else self.log_directory / (case.name + ".log")
        )
        if logfile is not None:
            logfile.parent.mkdir(parents=True, exist_ok=True)
            options.update(LogFile=str(logfile.resolve()), OutputFlag=1, LogToConsole=0)
        raw = worker.solve(case, options)
        seconds = perf_counter() - started
        if logfile is not None:
            logfile.with_suffix(".result.json").write_text(
                json.dumps(
                    dict(raw, backend=self.identity(), versions=worker.versions),
                    indent=2,
                )
                + "\n"
            )
        if not raw["feasible"]:
            return DcOpfResult(
                False,
                raw["status"],
                None,
                seconds,
                optimizer_seconds=raw.get("optimizer_seconds"),
            )
        native = raw["solution"]

        def value(table, index, field, active=True):
            if not active:
                return 0.0
            try:
                return float(native[table][str(index)][field])
            except (KeyError, TypeError) as error:
                raise RuntimeError(
                    f"OPF solution is missing {table}[{index}].{field}"
                ) from error

        # PowerModels returns radians and per-unit power. Existing extraction
        # consumes MATPOWER degrees and MW, with the original branch row order.
        flows = np.asarray(
            [
                value("branch", i + 1, "pf", row[10] > 0) * case.base_mva
                for i, row in enumerate(case.branch)
            ]
        )
        solution = DcOpfSolution(
            np.asarray(
                [
                    math.degrees(value("bus", int(row[0]), "va", row[1] != 4))
                    for row in case.bus
                ]
            ),
            flows,
            -flows,
            np.asarray(
                [
                    value("gen", i + 1, "pg", row[7] > 0) * case.base_mva
                    for i, row in enumerate(case.gen)
                ]
            ),
        )
        solution.validate_for(case)
        return DcOpfResult(
            True,
            raw["status"],
            float(raw["objective"]),
            seconds,
            solution,
            raw.get("optimizer_seconds"),
        )
