#!/usr/bin/env python3
"""Build the feasible PGLib source pool with strict base-case DC-OPF."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import re
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from planar_instance_generation.evaluation import (
    screen_source_cases,
    select_cases_by_bus_limit,
)
from planar_instance_generation.gates import InnoptemDcOpfGate
from planar_instance_generation.io import read_pglib


PGLIB_DIRECTORY = REPOSITORY_ROOT / "check/data/pglib"
RESULTS_DIRECTORY = REPOSITORY_ROOT / "check/results"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sources",
        nargs="*",
        type=Path,
        help="PGLib files; default: every .m file below check/data/pglib",
    )
    parser.add_argument("--max-buses", type=int, default=2_000)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--exclude-euler-bound-violations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "exclude simple topologies with |E| > 3|V|-6 (default: enabled; "
            "use --no-exclude-euler-bound-violations to retain them)"
        ),
    )
    parser.add_argument("--result-name", default="pglib_source_pool_innoptem")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not _SAFE_NAME.fullmatch(args.result_name):
        raise SystemExit("--result-name must be a single safe filename component")
    if args.threads < 1:
        raise SystemExit("--threads must be positive")

    paths = _input_paths(args.sources)
    cases = tuple(read_pglib(path) for path in paths)
    limited = select_cases_by_bus_limit(cases, max_buses=args.max_buses)
    run_directory = RESULTS_DIRECTORY / args.result_name
    if run_directory.exists() and any(run_directory.iterdir()):
        raise SystemExit(
            f"result run already exists: {run_directory}; choose another --result-name"
        )
    solution_directory = run_directory / "solutions"
    log_directory = run_directory / "logs"
    solution_directory.mkdir(parents=True, exist_ok=True)
    log_directory.mkdir(parents=True, exist_ok=True)

    gate = InnoptemDcOpfGate(
        time_limit=args.time_limit,
        solver_params={"Threads": args.threads},
        verbose=args.verbose,
        log_directory=log_directory,
    )
    pool = screen_source_cases(
        limited.included,
        gate,
        exclude_euler_bound_violations=args.exclude_euler_bound_violations,
    )

    solution_paths: dict[str, str] = {}
    for source in pool.sources:
        output = solution_directory / f"{source.case.name}.npz"
        np.savez_compressed(
            output,
            bus_angles=source.solution.bus_angles,
            branch_from_flows=source.solution.branch_from_flows,
            branch_to_flows=source.solution.branch_to_flows,
            generator_dispatch=source.solution.generator_dispatch,
        )
        solution_paths[source.case.name] = _display_path(output)

    records_path = run_directory / "source_screening.csv"
    _write_records(records_path, pool.records)
    manifest = {
        "schema_version": 1,
        "run": {
            "max_buses": args.max_buses,
            "time_limit": args.time_limit,
            "threads": args.threads,
            "exclude_euler_bound_violations": (
                args.exclude_euler_bound_violations
            ),
            "sources": [
                {
                    "path": _display_path(path),
                    "sha256": sha256(path.read_bytes()).hexdigest(),
                }
                for path in paths
            ],
        },
        "backend": gate.identity(),
        "attempted": pool.attempted_count,
        "feasible": pool.feasible_count,
        "excluded_euler_bound_violations": pool.euler_excluded_count,
        "total_dc_opf_wall_seconds": pool.total_dc_opf_seconds,
        "excluded_over_bus_limit": [
            {"name": case.name, "buses": int(case.bus.shape[0])}
            for case in limited.excluded
        ],
        "records": [asdict(record) for record in pool.records],
        "solutions": solution_paths,
        "records_csv": _display_path(records_path),
    }
    manifest_path = run_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Retained {pool.feasible_count}/{pool.attempted_count} cases; "
        f"Euler-excluded {pool.euler_excluded_count}; excluded "
        f"{len(limited.excluded)} cases above {args.max_buses} buses."
    )
    print(f"Results: {_display_path(run_directory)}")


def _input_paths(arguments: list[Path]) -> tuple[Path, ...]:
    candidates = arguments or sorted(PGLIB_DIRECTORY.glob("*.m"))
    resolved = []
    for candidate in candidates:
        path = (PGLIB_DIRECTORY / candidate if len(candidate.parts) == 1 else candidate).resolve()
        if not path.is_file():
            raise SystemExit(f"PGLib input does not exist: {path}")
        resolved.append(path)
    if not resolved:
        raise SystemExit(f"no PGLib files found below {PGLIB_DIRECTORY}")
    return tuple(resolved)


def _write_records(path: Path, records: tuple[object, ...]) -> None:
    rows = [asdict(record) for record in records]
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    main()
