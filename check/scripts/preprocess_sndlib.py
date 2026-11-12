#!/usr/bin/env python3
"""Build the SNDlib source pool with constructive capacities and exact MCMCF."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
import re
import sys
from time import perf_counter
import traceback


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from planar_instance_generation.evaluation.snd_source_pool import (
    NamedSndSource,
    screen_snd_source_cases,
)
from planar_instance_generation.evaluation.snd_source_pool_io import (
    write_snd_source_solution,
)
from planar_instance_generation.gates.mcmcf import (
    DEFAULT_MAX_VARIABLES,
    GurobiMcmcfGate,
)
from planar_instance_generation.io.sndlib import read_sndlib


SNDLIB_DIRECTORY = REPOSITORY_ROOT / "check/data/sndlib"
RESULTS_DIRECTORY = REPOSITORY_ROOT / "check/results"
EXPECTED_RAW_SOURCE_COUNT = 26
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sources",
        nargs="*",
        type=Path,
        help="raw SNDlib XML files; default: all 26 files in check/data/sndlib",
    )
    parser.add_argument("--max-variables", type=int, default=DEFAULT_MAX_VARIABLES)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--default-routing-cost", type=float, default=1.0)
    parser.add_argument(
        "--exclude-euler-bound-violations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "exclude simple topologies with |E| > 3|V|-6 (default: enabled; "
            "use --no-exclude-euler-bound-violations to retain them)"
        ),
    )
    parser.add_argument("--result-name", default="sndlib_source_pool")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    paths = _input_paths(args.sources)
    networks = tuple(
        NamedSndSource(source_name=path.stem, network=read_sndlib(path))
        for path in paths
    )

    run_directory = RESULTS_DIRECTORY / args.result_name
    if run_directory.exists():
        raise SystemExit(
            f"result run already exists: {run_directory}; choose another --result-name"
        )
    solution_directory = run_directory / "solutions"
    log_directory = run_directory / "logs"
    solution_directory.mkdir(parents=True)
    log_directory.mkdir()

    started_at = datetime.now(timezone.utc)
    source_entries = [
        {
            "source_name": path.stem,
            "path": _display_path(path),
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    ]
    config = {
        "schema_version": 1,
        "result_name": args.result_name,
        "started_at_utc": started_at.isoformat(),
        "command": [str(Path(sys.argv[0]).resolve()), *sys.argv[1:]],
        "source_selection": "explicit" if args.sources else "all_raw_sndlib",
        "source_count": len(paths),
        "max_variables": args.max_variables,
        "time_limit": args.time_limit,
        "threads": args.threads,
        "seed": args.seed,
        "default_routing_cost": args.default_routing_cost,
        "exclude_euler_bound_violations": args.exclude_euler_bound_violations,
        "sources": source_entries,
    }
    config_path = run_directory / "config.json"
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    process_log = log_directory / "preprocess.log"
    process_log.write_text(
        f"started_at_utc={started_at.isoformat()}\n"
        f"sources={len(paths)}\n"
        f"max_variables={args.max_variables}\n",
        encoding="utf-8",
    )

    gate = GurobiMcmcfGate(
        max_variables=args.max_variables,
        time_limit=args.time_limit,
        threads=args.threads,
        seed=args.seed,
        default_routing_cost=args.default_routing_cost,
        verbose=args.verbose,
        log_directory=log_directory,
    )
    wall_started = perf_counter()
    try:
        pool = screen_snd_source_cases(
            networks,
            gate,
            exclude_euler_bound_violations=args.exclude_euler_bound_violations,
        )
    except BaseException:
        with process_log.open("a", encoding="utf-8") as stream:
            stream.write("status=FAILED\n")
            stream.write(traceback.format_exc())
        raise
    wall_seconds = perf_counter() - wall_started
    if pool.attempted_count != len(paths):
        raise RuntimeError("source screening did not record every input")

    solution_paths: dict[str, str] = {}
    for source in pool.sources:
        output = solution_directory / f"{source.source_name}.json"
        write_snd_source_solution(output, source)
        solution_paths[source.source_name] = _display_path(output)

    record_rows = [_record_payload(record) for record in pool.records]
    records_path = run_directory / "source_screening.csv"
    _write_records(records_path, record_rows)
    status_counts: dict[str, int] = {}
    for record in pool.records:
        status_counts[record.status] = status_counts.get(record.status, 0) + 1
    finished_at = datetime.now(timezone.utc)
    manifest = {
        "schema_version": 1,
        "artifact_type": "sndlib_source_pool",
        "run": {
            **config,
            "config_path": _display_path(config_path),
            "process_log": _display_path(process_log),
            "finished_at_utc": finished_at.isoformat(),
            "wall_seconds": wall_seconds,
        },
        "backend": {
            "name": "gurobi-fixed-capacity-mcmcf",
            "gurobipy_version": _package_version("gurobipy"),
            "python_version": platform.python_version(),
            "formulation": "continuous signed flow plus absolute-flow epigraph",
            "link_semantics": "undirected; signed relative to XML orientation",
            "shared_capacity": "sum of absolute commodity flows",
            "capacities": "fixed constructive installed capacities",
            "constructive_method": "deterministic_shortest_path_module_rounding",
            "constructive_design_optimal": False,
        },
        "attempted": pool.attempted_count,
        "retained": pool.feasible_count,
        "excluded_euler_bound_violations": pool.euler_excluded_count,
        "resource_skipped": pool.resource_skipped_count,
        "unsupported_routing_restrictions": pool.unsupported_count,
        "status_counts": dict(sorted(status_counts.items())),
        "total_constructive_seconds": pool.total_constructive_seconds,
        "total_mcmcf_wall_seconds": pool.total_mcmcf_seconds,
        "total_source_preprocessing_seconds": pool.total_preprocessing_seconds,
        "records": record_rows,
        "solutions": solution_paths,
        "records_csv": _display_path(records_path),
    }
    manifest_path = run_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with process_log.open("a", encoding="utf-8") as stream:
        stream.write("status=COMPLETED\n")
        stream.write(f"finished_at_utc={finished_at.isoformat()}\n")
        stream.write(f"wall_seconds={wall_seconds:.9f}\n")
        stream.write(f"attempted={pool.attempted_count}\n")
        stream.write(f"retained={pool.feasible_count}\n")
        stream.write(f"euler_excluded={pool.euler_excluded_count}\n")
        stream.write(f"resource_skipped={pool.resource_skipped_count}\n")
        stream.write(f"unsupported={pool.unsupported_count}\n")
    print(
        f"Retained {pool.feasible_count}/{pool.attempted_count} SNDlib sources; "
        f"Euler-excluded {pool.euler_excluded_count}, "
        f"resource-skipped {pool.resource_skipped_count}, "
        f"unsupported {pool.unsupported_count}."
    )
    print(f"Results: {_display_path(run_directory)}")


def _validate_args(args: argparse.Namespace) -> None:
    if not _SAFE_NAME.fullmatch(args.result_name):
        raise SystemExit("--result-name must be a safe filename component")
    if args.max_variables < 1:
        raise SystemExit("--max-variables must be positive")
    if args.threads < 1:
        raise SystemExit("--threads must be positive")
    if args.seed < 0:
        raise SystemExit("--seed must be nonnegative")
    if args.time_limit is not None and (
        not math.isfinite(args.time_limit) or args.time_limit <= 0.0
    ):
        raise SystemExit("--time-limit must be positive")
    if not math.isfinite(args.default_routing_cost) or args.default_routing_cost < 0.0:
        raise SystemExit("--default-routing-cost must be nonnegative")


def _input_paths(arguments: list[Path]) -> tuple[Path, ...]:
    candidates = arguments or sorted(SNDLIB_DIRECTORY.glob("*.xml"))
    paths: list[Path] = []
    for candidate in candidates:
        path = (
            SNDLIB_DIRECTORY / candidate
            if len(candidate.parts) == 1
            else candidate
        ).resolve()
        if not path.is_file():
            raise SystemExit(f"SNDlib input does not exist: {path}")
        if path.parent != SNDLIB_DIRECTORY.resolve() or path.suffix.lower() != ".xml":
            raise SystemExit(
                f"SND preprocessing accepts only raw XML files in {SNDLIB_DIRECTORY}: {path}"
            )
        if not _SAFE_NAME.fullmatch(path.stem):
            raise SystemExit(f"SNDlib source filename has an unsafe stem: {path.name}")
        paths.append(path)
    if not paths:
        raise SystemExit(f"no SNDlib XML files found below {SNDLIB_DIRECTORY}")
    if len(set(paths)) != len(paths) or len({path.stem for path in paths}) != len(paths):
        raise SystemExit("SNDlib inputs and source names must be unique")
    if not arguments and len(paths) != EXPECTED_RAW_SOURCE_COUNT:
        raise SystemExit(
            f"expected all {EXPECTED_RAW_SOURCE_COUNT} raw SNDlib sources, found {len(paths)}"
        )
    return tuple(paths)


def _record_payload(record: object) -> dict[str, object]:
    payload = asdict(record)
    if payload["log_path"] is not None:
        payload["log_path"] = _display_path(Path(str(payload["log_path"])))
    payload["restricted_demand_ids"] = list(payload["restricted_demand_ids"])
    return payload


def _write_records(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty source-screening table")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {
                key: (
                    json.dumps(value, sort_keys=True)
                    if isinstance(value, (list, tuple, dict))
                    else value
                )
                for key, value in row.items()
            }
            for row in rows
        )


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
