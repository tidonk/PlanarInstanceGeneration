#!/usr/bin/env python3
"""Evaluate a PGLib composite using shared, independently solved pieces.

Workers for the configured strategies reuse locked piece checkpoints, then apply
the same piece constraints and warm-start construction to their final graph.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import build_partial_pglib_report as partial
from planar_instance_generation.io import read_pglib
from planar_instance_generation.evaluation.planarization_batch import (
    planarization_result_row as _result_row,
    PlanarizationTask,
    solve_partitioned_batch,
    source_mappings,
)

RESULTS_DIRECTORY = REPOSITORY_ROOT / "check/results"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("cohort", type=int)
    parser.add_argument("strategy", choices=partial.STRATEGIES)
    parser.add_argument(
        "--output-name",
        default=None,
        help="checkpoint result directory (default: <experiment>_pglib_exact)",
    )
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_name = args.output_name or f"{args.experiment}_pglib_exact"
    if not _SAFE_NAME.fullmatch(args.experiment) or not _SAFE_NAME.fullmatch(
        output_name
    ):
        raise SystemExit("experiment and output names must be safe path components")
    if args.cohort < 1:
        raise SystemExit("cohort must be positive")
    if args.time_limit <= 0:
        raise SystemExit("invalid exact-solver time settings")

    selected = [
        cohort
        for cohort in partial._discover_cohorts(
            args.experiment, index=args.cohort, use_configured_strategies=True
        )
        if cohort.index == args.cohort
    ]
    if not selected:
        raise SystemExit(f"i{args.cohort:03d} is incomplete for its configured methods")
    cohort = selected[0]
    selected_rows = [row for row in cohort.rows if row["strategy"] == args.strategy]
    if len(selected_rows) != 1:
        raise SystemExit(f"requested method {args.strategy} is not in the saved input")

    output = RESULTS_DIRECTORY / output_name
    checkpoints = output / "evaluation/cohort_checkpoints" / f"i{cohort.index:03d}"
    solver_logs = output / "logs/skewness" / f"i{cohort.index:03d}"
    checkpoints.mkdir(parents=True, exist_ok=True)
    solver_logs.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoints / f"{args.strategy}.json"
    from planar_instance_generation.core.ogdf import build_identity

    settings = {
        "backend": build_identity(),
        "mode": "exact_branch_and_cut",
        "time_limit_seconds": float(args.time_limit),
        "heuristic_runs": 10,
        "solver_threads": args.threads,
        "seed": int(args.seed),
        "shared_piece_policy": 5,
    }
    if checkpoint.is_file() and not args.overwrite:
        try:
            cached = json.loads(checkpoint.read_text(encoding="utf-8"))
            if cached.get("settings") != settings:
                raise SystemExit(
                    f"checkpoint exists with different settings: {checkpoint}"
                )
        except (OSError, json.JSONDecodeError):
            pass

    all_specs = partial._task_specs((cohort,), 10, args.seed)
    final_stem = selected_rows[0]["artifact_stem"]
    final_key = f"final:{cohort.index}:{final_stem}"
    source_keys = [f"source:{cohort.index}:{stem}" for stem in cohort.source_stems]
    keys = [*source_keys, final_key]
    tasks = [
        PlanarizationTask(
            read_pglib(Path(all_specs[key][0])).topology(active_only=True),
            seed=all_specs[key][2],
            log_path=solver_logs / f"{_slug(key)}.log",
            cache_key=_slug(key),
        )
        for key in keys
    ]
    metadata_path = cohort.directory / "plots/composite" / final_stem / "metadata.json"
    if not metadata_path.is_file():
        metadata_path = cohort.directory / "plots" / final_stem / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dependencies = {
        _slug(final_key): [
            (_slug(f"source:{cohort.index}:{stem}"), mapping)
            for stem, mapping in source_mappings(metadata)
        ]
    }
    print(
        f"i{cohort.index:03d} {args.strategy}: solving {len(tasks)} "
        f"graph(s), OGDF with a {args.threads}-thread heuristic cap.",
        flush=True,
    )
    generation_metadata = []
    for row in cohort.rows:
        path = (
            cohort.directory
            / "plots/composite"
            / row["artifact_stem"]
            / "metadata.json"
        )
        record = json.loads(path.read_text(encoding="utf-8"))
        generation_metadata.append(
            (
                record,
                [
                    _slug(f"source:{cohort.index}:{stem}")
                    for stem, _mapping in source_mappings(record)
                ],
            )
        )
    solved = solve_partitioned_batch(
        tasks,
        dependencies,
        generation_metadata=generation_metadata,
        time_limit_seconds=float(args.time_limit),
        solver_threads=args.threads,
        workers=1,
        checkpoint_directory=output / "evaluation/shared_piece_checkpoints",
    )
    results = {
        key: _result_row(result) for key, result in zip(keys, solved, strict=True)
    }

    payload = {
        "schema_version": 1,
        "experiment": args.experiment,
        "cohort": cohort.index,
        "settings": settings,
        "strategy": args.strategy,
        "graph_results": results,
    }
    temporary = checkpoint.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(checkpoint)
    print(f"Checkpoint: {checkpoint.relative_to(REPOSITORY_ROOT)}", flush=True)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


if __name__ == "__main__":
    main()
