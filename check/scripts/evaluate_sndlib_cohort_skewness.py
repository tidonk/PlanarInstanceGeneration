#!/usr/bin/env python3
"""Evaluate one generated SNDlib composite with shared constituent certificates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from planar_instance_generation.evaluation import load_generated_snd_artifacts
from planar_instance_generation.evaluation.planarization_batch import (
    planarization_result_row as _result_row,
    PlanarizationTask,
    solve_partitioned_batch,
    source_mappings,
)

RESULTS_DIRECTORY = REPOSITORY_ROOT / "check/results"
STRATEGIES = (
    "random",
    "high_degree",
    "kamada_kawai",
    "heuristic_outer_face",
    "exact_outer_face",
)
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("cohort", type=int)
    parser.add_argument("strategy", choices=STRATEGIES)
    parser.add_argument(
        "--output-name",
        default=None,
        help="checkpoint result directory (default: <experiment>_snd_exact)",
    )
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_name = args.output_name or f"{args.experiment}_snd_exact"
    if not _SAFE_NAME.fullmatch(args.experiment) or not _SAFE_NAME.fullmatch(
        output_name
    ):
        raise SystemExit("experiment and output names must be safe path components")
    if args.cohort < 1 or args.seed < 0:
        raise SystemExit("cohort must be positive and seed nonnegative")
    if args.time_limit <= 0:
        raise SystemExit("invalid exact-solver time settings")

    run = _cohort_directory(args.experiment, args.cohort)
    generated = load_generated_snd_artifacts(run)
    matching = [final for final in generated.finals if final.strategy == args.strategy]
    if len(matching) != 1:
        raise SystemExit(
            f"i{args.cohort:03d} must contain one {args.strategy} composite; "
            f"found {len(matching)}"
        )
    final = matching[0]
    extracted = {artifact.stem: artifact for artifact in generated.extracted}

    output = RESULTS_DIRECTORY / output_name
    checkpoint_directory = (
        output / "evaluation/cohort_checkpoints" / f"i{args.cohort:03d}"
    )
    solver_logs = output / "logs/skewness" / f"i{args.cohort:03d}"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    solver_logs.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_directory / f"{args.strategy}.json"
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
        cached = _read_json(checkpoint)
        if cached.get("settings") != settings:
            raise SystemExit(f"checkpoint exists with different settings: {checkpoint}")
        print(f"Reusing checkpoint: {checkpoint.relative_to(REPOSITORY_ROOT)}")
        return

    source_keys = [f"source:{args.cohort}:{stem}" for stem in final.source_stems]
    final_key = f"final:{args.cohort}:{final.stem}"
    keys = [*source_keys, final_key]
    tasks = [
        PlanarizationTask(
            graph=extracted[stem].graph,
            seed=args.seed,
            log_path=solver_logs / f"{_slug(stem)}.log",
            cache_key=_slug(key),
        )
        for stem, key in zip(final.source_stems, source_keys, strict=True)
    ]
    tasks.append(
        PlanarizationTask(
            graph=final.graph,
            seed=args.seed,
            log_path=solver_logs / f"{_slug(final.stem)}.log",
            cache_key=_slug(final_key),
        )
    )
    dependencies = {
        _slug(final_key): [
            (_slug(f"source:{args.cohort}:{stem}"), mapping)
            for stem, mapping in source_mappings(final.metadata)
        ]
    }
    solved = solve_partitioned_batch(
        tasks,
        dependencies,
        generation_metadata=[
            (
                item.metadata,
                [_slug(f"source:{args.cohort}:{stem}") for stem in item.source_stems],
            )
            for item in generated.finals
        ],
        time_limit_seconds=float(args.time_limit),
        solver_threads=args.threads,
        workers=1,
        checkpoint_directory=output / "evaluation/shared_piece_checkpoints",
    )
    names = [*final.source_stems, final.stem]
    graph_results = {
        key: {"instance": name, **_result_row(result)}
        for key, name, result in zip(keys, names, solved, strict=True)
    }
    payload = {
        "schema_version": 1,
        "experiment": args.experiment,
        "cohort": args.cohort,
        "settings": settings,
        "strategy": args.strategy,
        "graph_results": graph_results,
    }
    temporary = checkpoint.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(checkpoint)
    print(f"Checkpoint: {checkpoint.relative_to(REPOSITORY_ROOT)}", flush=True)


def _cohort_directory(experiment: str, cohort: int) -> Path:
    name = f"{experiment}_snd_job_i{cohort:03d}"
    candidates = (
        RESULTS_DIRECTORY / name,
        RESULTS_DIRECTORY / "_cohorts" / f"{experiment}_snd" / name,
    )
    found = [path for path in candidates if path.is_dir()]
    if len(found) != 1:
        raise SystemExit(
            f"expected one active or archived SND cohort directory for i{cohort:03d}; "
            f"found {len(found)}"
        )
    return found[0]


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read checkpoint {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"checkpoint must contain a JSON object: {path}")
    return payload


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


if __name__ == "__main__":
    main()
