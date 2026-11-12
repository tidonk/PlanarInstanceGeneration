#!/usr/bin/env python3
"""Write the HTCondor job tables for the main generation experiment.

Each dataset gets one whitespace-separated table with one line per requested
composite holding ``instance seed``. Each job runs every configured merge-node
strategy together so all strategies reuse one extracted subinstance cohort.

Every instance number ``i`` (one-based) is paired with the seed
``base_seed + i``. The job passes all configured strategies to one generation
process, which samples and extracts once per attempt before running any
strategy-specific merge work.
"""

from __future__ import annotations

import argparse
from pathlib import Path


STRATEGIES = (
    "random",
    "high_degree",
    "kamada_kawai",
    "heuristic_outer_face",
    "exact_outer_face",
)
DATASETS = ("pglib", "snd")
EVAL_DIRECTORY = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="main")
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=STRATEGIES,
        default=list(STRATEGIES),
    )
    parser.add_argument("--output-dir", type=Path, default=EVAL_DIRECTORY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.instances < 1:
        raise SystemExit("--instances must be positive")
    if args.base_seed < 0:
        raise SystemExit("--base-seed must be nonnegative")
    if len(set(args.methods)) != len(args.methods):
        raise SystemExit("--methods must not contain duplicates")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    methods = [method for method in STRATEGIES if method in args.methods]
    for dataset in args.datasets:
        # The job table (instance seed) is experiment-independent, and
        # HTCondor does not expand $(...) inside a `queue ... from` path, so the
        # submit files reference a fixed per-dataset filename.
        path = args.output_dir / f"{dataset}.jobs"
        lines = [
            f"{instance} {args.base_seed + instance}"
            for instance in range(1, args.instances + 1)
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        # Condor does not create missing directories for output/error/log.
        log_directory = args.output_dir / "logs" / args.experiment / dataset
        log_directory.mkdir(
            parents=True, exist_ok=True
        )
        if dataset == "pglib":
            (log_directory / "skewness").mkdir(exist_ok=True)
        print(
            f"{dataset}: wrote {len(lines)} jobs "
            f"({args.instances} paired instances x {len(methods)} methods) to {path}"
        )


if __name__ == "__main__":
    main()
