#!/usr/bin/env python3
"""Run DC-SCOPF/SCOTS through TopologyOptimizer's Benders application."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from planar_instance_generation.evaluation.scopf_application import (
    ScopfBendersConfiguration,
    execute_scopf_benders_run,
    prepare_scopf_benders_run,
)


RESULTS_DIRECTORY = REPOSITORY_ROOT / "check/results"
DEFAULT_INNOPTEM_REPOSITORY = (
    REPOSITORY_ROOT / "build/TopologyOptimizer"
)
ADAPTER = Path(__file__).with_name("run_scopf_benders.jl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path, help="generated PGLib case below check/results")
    parser.add_argument("--run", required=True, help="result directory name below check/results")
    parser.add_argument("--topology-repository", "--innoptem-repository", dest="innoptem_repository", type=Path, default=DEFAULT_INNOPTEM_REPOSITORY)
    parser.add_argument("--julia", default="julia", help="Juliaup launcher or Julia executable")
    julia_selection = parser.add_mutually_exclusive_group()
    julia_selection.add_argument(
        "--ignore-manifest-julia",
        action="store_true",
        help="do not pass +<julia_version> from the TopologyOptimizer Manifest.toml",
    )
    julia_selection.add_argument(
        "--julia-channel",
        help="explicit Juliaup version, overriding the external manifest version",
    )
    parser.add_argument("--time-limit", type=float, default=10_800.0, help="Solver seconds; 0 disables the solver time limit")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topological-action-limit", type=int, default=0)
    parser.add_argument("--scopf-warmstart", action="store_true")
    parser.add_argument("--warmstart-time-limit", type=float, default=10800.0)
    parser.add_argument("--benders-settings", type=str)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run = prepare_scopf_benders_run(
            args.instance,
            run_name=args.run,
            results_directory=RESULTS_DIRECTORY,
            innoptem_repository=args.innoptem_repository,
            adapter=ADAPTER,
            julia=args.julia,
            use_manifest_julia=not args.ignore_manifest_julia,
            julia_channel=args.julia_channel,
            config=ScopfBendersConfiguration(
                time_limit_seconds=args.time_limit,
                threads=args.threads,
                seed=args.seed,
                topological_action_limit=args.topological_action_limit,
                scopf_warmstart=args.scopf_warmstart,
                warmstart_time_limit_seconds=args.warmstart_time_limit,
                benders_settings=args.benders_settings,
            ),
        )
        summary = execute_scopf_benders_run(run, dry_run=args.dry_run)
    except (FileExistsError, OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"State: {summary['state']}")
    print(f"Summary: {run.paths.summary}")
    if summary["state"] == "adapter_error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
