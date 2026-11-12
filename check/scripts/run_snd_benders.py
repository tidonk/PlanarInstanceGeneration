#!/usr/bin/env python3
"""Run the full N-1 SND application through BendersNetworkDesign.jl."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from planar_instance_generation.evaluation.snd_application import (
    SndBendersConfiguration,
    execute_snd_benders_run,
    prepare_snd_benders_run,
)


RESULTS_DIRECTORY = REPOSITORY_ROOT / "check/results"
DEFAULT_BENDERS_REPOSITORY = (
    REPOSITORY_ROOT / "third_party/BendersNetworkDesign"
)
ADAPTER = Path(__file__).with_name("run_snd_benders.jl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path, help="generated SNDlib XML below check/results")
    parser.add_argument("--run", required=True, help="result directory name below check/results")
    parser.add_argument("--benders-repository", type=Path, default=DEFAULT_BENDERS_REPOSITORY)
    parser.add_argument("--julia", default="julia", help="Juliaup launcher or Julia executable")
    parser.add_argument(
        "--ignore-manifest-julia",
        action="store_true",
        help="do not pass +<julia_version> from the external Manifest.toml",
    )
    parser.add_argument("--time-limit", type=float, default=10_800.0)
    parser.add_argument(
        "--wall-clock-grace",
        type=float,
        default=60.0,
        help="extra seconds allowed for Julia startup and result serialization",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--method",
        choices=("standard", "adaptive_ml"),
        default="standard",
        help="Benders subproblem-selection configuration",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run = prepare_snd_benders_run(
            args.instance,
            run_name=args.run,
            results_directory=RESULTS_DIRECTORY,
            benders_repository=args.benders_repository,
            adapter=ADAPTER,
            julia=args.julia,
            use_manifest_julia=not args.ignore_manifest_julia,
            config=SndBendersConfiguration(
                time_limit_seconds=args.time_limit,
                wall_clock_grace_seconds=args.wall_clock_grace,
                threads=args.threads,
                seed=args.seed,
                method=args.method,
            ),
        )
        summary = execute_snd_benders_run(run, dry_run=args.dry_run)
    except (FileExistsError, OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"State: {summary['state']}")
    print(f"Summary: {run.paths.summary}")
    if summary["state"] == "adapter_error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
