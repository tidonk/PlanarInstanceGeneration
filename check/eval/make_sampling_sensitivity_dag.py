#!/usr/bin/env python3
"""Prepare the 600/3600 outer face sampling sensitivity DAG."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
OUTER_FACE_STRATEGIES = ("exact_outer_face",)
ALTERNATIVES = (
    ("equidistant", "equidistant", 0),
    ("adaptive_skip_one", "adaptive_skip_one", 0),
    ("prefer_non_leaf", "consecutive", 1),
)
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def build_dag(
    prefix: str,
    settings: Path,
    workflow: Path,
    *,
    instances: int = 100,
    priority: int = 30,
    nice_user: bool = False,
) -> str:
    if not _SAFE.fullmatch(prefix):
        raise ValueError("unsafe experiment prefix")
    if instances < 1:
        raise ValueError("instances must be positive")
    if any(character in str(path) for path in (settings, workflow)
           for character in '\n\r" '):
        raise ValueError("settings and workflow paths must not contain whitespace")

    lines = [
        "# Exact outer face sampling sensitivity aligned with the final log2 experiment.",
        "# Merge planarization: 600 s; final skewness: 3600 s; SCOPF: no solver time limit.",
        "# The completed consecutive branch is reused as the reference.",
    ]
    for suffix, sampling, prefer_non_leaf in ALTERNATIVES:
        experiment = f"{prefix}_{suffix}"
        generation_nodes: list[str] = []
        for index in range(1, instances + 1):
            generation = f"gen_{suffix}_i{index:03d}"
            generation_nodes.append(generation)
            lines.extend(
                (
                    f"JOB {generation} check/eval/pglib_one.sub",
                    (
                        f'VARS {generation} experiment="{experiment}" '
                        f'dataset="pglib" instance="{index}" seed="{index}" '
                        f'sampling="{sampling}" prefer_non_leaf="{prefer_non_leaf}" '
                        f'settings_file="{settings}" max_runtime="172800" '
                        f'job_priority="{priority}" nice_user="{str(nice_user)}"'
                    ),
                )
            )
            for strategy in OUTER_FACE_STRATEGIES:
                exact = f"skew_{suffix}_i{index:03d}_{strategy}"
                lines.extend(
                    (
                        f"JOB {exact} check/eval/pglib_cohort_skewness.sub",
                        (
                            f'VARS {exact} experiment="{experiment}" '
                            f'output_name="{experiment}_pglib_exact" '
                            f'instance="{index}" seed="0" strategy="{strategy}" '
                            f'time_limit="3600" '
                            f'max_runtime="86400" hold_margin="3600" '
                            f'job_priority="{priority}" nice_user="{str(nice_user)}"'
                        ),
                        f"PARENT {generation} CHILD {exact}",
                    )
                )

        application = f"app_{suffix}"
        jobs_file = (
            workflow
            / "application_jobs"
            / experiment
            / "pglib_all.jobs"
        )
        lines.extend(
            (
                f"JOB {application} check/eval/pglib_application.sub",
                (
                    f'VARS {application} experiment="{experiment}" cohort="all" '
                    f'settings_file="{settings}" application_time_limit="0" unlimited_scopf="True" '
                    f'jobs_file="{jobs_file}" job_priority="{priority}" nice_user="{str(nice_user)}"'
                ),
                (
                    f"SCRIPT PRE {application} /usr/bin/python3 "
                    f"check/eval/make_application_jobs.py pglib --experiment {experiment} "
                    f"--instances {instances} --base-seed 0 --methods "
                    f"{' '.join(OUTER_FACE_STRATEGIES)} --output-dir {workflow}"
                ),
                (
                    f"SCRIPT POST {application} /usr/bin/python3 "
                    f"check/eval/summarize_applications.py pglib --experiment {experiment}"
                ),
                f"PARENT {' '.join(generation_nodes)} CHILD {application}",
            )
        )

    lines.append("# Exact jobs are independent of the SCOPF application arrays.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix")
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--priority", type=int, default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--settings", type=Path, default=ROOT / "check/eval/sampling_sensitivity_600_3600.env")
    parser.add_argument("--nice-user", action="store_true")
    args = parser.parse_args()

    settings = args.settings.resolve()
    workflow = (ROOT / "check/results" / f"{args.prefix}_workflow").resolve()
    output = args.output or ROOT / "check/eval" / f"{args.prefix}.dag"
    if output.exists():
        raise SystemExit(f"DAG already exists: {output}")
    for suffix, _sampling, _preference in ALTERNATIVES:
        experiment = f"{args.prefix}_{suffix}"
        if (ROOT / "check/results" / f"{experiment}_pglib_job_i001").exists():
            raise SystemExit(f"experiment already exists: {experiment}")

    for suffix, _sampling, _preference in ALTERNATIVES:
        experiment = f"{args.prefix}_{suffix}"
        for relative in ("", "skewness", "application"):
            (ROOT / "check/eval/logs" / experiment / "pglib" / relative).mkdir(
                parents=True, exist_ok=True
            )
    workflow.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        build_dag(
            args.prefix,
            settings,
            workflow,
            instances=args.instances,
            priority=args.priority,
            nice_user=args.nice_user,
        ),
        encoding="utf-8",
    )
    print(
        f"Prepared {output}: {3 * args.instances} generation jobs, "
        f"{3 * args.instances} exact skewness jobs, and three SCOPF arrays "
        f"with {args.instances} jobs each."
    )


if __name__ == "__main__":
    main()
