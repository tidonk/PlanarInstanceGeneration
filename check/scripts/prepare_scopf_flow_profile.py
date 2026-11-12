#!/usr/bin/env python3
"""Prepare a focused rerun that records branch flows for feasible SCOPF cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "check/results"
EVAL = ROOT / "check/eval"
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-run", required=True)
    parser.add_argument("--job-experiment", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for label, value in (
        ("source run", args.source_run),
        ("output run", args.output_run),
        ("job experiment", args.job_experiment),
    ):
        if not SAFE_NAME.fullmatch(value):
            raise SystemExit(f"{label} must be a safe path component")

    source_root = RESULTS / args.source_run
    source_summary = source_root / "summary.csv"
    if not source_summary.is_file():
        raise SystemExit(f"missing source summary: {source_summary}")
    with source_summary.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 500:
        raise SystemExit(f"expected 500 source rows, found {len(rows)}")

    selected = [
        row
        for row in rows
        if row["state"] == "complete" and row["scopf_feasible"] == "True"
    ]
    if not selected:
        raise SystemExit("source summary has no feasible SCOPF rows")
    if len({row["artifact_stem"] for row in selected}) != len(selected):
        raise SystemExit("source summary contains duplicate feasible artifact stems")

    output_root = RESULTS / args.output_run
    job_directory = EVAL / "application_jobs" / args.job_experiment
    if output_root.exists():
        raise SystemExit(f"output result directory already exists: {output_root}")
    if job_directory.exists():
        raise SystemExit(f"job directory already exists: {job_directory}")
    output_root.mkdir(parents=True)
    job_directory.mkdir(parents=True)

    manifest_fields = (
        "dataset",
        "instance_index",
        "strategy",
        "artifact_stem",
        "instance_file",
        "application_seed",
        "run_name",
        "snd_method",
    )
    manifest_rows: list[dict[str, str]] = []
    job_lines: list[str] = []
    for row in selected:
        instance = ROOT / row["instance_file"]
        if not instance.is_file():
            raise SystemExit(f"missing source instance: {instance}")
        record = {
            "dataset": "pglib",
            "instance_index": row["instance_index"],
            "strategy": row["strategy"],
            "artifact_stem": row["artifact_stem"],
            "instance_file": row["instance_file"],
            "application_seed": row["application_seed"],
            "run_name": args.output_run,
            "snd_method": "",
        }
        manifest_rows.append(record)
        job_lines.append(
            " ".join(
                (
                    record["artifact_stem"],
                    record["instance_file"],
                    record["run_name"],
                    record["application_seed"],
                    record["strategy"],
                    record["instance_index"],
                )
            )
        )

    with (output_root / "jobs.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    jobs_path = job_directory / "pglib_all.jobs"
    jobs_path.write_text("\n".join(job_lines) + "\n", encoding="utf-8")
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "record base-case coupling and non-coupling branch flows",
                "source_run": args.source_run,
                "output_run": args.output_run,
                "selection": "completed rows with scopf_feasible=true",
                "selected_count": len(selected),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (EVAL / "logs" / args.job_experiment / "pglib/application").mkdir(
        parents=True, exist_ok=True
    )
    print(f"Wrote {len(selected)} SCOPF flow-profile jobs to {jobs_path}")


if __name__ == "__main__":
    main()
