#!/usr/bin/env python3
"""Prepare standalone evaluations of compensated SND subinstances."""

from __future__ import annotations

import argparse
import csv
import filecmp
import json
from pathlib import Path
import re
import shutil


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIRECTORY = REPOSITORY_ROOT / "check/results"
EVAL_DIRECTORY = REPOSITORY_ROOT / "check/eval"
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
    parser.add_argument("--source-experiment", required=True)
    parser.add_argument("--output-experiment", required=True)
    parser.add_argument("--base-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for label, value in (
        ("source experiment", args.source_experiment),
        ("output experiment", args.output_experiment),
    ):
        if not _SAFE_NAME.fullmatch(value):
            raise SystemExit(f"{label} must be a safe path component")
    if args.base_seed < 0:
        raise SystemExit("--base-seed must be nonnegative")

    source_root = RESULTS_DIRECTORY / f"{args.source_experiment}_snd"
    accepted_path = source_root / "tables/accepted_instances.csv"
    if not accepted_path.is_file():
        raise SystemExit(f"missing accepted-instance table: {accepted_path}")
    with accepted_path.open(encoding="utf-8", newline="") as stream:
        accepted = list(csv.DictReader(stream))

    rows_by_index: dict[int, list[dict[str, str]]] = {}
    for row in accepted:
        index = int(row["output_index"])
        rows_by_index.setdefault(index, []).append(row)
    if sorted(rows_by_index) != list(range(1, 101)):
        raise SystemExit("expected accepted SND inputs numbered 1 through 100")

    run_name = f"{args.output_experiment}_snd_pieces_benders"
    output_root = RESULTS_DIRECTORY / run_name
    job_directory = EVAL_DIRECTORY / "application_jobs" / args.output_experiment
    if output_root.exists():
        raise SystemExit(f"piece result directory already exists: {output_root}")
    if job_directory.exists():
        raise SystemExit(f"piece job directory already exists: {job_directory}")
    instance_root = output_root / "instances"
    instance_root.mkdir(parents=True)
    job_directory.mkdir(parents=True)

    records: list[dict[str, object]] = []
    job_lines: list[str] = []
    seen_stems: set[str] = set()
    for index in sorted(rows_by_index):
        rows = rows_by_index[index]
        by_strategy = {row["strategy"]: row for row in rows}
        if len(rows) != len(STRATEGIES) or set(by_strategy) != set(STRATEGIES):
            raise SystemExit(
                f"input {index} has strategies {sorted(by_strategy)}; "
                f"expected {list(STRATEGIES)}"
            )
        source_lists = {row["source_artifact_stems"] for row in rows}
        if len(source_lists) != 1:
            raise SystemExit(
                f"input {index} does not share the same extracted subinstances"
            )
        stems = next(iter(source_lists)).split(";")
        if not stems or any(not _SAFE_NAME.fullmatch(stem) for stem in stems):
            raise SystemExit(f"input {index} has invalid subinstance stems")

        for piece_number, stem in enumerate(stems, start=1):
            if stem in seen_stems:
                raise SystemExit(f"duplicate subinstance stem: {stem}")
            seen_stems.add(stem)
            source = source_root / "subinstances" / f"{stem}.xml"
            if not source.is_file():
                raise SystemExit(f"missing compensated subinstance: {source}")
            target = instance_root / source.name
            _materialize(source, target)
            relative_target = target.relative_to(REPOSITORY_ROOT).as_posix()
            seed = args.base_seed + index
            records.append(
                {
                    "dataset": "snd",
                    "parent_instance_index": index,
                    "piece_number": piece_number,
                    "artifact_stem": stem,
                    "instance_file": relative_target,
                    "application_seed": seed,
                    "run_name": run_name,
                    "snd_method": "standard",
                }
            )
            job_lines.append(
                f"{stem} {relative_target} {run_name} {seed} piece {index}"
            )

    _write_csv(output_root / "jobs.csv", records)
    job_path = job_directory / "snd_pieces_all.jobs"
    job_path.write_text("\n".join(job_lines) + "\n", encoding="utf-8")
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_experiment": args.source_experiment,
                "output_experiment": args.output_experiment,
                "run_name": run_name,
                "parent_instance_count": len(rows_by_index),
                "piece_count": len(records),
                "input_kind": "compensated extracted SND subinstance",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (EVAL_DIRECTORY / "logs" / args.output_experiment / "snd/application").mkdir(
        parents=True, exist_ok=True
    )
    print(f"Wrote {len(records)} standalone piece jobs to {job_path}")
    print(f"Application run: {output_root}")


def _materialize(source: Path, target: Path) -> None:
    if target.exists():
        if target.is_file() and filecmp.cmp(source, target, shallow=False):
            return
        raise SystemExit(f"application input already exists with different data: {target}")
    try:
        target.hardlink_to(source)
    except OSError:
        shutil.copy2(source, target)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
