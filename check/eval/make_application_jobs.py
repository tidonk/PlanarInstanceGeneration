#!/usr/bin/env python3
"""Build one post-generation application job per accepted final composite."""

from __future__ import annotations

import argparse
import csv
import filecmp
import io
import os
from pathlib import Path
import re
import shutil


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIRECTORY = REPOSITORY_ROOT / "check/results"
EVAL_DIRECTORY = Path(__file__).resolve().parent
DATASETS = {"pglib": ".m", "snd": ".xml"}
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
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--run-name",
        help="override the application result directory name",
    )
    parser.add_argument("--instance-index", type=int)
    parser.add_argument(
        "--methods", nargs="+", choices=STRATEGIES, default=list(STRATEGIES)
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIRECTORY)
    parser.add_argument("--output-dir", type=Path, default=EVAL_DIRECTORY)
    parser.add_argument(
        "--snd-method",
        choices=("standard", "adaptive_ml"),
        default="standard",
        help="SND Benders configuration; ignored for PGLib",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not _SAFE_NAME.fullmatch(args.experiment):
        raise SystemExit("--experiment must be a safe filename component")
    if args.instances < 1:
        raise SystemExit("--instances must be positive")
    if args.base_seed < 0:
        raise SystemExit("--base-seed must be nonnegative")
    if args.instance_index is not None and not (
        1 <= args.instance_index <= args.instances
    ):
        raise SystemExit("--instance-index must lie between 1 and --instances")
    if len(set(args.methods)) != len(args.methods):
        raise SystemExit("--methods must not contain duplicates")
    if args.dataset != "snd" and args.snd_method != "standard":
        raise SystemExit("--snd-method applies only to SND")

    methods = tuple(method for method in STRATEGIES if method in args.methods)
    run_suffix = "scopf" if args.dataset == "pglib" else "benders"
    run_name = args.run_name or f"{args.experiment}_{args.dataset}_{run_suffix}"
    if not _SAFE_NAME.fullmatch(run_name):
        raise SystemExit("--run-name must be a safe filename component")
    if args.dataset == "snd" and args.snd_method == "adaptive_ml":
        run_name += "_adaptive_ml"
    records: list[dict[str, str]] = []
    job_lines: list[str] = []
    results = args.results_dir.resolve()
    application_root = results / run_name
    application_input_root = application_root / "instances"
    application_input_root.mkdir(parents=True, exist_ok=True)

    instance_indices = (
        (args.instance_index,)
        if args.instance_index is not None
        else range(1, args.instances + 1)
    )
    for instance_index in instance_indices:
        name = f"{args.experiment}_{args.dataset}_job_i{instance_index:03d}"
        active_job = results / name
        archived_job = (
            results / "_cohorts" / f"{args.experiment}_{args.dataset}" / name
        )
        if active_job.is_dir() and archived_job.is_dir():
            raise SystemExit(
                f"cohort exists in active and archived locations: {name}"
            )
        job = archived_job if archived_job.is_dir() else active_job
        accepted_path = job / "tables" / "accepted_instances.csv"
        if not accepted_path.is_file():
            raise SystemExit(f"missing accepted-instance table: {accepted_path}")
        with accepted_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        by_strategy = _strategy_rows(rows, methods, accepted_path)
        seed = args.base_seed + instance_index
        for strategy in methods:
            row = by_strategy[strategy]
            artifact_stem = row.get("artifact_stem", "")
            raw_instance = row.get("instance_file", "")
            if not _SAFE_NAME.fullmatch(artifact_stem):
                raise SystemExit(f"invalid artifact stem in {accepted_path}")
            source = Path(raw_instance)
            if not source.is_absolute():
                source = REPOSITORY_ROOT / source
            source = source.resolve()
            expected_parent = (job / "instances").resolve()
            if not source.is_file():
                source = (
                    expected_parent
                    / f"{artifact_stem}{DATASETS[args.dataset]}"
                ).resolve()
            if (
                source.parent != expected_parent
                or source.suffix != DATASETS[args.dataset]
            ):
                raise SystemExit(
                    f"invalid final instance path in {accepted_path}: {source}"
                )
            if not source.is_file() or source.stem != artifact_stem:
                raise SystemExit(f"missing final accepted composite: {source}")
            source = _materialize_application_input(
                source, application_input_root / source.name
            )
            if args.dataset == "pglib":
                _materialize_pglib_metadata(
                    job, application_root, artifact_stem
                )
            try:
                relative_source = source.relative_to(REPOSITORY_ROOT).as_posix()
            except ValueError:
                relative_source = source.as_posix()
            records.append(
                {
                    "dataset": args.dataset,
                    "instance_index": str(instance_index),
                    "strategy": strategy,
                    "artifact_stem": artifact_stem,
                    "instance_file": relative_source,
                    "application_seed": str(seed),
                    "run_name": run_name,
                    "snd_method": args.snd_method if args.dataset == "snd" else "",
                }
            )
            job_lines.append(
                f"{artifact_stem} {relative_source} {run_name} "
                f"{seed} {strategy} {instance_index}"
            )

    cohort = (
        f"i{args.instance_index:03d}"
        if args.instance_index is not None
        else "all"
    )
    job_directory = args.output_dir / "application_jobs" / args.experiment
    job_directory.mkdir(parents=True, exist_ok=True)
    job_stem = args.dataset
    if args.dataset == "snd" and args.snd_method == "adaptive_ml":
        job_stem += "_adaptive"
    job_path = job_directory / f"{job_stem}_{cohort}.jobs"
    job_path.write_text("\n".join(job_lines) + "\n", encoding="utf-8")
    (args.output_dir / "logs" / args.experiment / args.dataset / "application").mkdir(
        parents=True, exist_ok=True
    )

    manifest_path = (
        application_root / "jobs.csv"
        if args.instance_index is None
        else application_root / "job_manifests" / f"{cohort}.csv"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=tuple(records[0]), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(records)
    manifest_text = buffer.getvalue()
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != manifest_text:
        raise SystemExit(
            "application manifest already exists with different data: "
            f"{manifest_path}"
        )
    manifest_path.write_text(manifest_text, encoding="utf-8")
    print(
        f"{args.dataset}: wrote {len(records)} post-generation evaluation jobs "
        f"to {job_path}"
    )


def _strategy_rows(
    rows: list[dict[str, str]],
    methods: tuple[str, ...],
    accepted_path: Path,
) -> dict[str, dict[str, str]]:
    by_strategy = {row.get("strategy", ""): row for row in rows}
    if len(by_strategy) != len(rows):
        raise SystemExit(f"{accepted_path} contains duplicate strategy rows")
    missing_methods = set(methods) - set(by_strategy)
    if missing_methods:
        raise SystemExit(
            f"{accepted_path} has strategies {sorted(by_strategy)}; "
            f"missing requested strategies {sorted(missing_methods)}"
        )
    return by_strategy


def _materialize_application_input(source: Path, target: Path) -> Path:
    if target.exists():
        if target.is_file() and filecmp.cmp(source, target, shallow=False):
            return target.resolve()
        raise SystemExit(
            f"application input already exists with different data: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return target.resolve()


def _materialize_pglib_metadata(
    cohort: Path, application_root: Path, artifact_stem: str
) -> Path:
    source = (
        cohort
        / "plots"
        / "composite"
        / artifact_stem
        / "metadata.json"
    )
    if not source.is_file():
        raise SystemExit(f"missing PGLib coupling metadata: {source}")
    target = (
        application_root
        / "plots"
        / "composite"
        / artifact_stem
        / "metadata.json"
    )
    return _materialize_application_input(source, target)


if __name__ == "__main__":
    main()
