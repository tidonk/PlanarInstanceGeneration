#!/usr/bin/env python3
"""Consolidate per-job generation runs and build the combined review report.

The main experiment submits one HTCondor job per requested instance. Each job
runs all configured merge-node strategies together and writes an isolated
result directory below ``check/results`` holding the paired composites, their
shared extraction provenance, plots, and per-run CSV tables.  This script gathers all of those job directories for one dataset into
a single consolidated run directory that mirrors the layout a monolithic
generation run would have produced, concatenating the result CSVs and
synthesising the generation manifest.  With ``--report`` it then invokes the
existing ``evaluate_*`` script on the consolidated run so the full A4 review
report is generated exactly once, after every experiment has finished.

Artifact stems are made globally unique by the per-instance ``--suffix``. Every
job records ``output_index = 1``; here that value is rewritten to its instance
number so the paired-strategy report groups the five shared-extraction outcomes
into one cohort.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIRECTORY = REPOSITORY_ROOT / "check/results"
SCRIPTS_DIRECTORY = REPOSITORY_ROOT / "check/scripts"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

STRATEGIES = (
    "random",
    "high_degree",
    "kamada_kawai",
    "heuristic_outer_face",
    "exact_outer_face",
)

DATASETS = {
    "pglib": {
        "extension": ".m",
        "artifact_type": None,
        "evaluate_script": "evaluate_pglib.py",
        "tables": (
            "accepted_instances",
            "attempts",
            "merge_stages",
            "extractions",
        ),
    },
    "snd": {
        "extension": ".xml",
        "artifact_type": "sndlib_generation",
        "evaluate_script": "evaluate_sndlib.py",
        "tables": (
            "accepted_instances",
            "attempts",
            "merge_stages",
            "extractions",
            "routing_unit_strata",
        ),
    },
}

# Tables whose ``output_index`` must be rewritten from the per-job value of 1 to
# the instance number so cohorts stay distinct after consolidation.
_OUTPUT_INDEX_TABLES = frozenset({"accepted_instances", "attempts"})

_JOB_DIRECTORY = re.compile(r"_job_i(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument(
        "--experiment",
        default="main",
        help="experiment tag shared by the submitted jobs",
    )
    parser.add_argument(
        "--consolidated-name",
        default=None,
        help="result name for the merged run (default: <experiment>_<dataset>)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIRECTORY,
        help="directory holding the per-job result runs",
    )
    parser.add_argument("--source-pool", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--report",
        action="store_true",
        help="run the dataset evaluate script on the consolidated run",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="interpreter used to invoke the evaluate script under --report",
    )
    # Forwarded verbatim to the evaluate script when --report is set.
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--skewness-workers", type=int, default=1)
    parser.add_argument("--skewness-time-limit", type=float, default=300.0)
    parser.add_argument(
        "--pglib-cohort-checkpoints",
        type=Path,
        help="distributed exact-skewness checkpoint directory for PGLib",
    )
    parser.add_argument(
        "--snd-cohort-checkpoints",
        type=Path,
        help="distributed exact-skewness checkpoint directory for SNDlib",
    )
    parser.add_argument("--betweenness-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--reuse-completed-evaluation",
        action="store_true",
        help=(
            "validate and reuse an existing combined evaluation without "
            "reconsolidating, reevaluating, or compiling it"
        ),
    )
    parser.add_argument(
        "--keep-job-runs",
        action="store_true",
        help=(
            "leave complete per-cohort directories at the results root after "
            "the combined review report is compiled"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = DATASETS[args.dataset]
    consolidated_name = args.consolidated_name or f"{args.experiment}_{args.dataset}"
    if not _SAFE_NAME.fullmatch(consolidated_name):
        raise SystemExit("--consolidated-name must be a single safe filename component")
    if not _SAFE_NAME.fullmatch(args.experiment):
        raise SystemExit("--experiment must be a single safe filename component")

    jobs = _discover_jobs(args.results_dir, args.experiment, args.dataset)
    if not jobs:
        raise SystemExit(
            f"no job runs found for experiment {args.experiment!r} dataset "
            f"{args.dataset!r} below {args.results_dir}"
        )

    consolidated = (args.results_dir / consolidated_name).resolve()
    if args.reuse_completed_evaluation:
        _validate_completed_evaluation(consolidated)
        print(f"Reusing completed evaluation at {_rel(consolidated / 'evaluation')}")
        return
    previous_run = None
    if consolidated.exists():
        if not args.overwrite:
            raise SystemExit(
                f"consolidated run already exists: {consolidated}; pass --overwrite"
            )
        archive_root = args.results_dir / "_superseded"
        archive_root.mkdir(parents=True, exist_ok=True)
        archive = Path(tempfile.mkdtemp(prefix=consolidated_name + "-", dir=archive_root))
        previous_run = archive / consolidated_name
        shutil.move(str(consolidated), str(previous_run))
        print(f"Preserved previous combined run at {previous_run}")

    complete, incomplete = _partition_jobs(jobs, dataset)
    if not complete:
        raise SystemExit("no job produced an accepted composite; nothing to consolidate")

    _build_consolidated_run(complete, consolidated, dataset)
    if previous_run is not None:
        for relative in ("evaluation/planarization_checkpoints", "logs/skewness"):
            source = previous_run / relative
            if source.is_dir():
                shutil.copytree(source, consolidated / relative, dirs_exist_ok=True)
    manifest_path = _write_manifest(complete, consolidated, dataset)

    print(f"Consolidated {len(complete)} accepted jobs into {_rel(consolidated)}")
    if incomplete:
        print(f"Skipped {len(incomplete)} job(s) without a complete cohort:")
        for instance, directory in sorted(incomplete):
            print(f"  i{instance:03d}: {_rel(directory)}")
    accepted_rows = _read_rows(consolidated / "tables" / "accepted_instances.csv")
    for method in STRATEGIES:
        count = sum(row.get("strategy") == method for row in accepted_rows)
        if count:
            print(f"  {method}: {count} accepted")

    if not args.report:
        print(f"Manifest: {_rel(manifest_path)}")
        print("Re-run with --report (or submit report.sub) to build the review report.")
        return

    _run_report(args, dataset, consolidated_name)
    if not args.no_compile and not args.keep_job_runs:
        report = consolidated / "review_report.pdf"
        if not report.is_file():
            raise SystemExit(f"compiled combined report is missing: {_rel(report)}")
        _archive_jobs(complete, args.results_dir, consolidated_name, manifest_path)


class _Job:
    __slots__ = ("instance", "directory")

    def __init__(self, instance: int, directory: Path) -> None:
        self.instance = instance
        self.directory = directory


def _discover_jobs(results_dir: Path, experiment: str, dataset: str) -> list[_Job]:
    prefix = f"{experiment}_{dataset}_job_i"
    by_instance: dict[int, _Job] = {}
    archive = _archive_root(results_dir, f"{experiment}_{dataset}")
    for root in (results_dir, archive):
        for entry in sorted(root.glob(f"{experiment}_{dataset}_job_i*")):
            if not entry.is_dir() or not entry.name.startswith(prefix):
                continue
            match = _JOB_DIRECTORY.search(entry.name)
            if match is None:
                continue
            instance = int(match.group(1))
            previous = by_instance.get(instance)
            if previous is not None and previous.directory != entry:
                raise SystemExit(
                    f"duplicate active and archived cohort i{instance:03d}: "
                    f"{_rel(previous.directory)} and {_rel(entry)}"
                )
            by_instance[instance] = _Job(instance, entry)
    return [by_instance[index] for index in sorted(by_instance)]


def _partition_jobs(
    jobs: list[_Job], dataset: dict[str, object]
) -> tuple[list[_Job], list[tuple[int, Path]]]:
    complete: list[_Job] = []
    incomplete: list[tuple[int, Path]] = []
    for job in jobs:
        accepted = job.directory / "tables" / "accepted_instances.csv"
        manifest = job.directory / "manifest.json"
        if manifest.is_file() and accepted.is_file() and _has_rows(accepted):
            complete.append(job)
        else:
            incomplete.append((job.instance, job.directory))
    return complete, incomplete


def _archive_root(results_dir: Path, consolidated_name: str) -> Path:
    return results_dir.resolve() / "_cohorts" / consolidated_name


def _archive_jobs(
    jobs: list[_Job],
    results_dir: Path,
    consolidated_name: str,
    manifest_path: Path,
) -> None:
    archive = _archive_root(results_dir, consolidated_name)
    archive.mkdir(parents=True, exist_ok=True)
    archived_manifests = []
    moved = 0
    for job in jobs:
        target = archive / job.directory.name
        if job.directory.resolve() != target.resolve():
            if target.exists():
                raise SystemExit(
                    f"refusing to overwrite archived cohort: {_rel(target)}"
                )
            shutil.move(str(job.directory), str(target))
            moved += 1
        archived_manifests.append(_rel(target / "manifest.json"))

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    consolidated = payload.get("consolidated")
    if not isinstance(consolidated, dict):
        raise SystemExit(f"invalid consolidated manifest: {_rel(manifest_path)}")
    consolidated["source_job_manifests"] = archived_manifests
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    print(f"Archived {moved} complete cohort directories below {_rel(archive)}")


def _build_consolidated_run(
    jobs: list[_Job], consolidated: Path, dataset: dict[str, object]
) -> None:
    for path in (
        consolidated / "instances",
        consolidated / "subinstances",
        consolidated / "plots" / "original",
        consolidated / "plots" / "merged",
        consolidated / "plots" / "composite",
        consolidated / "logs" / "original",
        consolidated / "logs" / "merged",
        consolidated / "logs" / "composite",
        consolidated / "logs" / "attempts",
        consolidated / "tables",
    ):
        path.mkdir(parents=True, exist_ok=True)

    table_rows: dict[str, list[dict[str, str]]] = {
        name: [] for name in dataset["tables"]
    }
    seen_stems: dict[str, Path] = {}
    for job in jobs:
        tag = f"i{job.instance:03d}"
        aliases = _subinstance_aliases(job)
        _copy_artifacts(job, consolidated, seen_stems, aliases)
        _copy_plots(job, consolidated, aliases)
        _copy_logs(job, consolidated, aliases, tag)
        for name in dataset["tables"]:
            source = job.directory / "tables" / f"{name}.csv"
            for row in _read_rows(source):
                if name in _OUTPUT_INDEX_TABLES:
                    row = {**row, "output_index": str(job.instance)}
                row = _normalize_table_row(
                    name, row, aliases, consolidated, dataset
                )
                table_rows[name].append(row)

    for name, rows in table_rows.items():
        if name == "routing_unit_strata":
            rows = _dedup_rows(rows)
            table_rows[name] = rows
        _write_rows(consolidated / "tables" / f"{name}.csv", rows)
    _validate_artifact_trees(consolidated, table_rows)


def _subinstance_aliases(job: _Job) -> dict[str, str]:
    accepted = _read_rows(job.directory / "tables" / "accepted_instances.csv")
    output_by_final = {
        row.get("artifact_stem", ""): row.get("output_index", "")
        for row in accepted
    }
    canonical_by_key: dict[tuple[str, ...], str] = {}
    aliases: dict[str, str] = {}
    for row in _read_rows(job.directory / "tables" / "extractions.csv"):
        stem = row.get("subinstance_artifact_stem", "")
        if not stem:
            raise SystemExit(f"missing subinstance artifact stem in {_rel(job.directory)}")
        key = (
            output_by_final.get(row.get("final_artifact_stem", ""), ""),
            row.get("source_position", ""),
            row.get("source_name", ""),
            row.get("source_node_count", ""),
            row.get("target_nodes", ""),
            row.get("seed", ""),
        )
        aliases[stem] = canonical_by_key.setdefault(key, stem)
    return aliases


def _copy_artifacts(
    job: _Job,
    consolidated: Path,
    seen_stems: dict[str, Path],
    aliases: dict[str, str],
) -> None:
    for source_tree in ("instances", "subinstances"):
        source_dir = job.directory / source_tree
        if not source_dir.is_dir():
            continue
        for path in sorted(source_dir.iterdir()):
            if not path.is_file():
                continue
            metadata = _artifact_metadata(job, path.stem)
            kind = metadata.get("kind")
            if kind in {"extracted_subinstance", "extracted_snd_subinstance"}:
                destination_tree = "subinstances"
                destination_stem = aliases.get(path.stem, path.stem)
                if destination_stem != path.stem:
                    continue
            elif kind in {"final_composite", "final_snd_composite"}:
                destination_tree = "instances"
                destination_stem = path.stem
            else:
                raise SystemExit(f"unknown artifact kind {kind!r} for {_rel(path)}")
            if destination_stem in seen_stems:
                raise SystemExit(
                    f"duplicate artifact stem {destination_stem!r} in {_rel(path)} and "
                    f"{_rel(seen_stems[destination_stem])}; suffixes are not unique"
                )
            seen_stems[destination_stem] = path
            target = consolidated / destination_tree / f"{destination_stem}{path.suffix}"
            shutil.copy2(path, target)


def _job_metadata_paths(job: _Job) -> list[Path]:
    plot_root = job.directory / "plots"
    nested = [
        *plot_root.joinpath("original").glob("*/metadata.json"),
        *plot_root.joinpath("composite").glob("*/metadata.json"),
    ]
    return sorted(nested or plot_root.glob("*/metadata.json"))


def _artifact_metadata(job: _Job, stem: str) -> dict[str, object]:
    candidates = (
        job.directory / "plots" / "original" / stem / "metadata.json",
        job.directory / "plots" / "composite" / stem / "metadata.json",
        job.directory / "plots" / stem / "metadata.json",
    )
    for path in candidates:
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                break
            return payload
    raise SystemExit(f"missing artifact metadata for {stem!r} in {_rel(job.directory)}")


def _copy_plots(
    job: _Job, consolidated: Path, aliases: dict[str, str]
) -> None:
    for metadata_path in _job_metadata_paths(job):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        stem = str(metadata.get("artifact_stem", ""))
        kind = metadata.get("kind")
        source = metadata_path.parent
        if kind in {"extracted_subinstance", "extracted_snd_subinstance"}:
            canonical = aliases.get(stem, stem)
            if canonical != stem:
                continue
            target = consolidated / "plots" / "original" / canonical
            _copy_plot_directory(source, target, include_stages=False)
            metadata["artifact_stem"] = canonical
            metadata["plot_tree"] = "original"
            _write_metadata(target / "metadata.json", metadata, aliases)
            continue
        if kind not in {"final_composite", "final_snd_composite"}:
            raise SystemExit(f"unknown artifact kind {kind!r} in {_rel(metadata_path)}")
        composite_target = consolidated / "plots" / "composite" / stem
        _copy_plot_directory(source, composite_target, include_stages=False)
        metadata["plot_tree"] = "composite"
        metadata["merge_plot_tree"] = "merged"
        _write_metadata(composite_target / "metadata.json", metadata, aliases)

        merged_source = job.directory / "plots" / "merged" / stem
        merged_target = consolidated / "plots" / "merged" / stem
        if merged_source.is_dir():
            _copy_plot_directory(merged_source, merged_target, include_stages=True)
        else:
            merged_target.mkdir(parents=True, exist_ok=True)
            for stage in sorted(source.glob("stage_*")):
                target = merged_target / stage.name
                if target.exists():
                    raise SystemExit(f"stem collision while copying {_rel(stage)}")
                shutil.copytree(stage, target)


def _copy_plot_directory(
    source: Path, target: Path, *, include_stages: bool
) -> None:
    if target.exists():
        raise SystemExit(f"stem collision while copying {_rel(source)}")
    target.mkdir(parents=True)
    for child in sorted(source.iterdir()):
        if not include_stages and child.name.startswith("stage_"):
            continue
        destination = target / child.name
        if child.is_dir():
            shutil.copytree(child, destination)
        else:
            shutil.copy2(child, destination)


def _write_metadata(
    path: Path, metadata: dict[str, object], aliases: dict[str, str]
) -> None:
    source_stems = metadata.get("source_artifact_stems")
    if isinstance(source_stems, list):
        metadata["source_artifact_stems"] = [
            aliases.get(str(stem), str(stem)) for stem in source_stems
        ]
    for key in ("source_bus_mappings", "source_mappings"):
        mappings = metadata.get(key)
        if not isinstance(mappings, list):
            continue
        for mapping in mappings:
            if not isinstance(mapping, dict):
                continue
            stem = mapping.get("source_artifact_stem")
            if isinstance(stem, str):
                mapping["source_artifact_stem"] = aliases.get(stem, stem)
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _copy_logs(
    job: _Job, consolidated: Path, aliases: dict[str, str], tag: str
) -> None:
    source_root = job.directory / "logs"
    _copy_attempt_logs(source_root, consolidated / "logs", tag)
    for category in ("original", "merged", "composite"):
        category_root = source_root / category
        if not category_root.is_dir():
            continue
        for child in sorted(category_root.iterdir()):
            if not child.is_dir():
                continue
            target_stem = aliases.get(child.name, child.name)
            if category == "original" and target_stem != child.name:
                continue
            _copy_tree(child, consolidated / "logs" / category / target_stem)

    reserved = {"attempts", "original", "merged", "composite"}
    for child in sorted(source_root.iterdir()) if source_root.is_dir() else ():
        if not child.is_dir() or child.name in reserved:
            continue
        try:
            metadata = _artifact_metadata(job, child.name)
        except SystemExit:
            _copy_tree(child, consolidated / "logs" / "other" / tag / child.name)
            continue
        kind = metadata.get("kind")
        if kind in {"extracted_subinstance", "extracted_snd_subinstance"}:
            canonical = aliases.get(child.name, child.name)
            if canonical != child.name:
                continue
            _copy_tree(child, consolidated / "logs" / "original" / canonical)
        elif kind in {"final_composite", "final_snd_composite"}:
            composite_target = consolidated / "logs" / "composite" / child.name
            merged_target = consolidated / "logs" / "merged" / child.name
            composite_target.mkdir(parents=True, exist_ok=True)
            merged_target.mkdir(parents=True, exist_ok=True)
            for entry in sorted(child.iterdir()):
                target_root = merged_target if entry.name == "selection" else composite_target
                _copy_entry(entry, target_root / entry.name)
        else:
            raise SystemExit(f"unknown artifact kind {kind!r} for {_rel(child)}")


def _copy_tree(source: Path, target: Path) -> None:
    if target.exists():
        raise SystemExit(f"stem collision while copying {_rel(source)}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)


def _copy_entry(source: Path, target: Path) -> None:
    if target.exists():
        raise SystemExit(f"artifact collision while copying {_rel(source)}")
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _copy_attempt_logs(source: Path, destination: Path, tag: str) -> None:
    attempts = source / "attempts"
    if not attempts.is_dir():
        return
    shutil.copytree(attempts, destination / "attempts" / tag)


def _normalize_table_row(
    name: str,
    row: dict[str, str],
    aliases: dict[str, str],
    consolidated: Path,
    dataset: dict[str, object],
) -> dict[str, str]:
    extension = str(dataset["extension"])
    if name == "accepted_instances":
        stem = row.get("artifact_stem", "")
        row["instance_file"] = _rel(consolidated / "instances" / f"{stem}{extension}")
        row["plot_directory"] = _rel(consolidated / "plots" / "composite" / stem)
        row["log_directory"] = _rel(consolidated / "logs" / "composite" / stem)
        sources = row.get("source_artifact_stems", "").split(";")
        row["source_artifact_stems"] = ";".join(
            aliases.get(stem, stem) for stem in sources if stem
        )
    elif name == "extractions":
        stem = row.get("subinstance_artifact_stem", "")
        canonical = aliases.get(stem, stem)
        row["subinstance_artifact_stem"] = canonical
        row["subinstance_file"] = _rel(
            consolidated / "subinstances" / f"{canonical}{extension}"
        )
    return row


def _validate_artifact_trees(
    consolidated: Path, table_rows: dict[str, list[dict[str, str]]]
) -> None:
    expected_finals = {
        row.get("artifact_stem", "") for row in table_rows["accepted_instances"]
    }
    actual_finals = {
        path.stem
        for path in (consolidated / "instances").iterdir()
        if path.is_file()
    }
    if actual_finals != expected_finals:
        raise SystemExit(
            "consolidated instances/ must contain final composites only; "
            f"expected={sorted(expected_finals)}, actual={sorted(actual_finals)}"
        )
    expected_pieces = {
        row.get("subinstance_artifact_stem", "")
        for row in table_rows["extractions"]
    }
    actual_pieces = {
        path.stem
        for path in (consolidated / "subinstances").iterdir()
        if path.is_file()
    }
    if actual_pieces != expected_pieces:
        raise SystemExit(
            "consolidated subinstances/ does not match extraction records; "
            f"expected={sorted(expected_pieces)}, actual={sorted(actual_pieces)}"
        )
    for category, expected in (
        ("original", expected_pieces),
        ("composite", expected_finals),
        ("merged", expected_finals),
    ):
        actual = {
            path.name
            for path in (consolidated / "plots" / category).iterdir()
            if path.is_dir()
        }
        if actual != expected:
            raise SystemExit(
                f"plots/{category}/ disagrees with generation tables; "
                f"expected={sorted(expected)}, actual={sorted(actual)}"
            )


def _write_manifest(
    jobs: list[_Job], consolidated: Path, dataset: dict[str, object]
) -> Path:
    template = json.loads(
        (jobs[0].directory / "manifest.json").read_text(encoding="utf-8")
    )
    run = dict(template.get("run") or {})
    accepted_rows = _read_rows(consolidated / "tables" / "accepted_instances.csv")
    present = [
        method
        for method in STRATEGIES
        if any(row.get("strategy") == method for row in accepted_rows)
    ]
    run["strategies"] = present
    run["result_name"] = consolidated.name
    run["suffix"] = None
    run["requested_per_strategy"] = None

    attempt_rows = _read_rows(consolidated / "tables" / "attempts.csv")

    tables = {
        name: _rel(consolidated / "tables" / f"{name}.csv")
        for name in dataset["tables"]
    }
    manifest = dict(template)
    manifest["run"] = run
    manifest["consolidated"] = {
        "experiment_jobs": len(jobs),
        "instances": sorted({job.instance for job in jobs}),
        "strategies": present,
        "source_job_manifests": [_rel(job.directory / "manifest.json") for job in jobs],
    }
    manifest["accepted"] = accepted_rows
    manifest["failures"] = []
    manifest["acceptance_rates"] = _acceptance_rates(attempt_rows, present)
    manifest["tables"] = tables
    if dataset["artifact_type"] is not None:
        manifest["artifact_type"] = dataset["artifact_type"]

    manifest_path = consolidated / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _acceptance_rates(
    attempt_rows: list[dict[str, str]], strategies: list[str]
) -> dict[str, float]:
    totals: dict[str, int] = {strategy: 0 for strategy in strategies}
    accepted: dict[str, int] = {strategy: 0 for strategy in strategies}
    for row in attempt_rows:
        strategy = row.get("strategy", "")
        if strategy not in totals:
            continue
        totals[strategy] += 1
        if row.get("status", "").strip().lower() == "accepted":
            accepted[strategy] += 1
    return {
        strategy: (accepted[strategy] / totals[strategy]) if totals[strategy] else 0.0
        for strategy in strategies
    }


def _run_report(
    args: argparse.Namespace, dataset: dict[str, object], consolidated_name: str
) -> None:
    command = [
        args.python,
        str(SCRIPTS_DIRECTORY / dataset["evaluate_script"]),
        consolidated_name,
        "--threads",
        str(args.threads),
        "--skewness-workers",
        str(args.skewness_workers),
        "--skewness-time-limit",
        str(args.skewness_time_limit),
        "--betweenness-samples",
        str(args.betweenness_samples),
        "--seed",
        str(args.seed),
        "--overwrite",
    ]
    if args.source_pool is not None:
        command += ["--source-pool", str(args.source_pool)]
    if args.pglib_cohort_checkpoints is not None:
        if args.dataset != "pglib":
            raise SystemExit(
                "--pglib-cohort-checkpoints is only valid for the PGLib report"
            )
        command += [
            "--cohort-checkpoints", str(args.pglib_cohort_checkpoints)
        ]
    if args.snd_cohort_checkpoints is not None:
        if args.dataset != "snd":
            raise SystemExit(
                "--snd-cohort-checkpoints is only valid for the SNDlib report"
            )
        command += [
            "--cohort-checkpoints", str(args.snd_cohort_checkpoints)
        ]
    if args.no_compile:
        command.append("--no-compile")
    print("Running report:", " ".join(command))
    completed = subprocess.run(command, cwd=str(REPOSITORY_ROOT), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _validate_completed_evaluation(consolidated: Path) -> None:
    required = (
        consolidated / "manifest.json",
        consolidated / "evaluation" / "manifest.json",
        consolidated / "evaluation" / "skewness.csv",
        consolidated / "evaluation" / "structural.csv",
        consolidated / "evaluation" / "paired_strategy_outcomes.csv",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        formatted = ", ".join(_rel(path) for path in missing)
        raise SystemExit(f"completed evaluation is missing required files: {formatted}")
    for path in required[2:]:
        if not _has_rows(path):
            raise SystemExit(f"completed evaluation table has no data rows: {_rel(path)}")


def _has_rows(path: Path) -> bool:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header:
            return False
        return next(reader, None) is not None


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            return []
        return [dict(row) for row in reader]


def _dedup_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[tuple[str, str], ...]] = set()
    unique: list[dict[str, str]] = []
    for row in rows:
        key = tuple(sorted(row.items()))
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _rel(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    main()
