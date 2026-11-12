#!/usr/bin/env python3
"""Materialize component-normalized PGLib composites and SCOPF jobs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import median
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from planar_instance_generation.evaluation.power_normalization import (
    normalize_component_voltage_aware_load,
)
from planar_instance_generation.io import read_pglib, write_pglib


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
_COHORT = re.compile(r"_i(\d{3})(?:_\d+)?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-experiment",
        default="full_shared4_300_consecutive",
        help="generation experiment whose combined PGLib report supplies the cases",
    )
    parser.add_argument(
        "--output-experiment",
        default=None,
        help="sensitivity experiment name (default: <source>_voltage_aware_normalized)",
    )
    parser.add_argument("--base-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_experiment = (
        args.output_experiment
        or f"{args.source_experiment}_voltage_aware_normalized"
    )
    for label, value in (
        ("source experiment", args.source_experiment),
        ("output experiment", output_experiment),
    ):
        if not _SAFE_NAME.fullmatch(value):
            raise SystemExit(f"{label} must be a safe path component")
    if args.base_seed < 0:
        raise SystemExit("--base-seed must be nonnegative")

    source_root = RESULTS_DIRECTORY / f"{args.source_experiment}_pglib"
    table = source_root / "tables/accepted_instances.csv"
    if not table.is_file():
        raise SystemExit(f"missing combined accepted table: {table}")
    with table.open(encoding="utf-8", newline="") as stream:
        accepted = list(csv.DictReader(stream))
    if not accepted:
        raise SystemExit(f"combined accepted table is empty: {table}")

    run_name = f"{output_experiment}_pglib_scopf"
    output_root = RESULTS_DIRECTORY / run_name
    if output_root.exists():
        raise SystemExit(f"sensitivity result directory already exists: {output_root}")
    instances = output_root / "instances"
    metadata_root = output_root / "plots/composite"
    instances.mkdir(parents=True)

    records: list[dict[str, object]] = []
    jobs: list[str] = []
    for row in accepted:
        artifact = str(row["artifact_stem"])
        strategy = str(row["strategy"])
        match = _COHORT.search(artifact)
        if not _SAFE_NAME.fullmatch(artifact) or match is None:
            raise SystemExit(f"cannot recover cohort index from artifact {artifact!r}")
        if strategy not in STRATEGIES:
            raise SystemExit(f"unknown strategy {strategy!r}")
        cohort = int(match.group(1))
        source = source_root / "instances" / f"{artifact}.m"
        metadata_source = source_root / "plots/composite" / artifact / "metadata.json"
        if not source.is_file() or not metadata_source.is_file():
            raise SystemExit(f"missing source case or metadata for {artifact}")

        case = read_pglib(source)
        metadata = json.loads(metadata_source.read_text(encoding="utf-8"))
        if metadata.get("artifact_stem") != artifact:
            raise SystemExit(f"metadata artifact mismatch for {artifact}")
        mappings = metadata.get("source_bus_mappings")
        coupling_rows = metadata.get("coupling_branch_rows_zero_based")
        if not isinstance(mappings, list) or not isinstance(coupling_rows, list):
            raise SystemExit(f"incomplete source/coupling metadata for {artifact}")
        groups = {
            str(mapping["source_artifact_stem"]): tuple(
                int(bus["composite_bus"]) for bus in mapping["buses"]
            )
            for mapping in mappings
        }
        result = normalize_component_voltage_aware_load(
            case,
            groups,
            (int(value) for value in coupling_rows),
            name=case.name,
        )
        destination = instances / source.name
        write_pglib(result.case, destination)

        audit = {
            "policy": (
                "scale each source piece to the within-composite median voltage-aware "
                "apparent-demand density sum(hypot(Pd,Qd)/baseKV)/bus_count; scale "
                "its bus injections, generator powers, base and limits, branch MVA "
                "ratings and charging, and cost-curve power coordinates; inversely "
                "scale source resistance and reactance; preserve every bus base-kV, "
                "voltage limit, transformer tap, topology, endpoint, and coupling branch"
            ),
            "source_path": str(source.relative_to(REPOSITORY_ROOT)),
            "source_sha256": sha256(source.read_bytes()).hexdigest(),
            "normalized_sha256": sha256(destination.read_bytes()).hexdigest(),
            "components": [asdict(component) for component in result.components],
            "coupling_voltage": asdict(result.voltage),
        }
        metadata["normalization_sensitivity"] = audit
        target_metadata = metadata_root / artifact / "metadata.json"
        target_metadata.parent.mkdir(parents=True)
        target_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        factors = [component.factor for component in result.components]
        record = {
            "cohort": cohort,
            "strategy": strategy,
            "artifact_stem": artifact,
            "source_instance": str(source.relative_to(REPOSITORY_ROOT)),
            "normalized_instance": str(destination.relative_to(REPOSITORY_ROOT)),
            "source_sha256": audit["source_sha256"],
            "normalized_sha256": audit["normalized_sha256"],
            "component_count": len(result.components),
            "minimum_scale_factor": min(factors),
            "median_scale_factor": median(factors),
            "maximum_scale_factor": max(factors),
            **asdict(result.voltage),
        }
        records.append(record)
        jobs.append(
            f"{artifact} {destination.relative_to(REPOSITORY_ROOT).as_posix()} "
            f"{run_name} {args.base_seed + cohort} {strategy} {cohort}"
        )

    records.sort(
        key=lambda item: (
            int(item["cohort"]),
            STRATEGIES.index(str(item["strategy"])),
        )
    )
    job_by_artifact = {line.split(maxsplit=1)[0]: line for line in jobs}
    jobs = [job_by_artifact[str(record["artifact_stem"])] for record in records]
    application_records = [
        {
            "dataset": "pglib",
            "instance_index": str(record["cohort"]),
            "strategy": str(record["strategy"]),
            "artifact_stem": str(record["artifact_stem"]),
            "instance_file": str(record["normalized_instance"]),
            "application_seed": str(args.base_seed + int(record["cohort"])),
            "run_name": run_name,
        }
        for record in records
    ]
    _write_csv(output_root / "jobs.csv", application_records)
    _write_csv(output_root / "normalization_sensitivity.csv", records)
    (output_root / "normalization_sensitivity.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_experiment": args.source_experiment,
                "output_experiment": output_experiment,
                "run_name": run_name,
                "case_count": len(records),
                "normalization": (
                    "median component apparent-demand/current density "
                    "sum(hypot(Pd,Qd)/baseKV)/bus_count"
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    job_directory = EVAL_DIRECTORY / "application_jobs" / output_experiment
    job_directory.mkdir(parents=True, exist_ok=True)
    job_path = job_directory / "pglib_all.jobs"
    job_path.write_text("\n".join(jobs) + "\n", encoding="utf-8")
    log_directory = EVAL_DIRECTORY / "logs" / output_experiment / "pglib/application"
    log_directory.mkdir(parents=True, exist_ok=True)
    print(f"Wrote {len(records)} normalized cases and jobs to {output_root}")
    print(
        "Submit with: condor_submit "
        f"-append 'experiment = {output_experiment}' "
        "check/eval/pglib_application.sub"
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
