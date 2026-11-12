#!/usr/bin/env python3
"""Combine independently screened PGLib sources into the generation input pool."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planar_instance_generation.evaluation import load_pglib_source_pool


def merge_source_pools(jobs, output, *, root=ROOT):
    if not jobs:
        raise ValueError("source job list is empty")
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    combined = None
    seen_paths, seen_names = set(), set()
    for job in jobs:
        path = root / "check/results" / job["result_name"] / "manifest.json"
        part = json.loads(path.read_text())
        if part.get("schema_version") != 1:
            raise ValueError(f"unsupported source result: {path}")
        settings = {
            key: value for key, value in part["run"].items() if key != "sources"
        }
        entries = part["run"]["sources"]
        if len(entries) != 1:
            raise ValueError(f"expected one source in {path}")
        source = (root / entries[0]["path"]).resolve()
        if source != (root / job["source"]).resolve() or source in seen_paths:
            raise ValueError(f"wrong or duplicate source in {path}")
        seen_paths.add(source)
        if combined is None:
            combined = {
                "schema_version": 1,
                "run": {**settings, "sources": []},
                "backend": part["backend"],
                "attempted": 0,
                "feasible": 0,
                "excluded_euler_bound_violations": 0,
                "total_dc_opf_wall_seconds": 0,
                "excluded_over_bus_limit": [],
                "records": [],
                "solutions": {},
            }
        elif (
            settings != {k: v for k, v in combined["run"].items() if k != "sources"}
            or part["backend"] != combined["backend"]
        ):
            raise ValueError(f"inconsistent settings or OPF implementation in {path}")
        if part["attempted"] != len(part["records"]) or part["feasible"] != len(
            part["solutions"]
        ):
            raise ValueError(f"inconsistent source counts in {path}")
        names = [r["source_name"] for r in part["records"]] + [
            r["name"] for r in part["excluded_over_bus_limit"]
        ]
        if len(names) != 1 or seen_names.intersection(names):
            raise ValueError(f"missing or duplicate source record in {path}")
        seen_names.update(names)
        for field in (
            "attempted",
            "feasible",
            "excluded_euler_bound_violations",
            "total_dc_opf_wall_seconds",
        ):
            combined[field] += part[field]
        for field in ("excluded_over_bus_limit", "records"):
            combined[field].extend(part[field])
        combined["run"]["sources"].extend(entries)
        combined["solutions"].update(part["solutions"])
    # Publish the completion file only after the existing loader has checked
    # source hashes, solution vectors, feasibility records, and the total count.
    output.mkdir(parents=True)
    records = output / "source_screening.csv"
    combined["records_csv"] = str(records.resolve())
    if combined["records"]:
        with records.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(combined["records"][0]))
            writer.writeheader()
            writer.writerows(combined["records"])
    temporary = output / "pool.json.tmp"
    temporary.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n")
    load_pglib_source_pool(temporary, path_root=root)
    temporary.replace(output / "manifest.json")
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "check/results/pglib_source_pool_innoptem"
    )
    args = parser.parse_args()
    result = merge_source_pools(json.loads(args.jobs.read_text()), args.output)
    print(
        f'Combined {len(result["run"]["sources"])} screened sources; retained {result["feasible"]}.'
    )


if __name__ == "__main__":
    main()
