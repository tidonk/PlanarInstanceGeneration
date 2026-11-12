#!/usr/bin/env python3
"""Compare optimal composite objectives with summed standalone-piece optima."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piece-run", type=Path, required=True)
    parser.add_argument("--piece-jobs", type=Path, required=True)
    parser.add_argument("--composite-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected: dict[int, list[str]] = defaultdict(list)
    for line in args.piece_jobs.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        stem, _instance, _run, _seed, _strategy, parent = line.split()
        expected[int(parent)].append(stem)

    piece_values: dict[int, list[float]] = defaultdict(list)
    piece_bounds: dict[int, list[float]] = defaultdict(list)
    for parent, stems in expected.items():
        for stem in stems:
            summary_file = args.piece_run / "logs" / stem / "summary.json"
            if not summary_file.exists():
                continue
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            result = summary.get("result", {})
            if summary.get("state") != "complete" or result.get("termination_status") != "OPTIMAL":
                continue
            piece_values[parent].append(float(result["objective_value"]))
            piece_bounds[parent].append(float(result["objective_bound"]))

    output_rows: list[dict[str, object]] = []
    with args.composite_summary.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["termination_status"] != "OPTIMAL":
                continue
            parent = int(row["instance_index"])
            if len(piece_values[parent]) != len(expected[parent]):
                continue
            standalone_objective = sum(piece_values[parent])
            standalone_bound = sum(piece_bounds[parent])
            composite_objective = float(row["objective_value"])
            composite_bound = float(row["objective_bound"])
            delta = composite_objective - standalone_objective
            output_rows.append(
                {
                    "instance_index": parent,
                    "strategy": row["strategy"],
                    "piece_count": len(expected[parent]),
                    "standalone_objective_sum": standalone_objective,
                    "standalone_bound_sum": standalone_bound,
                    "composite_objective": composite_objective,
                    "composite_bound": composite_bound,
                    "composite_minus_standalone": delta,
                    "relative_delta_percent": 100.0 * delta / standalone_objective,
                }
            )

    output_rows.sort(key=lambda row: (int(row["instance_index"]), str(row["strategy"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0]) if output_rows else [
        "instance_index",
        "strategy",
        "piece_count",
        "standalone_objective_sum",
        "standalone_bound_sum",
        "composite_objective",
        "composite_bound",
        "composite_minus_standalone",
        "relative_delta_percent",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} comparisons to {args.output}")


if __name__ == "__main__":
    main()
