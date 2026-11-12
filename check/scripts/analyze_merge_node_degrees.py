#!/usr/bin/env python3
"""Recover merge-node degree distributions from saved merge-stage plots."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result_roots",
        nargs="+",
        type=Path,
        help="Consolidated result directories containing tables/ and plots/.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional destination for the tidy degree-distribution CSV.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional destination for the complete summary JSON.",
    )
    parser.add_argument(
        "--plot-data-directory",
        type=Path,
        help="Optional destination for one degree/fraction data file per series.",
    )
    return parser.parse_args()


def canonical_edge(edge: dict[str, object]) -> tuple[str, str]:
    endpoints = sorted((str(edge["source"]), str(edge["target"])))
    return endpoints[0], endpoints[1]


def load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def stage_degrees(stage_directory: Path) -> tuple[list[int], int]:
    stage_name = stage_directory.name
    stage_index = int(stage_name.removeprefix("stage_"))
    prefix = f"merge_stage_{stage_index:03d}"
    original = load_json(stage_directory / f"{prefix}_original.json")
    final = load_json(stage_directory / f"{prefix}_final.json")

    original_edges = {
        canonical_edge(edge) for edge in original["edges"]  # type: ignore[index]
    }
    final_merge_edges = {
        canonical_edge(edge)
        for edge in final["edges"]  # type: ignore[index]
        if edge["merge_edge"]
    }
    new_edges = final_merge_edges - original_edges

    degrees: Counter[str] = Counter()
    for first, second in original_edges:
        degrees[first] += 1
        degrees[second] += 1

    selected_degrees = [
        degrees[node]
        for edge in sorted(new_edges)
        for node in edge
    ]
    return selected_degrees, len(new_edges)


def accepted_strategies(result_root: Path) -> dict[str, str]:
    table = result_root / "tables" / "accepted_instances.csv"
    with table.open(newline="", encoding="utf-8") as stream:
        return {
            row["artifact_stem"]: row["strategy"]
            for row in csv.DictReader(stream)
        }


def summarize(result_roots: list[Path]) -> dict[str, object]:
    counts: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    pair_counts: Counter[tuple[str, str]] = Counter()
    stage_counts: Counter[tuple[str, str]] = Counter()
    artifacts_seen: Counter[tuple[str, str]] = Counter()

    for result_root in result_roots:
        domain = str(load_json(result_root / "manifest.json").get("domain", result_root.name))
        strategies = accepted_strategies(result_root)
        for artifact_stem, strategy in strategies.items():
            artifact = result_root / "plots" / "merged" / artifact_stem
            stage_directories = sorted(artifact.glob("stage_*"))
            if not stage_directories:
                raise ValueError(f"no merge stages found below {artifact}")
            artifacts_seen[domain, strategy] += 1
            for stage_directory in stage_directories:
                degrees, pair_count = stage_degrees(stage_directory)
                if len(degrees) != 2 * pair_count:
                    raise AssertionError(f"invalid endpoint count in {stage_directory}")
                counts[domain, strategy].update(degrees)
                pair_counts[domain, strategy] += pair_count
                stage_counts[domain, strategy] += 1

    rows = []
    groups = []
    for domain, strategy in sorted(counts):
        distribution = counts[domain, strategy]
        endpoint_count = distribution.total()
        group_rows = [
            {
                "domain": domain,
                "strategy": strategy,
                "degree": degree,
                "selected_endpoint_count": count,
                "selected_endpoint_fraction": count / endpoint_count,
            }
            for degree, count in sorted(distribution.items())
        ]
        rows.extend(group_rows)
        groups.append(
            {
                "domain": domain,
                "strategy": strategy,
                "artifact_count": artifacts_seen[domain, strategy],
                "stage_count": stage_counts[domain, strategy],
                "pair_count": pair_counts[domain, strategy],
                "selected_endpoint_count": endpoint_count,
                "degree_one_count": distribution[1],
                "degree_one_fraction": distribution[1] / endpoint_count,
                "degree_distribution": {
                    str(degree): count for degree, count in sorted(distribution.items())
                },
            }
        )
    return {"groups": groups, "distribution": rows}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plot_data(path: Path, rows: list[dict[str, object]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for row in rows:
        domain = "pglib" if str(row["domain"]).startswith("PGLib") else "snd"
        grouped[domain, str(row["strategy"])][int(row["degree"])] = float(
            row["selected_endpoint_fraction"]
        )
    for (domain, strategy), distribution in grouped.items():
        lines = ["degree fraction"]
        lines.extend(
            f"{degree} {distribution.get(degree, 0.0):.12g}"
            for degree in range(1, max(distribution) + 1)
        )
        (path / f"merge-node-degree-{domain}-{strategy}.dat").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


def main() -> None:
    args = parse_args()
    summary = summarize([path.resolve() for path in args.result_roots])
    if args.csv:
        write_csv(args.csv, summary["distribution"])
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.plot_data_directory:
        write_plot_data(args.plot_data_directory, summary["distribution"])
    print(json.dumps(summary["groups"], indent=2))


if __name__ == "__main__":
    main()
