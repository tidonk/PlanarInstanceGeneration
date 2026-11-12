#!/usr/bin/env python3
"""Compare optimal SND composites with independently optimized pieces."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIRECTORY = REPOSITORY_ROOT / "check/results"
STRATEGIES = (
    "random",
    "high_degree",
    "kamada_kawai",
    "heuristic_outer_face",
    "exact_outer_face",
)
STRATEGY_LABELS = {
    "random": "R",
    "high_degree": "HD",
    "kamada_kawai": "KK",
    "heuristic_outer_face": "OF-H",
    "exact_outer_face": "OF-B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--composite-run",
        type=Path,
        default=RESULTS_DIRECTORY / "merge_count_600_3600_20260911_log2_snd_benders",
    )
    parser.add_argument(
        "--piece-run",
        type=Path,
        default=(
            RESULTS_DIRECTORY
            / "merge_count_600_3600_20260911_log2_piece_comparison_snd_pieces_benders"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="comparison CSV; defaults to PIECE_RUN/piece_objective_comparison.csv",
    )
    parser.add_argument(
        "--paper-table",
        type=Path,
        help="write a LaTeX table body for inclusion in the paper",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _optimal_objective(run: Path, stem: str) -> tuple[float, float] | None:
    path = run / "logs" / stem / "summary.json"
    if not path.is_file():
        raise SystemExit(f"missing application summary: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    result = summary.get("result") or {}
    if summary.get("state") != "complete" or result.get("termination_status") != "OPTIMAL":
        return None
    objective = result.get("objective_value")
    bound = result.get("objective_bound")
    if not _finite(objective) or not _finite(bound):
        return None
    return float(objective), float(bound)


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def comparison_rows(composite_run: Path, piece_run: Path) -> list[dict[str, object]]:
    piece_jobs = _read_csv(piece_run / "jobs.csv")
    pieces_by_parent: dict[int, list[str]] = {}
    for row in piece_jobs:
        pieces_by_parent.setdefault(int(row["parent_instance_index"]), []).append(
            row["artifact_stem"]
        )

    piece_totals: dict[int, tuple[int, float, float]] = {}
    for parent, stems in pieces_by_parent.items():
        values = [_optimal_objective(piece_run, stem) for stem in stems]
        if all(value is not None for value in values):
            optimal = [value for value in values if value is not None]
            piece_totals[parent] = (
                len(optimal),
                sum(value[0] for value in optimal),
                sum(value[1] for value in optimal),
            )

    result: list[dict[str, object]] = []
    for row in _read_csv(composite_run / "jobs.csv"):
        parent = int(row["instance_index"])
        if parent not in piece_totals:
            continue
        composite = _optimal_objective(composite_run, row["artifact_stem"])
        if composite is None:
            continue
        piece_count, standalone_objective, standalone_bound = piece_totals[parent]
        difference = composite[0] - standalone_objective
        result.append(
            {
                "instance_index": parent,
                "strategy": row["strategy"],
                "piece_count": piece_count,
                "standalone_objective_sum": standalone_objective,
                "standalone_bound_sum": standalone_bound,
                "composite_objective": composite[0],
                "composite_bound": composite[1],
                "composite_minus_standalone": difference,
                "relative_delta_percent": 100.0 * difference / standalone_objective,
            }
        )
    return sorted(result, key=lambda row: (int(row["instance_index"]), STRATEGIES.index(str(row["strategy"]))))


def summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for strategy in (*STRATEGIES, "all"):
        subset = rows if strategy == "all" else [row for row in rows if row["strategy"] == strategy]
        differences = [float(row["composite_minus_standalone"]) for row in subset]
        relative = [float(row["relative_delta_percent"]) for row in subset]
        summaries.append(
            {
                "strategy": strategy,
                "count": len(subset),
                "lower": sum(value < 0.0 for value in differences),
                "equal": sum(value == 0.0 for value in differences),
                "higher": sum(value > 0.0 for value in differences),
                "median_relative_delta_percent": (
                    statistics.median(relative) if relative else None
                ),
            }
        )
    return summaries


def paper_table(summaries: list[dict[str, object]]) -> str:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"  \toprule",
        r"  & $N$ & Lower & Equal & Higher & Median change (\si{\percent}) \\",
        r"  \midrule",
    ]
    for row in summaries:
        strategy = str(row["strategy"])
        label = "All" if strategy == "all" else STRATEGY_LABELS[strategy]
        median = row["median_relative_delta_percent"]
        median_text = "--" if median is None else f"{float(median):.1f}"
        lines.append(
            f"  {label:<5} & {int(row['count'])} & {int(row['lower'])} & "
            f"{int(row['equal'])} & {int(row['higher'])} & "
            f"{median_text} \\\\"
        )
    lines.extend((r"  \bottomrule", r"\end{tabular}", ""))
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise SystemExit("no jointly optimal composite-piece comparisons")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    composite_run = args.composite_run.resolve()
    piece_run = args.piece_run.resolve()
    rows = comparison_rows(composite_run, piece_run)
    output = args.output or piece_run / "piece_objective_comparison.csv"
    _write_csv(output, rows)
    summaries = summary_rows(rows)
    if args.paper_table is not None:
        args.paper_table.parent.mkdir(parents=True, exist_ok=True)
        args.paper_table.write_text(paper_table(summaries), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
