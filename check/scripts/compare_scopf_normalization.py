#!/usr/bin/env python3
"""Compare baseline and voltage-aware component-normalized SCOPF outcomes."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from statistics import mean, median


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPOSITORY_ROOT / "check/results"
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
        "--baseline",
        type=Path,
        default=RESULTS / "full_shared4_300_consecutive_pglib_scopf/summary.csv",
    )
    parser.add_argument(
        "--normalized",
        type=Path,
        default=RESULTS
        / "full_shared4_300_consecutive_voltage_aware_normalized_pglib_scopf"
        / "summary.csv",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=RESULTS
        / "full_shared4_300_consecutive_voltage_aware_normalized_pglib_scopf"
        / "normalization_sensitivity.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS
        / "full_shared4_300_consecutive_voltage_aware_normalized_pglib_scopf"
        / "paired_sensitivity",
    )
    parser.add_argument(
        "--paper-table",
        type=Path,
        help="write the LaTeX feasibility-count table used by the paper",
    )
    return parser.parse_args()


def _read(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"missing input: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    keyed = {row["artifact_stem"]: row for row in rows}
    if len(keyed) != len(rows):
        raise SystemExit(f"duplicate artifact_stem in {path}")
    return keyed


def _truth(value: str | None) -> bool:
    return value == "True"


def _outcome(row: dict[str, str]) -> str:
    if row.get("state") == "complete" and _truth(row.get("scopf_feasible")):
        return "feasible"
    if (
        row.get("state") == "complete"
        and row.get("termination_status") == "INFEASIBLE"
    ):
        return "infeasible"
    return "other"


def _number(row: dict[str, str], field: str) -> float:
    return float(row[field])


def _summary(
    name: str,
    records: list[dict[str, object]],
) -> dict[str, object]:
    transitions = Counter(
        (str(record["baseline_outcome"]), str(record["normalized_outcome"]))
        for record in records
    )
    baseline = Counter(str(record["baseline_outcome"]) for record in records)
    normalized = Counter(str(record["normalized_outcome"]) for record in records)
    scale_ratios = [float(record["component_scale_ratio"]) for record in records]
    voltage_ratios = [float(record["median_voltage_ratio"]) for record in records]
    maximum_voltage_ratios = [
        float(record["maximum_voltage_ratio"]) for record in records
    ]
    same_voltage = [float(record["same_voltage_fraction"]) for record in records]
    within_ten = [
        float(record["within_ten_percent_fraction"]) for record in records
    ]
    return {
        "scope": name,
        "job_count": len(records),
        "baseline_feasible": baseline["feasible"],
        "normalized_feasible": normalized["feasible"],
        "feasible_retained": transitions[("feasible", "feasible")],
        "feasible_lost": sum(
            transitions[("feasible", target)]
            for target in ("infeasible", "other")
        ),
        "feasible_gained": sum(
            transitions[(source, "feasible")]
            for source in ("infeasible", "other")
        ),
        "neither_feasible": sum(
            count
            for (source, target), count in transitions.items()
            if source != "feasible" and target != "feasible"
        ),
        "baseline_certified_infeasible": baseline["infeasible"],
        "normalized_certified_infeasible": normalized["infeasible"],
        "baseline_other": baseline["other"],
        "normalized_other": normalized["other"],
        "baseline_overload_free": sum(
            bool(record["baseline_overload_free"]) for record in records
        ),
        "normalized_overload_free": sum(
            bool(record["normalized_overload_free"]) for record in records
        ),
        "median_component_scale_ratio": median(scale_ratios),
        "median_coupling_voltage_ratio": median(voltage_ratios),
        "median_maximum_coupling_voltage_ratio": median(maximum_voltage_ratios),
        "mean_same_voltage_fraction": mean(same_voltage),
        "mean_within_ten_percent_fraction": mean(within_ten),
    }


def _paper_table(summaries: list[dict[str, object]]) -> str:
    by_scope = {str(summary["scope"]): summary for summary in summaries}
    columns = [by_scope[strategy] for strategy in STRATEGIES]
    baseline = [int(summary["baseline_feasible"]) for summary in columns]
    normalized = [int(summary["normalized_feasible"]) for summary in columns]
    labels = [STRATEGY_LABELS[strategy] for strategy in STRATEGIES]

    def row(label: str, values: list[int]) -> str:
        fields = " & ".join(str(value) for value in values)
        return f"  {label:<21} & {fields} & {sum(values)} \\\\"

    return "\n".join(
        (
            r"\begin{tabular}{lrrrrrr}",
            r"  \toprule",
            "  & " + " & ".join(labels) + r" & Total \\",
            r"  \midrule",
            row("Without normalization", baseline),
            row("With normalization", normalized),
            r"  \bottomrule",
            r"\end{tabular}",
            "",
        )
    )


def main() -> None:
    args = parse_args()
    baseline = _read(args.baseline.resolve())
    normalized = _read(args.normalized.resolve())
    audit = _read(args.audit.resolve())
    if set(baseline) != set(normalized) or set(baseline) != set(audit):
        raise SystemExit("baseline, normalized, and audit artifact sets differ")

    records: list[dict[str, object]] = []
    for artifact_stem in baseline:
        before = baseline[artifact_stem]
        after = normalized[artifact_stem]
        parameters = audit[artifact_stem]
        if len(
            {
                before["strategy"],
                after["strategy"],
                parameters["strategy"],
            }
        ) != 1:
            raise SystemExit(f"strategy mismatch for {artifact_stem}")
        minimum_scale = _number(parameters, "minimum_scale_factor")
        maximum_scale = _number(parameters, "maximum_scale_factor")
        records.append(
            {
                "artifact_stem": artifact_stem,
                "cohort": parameters["cohort"],
                "strategy": before["strategy"],
                "baseline_outcome": _outcome(before),
                "normalized_outcome": _outcome(after),
                "baseline_overload_free": _truth(
                    before.get("scopf_feasible_without_overload")
                ),
                "normalized_overload_free": _truth(
                    after.get("scopf_feasible_without_overload")
                ),
                "component_scale_ratio": maximum_scale / minimum_scale,
                "same_voltage_fraction": _number(
                    parameters, "same_voltage_fraction"
                ),
                "within_ten_percent_fraction": _number(
                    parameters, "within_ten_percent_fraction"
                ),
                "median_voltage_ratio": _number(
                    parameters, "median_voltage_ratio"
                ),
                "maximum_voltage_ratio": _number(
                    parameters, "maximum_voltage_ratio"
                ),
            }
        )

    summaries = [_summary("all", records)]
    summaries.extend(
        _summary(
            strategy,
            [record for record in records if record["strategy"] == strategy],
        )
        for strategy in STRATEGIES
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary_path = args.output.with_name(args.output.name + "_summary.csv")
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    json_path = args.output.with_name(args.output.name + "_summary.json")
    json_path.write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.paper_table is not None:
        args.paper_table.parent.mkdir(parents=True, exist_ok=True)
        args.paper_table.write_text(_paper_table(summaries), encoding="utf-8")
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
