#!/usr/bin/env python3
"""Export full per-instance added-skewness tables for the paper appendix."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STRATEGIES = (
    "random",
    "high_degree",
    "kamada_kawai",
    "heuristic_outer_face",
    "exact_outer_face",
)
STRATEGY_LABELS = {
    "random": "Random",
    "high_degree": "High degree",
    "kamada_kawai": "Kamada--Kawai",
    "heuristic_outer_face": "Heuristic face",
    "exact_outer_face": "Exact face",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        type=Path,
        help="destination .tex file",
    )
    parser.add_argument("--pglib", type=Path, required=True)
    parser.add_argument("--snd", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path) -> dict[int, dict[str, tuple[float, float]]]:
    by_instance: dict[int, dict[str, tuple[float, float]]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            index = int(row.get("output_index") or row["cohort"])
            strategy = row["strategy"]
            if strategy not in STRATEGIES:
                raise ValueError(f"unknown strategy {strategy!r} in {path}")
            lower = float(
                row.get("skewness_inflation_lower_bound")
                or row["inflation_lower_bound"]
            )
            upper = float(
                row.get("skewness_inflation_upper_bound")
                or row["inflation_upper_bound"]
            )
            if not (math.isfinite(lower) and math.isfinite(upper)) or lower < 0 or upper < lower:
                raise ValueError(
                    f"invalid added-skewness interval [{lower}, {upper}] in {path}"
                )
            values = by_instance.setdefault(index, {})
            if strategy in values:
                raise ValueError(f"duplicate {strategy} result for instance i{index:03d} in {path}")
            values[strategy] = (lower, upper)
    expected_indices = set(range(1, 101))
    if set(by_instance) != expected_indices:
        missing = sorted(expected_indices - set(by_instance))
        extra = sorted(set(by_instance) - expected_indices)
        raise ValueError(f"expected i001--i100 in {path}; missing={missing}, extra={extra}")
    expected_strategies = set(STRATEGIES)
    for index, values in by_instance.items():
        if set(values) != expected_strategies:
            missing = sorted(expected_strategies - set(values))
            raise ValueError(f"i{index:03d} in {path} is missing {missing}")
    return by_instance


def _shifted_geometric_mean(values: list[float]) -> float:
    return math.exp(math.fsum(math.log1p(value) for value in values) / len(values)) - 1


def _format_value(value: float, *, summary: bool = False) -> str:
    if not summary and value.is_integer():
        return str(int(value))
    return f"{value:.3f}"


def _format_interval(interval: tuple[float, float], *, summary: bool = False) -> str:
    lower, upper = interval
    if math.isclose(lower, upper, abs_tol=5e-13):
        return _format_value(lower, summary=summary)
    return f"[{_format_value(lower, summary=summary)},\,{_format_value(upper, summary=summary)}]"


def _provable_winners(
    intervals: dict[str, tuple[float, float]],
) -> set[str]:
    """Return outer-face methods provably better than every non-outer method."""
    non_outer_strategies = STRATEGIES[:3]
    outer_face_strategies = STRATEGIES[3:]
    return {
        strategy
        for strategy in outer_face_strategies
        if all(
            intervals[strategy][1] < intervals[other][0]
            for other in non_outer_strategies
        )
    }


def _summary_interval(
    values: dict[int, dict[str, tuple[float, float]]],
    strategy: str,
    statistic: str,
) -> tuple[float, float]:
    lowers = [values[index][strategy][0] for index in sorted(values)]
    uppers = [values[index][strategy][1] for index in sorted(values)]
    if statistic == "mean":
        operation = statistics.fmean
    elif statistic == "median":
        operation = statistics.median
    elif statistic == "sgm":
        operation = _shifted_geometric_mean
    else:
        raise ValueError(f"unknown statistic {statistic!r}")
    return float(operation(lowers)), float(operation(uppers))


def _table(domain: str, values: dict[int, dict[str, tuple[float, float]]]) -> str:
    headers = (
        "Instance",
        "Random",
        r"\makecell{High\\degree}",
        r"\makecell{Kamada--\\Kawai}",
        r"\makecell{Outer face\\(heuristic)}",
        r"\makecell{Outer face\\(branch-and-cut)}",
    )
    header = " & ".join(headers)
    lines = [
        "\\begingroup\n",
        "\\scriptsize\n",
        "\\setlength{\\tabcolsep}{3.2pt}\n",
        "\\begin{longtable}{lrrrrr}\n",
        fr"\caption{{Certified added skewness intervals for 100 {domain} inputs. "
        "Single values are exact; bold marks an outer face upper bound below all R, HD, and KK lower bounds.}"
        fr"\label{{tab:app-full-skewness-{domain.lower()}}}\\" + "\n",
        "\\toprule\n",
        f"{header} \\\\\n",
        "\\midrule\n",
        "\\endfirsthead\n",
        f"\\multicolumn{{6}}{{l}}{{\\tablename~\\thetable\ continued}}\\\\\n",
        "\\toprule\n",
        f"{header} \\\\\n",
        "\\midrule\n",
        "\\endhead\n",
        "\\midrule\n",
        "\\multicolumn{6}{r}{Continued on next page}\\\\\n",
        "\\endfoot\n",
        "\\bottomrule\n",
        "\\endlastfoot\n",
    ]
    for index in sorted(values):
        winners = _provable_winners(values[index])
        formatted = {
            strategy: _format_interval(values[index][strategy])
            for strategy in STRATEGIES
        }
        cells = [f"$i{index:03d}$"] + [
            (
                f"\\textbf{{{formatted[strategy]}}}"
                if strategy in winners
                else formatted[strategy]
            )
            for strategy in STRATEGIES
        ]
        lines.append(" & ".join(cells) + " \\\\\n")
    lines.append("\\midrule\n")
    for label, statistic in (
        ("Mean", "mean"),
        ("Median", "median"),
        ("Shifted geometric mean", "sgm"),
    ):
        display_label = (
            r"\makecell[l]{Shifted\\geometric mean}"
            if statistic == "sgm"
            else label
        )
        cells = [display_label] + [
            _format_interval(
                _summary_interval(values, strategy, statistic), summary=True
            )
            for strategy in STRATEGIES
        ]
        lines.append(" & ".join(cells) + " \\\\\n")
    lines.extend(["\\end{longtable}\n", "\\endgroup\n"])
    return "".join(lines)


def main() -> None:
    args = parse_args()
    domains = (("PGLib", args.pglib), ("SNDlib", args.snd))
    text = "% Generated by check/scripts/export_paper_skewness_tables.py.\n"
    text += "\n\\clearpage\n\n".join(
        _table(domain, _read(path)) for domain, path in domains
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
