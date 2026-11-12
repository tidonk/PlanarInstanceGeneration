#!/usr/bin/env python3
"""Export structural and application runtime summaries for the paper."""
import argparse
import csv
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("random", "high_degree", "kamada_kawai", "heuristic_outer_face", "exact_outer_face")
LABELS = ("Random", "High degree", "Kamada--Kawai", "Outer face (heuristic)", "Outer face (branch-and-cut)")
SHORT_LABELS = ("R", "HD", "KK", "OF-H", "OF-B")
SOURCE_LABEL = "Original sources"
SOURCE_SHORT_LABEL = "Source"
RUNTIME_FIELDS = {"scopf": "solver_solve_seconds", "snd": "solver_total_seconds"}
RUNTIME_FIGURES = {
    "scopf": "fig-eval-application-runtime-scots.tikz",
    "snd": "fig-eval-application-runtime-snd.tikz",
}


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def select_source_rows(data, generation_root):
    evaluation = generation_root / "evaluation"
    structural_path = evaluation / "structural_exact.csv"
    if not structural_path.is_file():
        structural_path = evaluation / "structural.csv"
    names = {row["instance"] for row in rows(structural_path)
             if row["group"] == "Original sources"}
    selected = [row for row in data
                if row["strategy"] == "source" and row["artifact_stem"] in names]
    observed = {row["artifact_stem"] for row in selected}
    if len(names) != 16 or observed != names or len(selected) != len(names):
        raise ValueError("source results must cover each of the 16 generation sources exactly once")
    return selected


def shifted_mean(values):
    assert values and all(math.isfinite(x) and x >= 0 for x in values)
    return math.expm1(sum(math.log1p(x) for x in values) / len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper", type=Path)
    parser.add_argument("--domain", choices=("all", "scots", "snd"), default="all")
    parser.add_argument(
        "--pglib-root", type=Path,
        default=ROOT / "check/results/full_shared4_300_consecutive_pglib",
    )
    parser.add_argument(
        "--snd-root", type=Path,
        default=ROOT / "check/results/full_shared4_300_consecutive_snd",
    )
    parser.add_argument(
        "--scots-summary", "--scopf-summary", dest="scots_summary", type=Path,
        default=ROOT / "check/results/full_shared4_300_consecutive_pglib_scots_tal1_10800/summary.csv",
    )
    parser.add_argument(
        "--snd-summary", type=Path,
        default=ROOT / "check/results/full_shared4_300_consecutive_snd_benders_10800/summary.csv",
    )
    parser.add_argument(
        "--scots-source-summary", type=Path,
        default=ROOT / "check/results/pglib_source_scots_tal1_10800/summary.csv",
    )
    parser.add_argument(
        "--snd-source-summary", type=Path,
        default=ROOT / "check/results/snd_source_benders_10800/summary.csv",
    )
    parser.add_argument("--scots-include-warmstart", action="store_true",
                        help="plot warmstart plus main SCOTS solve time")
    parser.add_argument("--omit-scots-sources", action="store_true")
    args = parser.parse_args()
    out = args.paper / "resources/evaluation"
    roots = {"pglib": args.pglib_root, "snd": args.snd_root}
    if args.domain == "all":
        lines = [r"\begin{table}[htbp]\centering\small",
                 r"\caption{Median global characteristics over 100 composites per method. "
                 r"$d$ is diameter, $\lambda_2$ algebraic connectivity, "
                 r"and $r$ degree assortativity.}\label{tab:eval-structure}",
                 r"\begin{tabular}{llrrr}\toprule",
                 r"Domain & method & $d$ & $\lambda_2$ & $r$\\",
                 r"\midrule"]
        for domain in ("pglib", "snd"):
            evaluation = roots[domain] / "evaluation"
            paired_path = evaluation / "paired_strategy_outcomes_exact.csv"
            if not paired_path.is_file():
                paired_path = evaluation / "paired_strategy_outcomes.csv"
            data = rows(paired_path)
            assert len(data) == 500
            for method, label in zip(METHODS, LABELS):
                subset = [r for r in data if r["strategy"] == method]
                assert len(subset) == 100
                def med(key):
                    return median(float(r["structural_" + key]) for r in subset)
                title = ("PGLib" if domain == "pglib" else "SNDlib") if method == "random" else ""
                lines.append(f"{title} & {label} & {med('diameter'):g} & "
                             f"{med('algebraic_connectivity'):.6f} & "
                             f"{med('degree_assortativity'):.6f}" + r"\\")
            lines.append(r"\midrule" if domain == "pglib" else r"\bottomrule")
        lines.extend([r"\end{tabular}\end{table}", ""])
        (out / "exact-structure-table.tex").write_text("\n".join(lines))

    data_directory = out / "plot-data"
    data_directory.mkdir(exist_ok=True)
    marks = ("*", "triangle*", "square*", "diamond*", "triangle*")
    for domain, summary, source_summary in (
        ("scopf", args.scots_summary, args.scots_source_summary),
        ("snd", args.snd_summary, args.snd_source_summary),
    ):
        if args.domain != "all" and domain != {"scots": "scopf", "snd": "snd"}[args.domain]:
            continue
        data = rows(summary)
        assert len(data) == 500
        method_rows = []
        all_times = []
        for method_index, (method, label) in enumerate(zip(METHODS, LABELS)):
            subset = [
                row for row in data
                if row["strategy"] == method
                and row["state"] == "complete"
                and row["termination_status"] == "OPTIMAL"
            ]
            times = sorted(float(row["solver_solve_with_warmstart_seconds" if domain == "scopf" and args.scots_include_warmstart else RUNTIME_FIELDS[domain]]) for row in subset)
            all_times.extend(times)
            points = ["runtime y"]
            for index, runtime in enumerate(times):
                offset = 0.0
                points.append(f"{runtime:.12g} {method_index + offset:.4f}")
            data_path = data_directory / f"application-runtime-{domain}-{method}.dat"
            data_path.write_text("\n".join(points) + "\n")
            relative = data_path.relative_to(out.parents[1]).as_posix()
            method_rows.append((label, len(times), relative, shifted_mean(times), marks[method_index]))

        include_sources = not (domain == "scopf" and args.omit_scots_sources)
        if include_sources:
            source_data = select_source_rows(
                rows(source_summary), roots["pglib" if domain == "scopf" else "snd"]
            )
            source_times = sorted(
                float(row["solver_solve_with_warmstart_seconds" if domain == "scopf" and args.scots_include_warmstart else RUNTIME_FIELDS[domain]])
                for row in source_data
                if row["strategy"] == "source"
                and row["state"] == "complete"
                and row["termination_status"] == "OPTIMAL"
            )
            all_times.extend(source_times)
            assert source_times
            source_points = ["runtime y"]
            for index, runtime in enumerate(source_times):
                offset = 0.0
                source_points.append(f"{runtime:.12g} {-1.5 + offset:g}")
            source_path = data_directory / f"application-runtime-{domain}-source.dat"
            source_path.write_text("\n".join(source_points) + "\n")
            source_relative = source_path.relative_to(out.parents[1]).as_posix()
            method_rows.append(
                (
                    SOURCE_LABEL,
                    len(source_times),
                    source_relative,
                    shifted_mean(source_times),
                    "pentagon*",
                )
            )
        short_labels = ((SOURCE_SHORT_LABEL,) if include_sources else ()) + SHORT_LABELS
        yticklabels = ",".join(short_labels)
        tex = [
            "% Generated by check/scripts/export_paper_exact_tables.py.",
            r"\begingroup",
            r"\pigSetRowSpacing",
            r"\begin{tikzpicture}",
            r"\begin{axis}[",
            r"  width=.96\linewidth,height=\pigSourcePlotHeight,",
            f"  xmode=log,xmin={min(4, min(all_times)*.95):.12g},xmax={max(10800, max(all_times)*1.02):.12g},",
            r"  xtick={5,30,180,1080,10800},",
            r"  xticklabels={5,30,180,1080,10800},",
            (r"  ymin=\pigSourceYMin,ymax=4.64,ytick={\pigSourceY,0,1,2,3,4},y dir=reverse,"
             if include_sources else r"  ymin=-.64,ymax=4.64,ytick={0,1,2,3,4},y dir=reverse,"),
            "  yticklabels={" + yticklabels + "},",
            r"  axis lines*=left,axis line style={line width=.5pt},",
            r"  tick label style={font=\scriptsize},label style={font=\scriptsize},",
            r"  xlabel={Reported solver time [s]},",
            r"  xmajorgrids=true,grid style={gray!20,line width=.3pt},",
            r"]",
        ]
        for method_index, (_, _, relative, shifted, mark) in enumerate(method_rows):
            rotation = ",mark options={rotate=180}" if method_index == 4 else ""
            y_column = r"y expr=\pigSourceY" if method_index == len(METHODS) else "y=y"
            row_position = r"\pigSourceY" if method_index == len(METHODS) else method_index
            tex.append(
                r"\addplot[only marks,mark=" + mark + rotation
                + r",mark size=1.25pt,draw=pigMethod" + str(method_index)
                + r",fill=pigMethod" + str(method_index) + r",opacity=.65] "
                + "table[x=runtime," + y_column + "] {" + relative + "};"
            )
            tex.append(
                r"\addplot[only marks,mark=|,mark size=5pt,line width=1.1pt,color=pigMethod"
                + str(method_index) + r"] "
                + f"coordinates {{({shifted:.12g},{row_position})}};"
            )
        tex.extend([r"\end{axis}", r"\end{tikzpicture}", r"\endgroup", ""])
        (out / RUNTIME_FIGURES[domain]).write_text("\n".join(tex))


if __name__ == "__main__":
    main()
