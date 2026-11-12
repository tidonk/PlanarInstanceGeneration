#!/usr/bin/env python3
"""Export the manuscript's structural evaluation figures as native PGFPlots/TikZ."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
METHODS = (
    "random",
    "high_degree",
    "kamada_kawai",
    "heuristic_outer_face",
    "exact_outer_face",
)
LABELS = ("R", "HD", "KK", "OF-H", "OF-B")
COLORS = ("648FFF", "785EF0", "DC267F", "FE6100", "FFB000")


def rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def num(row, key):
    return float(row[key])


def top_five_percent_mean(values):
    values = np.sort(np.asarray(values, dtype=float))
    count = max(1, int(np.ceil(.05 * len(values))))
    return float(values[-count:].mean())


def load_domain(root, *, include_betweenness=False):
    evaluation = root / "evaluation"
    structural_path = evaluation / "structural_exact.csv"
    if not structural_path.is_file():
        structural_path = evaluation / "structural.csv"
    structural = rows(structural_path)
    source = [row for row in structural if row["group"] == "Original sources"]

    paired_path = evaluation / "paired_strategy_outcomes_exact.csv"
    if not paired_path.is_file():
        paired_path = evaluation / "paired_strategy_outcomes.csv"
    paired = rows(paired_path)

    # Retain the optional measurements for separate diagnostic analyses.
    # The manuscript figures and tables do not require these checkpoints.
    if include_betweenness:
        top_five_by_instance = {}
        for row in structural:
            checkpoint_name = (
                row["group"].replace(":", "").replace(" ", "_")
                + "_" + row["instance"] + ".json"
            )
            checkpoint = json.loads(
                (evaluation / "exact_betweenness" / checkpoint_name).read_text()
            )
            values = [entry["betweenness"] for entry in checkpoint["nodes"]]
            assert len(values) == int(row["node_count"])
            top_five_by_instance[row["instance"]] = top_five_percent_mean(values)
            row["betweenness_top_five_mean"] = top_five_by_instance[row["instance"]]
        for row in paired:
            row["structural_betweenness_top_five_mean"] = (
                top_five_by_instance[row["artifact_stem"]]
            )

    return {
        "structural": structural,
        "source": source,
        "paired": paired,
        "audits": rows(evaluation / "preservation_properties.csv"),
        "distributions": rows(evaluation / "preservation_degree_distribution.csv"),
    }


def definitions():
    return [
        rf"\definecolor{{pigMethod{index}}}{{HTML}}{{{color}}}"
        for index, color in enumerate(COLORS)
    ]


def mark_options(index):
    marks = ("*", "triangle*", "square*", "diamond*", "triangle*")
    rotation = ",mark options={rotate=180}" if index == 4 else ""
    return marks[index], rotation


def write_plot_data(destination, name, x_values, y_values):
    data_directory = destination / "plot-data"
    data_directory.mkdir(exist_ok=True)
    path = data_directory / (name + ".dat")
    lines = ["x y"] + [
        f"{x:.12g} {y:.12g}" for x, y in zip(x_values, y_values)
    ]
    path.write_text("\n".join(lines) + "\n")
    return path.relative_to(destination.parents[1]).as_posix()


def symlog(values):
    """Linear on [-1, 1] and base-10 logarithmic outside."""
    values = np.asarray(values, dtype=float)
    transformed = values.copy()
    outside = np.abs(values) > 1
    transformed[outside] = np.sign(values[outside]) * (
        1 + np.log10(np.abs(values[outside]))
    )
    return transformed


def boxplot_summary(values):
    values = np.sort(np.asarray(values, dtype=float))
    q1, median, q3 = np.quantile(values, [.25, .5, .75])
    iqr = q3 - q1
    lower = values[values >= q1 - 1.5 * iqr][0]
    upper = values[values <= q3 + 1.5 * iqr][-1]
    return lower, q1, median, q3, upper


def add_box(lines, center, values, color, half_width, cap_width, transform):
    lower, q1, median, q3, upper = transform(boxplot_summary(values))
    lines.extend([
        rf"\draw[color={color},line width=.45pt] (axis cs:{center:.4f},{lower:.12g}) -- (axis cs:{center:.4f},{q1:.12g});",
        rf"\draw[color={color},line width=.45pt] (axis cs:{center:.4f},{q3:.12g}) -- (axis cs:{center:.4f},{upper:.12g});",
        rf"\draw[color={color},line width=.45pt] (axis cs:{center-cap_width:.4f},{lower:.12g}) -- (axis cs:{center+cap_width:.4f},{lower:.12g});",
        rf"\draw[color={color},line width=.45pt] (axis cs:{center-cap_width:.4f},{upper:.12g}) -- (axis cs:{center+cap_width:.4f},{upper:.12g});",
        rf"\path[draw={color},fill={color},fill opacity=.16,draw opacity=1,line width=.5pt] (axis cs:{center-half_width:.4f},{q1:.12g}) rectangle (axis cs:{center+half_width:.4f},{q3:.12g});",
        rf"\draw[black,line width=.65pt] (axis cs:{center-half_width:.4f},{median:.12g}) -- (axis cs:{center+half_width:.4f},{median:.12g});",
    ])


def export_bounded(destination, inputs):
    """Export bounded structural changes with a shared legend."""
    for domain_index, domain in enumerate(("pglib", "snd")):
        data = inputs[domain]
        paired = data["paired"]
        strategy = {row["artifact_stem"]: row["strategy"] for row in paired}

        degree_change = {}
        for row in data["distributions"]:
            value = num(row, "absolute_change") / num(row, "paper_upper_bound")
            degree_change[row["instance"]] = max(
                degree_change.get(row["instance"], 0), value
            )

        paired_by_instance = {row["artifact_stem"]: row for row in paired}
        subgraph_degrees = {
            row["instance"]: num(row, "maximum_degree")
            for row in data["structural"] if row["group"] == "Extracted subinstances"
        }
        bounded = {method: [[], [], []] for method in METHODS}
        for name, value in degree_change.items():
            bounded[strategy[name]][0].append(100 * value)
        for row in data["audits"]:
            if row["property"] == "average_clustering_difference":
                bounded[strategy[row["instance"]]][1].append(
                    100 * num(row, "observed_value") / num(row, "paper_upper_bound")
                )
            elif row["property"] == "maximum_degree":
                source_names = paired_by_instance[row["instance"]]["source_artifact_stems"]
                baseline = max(subgraph_degrees[name] for name in source_names.split(";"))
                coupling_degree = num(row, "paper_upper_bound") - baseline
                assert coupling_degree > 0
                assert num(row, "paper_lower_bound") == max(baseline, coupling_degree)
                bounded[strategy[row["instance"]]][2].append(
                    100 * (num(row, "observed_value") - baseline) / coupling_degree
                )

        rng = np.random.default_rng(20260920 + domain_index)
        maximum_degree_rng = np.random.default_rng(20260922 + domain_index)
        lines = [
            "% Generated by check/scripts/export_paper_structure_tikz.py.",
            r"\begingroup",
            *definitions(),
            r"\begin{tikzpicture}",
            r"\begin{axis}[",
            r"  width=.97\linewidth,height=.5567\linewidth,",
            r"  xmin=-.48,xmax=2.48,ymin=-5,ymax=108,",
            r"  xtick={0,1,2},xticklabels={{\shortstack{Degree distribution\\change}},{\shortstack{Mean\\clustering}},{\shortstack{Maximum\\degree}}},",
            r"  ytick={0,25,50,75,100},",
            r"  axis lines*=left,axis line style={line width=.5pt},",
            r"  tick label style={font=\scriptsize},label style={font=\scriptsize},ylabel style={font=\scriptsize},",
            *([r"  legend to name=boundedlegend,"] if domain_index == 0 else []),
            r"  ylabel={\shortstack{Observed change\\relative to upper bound (\%)}},",
            r"  legend columns=5,legend style={font=\scriptsize,draw=none,",
            r"    /tikz/every even column/.append style={column sep=2mm}},",
            r"]",
            r"\path[fill=gray!10] (axis cs:-.48,0) rectangle (axis cs:2.48,100);",
            r"\draw[black,dashed,line width=.55pt] (axis cs:-.48,100) -- (axis cs:2.48,100);",
        ]
        for method_index, method in enumerate(METHODS):
            mark, rotation = mark_options(method_index)
            for property_index, values in enumerate(bounded[method]):
                values = np.asarray(values)
                assert len(values) == 100
                assert values.min() >= -1e-7 and values.max() <= 100 + 1e-7
                center = property_index + (method_index - 2) * .14
                point_rng = maximum_degree_rng if property_index == 2 else rng
                path = write_plot_data(
                    destination,
                    f"structure-bounds-{domain}-{method}-{property_index}",
                    center + point_rng.uniform(-.045, .045, len(values)),
                    values,
                )
                legend_option = "" if property_index == 0 and domain_index == 0 else ",forget plot"
                lines.append(
                    rf"\addplot[only marks,mark={mark},mark size=1.15pt,"
                    rf"color=pigMethod{method_index},opacity=.45{rotation}"
                    rf"{legend_option}] table[x=x,y=y] {{{path}}};"
                )
                if property_index == 0 and domain_index == 0:
                    lines.append(r"\addlegendentry{" + LABELS[method_index] + "}")
                add_box(
                    lines,
                    center,
                    values,
                    f"pigMethod{method_index}",
                    .045,
                    .025,
                    np.asarray,
                )
        lines.extend([r"\end{axis}", r"\end{tikzpicture}", r"\endgroup", ""])
        for name in (f"fig-eval-bounded-{domain}.tikz",):
            (destination / name).write_text("\n".join(lines))


def export_global(destination, inputs):
    keys = ("diameter", "algebraic_connectivity", "degree_assortativity")
    lines = ["% Generated by check/scripts/export_paper_structure_tikz.py.",
             r"\pgfplotslegendfromname{structurelegend}\par\smallskip"]
    for domain_index, domain in enumerate(("pglib", "snd")):
        source = inputs[domain]["source"]
        paired = inputs[domain]["paired"]
        title = "PGLib" if domain == "pglib" else "SNDlib"
        if domain_index:
            lines.append(r"\hfill")
        lines.extend([
            r"\begin{subfigure}[t]{0.485\linewidth}",
            r"\centering",
            r"\begingroup",
            *definitions(),
            r"\begin{tikzpicture}",
            r"\begin{axis}[",
            r"  width=.97\linewidth,height=.5567\linewidth,",
            r"  xmin=-.48,xmax=2.48,ymin=-2.15,ymax=2.18,",
            r"  xtick={0,1,2},xticklabels={{\shortstack{Diameter}},{\shortstack{Algebraic\\connectivity}},{\shortstack{Degree\\assortativity}}},",
            r"  ytick={-2,-1,-.5,0,.5,1,2},yticklabels={$-10$,$-1$,$-0.5$,$0$,$0.5$,$1$,$10$},",
            r"  axis lines*=left,axis line style={line width=.5pt},",
            r"  tick label style={font=\scriptsize},label style={font=\scriptsize},ylabel style={font=\scriptsize},",
            *([r"  legend to name=structurelegend,"] if domain_index == 0 else []),
            r"  ylabel={Deviation in source IQRs},",
            r"  legend columns=5,legend style={font=\scriptsize,draw=none,",
            r"    /tikz/every even column/.append style={column sep=2mm}},",
            r"]",
        ])

        standards = {}
        for property_index, key in enumerate(keys):
            reference = np.array([num(row, key) for row in source])
            q1, median, q3 = np.quantile(reference, [.25, .5, .75])
            scale = q3 - q1
            standards[key] = (median, scale)
            band = symlog([(q1 - median) / scale, (q3 - median) / scale])
            lines.append(
                rf"\path[fill=gray!20] (axis cs:{property_index-.43:.2f},{band[0]:.12g}) rectangle (axis cs:{property_index+.43:.2f},{band[1]:.12g});"
            )
        lines.append(
            r"\draw[black,line width=.55pt] (axis cs:-.48,0) -- (axis cs:2.48,0);"
        )

        rng = np.random.default_rng(20260930 + domain_index)
        for method_index, method in enumerate(METHODS):
            mark, rotation = mark_options(method_index)
            for property_index, key in enumerate(keys):
                median, scale = standards[key]
                values = np.array([
                    num(row, "structural_" + key)
                    for row in paired if row["strategy"] == method
                ])
                standardized = (values - median) / scale
                values = symlog(standardized)
                center = property_index + (method_index - 2) * .14
                path = write_plot_data(
                    destination,
                    f"global-structure-{domain}-{method}-{property_index}",
                    center + rng.uniform(-.045, .045, len(values)),
                    values,
                )
                legend_option = (
                    "" if property_index == 0 and domain_index == 0
                    else ",forget plot"
                )
                lines.append(
                    rf"\addplot[only marks,mark={mark},mark size=1.05pt,"
                    rf"color=pigMethod{method_index},opacity=.4{rotation}"
                    rf"{legend_option}] table[x=x,y=y] {{{path}}};"
                )
                if property_index == 0 and domain_index == 0:
                    lines.append(r"\addlegendentry{" + LABELS[method_index] + "}")
                add_box(
                    lines,
                    center,
                    standardized,
                    f"pigMethod{method_index}",
                    .045,
                    .025,
                    symlog,
                )
        plot_start = max(i for i, line in enumerate(lines) if line == r"\begin{tikzpicture}")
        (destination / f"fig-eval-structure-{domain}.tikz").write_text(
            "\n".join(lines[plot_start:] + [r"\end{axis}", r"\end{tikzpicture}", ""])
        )
        lines.extend([
            r"\end{axis}",
            r"\end{tikzpicture}",
            r"\endgroup",
            rf"\caption{{{title}}}",
            r"\end{subfigure}",
        ])
    lines.append("")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper", type=Path)
    parser.add_argument(
        "--pglib-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--snd-root",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    destination = args.paper.resolve() / "resources/evaluation"
    inputs = {
        "pglib": load_domain(args.pglib_root),
        "snd": load_domain(args.snd_root),
    }
    export_bounded(destination, inputs)
    export_global(destination, inputs)
    print("Exported native structural TikZ figures to", destination)


if __name__ == "__main__":
    main()
