#!/usr/bin/env python3
"""Export the paper's stacked runtime figure from paired experiment rows."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METHODS = (
    ("random", "Random"),
    ("high_degree", "High degree"),
    ("kamada_kawai", "Kamada--Kawai"),
    ("heuristic_outer_face", "Outer face (heuristic)"),
    ("exact_outer_face", "Outer face (branch-and-cut)"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper", type=Path)
    parser.add_argument(
        "--pglib",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--snd",
        type=Path,
        required=True,
    )
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
    return parser.parse_args()


def shifted_geometric_mean(values: list[float]) -> float:
    return math.exp(math.fsum(math.log1p(value) for value in values) / len(values)) - 1


def _required_field(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name, "")
        if value != "":
            return value
    raise ValueError(f"none of the required fields {names!r} is present")


def _boolean_field(row: dict[str, str], *names: str) -> bool:
    value = _required_field(row, *names).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError(f"expected a Boolean value in one of {names!r}, found {value!r}")


def summarize(
    path: Path, experiment_root: Path
) -> dict[str, tuple[float, float, float, float, int]]:
    """Return selection, feasibility check, assembly, exact evaluation, timeouts.

    Aggregate each measured component separately. In particular, selection
    includes planarization, and the feasibility check includes model setup and
    solution processing as well as optimization. Extraction and assembly is
    the remaining generation time, calculated per instance before aggregation.
    """
    with path.open(encoding="utf-8", newline="") as stream:
        evaluation_rows = list(csv.DictReader(stream))
    with (experiment_root / "tables" / "accepted_instances.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        generation_rows = list(csv.DictReader(stream))
    result = {}
    for method, _ in METHODS:
        generated = [row for row in generation_rows if row["strategy"] == method]
        evaluated = [row for row in evaluation_rows if row["strategy"] == method]
        if len(generated) != 100 or len(evaluated) != 100:
            raise ValueError(
                f"expected 100 {method} rows in both generation and evaluation "
                f"tables, found {len(generated)} and {len(evaluated)}"
            )
        selection = [float(row["merge_node_selection_seconds"]) for row in generated]
        gate = [float(row["gate_seconds"]) for row in generated]
        generation = [float(row["generation_and_gate_seconds"]) for row in generated]
        overhead = [
            max(0.0, total - selected_time - gate_time)
            for total, selected_time, gate_time
            in zip(generation, selection, gate)
        ]
        runtime_field = (
            "skewness_runtime_seconds"
            if "skewness_runtime_seconds" in evaluated[0]
            else "runtime_seconds"
        )
        exact_evaluations = [
            row
            for row in evaluated
            if _boolean_field(row, "skewness_exact", "exact")
        ]
        if not exact_evaluations:
            raise ValueError(f"found no exact final evaluations for {method} in {path}")
        final = [float(row[runtime_field]) for row in exact_evaluations]
        timeouts = sum(
            _required_field(row, "skewness_status", "status") == "TIME_LIMIT"
            for row in evaluated
        )
        result[method] = (
            *tuple(
                shifted_geometric_mean(values)
                for values in (selection, gate, overhead, final)
            ),
            timeouts,
        )
    return result



def component_plots(
    summary: dict[str, tuple[float, float, float, float, int]],
    field: int,
    shade: int,
    *,
    dashed: bool = False,
    hatched: bool = False,
) -> str:
    plots = []
    for method_index, (target_method, _label) in enumerate(METHODS):
        if hatched:
            style = (
                f"draw=pigMethod{method_index},pattern=north east lines,"
                f"pattern color=pigMethod{method_index},line width=.35pt"
            )
        else:
            style = (
                f"draw=pigMethod{method_index},"
                f"fill=pigMethod{method_index}!{shade}!white"
            )
            if dashed:
                style += ",dashed,line width=.8pt"
            else:
                style = "draw=none," + style.split(",", 1)[1]
        points = " ".join(
            f"({summary[method][field] if method == target_method else 0.0:.3f},"
            f"{method})"
            for method, _method_label in METHODS
        )
        plots.append(rf"\addplot[{style},forget plot] coordinates {{{points}}};")
    return "\n".join(plots)


def final_evaluation_nodes(
    summary: dict[str, tuple[float, float, float, float, int]],
    axis_maximum: float,
) -> str:
    nodes = []
    nodes.append(
        r"\node[anchor=south east,font=\scriptsize,xshift=16pt] "
        r"at (axis description cs:1,1.02) {\textsc{TO}};"
    )
    nodes.extend(
        rf"\node[anchor=east,font=\scriptsize,xshift=16pt] "
        rf"at (axis cs:{axis_maximum:g},{method}) "
        rf"{{{summary[method][4]}}};"
        for method, _label in METHODS
    )
    return "\n".join(nodes)


def nice_axis(maximum: float) -> tuple[float, str]:
    if not math.isfinite(maximum) or maximum < 0:
        raise ValueError("axis maximum must be finite and nonnegative")
    if maximum == 0:
        return 1.0, "0,0.25,0.5,0.75,1"
    raw_step = maximum / 4.0
    magnitude = 10.0 ** math.floor(math.log10(raw_step))
    step = next(
        factor * magnitude
        for factor in (1.0, 2.0, 5.0, 10.0)
        if factor * magnitude >= raw_step
    )
    upper = step * math.ceil(maximum / step)
    ticks = ",".join(
        f"{index * step:g}" for index in range(round(upper / step) + 1)
    )
    return upper, ticks


def main() -> None:
    args = parse_args()
    pglib = summarize(args.pglib, args.pglib_root)
    snd = summarize(args.snd, args.snd_root)
    pglib_generation_axis = nice_axis(
        max(sum(pglib[method][:3]) for method, _ in METHODS)
    )
    pglib_final_axis = nice_axis(max(pglib[method][3] for method, _ in METHODS))
    snd_generation_axis = nice_axis(
        max(sum(snd[method][:3]) for method, _ in METHODS)
    )
    snd_final_axis = nice_axis(max(snd[method][3] for method, _ in METHODS))
    template = r"""% Generated by check/scripts/export_paper_runtime_plot.py.
\begingroup
\pigSetRowSpacing
\definecolor{pigMethod0}{HTML}{648FFF}
\definecolor{pigMethod1}{HTML}{785EF0}
\definecolor{pigMethod2}{HTML}{DC267F}
\definecolor{pigMethod3}{HTML}{FE6100}
\definecolor{pigMethod4}{HTML}{FFB000}
\pgfplotslegendfromname{runtimelegend}\par\smallskip
\begin{subfigure}[t]{0.485\linewidth}
\centering
\begin{tikzpicture}
\begin{groupplot}[
  group style={group size=2 by 1,horizontal sep=6mm},
  xbar stacked,/pgf/bar width=\pigBarWidth,
  scale only axis,width=.453333333\linewidth,height=\pigMethodAxisHeight,
  scaled x ticks=false,xlabel={Time [s]},
  symbolic y coords={random,high_degree,kamada_kawai,heuristic_outer_face,exact_outer_face},
  y dir=reverse,ytick=data,yticklabels={R,HD,KK,OF-H,OF-B},
  enlarge y limits=.16,
  axis lines*=left,axis line style={line width=.5pt},
  tick label style={font=\scriptsize},label style={font=\scriptsize},
  xmajorgrids=true,grid style={gray!18,line width=.3pt},
  legend style={font=\scriptsize,legend columns=4,draw=none,
    /tikz/every even column/.append style={column sep=1mm}},
]
\nextgroupplot[legend to name=runtimelegend,
  xmin=0,xmax=__PGLIB_GENERATION_XMAX__,xtick={__PGLIB_GENERATION_XTICKS__},
]
\addlegendimage{area legend,draw=none,fill=black}
\addlegendentry{Merge vertex selection}
\addlegendimage{area legend,draw=black,pattern=north east lines,pattern color=black,line width=.35pt}
\addlegendentry{Feasibility check}
\addlegendimage{area legend,draw=none,fill=black!25}
\addlegendentry{Extraction and assembly}
\addlegendimage{area legend,draw=black,fill=black!10,dashed,line width=.8pt}
\addlegendentry{Final skewness evaluation}
__PGLIB_SELECTION__
__PGLIB_GATE__
__PGLIB_REMAINDER__
\nextgroupplot[clip=false,width=.226666667\linewidth,
  xmin=0,xmax=__PGLIB_FINAL_XMAX__,xtick={__PGLIB_FINAL_XTICKS__},
  yticklabels=\empty]
__PGLIB_FINAL__
__PGLIB_FINAL_LABELS__
\end{groupplot}
\end{tikzpicture}
\caption{PGLib}
\end{subfigure}
\hfill
\begin{subfigure}[t]{0.485\linewidth}
\centering
\begin{tikzpicture}
\begin{groupplot}[
  group style={group size=2 by 1,horizontal sep=6mm},
  xbar stacked,/pgf/bar width=\pigBarWidth,
  scale only axis,width=.453333333\linewidth,height=\pigMethodAxisHeight,
  scaled x ticks=false,xlabel={Time [s]},
  symbolic y coords={random,high_degree,kamada_kawai,heuristic_outer_face,exact_outer_face},
  y dir=reverse,ytick=data,yticklabels={R,HD,KK,OF-H,OF-B},
  enlarge y limits=.16,
  axis lines*=left,axis line style={line width=.5pt},
  tick label style={font=\scriptsize},label style={font=\scriptsize},
  xmajorgrids=true,grid style={gray!18,line width=.3pt},
  legend style={font=\scriptsize,legend columns=4,draw=none,
    /tikz/every even column/.append style={column sep=1mm}},
]
\nextgroupplot[xmin=0,xmax=__SND_GENERATION_XMAX__,xtick={__SND_GENERATION_XTICKS__},
]
__SND_SELECTION__
__SND_GATE__
__SND_REMAINDER__
\nextgroupplot[clip=false,width=.226666667\linewidth,
  xmin=0,xmax=__SND_FINAL_XMAX__,xtick={__SND_FINAL_XTICKS__},
  yticklabels=\empty]
__SND_FINAL__
__SND_FINAL_LABELS__
\end{groupplot}
\end{tikzpicture}
\caption{SNDlib}
\end{subfigure}
\endgroup
"""
    replacements = {
        "__PGLIB_SELECTION__": component_plots(pglib, 0, 100),
        "__PGLIB_GATE__": component_plots(pglib, 1, 0, hatched=True),
        "__PGLIB_REMAINDER__": component_plots(pglib, 2, 25),
        "__PGLIB_FINAL__": component_plots(pglib, 3, 10, dashed=True),
        "__SND_SELECTION__": component_plots(snd, 0, 100),
        "__SND_GATE__": component_plots(snd, 1, 0, hatched=True),
        "__SND_REMAINDER__": component_plots(snd, 2, 25),
        "__PGLIB_FINAL_LABELS__": final_evaluation_nodes(
            pglib, pglib_final_axis[0]
        ),
        "__SND_FINAL__": component_plots(snd, 3, 10, dashed=True),
        "__PGLIB_GENERATION_XMAX__": f"{pglib_generation_axis[0]:g}",
        "__PGLIB_GENERATION_XTICKS__": pglib_generation_axis[1],
        "__PGLIB_FINAL_XMAX__": f"{pglib_final_axis[0]:g}",
        "__PGLIB_FINAL_XTICKS__": f"0,{pglib_final_axis[0] / 2:g},{pglib_final_axis[0]:g}",
        "__SND_FINAL_LABELS__": final_evaluation_nodes(snd, snd_final_axis[0]),
        "__SND_GENERATION_XMAX__": f"{snd_generation_axis[0]:g}",
        "__SND_GENERATION_XTICKS__": snd_generation_axis[1],
        "__SND_FINAL_XMAX__": f"{snd_final_axis[0]:g}",
        "__SND_FINAL_XTICKS__": f"0,{snd_final_axis[0] / 2:g},{snd_final_axis[0]:g}",
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    destination = args.paper.resolve() / "resources/evaluation/fig-eval-runtime.tikz"
    destination.write_text(template)
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
