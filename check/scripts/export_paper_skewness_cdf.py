"""Export the two normalized added-skewness CDF panels used in the paper."""
import csv
from pathlib import Path
import numpy as np
from check.scripts.export_paper_structure_tikz import METHODS, LABELS, COLORS, rows, num

def normalized_skewness_upper(row):
    return num(
        row,
        "skewness_normalized_inflation_upper_bound"
        if "skewness_normalized_inflation_upper_bound" in row
        else "normalized_inflation_upper_bound",
    )


def export_skewness_pgf(destination, skewness_inputs):
    """Editable PGFPlots ECDFs, split into two LaTeX subfigures."""
    data_directory = destination / "plot-data"
    data_directory.mkdir(exist_ok=True)
    colors = [
        r"\definecolor{pigMethod" + str(i) + "}{HTML}{" + color.lstrip("#") + "}"
        for i, color in enumerate(COLORS)
    ]
    styles = ("dash pattern=on .6pt off 2.4pt", "solid", "solid",
              "dashed", "solid")
    marks = ("o", "triangle", "square", "diamond", "triangle")
    legend_marks = ("*", "triangle*", "square*", "diamond*", "triangle*")
    tex = [
        "% Generated PGFPlots source; fonts are inherited from the manuscript.",
        r"\begingroup",
        *colors,
    ]

    for di, domain in enumerate(("pglib", "snd")):
        paired = rows(skewness_inputs[domain])
        tex.extend([
            r"\begin{subfigure}[t]{.485\linewidth}",
            r"\centering",
            r"\begin{tikzpicture}",
            r"\begin{axis}[",
            r"width=.97\linewidth,height=.5567\linewidth,",
            r"axis lines*=left,axis line style={line width=.5pt},",
            r"xmin=-.015,xmax=1.015,ymin=-.015,ymax=1.03,",
            r"xtick={0,.2,.4,.6,.8,1},ytick={0,.25,.5,.75,1},",
            r"scaled ticks=false,tick label style={font=\scriptsize},",
            r"label style={font=\scriptsize},",
            r"xlabel={Normalized added skewness bound, $\bar\eta$},",
            r"ymajorgrids=true,grid style={gray!20,line width=.3pt},",
            r"legend columns=5,legend cell align=left,",
            r"legend style={font=\scriptsize,draw=none,",
            r"/tikz/every even column/.append style={column sep=3mm}},",
        ])
        if di == 0:
            tex.extend([
                r"ylabel={Fraction of instances},",
                r"legend to name=skewnesslegend,",
            ])
        else:
            tex.append(r"yticklabels={},")
        tex.append(r"]")

        for mi, method in enumerate(METHODS):
            values = np.sort([normalized_skewness_upper(r)
                              for r in paired if r["strategy"] == method])
            assert len(values) == 100 and np.all((values >= 0) & (values <= 1))
            distinct = np.unique(np.r_[0, values, 1])
            cumulative = np.searchsorted(values, distinct, side="right") / len(values)
            data = ["x y", f"0 {cumulative[0]:.12g}"]
            for index in range(1, len(distinct)):
                data.extend([f"{distinct[index]:.12g} {cumulative[index-1]:.12g}",
                             f"{distinct[index]:.12g} {cumulative[index]:.12g}"])
            path = data_directory / f"skewness-{domain}-{method}.dat"
            path.write_text("\n".join(data) + "\n")
            relative = path.relative_to(destination.parents[1]).as_posix()
            tex.append(r"\addplot+[no marks,line width=.9pt,color=pigMethod" + str(mi)
                       + "," + styles[mi] + ",forget plot"
                       + r"] table[x=x,y=y] {" + relative + "};")
            rotation = ",mark options={rotate=180}" if mi == 4 else ""
            tex.append(r"\addplot[only marks,forget plot,mark=" + marks[mi]
                       + r",mark size=1.8pt,color=pigMethod" + str(mi) + rotation
                       + "] coordinates {(0," + f"{cumulative[0]:.12g}" + ")};")

        if di == 0:
            for mi, label_text in enumerate(LABELS):
                rotation = ",mark options={rotate=180}" if mi == 4 else ""
                tex.append(r"\addlegendimage{line width=.9pt,color=pigMethod" + str(mi)
                           + "," + styles[mi] + ",mark=" + legend_marks[mi]
                           + rotation + "}")
                tex.append(r"\addlegendentry{" + label_text + "}")

        title = "PGLib" if di == 0 else "SNDlib"
        label = "pglib" if di == 0 else "snd"
        ending = r"\end{subfigure}\hfill" if di == 0 else r"\end{subfigure}"
        tex.extend([
            r"\end{axis}",
            r"\end{tikzpicture}",
            r"\caption{" + title + r"}\label{fig:eval-skewness-" + label + "}",
            ending,
        ])

    tex.extend([
        r"\par\smallskip",
        r"\pgfplotslegendfromname{skewnesslegend}",
        r"\endgroup",
        "",
    ])
    text = "\n".join(tex)
    import re
    panels = re.findall(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", text, flags=re.S)
    if len(panels) != 2: raise ValueError("Expected two CDF panels")
    for domain, panel in zip(("pglib", "snd"), panels):
        (destination / f"fig-eval-skewness-{domain}.tikz").write_text(panel + "\n")
