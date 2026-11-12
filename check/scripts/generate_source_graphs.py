#!/usr/bin/env python3
"""Generate the source-network PDFs used by the supplementary material.

The source datasets are intentionally not vendored in this repository.  Point this
script at checkouts of PGLib-OPF and TopoHub to refresh the committed vector
figures, for example:

    scripts/generate_source_graphs.py \
        --pglib /path/to/pglib-opf \
        --topohub /path/to/topohub
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


PGLIB_CASES = (
    "case3_lmbd", "case5_pjm", "case14_ieee", "case24_ieee_rts",
    "case30_as", "case30_ieee", "case39_epri", "case57_ieee", "case60_c",
    "case73_ieee_rts", "case89_pegase", "case118_ieee",
    "case162_ieee_dtc", "case179_goc", "case197_snem", "case200_activ",
    "case240_pserc", "case300_ieee", "case500_goc", "case588_sdet",
    "case793_goc", "case1354_pegase", "case1888_rte", "case1951_rte",
    "case2000_goc",
)

SNDLIB_CASES = (
    "abilene", "atlanta", "cost266", "france", "geant", "germany50",
    "giul39", "india35", "janos-us-ca", "janos-us", "nobel-eu",
    "nobel-germany", "nobel-us", "norway", "sun", "zib54",
)


def matrix(text: str, name: str) -> list[list[str]]:
    match = re.search(rf"mpc\.{name}\s*=\s*\[(.*?)\];", text, re.DOTALL)
    if match is None:
        raise ValueError(f"matrix mpc.{name} not found")
    rows = []
    for raw_row in match.group(1).split(";"):
        row = raw_row.split("%", 1)[0].split()
        if row:
            rows.append(row)
    return rows


def pglib_graph(path: Path) -> tuple[set[str], list[tuple[str, str]]]:
    text = path.read_text(encoding="utf-8")
    nodes = {row[0] for row in matrix(text, "bus")}
    edges = set()
    for row in matrix(text, "branch"):
        # Column 11 is BR_STATUS in the MATPOWER branch format.
        if len(row) < 11 or float(row[10]) != 0:
            edges.add(tuple(sorted((row[0], row[1]))))
    return nodes, sorted(edges)


def sndlib_graph(path: Path) -> tuple[set[str], list[tuple[str, str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = {str(node.get("id", node.get("name"))) for node in data["nodes"]}
    edges = [(str(edge["source"]), str(edge["target"])) for edge in data["edges"]]
    return nodes, edges


def dot_quote(value: str) -> str:
    return json.dumps(value)


def render(nodes: set[str], edges: list[tuple[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "graph G {",
        "  graph [bgcolor=transparent, outputorder=edgesfirst, overlap=prism,",
        "         splines=false, pad=0.04, margin=0];",
        "  node [shape=point, width=0.045, height=0.045, label=\"\",",
        "        color=\"#202020\", fillcolor=\"#202020\", style=filled];",
        "  edge [color=\"#777777\", penwidth=0.35];",
    ]
    lines.extend(f"  {dot_quote(node)};" for node in sorted(nodes))
    lines.extend(f"  {dot_quote(u)} -- {dot_quote(v)};" for u, v in edges)
    lines.append("}")
    subprocess.run(
        ["sfdp", "-Tpdf", "-o", str(output)],
        input="\n".join(lines),
        text=True,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pglib", type=Path, required=True)
    parser.add_argument("--topohub", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        required=True,
        help="destination directory in the paper repository",
    )
    args = parser.parse_args()

    for case in PGLIB_CASES:
        source = args.pglib / f"pglib_opf_{case}.m"
        nodes, edges = pglib_graph(source)
        render(nodes, edges, args.output / "pglib" / f"{case}.pdf")

    sndlib_dir = args.topohub / "data" / "sndlib"
    for case in SNDLIB_CASES:
        nodes, edges = sndlib_graph(sndlib_dir / f"{case}.json")
        render(nodes, edges, args.output / "sndlib" / f"{case}.pdf")


if __name__ == "__main__":
    main()
