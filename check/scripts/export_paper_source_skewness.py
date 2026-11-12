#!/usr/bin/env python3
"""Export certified source-graph skewness intervals for the paper appendix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
def _snd_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["group"] == "Original sources"
        ]
    if len(rows) != 16:
        raise ValueError(f"expected 16 SNDlib original sources in {path}, got {len(rows)}")
    return rows


def _escape(value: str) -> str:
    return value.replace("_", "\\_")


def _interval(lower: object, upper: object) -> str:
    lower_int = int(float(lower))
    upper_int = int(float(upper))
    if lower_int == upper_int:
        return str(lower_int)
    return rf"[{lower_int},\,{upper_int}]"


def _percentage(lower: object, upper: object, edge_count: object) -> str:
    relation = "$\\leq$\\," if float(lower) != float(upper) else ""
    value = 100 * float(upper) / int(edge_count)
    return f"{relation}\\SI{{{value:.1f}}}{{\\percent}}"


def export(paper, pglib_directory, voltage_directory, snd_csv, source_names):
    pglib = []
    for name in source_names:
        original = json.loads((pglib_directory / (name + ".json")).read_text())
        prime = json.loads((voltage_directory / (name + ".json")).read_text())
        digest = hashlib.sha256((REPOSITORY_ROOT / "check/data/pglib" / (name + ".m")).read_bytes()).hexdigest()
        for record in (original, prime):
            if record["source"] != name or record.get("source_sha256", record.get("settings", {}).get("sha256")) != digest:
                raise ValueError(f"Source certificate does not match {name}")
        if not prime.get("voltage_separated", prime.get("settings", {}).get("edge_rule") is not None):
            raise ValueError(f"Expected voltage-separated certificate for {name}")
        lo = int(original["skewness_lower_bound"]); hi = int(original["skewness_upper_bound"])
        plo = int(prime.get("skewness_lower_bound", prime.get("lower_bound")))
        phi = min(hi, int(prime.get("skewness_upper_bound", prime.get("upper_bound"))))
        lo = max(lo, plo)
        if not 0 <= plo <= phi or not 0 <= lo <= hi:
            raise ValueError(f"Inconsistent bounds for {name}")
        pglib.append((original, lo, hi, plo, phi, prime.get("edge_count", prime.get("retained_edge_count"))))
    lines = [r"\begin{tabular}{lrrr>{\columncolor{gray!10}}rr>{\columncolor{gray!10}}r}", r"\toprule",
             r"Source & $|V|$ & $|E|$ & $[\underline s,\bar s]$ & $\bar s/|E|$ & $[\underline{s}',\bar{s}']$ & $\bar{s}'/|E'|$\\", r"\midrule"]
    for record, lo, hi, plo, phi, edges in sorted(pglib, key=lambda x: x[0]["node_count"]):
        name = _escape(record["source"].removeprefix("pglib_opf_"))
        lines.append(f"{name} & {record['node_count']} & {record['edge_count']} & ${_interval(lo, hi)}$ & "
                     f"{_percentage(lo, hi, record['edge_count'])} & ${_interval(plo, phi)}$ & {_percentage(plo, phi, edges)} " + r"\\")
    out = paper / "resources/evaluation"
    (out / "tab-app-source-skewness.tex").write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""]))
    lines = [r"\begin{tabular}{lrrr>{\columncolor{gray!10}}r}", r"\toprule",
             r"Source & $|V|$ & $|E|$ & $[\underline s,\bar s]$ & $\bar s/|E|$\\", r"\midrule"]
    for row in _snd_rows(snd_csv):
        lo, hi = float(row['skewness_lower_bound']), float(row['skewness_upper_bound'])
        if not 0 <= lo <= hi: raise ValueError("Inconsistent SND source bounds")
        lines.append(f"{_escape(row['instance'])} & {row['node_count']} & {row['edge_count']} & "
                     f"${_interval(lo,hi)}$ & {_percentage(lo,hi,row['edge_count'])} " + r"\\")
    (out / "tab-snd-source-skewness.tex").write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""]))
