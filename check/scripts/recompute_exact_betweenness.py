#!/usr/bin/env python3
"""Recompute node betweenness on saved graphs without rerunning optimization."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from planar_instance_generation.io import read_pglib
from planar_instance_generation.io.sndlib import read_sndlib


def read_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def measure(payload):
    domain, run, row = payload
    name, group = row["instance"], row["group"]
    suffix = ".m" if domain == "pglib" else ".xml"
    directory = (
        ROOT / "check/data" / ("pglib" if domain == "pglib" else "sndlib")
        if group == "Original sources"
        else run / ("subinstances" if group == "Extracted subinstances" else "instances")
    )
    path = directory / (name + suffix)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cache_dir = run / "evaluation/exact_betweenness"
    cache_dir.mkdir(exist_ok=True)
    checkpoint = cache_dir / (group.replace(":", "").replace(" ", "_") + "_" + name + ".json")
    coupling = set()
    if group.startswith("Composite:"):
        metadata = json.loads((run / "plots/composite" / name / "metadata.json").read_text())
        if domain == "snd":
            coupling = {node for edge in metadata["coupling_links"] for node in (edge["source"], edge["target"])}
        else:
            case = read_pglib(path)
            coupling = {
                int(case.branch[index, endpoint])
                for index in metadata["coupling_branch_rows_zero_based"] for endpoint in (0, 1)
            }
    settings = {"sha256": digest, "coupling_nodes": sorted(coupling), "normalized": True, "weighted": False}
    if checkpoint.is_file():
        cached = json.loads(checkpoint.read_text())
        if cached.get("settings") == settings:
            return {**row, **cached["statistics"]}
    graph = read_pglib(path).topology(active_only=True) if domain == "pglib" else read_sndlib(path).topology()
    assert graph.number_of_nodes() == int(row["node_count"])
    assert graph.number_of_edges() == int(row["edge_count"])
    started = perf_counter()
    values = nx.betweenness_centrality(graph, normalized=True, weight=None)
    elapsed = perf_counter() - started
    all_values = np.array(list(values.values()))
    selected = np.array([values[node] for node in coupling])
    statistics = {
        "mean_betweenness": float(all_values.mean()),
        "betweenness_p95": float(np.quantile(all_values, .95)),
        "betweenness_sample_count": "", "betweenness_exact": True,
        "betweenness_seed": "",
        "coupling_betweenness_mean": float(selected.mean()) if len(selected) else "",
        "coupling_betweenness_p95": float(np.quantile(selected, .95)) if len(selected) else "",
        "coupling_betweenness_max": float(selected.max()) if len(selected) else "",
        "betweenness_measurement_seconds": elapsed,
    }
    checkpoint.write_text(json.dumps({
        "settings": settings, "statistics": statistics,
        "nodes": [{"node": node, "betweenness": value} for node, value in values.items()],
    }, indent=2) + "\n")
    return {**row, **statistics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    for domain in ("pglib", "snd"):
        parser.add_argument(
            "--" + domain + "-root", type=Path,
            default=ROOT / ("check/results/full_shared4_300_consecutive_" + domain),
        )
    args = parser.parse_args()
    for domain in ("pglib", "snd"):
        run = getattr(args, domain + "_root").resolve()
        rows = read_rows(run / "evaluation/structural.csv")
        measured = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(measure, ((domain, run, row) for row in rows)):
                measured.append(result)
                if len(measured) % 100 == 0:
                    print(domain, len(measured), "/", len(rows), flush=True)
        write_rows(run / "evaluation/structural_exact.csv", measured)
        by_name = {row["instance"]: row for row in measured if row["group"].startswith("Composite:")}
        paired = read_rows(run / "evaluation/paired_strategy_outcomes.csv")
        for row in paired:
            for key, value in by_name[row["artifact_stem"]].items():
                if "structural_" + key in row:
                    row["structural_" + key] = value
        write_rows(run / "evaluation/paired_strategy_outcomes_exact.csv", paired)
        print(domain, "exact recomputation complete:", len(measured), "graphs", flush=True)


if __name__ == "__main__":
    main()
