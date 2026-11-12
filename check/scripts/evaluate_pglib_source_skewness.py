#!/usr/bin/env python3
"""Compute a certified skewness interval for one PGLib source graph."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from time import perf_counter


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from planar_instance_generation.core import maximum_planar_subgraph
from planar_instance_generation.core.ogdf import build_identity
from planar_instance_generation.core.planarization import ogdf_planar_subgraph
from planar_instance_generation.evaluation.voltage_certificate import lift_subgraph_certificate
from planar_instance_generation.io import read_pglib


_SAFE_SOURCE = re.compile(r"^pglib_opf_case[A-Za-z0-9_]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
    )
    parser.add_argument("--time-limit", type=float, default=50_400.0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--voltage-certificate-directory", type=Path,
                        default=REPOSITORY_ROOT / "check/results/ogdf_pglib_voltage_level_skewness_14h")
    parser.add_argument("--voltage-warm-start", action=argparse.BooleanOptionalAction, default=False,
                        help="reuse matching voltage-separated bounds and planar certificates for original graphs")
    parser.add_argument("--voltage-separated", action=argparse.BooleanOptionalAction, default=False,
                        help="omit active edges joining unequal endpoint base-kV levels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not _SAFE_SOURCE.fullmatch(args.source):
        raise SystemExit(f"invalid PGLib source name: {args.source!r}")
    if args.time_limit <= 0:
        raise SystemExit("invalid exact-solver settings")
    source_path = REPOSITORY_ROOT / "check/data/pglib" / f"{args.source}.m"
    if not source_path.is_file():
        raise SystemExit(f"source does not exist: {source_path}")
    source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    output_directory = args.output_directory or Path(
        "check/results/ogdf_pglib_voltage_level_skewness_14h" if args.voltage_separated
        else "check/results/ogdf_pglib_source_skewness_14h"
    )
    if not output_directory.is_absolute():
        output_directory = REPOSITORY_ROOT / output_directory
    checkpoint = output_directory / f"{args.source}.json"
    log_path = output_directory / "solver_logs" / f"{args.source}.log"
    settings = {
        "backend": build_identity(),
        "time_limit_seconds": float(args.time_limit),
        "heuristic_runs": 10,
        "threads": args.threads,
        "seed": int(args.seed),
    }
    if args.voltage_separated:
        settings["edge_rule"] = "retain iff endpoint baseKV values are equal"
    voltage_records = []
    if args.voltage_warm_start and not args.voltage_separated:
        for directory in (args.voltage_certificate_directory,):
            path = directory / f"{args.source}.json"
            if not path.is_file():
                continue
            content = path.read_bytes()
            record = json.loads(content)
            digest = record.get("source_sha256", record.get("settings", {}).get("sha256"))
            if (record.get("settings", {}).get("backend") != build_identity() or
                    record.get("source") != args.source or digest != source_digest or
                    record.get("settings", {}).get("edge_rule") != "retain iff endpoint baseKV values are equal"):
                raise ValueError(f"voltage certificate does not match input: {path}")
            voltage_records.append((path, content, record))
        settings["voltage_warm_start_version"] = 1
        settings["voltage_certificates"] = {
            str(path.relative_to(REPOSITORY_ROOT)): hashlib.sha256(content).hexdigest()
            for path, content, _ in voltage_records
        }
    if checkpoint.is_file() and not args.overwrite:
        try:
            cached = json.loads(checkpoint.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
        if (cached is not None and cached.get("settings") == settings and
                cached.get("source_sha256") == source_digest):
            print(f"Reusing {checkpoint}")
            return
        raise SystemExit(f"checkpoint exists with different settings or input: {checkpoint}")

    output_directory.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    case = read_pglib(source_path)
    graph = case.topology(active_only=True)
    original_edge_count = graph.number_of_edges()
    warm_started = perf_counter()
    certificates = ()
    if voltage_records:
        voltage_graph = graph.copy()
        kv = {int(bus[0]): float(bus[9]) for bus in case.bus}
        voltage_graph.remove_edges_from([(u, v) for u, v in graph.edges() if kv[u] != kv[v]])
        lower_bound = 0
        candidates = []
        for path, _, record in voltage_records:
            if int(record.get("edge_count", record.get("retained_edge_count"))) != voltage_graph.number_of_edges():
                raise ValueError(f"voltage edge count mismatch: {path}")
            lower_bound = max(lower_bound, int(record.get("skewness_lower_bound", record.get("lower_bound"))))
            if "retained_planar_edges" in record:
                planar = voltage_graph.copy()
                planar.remove_edges_from(list(planar.edges))
                planar.add_edges_from(record["retained_planar_edges"])
                candidates.append(planar)
        if not candidates:
            candidates.append(ogdf_planar_subgraph(
                voltage_graph, threads=args.threads, seed=args.seed
            ).planar_subgraph)
        lifted = [lift_subgraph_certificate(graph, voltage_graph, planar, lower_bound) for planar in candidates]
        certificates = (max(lifted, key=lambda piece: piece.planar_subgraph.number_of_edges()),)
    voltage_preparation_seconds = perf_counter() - warm_started
    if args.voltage_separated:
        kv = {int(bus[0]): float(bus[9]) for bus in case.bus}
        graph.remove_edges_from([(u, v) for u, v in graph.edges() if kv[u] != kv[v]])
    result = maximum_planar_subgraph(
        graph,
        time_limit_seconds=args.time_limit,

        seed=args.seed,
        threads=args.threads,
        log_path=log_path,
        certified_pieces=certificates,
    )
    payload = {
        "schema_version": 1,
        "source": args.source,
        "source_file": str(source_path.relative_to(REPOSITORY_ROOT)),
        "source_sha256": source_digest,
        "voltage_separated": args.voltage_separated,
        "original_edge_count": original_edge_count,
        "removed_cross_voltage_edges": original_edge_count - graph.number_of_edges(),
        "retained_planar_edges": [list(edge) for edge in result.planar_subgraph.edges()],
        "settings": settings,
        "node_count": result.original_node_count,
        "edge_count": result.original_edge_count,
        "skewness_lower_bound": result.omitted_lower_bound,
        "skewness_upper_bound": result.omitted_upper_bound,
        "skewness_exact": result.exact,
        "absolute_gap": result.omitted_upper_bound - result.omitted_lower_bound,
        "status": result.termination_status,
        "runtime_seconds": result.total_runtime_seconds + voltage_preparation_seconds,
        "voltage_preparation_seconds": voltage_preparation_seconds,
        "voltage_lower_bound": sum(piece.lower_bound for piece in certificates),
        "greedy_start_upper_bound": result.greedy_start_upper_bound,
        "piece_start_upper_bound": result.piece_start_upper_bound,
        "coupling_forest_edge_count": result.coupling_forest_edge_count,
        "composition_upper_bound": result.composition_upper_bound,
    }
    if result.optimizer is not None:
        payload["optimizer"] = {
            "status": result.optimizer.status,
            "mip_gap": result.optimizer.mip_gap,
            "best_bound": result.optimizer.best_bound,
            "incumbent_objective": result.optimizer.incumbent_objective,
            "objective_upper_bound": result.omitted_upper_bound,
        }
    temporary = checkpoint.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(checkpoint)
    print(
        f"{args.source}: [{result.omitted_lower_bound}, "
        f"{result.omitted_upper_bound}], exact={result.exact}"
    )


if __name__ == "__main__":
    main()
