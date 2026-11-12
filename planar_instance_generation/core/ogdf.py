"""OGDF planarization with static Euler rows and direct certificate transfer."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import heapq
import json
import math
from numbers import Integral
import os
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "5b6795655399b9d8e2921afec9d97bab9107d5ee"
POLICY = "ogdf-static-direct-v3"


@dataclass(frozen=True, slots=True)
class OgdfStatistics:
    solver: str
    solver_version: str
    status: str
    runtime_seconds: float
    node_count: int
    solution_count: int
    incumbent_objective: int
    best_bound: int
    mip_gap: float
    static_euler_constraints: int
    propagated_constraints: int
    inconsistent_dual_bounds: int


@lru_cache(maxsize=None)
def build_identity():
    directory = Path(os.environ.get("PIG_OGDF_BIN", ROOT / "build/ogdf"))
    paths = [directory / name for name in ("exact", "heuristic")]
    if any(not path.is_file() for path in paths):
        raise RuntimeError(
            "OGDF executables are missing. Run check/scripts/build_ogdf.py first."
        )
    return dict(
        policy=POLICY,
        ogdf_commit=COMMIT,
        executables={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        python_sources={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(__file__).with_name("planarization.py"))
        },
    )


def _run(
    graph, directory, *, exact, seed, threads, limit=None, rows=(), lower=0, kept=None
):
    build_identity()
    binary = Path(os.environ.get("PIG_OGDF_BIN", ROOT / "build/ogdf")) / (
        "exact" if exact else "heuristic"
    )
    nodes = {v: i for i, v in enumerate(graph)}
    edges = list(graph.edges)
    source, output = directory / "graph.txt", directory / "result.json"
    output.unlink(missing_ok=True)
    source.write_text(
        f"{len(nodes)} {len(edges)}\n"
        + "".join(f"{nodes[u]} {nodes[v]}\n" for u, v in edges)
    )
    if exact:
        indices = {frozenset(e): i for i, e in enumerate(edges)}
        removed = [i for i, e in enumerate(edges) if not kept.has_edge(*e)]
        certificate = directory / "bounds.txt"
        certificate.write_text(
            f"{lower} {len(removed)} {len(rows)}\n"
            + " ".join(map(str, removed))
            + "\n"
            + "".join(
                f"{lb} {len(support)} "
                + " ".join(str(indices[frozenset(e)]) for e in support)
                + "\n"
                for support, lb in rows
            )
        )
        command = [
            str(binary),
            str(source),
            str(output),
            str(limit),
            str(seed),
            str(certificate),
            str(threads),
        ]
    else:
        command = [str(binary), str(source), str(output), str(seed), str(threads)]
    environment = os.environ.copy()
    environment.update(
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    with (directory / "solver.log").open("w") as stream:
        completed = subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT, env=environment
        )
    if completed.returncode or not output.is_file():
        raise RuntimeError(
            f'OGDF failed with exit code {completed.returncode}: {directory / "solver.log"}'
        )
    record = json.loads(output.read_text())
    ids = record["deleted_edge_indices"]
    if any(type(i) is not int or not 0 <= i < len(edges) for i in ids) or len(
        set(ids)
    ) != len(ids):
        raise RuntimeError("OGDF returned invalid edge indices")
    result = graph.copy()
    result.remove_edges_from(edges[i] for i in ids)
    if not nx.is_connected(result) or not nx.check_planarity(result)[0]:
        raise RuntimeError("OGDF returned an invalid planar subgraph")
    if exact:
        lb, ub = record["deletion_lower_bound"], record["deletion_upper_bound"]
        if record["status"] == "ERROR" or not 0 <= lb <= ub == len(ids):
            raise RuntimeError("OGDF returned inconsistent deletion bounds")
        expected = {
            frozenset(frozenset(e) for e in graph.subgraph(v).edges)
            for v in nx.biconnected_components(graph)
            if not nx.check_planarity(graph.subgraph(v))[0]
        }
        actual = set()
        for block in record["blocks"]:
            support = frozenset(
                frozenset(edges[i]) for i in block["original_edge_indices"]
            )
            if (
                support in actual
                or support not in expected
                or block["m"] != len(support)
            ):
                raise RuntimeError("OGDF block coverage is inconsistent")
            if (
                len(set(ids).intersection(block["original_edge_indices"]))
                != block["deletion_upper_bound"]
            ):
                raise RuntimeError("OGDF block deletion count is inconsistent")
            actual.add(support)
        if actual != expected:
            raise RuntimeError("OGDF omitted an unsolved block")
    return result, record


def _validate_certificate(graph, piece):
    if not set(piece.graph) <= set(graph) or set(piece.planar_subgraph) != set(
        piece.graph
    ):
        raise ValueError("certificate nodes do not match the graph")
    supplied = {frozenset(e) for e in piece.graph.edges}
    if not supplied <= {frozenset(e) for e in graph.edges}:
        raise ValueError("certificate contains an edge absent from the graph")
    if not {frozenset(e) for e in piece.planar_subgraph.edges} <= supplied:
        raise ValueError("certificate retains edges outside its support")
    if not nx.check_planarity(piece.planar_subgraph)[
        0
    ] or nx.number_connected_components(
        piece.planar_subgraph
    ) != nx.number_connected_components(
        piece.graph
    ):
        raise ValueError("certificate must be planar and preserve connected components")
    if (
        type(piece.lower_bound) is not int
        or not 0
        <= piece.lower_bound
        <= len(supplied) - piece.planar_subgraph.number_of_edges()
    ):
        raise ValueError("invalid certified lower bound")
    for support, lower in piece.bound_rows:
        if len(set(map(frozenset, support))) != len(support):
            raise ValueError("duplicate edges in bound support")
        if (
            not set(map(frozenset, support)) <= supplied
            or type(lower) is not int
            or not 0
            <= lower
            <= sum(not piece.planar_subgraph.has_edge(*e) for e in support)
        ):
            raise ValueError("invalid bound in supplied certificate")


def direct_solution(graph, pieces):
    """Retain selected pieces and all coupling edges, or a connecting forest."""
    kept = graph.copy()
    kept.remove_edges_from(list(kept.edges))
    owner = {}
    for i, piece in enumerate(pieces):
        _validate_certificate(graph, piece)
        if set(owner).intersection(piece.graph):
            raise ValueError("certified pieces must have disjoint nodes")
        if {frozenset(e) for e in graph.subgraph(piece.graph).edges} != {
            frozenset(e) for e in piece.graph.edges
        }:
            raise ValueError("pieces must contain all internal graph edges")
        owner.update((v, i) for v in piece.graph)
        kept.add_edges_from(
            (u, v, graph.edges[u, v]) for u, v in piece.planar_subgraph.edges
        )
    if set(owner) != set(graph):
        raise ValueError("certified pieces must partition the graph")
    coupling = [(u, v) for u, v in graph.edges if owner[u] != owner[v]]
    complete = kept.copy()
    complete.add_edges_from((u, v, graph.edges[u, v]) for u, v in coupling)
    if nx.check_planarity(complete)[0]:
        return complete
    components = nx.utils.UnionFind(graph)
    for u, v in kept.edges:
        components.union(u, v)
    for u, v in coupling:
        if components[u] != components[v]:
            kept.add_edge(u, v, **graph.edges[u, v])
            components.union(u, v)
    return kept


def _disjoint_sum(rows):
    used, total = set(), 0
    # This is a valid bound, without solving a further packing problem.
    for support, lower in sorted(rows, key=lambda row: -row[1]):
        keys = set(map(frozenset, support))
        if not keys.intersection(used):
            used.update(keys)
            total += lower
    return total


def static_euler_rows(graph):
    """Keep the best valid cut of each Euler family along one peeling order.

    Both scores use the same minimum-degree vertex removal sequence. Track the
    number of remaining triangles incrementally, so the coefficient-2 family
    is considered only on triangle-free subsets. Record removal positions to
    reconstruct at most two subsets, avoiding copies at every improvement.
    """
    if len(graph) < 3:
        return ()
    order = {node: index for index, node in enumerate(graph)}
    adjacency = {node: set(graph[node]) for node in graph}
    degree = dict(graph.degree)
    present = set(graph)
    edge_count = graph.number_of_edges()
    triangle_count = sum(nx.triangles(graph).values()) // 3
    scores = {2: 0, 3: 0}
    best_steps = {}
    removed = []
    heap = [(degree[node], order[node], node) for node in graph]
    heapq.heapify(heap)
    while True:
        for coefficient in scores:
            if coefficient == 2 and triangle_count:
                continue
            score = edge_count - coefficient * (len(present) - 2)
            if score > scores[coefficient]:
                scores[coefficient] = score
                best_steps[coefficient] = len(removed)
        if len(present) == 3:
            break
        while True:
            current, _, node = heapq.heappop(heap)
            if node in present and current == degree[node]:
                break
        neighbors = adjacency[node] & present
        if triangle_count:
            # Each triangle containing node is counted at both its neighbors.
            triangle_count -= sum(len(adjacency[v] & neighbors) for v in neighbors) // 2
        present.remove(node)
        removed.append(node)
        edge_count -= current
        for neighbor in neighbors:
            degree[neighbor] -= 1
            heapq.heappush(heap, (degree[neighbor], order[neighbor], neighbor))
    rows = {}
    edge_indices = {frozenset(edge): index for index, edge in enumerate(graph.edges)}
    for coefficient, step in best_steps.items():
        nodes = set(graph).difference(removed[:step])
        indices = tuple(
            sorted(
                edge_indices[frozenset(edge)] for edge in graph.subgraph(nodes).edges
            )
        )
        rows[indices] = max(rows.get(indices, 0), scores[coefficient])
    edges = list(graph.edges)
    return tuple(
        (tuple(edges[i] for i in indices), lower)
        for indices, lower in sorted(rows.items(), key=lambda row: (-row[1], row[0]))
    )


def bound_rows(graph, pieces=(), prior=()):
    """Collect valid rows, localize solver cuts, and combine disjoint supports."""
    edges = list(graph.edges)
    index = {frozenset(e): i for i, e in enumerate(edges)}
    rows, static_count = {}, 0
    insertable = set()
    blocks = []
    for nodes in nx.biconnected_components(graph):
        block = nx.Graph()
        block.add_nodes_from(v for v in graph if v in nodes)
        block.add_edges_from(graph.subgraph(nodes).edges)
        if nx.check_planarity(block)[0]:
            continue
        block_support = tuple(block.edges)
        blocks.append(frozenset(map(frozenset, block_support)))
        for support, lb in static_euler_rows(block):
            key = tuple(sorted(index[frozenset(edge)] for edge in support))
            rows[key] = lb
            insertable.add(key)
            static_count += 1
        key = tuple(sorted(index[frozenset(e)] for e in block_support))
        rows[key] = max(rows.get(key, 0), 1)
    static_keys = set(rows)
    for piece in (*pieces, *prior):
        _validate_certificate(graph, piece)
        for support, lb in (
            *piece.bound_rows,
            (tuple(piece.graph.edges), piece.lower_bound),
        ):
            if lb:
                if (
                    type(lb) is not int
                    or lb < 0
                    or not support
                    or any(frozenset(e) not in index for e in support)
                ):
                    raise ValueError("invalid propagated bound support")
                key = tuple(sorted(index[frozenset(e)] for e in support))
                if len(set(key)) != len(key):
                    raise ValueError("duplicate edges in bound support")
                rows[key] = max(rows.get(key, 0), lb)
                insertable.add(key)
    all_rows = tuple((tuple(edges[i] for i in ids), lb) for ids, lb in rows.items())
    local = tuple(
        (support, lb)
        for support, lb in all_rows
        if tuple(sorted(index[frozenset(e)] for e in support)) in insertable
        and any(set(map(frozenset, support)) <= block for block in blocks)
    )
    lower = max(
        [
            _disjoint_sum(all_rows),
            sum(piece.lower_bound for piece in pieces),
            *(piece.lower_bound for piece in prior),
            sum(
                _disjoint_sum(
                    [
                        (support, lb)
                        for support, lb in local
                        if set(map(frozenset, support)) <= block
                    ]
                )
                for block in blocks
            ),
        ]
    )
    return all_rows, local, lower, static_count, len(set(rows) - static_keys)


def planarize(
    graph,
    *,
    exact=False,
    seed=0,
    threads=16,
    time_limit_seconds=600,
    log_path=None,
    certified_pieces=(),
    prior_certificates=(),
):
    from . import planarization as pz

    pz._validate_simple_graph(graph)
    if not isinstance(threads, Integral) or isinstance(threads, bool) or threads < 1:
        raise ValueError("threads must be a positive integer")
    if (
        not isinstance(seed, Integral)
        or isinstance(seed, bool)
        or not 0 <= seed <= 2**31 - 1
    ):
        raise ValueError("seed must be a nonnegative 32-bit signed integer")
    if exact and (not math.isfinite(time_limit_seconds) or time_limit_seconds <= 0):
        raise ValueError("time_limit_seconds must be finite and positive")
    started = perf_counter()
    direct = direct_solution(graph, certified_pieces) if certified_pieces else None
    reuse_seconds = perf_counter() - started
    forest_edges, composition_upper = (
        pz._piece_forest_bound(graph, certified_pieces)
        if certified_pieces
        else (0, None)
    )
    rows, local, lower, static_count, propagated_count = bound_rows(
        graph, certified_pieces, prior_certificates
    )
    build = (
        build_identity()
        if not nx.check_planarity(graph)[0]
        else dict(policy=POLICY, ogdf_commit=COMMIT)
    )
    with tempfile.TemporaryDirectory(prefix="pig-ogdf-") as temporary:
        directory = (
            Path(temporary) if log_path is None else Path(str(log_path) + ".ogdf")
        )
        directory.mkdir(parents=True, exist_ok=True)
        kept = graph.copy()
        components = [graph.subgraph(v).copy() for v in nx.connected_components(graph)]
        records = []
        for i, component in enumerate(components):
            if nx.check_planarity(component)[0]:
                continue
            folder = directory / f"heuristic-{i}"
            folder.mkdir(exist_ok=True)
            fresh, data = _run(
                component, folder, exact=False, seed=seed, threads=threads
            )
            kept.remove_edges_from(e for e in component.edges if not fresh.has_edge(*e))
            records.append(data)
        fresh_upper = graph.number_of_edges() - kept.number_of_edges()
        direct_upper = (
            None
            if direct is None
            else graph.number_of_edges() - direct.number_of_edges()
        )
        if direct is not None and direct.number_of_edges() > kept.number_of_edges():
            kept = direct
        for support, lb in rows:
            if sum(not kept.has_edge(*e) for e in support) < lb:
                raise ValueError(
                    "propagated lower bound contradicts the verified solution"
                )
        upper = graph.number_of_edges() - kept.number_of_edges()
        if lower > upper:
            raise ValueError("lower bound contradicts the verified solution")
        construction = perf_counter() - started
        exact_records, solve_seconds = [], 0.0
        if exact and lower < upper:
            component_bounds = []
            for i, component in enumerate(components):
                if nx.check_planarity(component)[0]:
                    continue
                folder = directory / f"exact-{i}"
                folder.mkdir(exist_ok=True)
                component_rows = [
                    (s, lb) for s, lb in local if all(component.has_edge(*e) for e in s)
                ]
                global_rows = [
                    (s, lb) for s, lb in rows if all(component.has_edge(*e) for e in s)
                ]
                current, data = _run(
                    component,
                    folder,
                    exact=True,
                    seed=seed,
                    threads=threads,
                    limit=max(0, time_limit_seconds - solve_seconds),
                    rows=component_rows,
                    lower=_disjoint_sum(global_rows),
                    kept=kept.subgraph(component),
                )
                solve_seconds += data["runtime_seconds"]
                kept.remove_edges_from(list(component.edges))
                kept.add_edges_from((u, v, graph.edges[u, v]) for u, v in current.edges)
                component_bounds.append(data["deletion_lower_bound"])
                component_edges = list(component.edges)
                for block in data["blocks"]:
                    rows += (
                        (
                            tuple(
                                component_edges[j]
                                for j in block["original_edge_indices"]
                            ),
                            block["deletion_lower_bound"],
                        ),
                    )
                exact_records.append(data)
            lower = max(lower, sum(component_bounds))
        omitted = tuple(e for e in graph.edges if not kept.has_edge(*e))
        upper = len(omitted)
        if (
            not 0 <= lower <= upper
            or not nx.check_planarity(kept)[0]
            or nx.number_connected_components(kept)
            != nx.number_connected_components(graph)
        ):
            raise RuntimeError("invalid final OGDF certificate")
        status = (
            "OPTIMAL" if lower == upper else ("TIME_LIMIT" if exact else "FEASIBLE")
        )
        metadata = dict(
            build,
            heuristic_runs=10,
            heuristic_threads=threads,
            search_threads=1 if exact else 0,
            extra_greedy_passes=0,
            static_euler_rows=static_count,
            propagated_rows=propagated_count,
            fresh_upper_bound=fresh_upper,
            direct_upper_bound=direct_upper,
            heuristic=records,
            exact=exact_records,
        )
        if log_path is not None:
            Path(log_path).write_text(json.dumps(metadata, indent=2) + "\n")
        common = dict(
            planar_subgraph=kept,
            omitted_edges=omitted,
            omitted_lower_bound=lower,
            omitted_upper_bound=upper,
            exact=lower == upper,
            termination_status=status,
            original_node_count=len(graph),
            original_edge_count=graph.number_of_edges(),
            seed=int(seed),
            total_runtime_seconds=perf_counter() - started,
            shared_piece_count=len(certified_pieces),
            shared_piece_lower_bound=sum(p.lower_bound for p in certified_pieces),
            greedy_start_upper_bound=fresh_upper,
            piece_start_upper_bound=direct_upper,
            prior_certificate_count=len(prior_certificates),
            bound_rows=rows,
            solver_metadata=metadata,
            coupling_forest_edge_count=forest_edges,
            composition_upper_bound=composition_upper,
        )
        if exact:
            blocks = [b for r in exact_records for b in r["blocks"]]
            optimizer = (
                OgdfStatistics(
                    "OGDF/ABACUS/CLP",
                    COMMIT,
                    status,
                    solve_seconds,
                    sum(b["subproblems"] for b in blocks),
                    1,
                    upper,
                    lower,
                    (upper - lower) / upper if upper else 0,
                    static_count,
                    propagated_count,
                    sum(bool(b.get("bound_warning")) for b in blocks),
                )
                if exact_records
                else None
            )
            return pz.PlanarizationResult(
                **common,
                threads=int(threads),
                time_limit_seconds=time_limit_seconds,
                incumbent_construction_seconds=construction,
                optimizer=optimizer,
                log_path=None if log_path is None else str(log_path),
            )
        return pz.HeuristicPlanarizationResult(
            **common,
            time_limit_seconds=None,
            reuse_construction_seconds=reuse_seconds,
            restarts_completed=10 if records else 0,
            largest_face_candidates=pz._largest_face_candidate_count(kept),
            iteration_limit=10,
        )
