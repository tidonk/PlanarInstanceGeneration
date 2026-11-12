"""Independent checks for native solutions, bounds, and block coverage."""

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import networkx as nx
from planar_instance_generation.core.planarization import CertifiedPlanarPiece as CertifiedPlanarPiece

ENV = dict(
    os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def validate(item, record):
    import networkx as nx

    lb, ub = record["deletion_lower_bound"], record["deletion_upper_bound"]
    if not isinstance(lb, (int, float)) or not math.isfinite(lb) or lb < 0 or lb > ub:
        raise ValueError(f"invalid bounds: {lb}, {ub}")
    deleted = record["deleted_edge_indices"]
    if len(set(deleted)) != len(deleted) or len(deleted) != ub:
        raise ValueError("deletion count differs from upper bound")
    if any(not isinstance(i, int) or not 0 <= i < item["m"] for i in deleted):
        raise ValueError("invalid deleted edge index")
    graph = nx.Graph()
    graph.add_nodes_from(range(item["n"]))
    removed = set(deleted)
    graph.add_edges_from(e for i, e in enumerate(item["edges"]) if i not in removed)
    if not nx.check_planarity(graph)[0]:
        raise ValueError("returned solution is nonplanar")
    if nx.number_connected_components(graph) != item["components"]:
        raise ValueError("returned solution changes connected components")
    if max(lb, item["historical_bounds"][0]) > min(ub, item["historical_bounds"][1]):
        raise ValueError("returned and historical bound intervals are disjoint")
    return {
        "planarity": True,
        "edge_count": True,
        "components": True,
        "historical_interval": True,
    }


def graph_input(g, path):
    nodes = {v: i for i, v in enumerate(g)}
    edges = [[nodes[u], nodes[v]] for u, v in g.edges]
    path.write_text(
        f"{len(g)} {len(edges)}\n" + "".join(f"{u} {v}\n" for u, v in edges)
    )
    return edges


def read_graph(path):
    x = list(map(int, Path(path).read_text().split()))
    g = nx.Graph()
    g.add_nodes_from(range(x[0]))
    g.add_edges_from(zip(x[2::2], x[3::2]))
    assert len(g) == x[0] and g.number_of_edges() == x[1] and nx.is_connected(g)
    return g


def check_blocks(graph, result):
    edges = list(graph.edges)
    expected = {
        frozenset(frozenset(e) for e in graph.subgraph(v).edges)
        for v in nx.biconnected_components(graph)
        if not nx.check_planarity(graph.subgraph(v))[0]
    }
    actual = set()
    seen = set()
    removed = set(result["deleted_edge_indices"])
    for block in result["blocks"]:
        ids = block["original_edge_indices"]
        assert len(ids) == block["m"] and not seen.intersection(ids)
        seen.update(ids)
        represented = frozenset(frozenset(edges[i]) for i in ids)
        assert represented in expected
        actual.add(represented)
        assert len(removed.intersection(ids)) == block["deletion_upper_bound"]
    assert actual == expected
    # An externally propagated global bound can exceed the sum of per-block
    # bounds when the shared time limit leaves some blocks unprocessed.
    assert (
        sum(b["deletion_lower_bound"] for b in result["blocks"])
        <= result["deletion_lower_bound"]
    )


def certificate_input(graph, pieces, prior, seed, path):
    # The known-optimum tests supply one full-graph certificate.
    (piece,) = pieces
    indices = {frozenset(e): i for i, e in enumerate(graph.edges)}
    removed = [
        i for edge, i in indices.items() if not piece.planar_subgraph.has_edge(*edge)
    ]
    path.write_text(
        f"0 {piece.lower_bound} {len(removed)} 0\n" + " ".join(map(str, removed)) + "\n"
    )


def invoke(driver, graph, certificate, out, limit, seed, mode, cpu, item=None):
    """Vary the supplied certificates, without changing native solver controls."""
    if out.exists():
        raise ValueError(f"refusing to overwrite {out}")
    data = list(map(int, certificate.read_text().split()))
    joint, lower, count, nrows = data[:4]
    assert joint == 0
    removed, rows = data[4 : 4 + count], data[4 + count :]
    if not mode & 1:
        g = read_graph(graph)
        tree = nx.minimum_spanning_tree(g)
        removed = [i for i, e in enumerate(g.edges) if not tree.has_edge(*e)]
    if not mode & 2:
        lower, nrows, rows = 0, 0, []
    supplied = out.with_suffix(".cert")
    supplied.write_text(
        " ".join(map(str, [lower, len(removed), nrows, *removed, *rows])) + "\n"
    )
    started = time.perf_counter()
    with out.with_suffix(".log").open("w") as log:
        proc = subprocess.run(
            [
                "taskset",
                "-c",
                str(cpu),
                str(driver),
                str(graph),
                str(out),
                str(limit),
                str(seed),
                str(supplied),
                "16",
            ],
            env=ENV,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=limit + 120,
        )
    if proc.returncode:
        raise RuntimeError(f"{out.name}: return code {proc.returncode}; see log")
    result = json.loads(out.read_text())
    if item:
        result["validation"] = validate(item, result)
    result.update(
        process_seconds=time.perf_counter() - started,
        cpu=cpu,
        reuse_mode=mode,
        driver_sha256=digest(driver),
        input_sha256=digest(graph),
        certificate_sha256=digest(certificate),
    )
    write_json(out, result)
    print(
        f"{out.stem}: [{result['deletion_lower_bound']}, {result['deletion_upper_bound']}]",
        flush=True,
    )
    return result
