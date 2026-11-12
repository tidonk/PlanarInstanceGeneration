"""Process-parallel, restartable maximum-planar-subgraph evaluation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields, replace
import fcntl
import hashlib
import json
from numbers import Integral
from pathlib import Path
import time
from typing import Mapping, Sequence

import networkx as nx

from planar_instance_generation.core.ogdf import (
    OgdfStatistics,
    build_identity,
    _validate_certificate,
)

from planar_instance_generation.core.planarization import (
    CertifiedPlanarPiece,
    PlanarizationResult,
    maximum_planar_subgraph,
    relabel_planar_piece,
)


@dataclass(frozen=True, slots=True)
class PlanarizationTask:
    """One independently seeded graph solve and its distinct artifact paths."""

    graph: nx.Graph
    seed: int
    log_path: Path
    cache_key: str
    certified_pieces: tuple[CertifiedPlanarPiece, ...] = ()
    prior_certificates: tuple[CertifiedPlanarPiece, ...] = ()


def source_mappings(metadata: Mapping[str, object]) -> tuple[tuple[str, dict], ...]:
    """Read PGLib or SND source labels without using strategy-specific data."""
    if "source_bus_mappings" in metadata:
        return tuple(
            (
                record["source_artifact_stem"],
                {
                    int(bus["source_bus"]): int(bus["composite_bus"])
                    for bus in record["buses"]
                },
            )
            for record in metadata["source_bus_mappings"]
        )
    return tuple(
        (record["source_artifact_stem"], dict(record["node_ids"]))
        for record in metadata["source_mappings"]
    )


def solve_partitioned_batch(
    tasks: Sequence[PlanarizationTask],
    dependencies: Mapping[str, Sequence[tuple[str, Mapping]]],
    *,
    generation_metadata: Sequence = (),
    **kwargs,
) -> tuple[PlanarizationResult, ...]:
    """Solve shared pieces first, then finals with identical certificate reuse."""
    from .planar_certificates import attach_generation_certificates

    tasks = attach_generation_certificates(tasks, dependencies, generation_metadata)
    by_name = {task.cache_key: task for task in tasks}
    if len(by_name) != len(tasks):
        raise ValueError("duplicate task cache keys")
    if not set(dependencies) <= set(by_name):
        raise ValueError("dependencies refer to an unknown final task")
    for pieces in dependencies.values():
        if not pieces or any(
            key not in by_name or key in dependencies for key, _ in pieces
        ):
            raise ValueError("final dependencies must name independent piece tasks")
    independent = tuple(task for task in tasks if task.cache_key not in dependencies)
    results = dict(
        zip(
            (task.cache_key for task in independent),
            solve_planarization_batch(independent, **kwargs),
            strict=True,
        )
    )
    finals = []
    for task in tasks:
        if task.cache_key not in dependencies:
            continue
        pieces = []
        for key, mapping in dependencies[task.cache_key]:
            source, result = by_name[key], results[key]
            if set(mapping) != set(source.graph) or len(set(mapping.values())) != len(
                mapping
            ):
                raise ValueError("source mapping must be a bijection on piece nodes")
            pieces.append(relabel_planar_piece(source.graph, result, mapping))
        # Matched methods receive the same seed, determined only by their
        # shared pieces, never by strategy or processing order of finals.
        seed_payload = [
            (key, by_name[key].seed) for key, _mapping in dependencies[task.cache_key]
        ]
        shared_seed = (
            int.from_bytes(
                hashlib.sha256(json.dumps(seed_payload).encode()).digest()[:4], "big"
            )
            & 0x7FFFFFFF
        )
        finals.append(replace(task, certified_pieces=tuple(pieces), seed=shared_seed))
    results.update(
        zip(
            (task.cache_key for task in finals),
            solve_planarization_batch(finals, **kwargs),
            strict=True,
        )
    )
    return tuple(results[task.cache_key] for task in tasks)


def solve_planarization_batch(
    tasks: Sequence[PlanarizationTask],
    *,
    time_limit_seconds: float,
    solver_threads: int,
    workers: int,
    checkpoint_directory: Path | None = None,
) -> tuple[PlanarizationResult, ...]:
    """Solve independent graphs concurrently while preserving task order."""

    if not isinstance(workers, Integral) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    for task in tasks:
        if Path(task.cache_key).name != task.cache_key or not task.cache_key:
            raise ValueError("task cache_key must be one safe path component")
    checkpoints = (
        None if checkpoint_directory is None else checkpoint_directory.resolve()
    )
    if checkpoints is not None:
        checkpoints.mkdir(parents=True, exist_ok=True)
    payloads = tuple(
        (
            task,
            float(time_limit_seconds),
            int(solver_threads),
            checkpoints,
        )
        for task in tasks
    )
    if workers == 1 or len(payloads) < 2:
        return tuple(_solve_task(payload) for payload in payloads)
    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        return tuple(executor.map(_solve_task, payloads))


def _solve_task(
    payload: tuple[PlanarizationTask, float, int, int, Path | None],
) -> PlanarizationResult:
    # Distributed workers may request the same piece simultaneously. Lock the
    # complete read/solve/write operation so they all use one certificate.
    task, _limit, _threads, checkpoints = payload
    if checkpoints is None:
        return _solve_task_locked(payload)
    lock_path = checkpoints / f"{task.cache_key}.lock"
    # Concurrent O_CREAT calls on the cluster's NFS mount can transiently
    # return EEXIST even though append mode does not request exclusivity.
    # Retry the open; the subsequent flock still serializes complete solves.
    for attempt in range(20):
        try:
            lock = lock_path.open("a")
            break
        except FileExistsError:
            if attempt == 19:
                raise
            time.sleep(0.05 * (attempt + 1))
    with lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _solve_task_locked(payload)


def _solve_task_locked(payload) -> PlanarizationResult:
    task, time_limit_seconds, solver_threads, checkpoints = payload
    settings = {
        "seed": task.seed,
        "time_limit_seconds": time_limit_seconds,
        "heuristic_runs": 10,
        "solver_threads": solver_threads,
        "shared_piece_policy": 5,
        "backend": build_identity(),
        "piece_certificates": _piece_signature(task),
        "prior_certificates": _piece_signature(
            replace(task, certified_pieces=task.prior_certificates)
        ),
    }
    checkpoint = None if checkpoints is None else checkpoints / f"{task.cache_key}.json"
    if checkpoint is not None:
        cached = _read_checkpoint(checkpoint, task, settings)
        if cached is not None:
            return cached

    result = maximum_planar_subgraph(
        task.graph,
        time_limit_seconds=time_limit_seconds,
        seed=task.seed,
        threads=solver_threads,
        log_path=task.log_path,
        certified_pieces=task.certified_pieces,
        prior_certificates=task.prior_certificates,
    )
    if checkpoint is not None:
        _write_checkpoint(checkpoint, task, settings, result)
    return result


def _write_checkpoint(
    path: Path,
    task: PlanarizationTask,
    settings: dict[str, object],
    result: PlanarizationResult,
) -> None:
    edge_indices = {
        frozenset((first, second)): index
        for index, (first, second) in enumerate(task.graph.edges())
    }
    payload = {
        "schema_version": 3,
        "graph_signature": _graph_signature(task.graph),
        "settings": settings,
        "result": {
            **{
                f.name: getattr(result, f.name)
                for f in fields(result)
                if f.name
                not in {"planar_subgraph", "omitted_edges", "optimizer", "bound_rows"}
            },
            "omitted_edge_indices": [
                edge_indices[frozenset(edge)] for edge in result.omitted_edges
            ],
            "bound_rows": [
                ([edge_indices[frozenset(e)] for e in support], lower)
                for support, lower in result.bound_rows
            ],
            "optimizer": None if result.optimizer is None else asdict(result.optimizer),
        },
    }

    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_checkpoint(
    path: Path,
    task: PlanarizationTask,
    settings: dict[str, object] | None,
) -> PlanarizationResult | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != 3
            or (settings is not None and payload.get("settings") != settings)
            or payload.get("graph_signature") != _graph_signature(task.graph)
        ):
            return None
        record = dict(payload["result"])
        edges = tuple(task.graph.edges())
        omitted_indices = record.pop("omitted_edge_indices")
        bound_indices = record.pop("bound_rows")
        for indices in (omitted_indices, *(ids for ids, _ in bound_indices)):
            if len(set(indices)) != len(indices) or any(
                type(i) is not int or not 0 <= i < len(edges) for i in indices
            ):
                return None
        omitted_edges = tuple(edges[i] for i in omitted_indices)
        planar = task.graph.copy()
        planar.remove_edges_from(omitted_edges)
        bounds = tuple(
            (tuple(edges[i] for i in ids), lower) for ids, lower in bound_indices
        )
        _validate_certificate(
            task.graph,
            CertifiedPlanarPiece(
                task.graph, planar, record["omitted_lower_bound"], bounds
            ),
        )
        optimizer = record.pop("optimizer")
        return PlanarizationResult(
            **record,
            planar_subgraph=planar,
            omitted_edges=omitted_edges,
            bound_rows=bounds,
            optimizer=None if optimizer is None else OgdfStatistics(**optimizer),
        )
    except (OSError, json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
        return None


def _piece_signature(task: PlanarizationTask) -> list:
    index = {node: i for i, node in enumerate(task.graph)}
    return [
        {
            "nodes": sorted(index[node] for node in piece.graph),
            "edges": sorted(
                sorted((index[u], index[v])) for u, v in piece.graph.edges()
            ),
            "retained": sorted(
                sorted((index[u], index[v])) for u, v in piece.planar_subgraph.edges()
            ),
            "lower_bound": int(piece.lower_bound),
            "bound_rows": [
                [sorted(sorted((index[u], index[v])) for u, v in support), lower]
                for support, lower in piece.bound_rows
            ],
        }
        for piece in task.certified_pieces
    ]


def _graph_signature(graph: nx.Graph) -> str:
    node_index = {node: index for index, node in enumerate(graph.nodes())}
    # Omission certificates are stored by edge position, so the signature
    # must distinguish insertion orders as well as graph topology.
    indexed_edges = list(
        (
            min(node_index[first], node_index[second]),
            max(node_index[first], node_index[second]),
        )
        for first, second in graph.edges()
    )
    payload = json.dumps(
        [graph.number_of_nodes(), graph.number_of_edges(), indexed_edges],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def planarization_result_row(result) -> dict[str, object]:
    row = {
        "node_count": result.original_node_count,
        "edge_count": result.original_edge_count,
        "skewness_lower_bound": result.omitted_lower_bound,
        "skewness_upper_bound": result.omitted_upper_bound,
        "skewness_exact": result.exact,
        "absolute_gap": result.omitted_upper_bound - result.omitted_lower_bound,
        "status": result.termination_status,
        "runtime_seconds": result.total_runtime_seconds,
        "seed": result.seed,
        "shared_piece_count": result.shared_piece_count,
        "prior_certificate_count": result.prior_certificate_count,
        "shared_piece_lower_bound": result.shared_piece_lower_bound,
        "greedy_start_upper_bound": result.greedy_start_upper_bound,
        "piece_start_upper_bound": result.piece_start_upper_bound,
        "coupling_forest_edge_count": result.coupling_forest_edge_count,
        "composition_upper_bound": result.composition_upper_bound,
    }
    if result.optimizer is not None:
        row.update(
            solver_status=result.optimizer.status,
            mip_gap=result.optimizer.mip_gap,
            best_bound=result.optimizer.best_bound,
            incumbent_objective=result.optimizer.incumbent_objective,
            objective_upper_bound=result.omitted_upper_bound,
        )
    return row
