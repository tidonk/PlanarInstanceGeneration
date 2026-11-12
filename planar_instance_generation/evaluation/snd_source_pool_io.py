"""Validated JSON persistence for preprocessed SNDlib source pools."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

from planar_instance_generation.core.snd import SndSolution, validate_snd_solution
from planar_instance_generation.evaluation.snd_source_pool import (
    ConstructiveSndCapacityPlan,
    FeasibleSndSource,
)
from planar_instance_generation.io.sndlib import read_sndlib


class SndSourcePoolManifestError(ValueError):
    """Raised when an SND source-pool artifact is incomplete or inconsistent."""


def write_snd_source_solution(
    path: str | Path,
    source: FeasibleSndSource,
) -> Path:
    """Write exact and constructive source flows without overwriting a file."""

    output = Path(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite SND source solution: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source_name": source.source_name,
        "exact_mcmcf_solution": _solution_payload(source.solution),
        "constructive_plan": {
            "method": source.constructive_plan.method,
            "design_optimal": source.constructive_plan.design_optimal,
            "solution": _solution_payload(source.constructive_plan.solution),
            "demand_paths": {
                demand_id: list(path)
                for demand_id, path in sorted(
                    source.constructive_plan.demand_paths.items()
                )
            },
            "module_copies": {
                link_id: list(copies)
                for link_id, copies in sorted(
                    source.constructive_plan.module_copies.items()
                )
            },
            "link_loads": dict(sorted(source.constructive_plan.link_loads.items())),
            "default_routing_cost": source.constructive_plan.default_routing_cost,
            "routing_cost": source.constructive_plan.routing_cost,
            "module_cost": source.constructive_plan.module_cost,
            "setup_cost": source.constructive_plan.setup_cost,
            "total_design_cost": source.constructive_plan.total_design_cost,
        },
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def load_snd_source_pool(
    manifest_path: str | Path,
    *,
    path_root: str | Path | None = None,
    verify_hashes: bool = True,
) -> tuple[FeasibleSndSource, ...]:
    """Load retained SNDlib sources and validate every cached flow/capacity."""

    path = Path(manifest_path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SndSourcePoolManifestError(f"cannot read manifest {path}: {error}") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("artifact_type") != "sndlib_source_pool"
    ):
        raise SndSourcePoolManifestError("unsupported SND source-pool manifest schema")

    root = Path.cwd() if path_root is None else Path(path_root)
    run = _mapping(manifest.get("run"), "run")
    source_entries = run.get("sources")
    records = manifest.get("records")
    solutions = _mapping(manifest.get("solutions"), "solutions")
    if not isinstance(source_entries, list) or not isinstance(records, list):
        raise SndSourcePoolManifestError("run.sources and records must be lists")
    record_by_name: dict[str, Mapping[str, Any]] = {}
    for raw in records:
        record = _mapping(raw, "record")
        name = record.get("source_name")
        if not isinstance(name, str) or not name:
            raise SndSourcePoolManifestError("every record needs a source_name")
        if name in record_by_name:
            raise SndSourcePoolManifestError(f"duplicate source record {name!r}")
        record_by_name[name] = record

    loaded: list[FeasibleSndSource] = []
    seen: set[str] = set()
    for raw_entry in source_entries:
        entry = _mapping(raw_entry, "source entry")
        name = entry.get("source_name")
        if not isinstance(name, str) or not name:
            raise SndSourcePoolManifestError("source entry needs a source_name")
        if name in seen:
            raise SndSourcePoolManifestError(f"duplicate source entry {name!r}")
        seen.add(name)
        source_path = _resolve(root, entry.get("path"), "source path")
        if verify_hashes:
            expected_hash = entry.get("sha256")
            if (
                not isinstance(expected_hash, str)
                or sha256(source_path.read_bytes()).hexdigest() != expected_hash
            ):
                raise SndSourcePoolManifestError(f"source hash mismatch: {source_path}")
        network = read_sndlib(source_path)
        record = record_by_name.get(name)
        if record is None:
            raise SndSourcePoolManifestError(f"missing screening record for {name!r}")
        retained = record.get("retained")
        if retained not in (True, False):
            raise SndSourcePoolManifestError(f"{name}: retained must be a bool")
        if not retained:
            if name in solutions:
                raise SndSourcePoolManifestError(
                    f"{name}: excluded source unexpectedly has a solution artifact"
                )
            continue
        solution_path = _resolve(root, solutions.get(name), "solution path")
        exact, plan = _load_solution_artifact(solution_path, name, network)
        log_path = (
            None
            if record.get("log_path") is None
            else _resolve(root, record.get("log_path"), "solver log path")
        )
        loaded.append(
            FeasibleSndSource(
                source_name=name,
                network=network,
                solution=exact,
                constructive_plan=plan,
                objective=_finite(record.get("objective"), f"{name} objective"),
                constructive_seconds=_nonnegative(
                    record.get("constructive_seconds"),
                    f"{name} constructive_seconds",
                ),
                mcmcf_seconds=_nonnegative(
                    record.get("mcmcf_seconds"), f"{name} mcmcf_seconds"
                ),
                preprocessing_seconds=_nonnegative(
                    record.get("preprocessing_seconds"),
                    f"{name} preprocessing_seconds",
                ),
                optimizer_seconds=(
                    None
                    if record.get("optimizer_seconds") is None
                    else _nonnegative(
                        record.get("optimizer_seconds"),
                        f"{name} optimizer_seconds",
                    )
                ),
                log_path=log_path,
            )
        )
    declared = manifest.get("retained")
    if declared != len(loaded):
        raise SndSourcePoolManifestError(
            f"manifest declares {declared!r} retained sources but loaded {len(loaded)}"
        )
    if not loaded:
        raise SndSourcePoolManifestError("SND source pool contains no retained sources")
    return tuple(loaded)


def _solution_payload(solution: SndSolution) -> dict[str, Any]:
    return {
        "installed_capacities": dict(sorted(solution.installed_capacities.items())),
        "flows": [
            {
                "demand_id": demand_id,
                "link_id": link_id,
                "value": value,
            }
            for (demand_id, link_id), value in sorted(solution.flows.items())
        ],
    }


def _load_solution_artifact(
    path: Path,
    expected_name: str,
    network: Any,
) -> tuple[SndSolution, ConstructiveSndCapacityPlan]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SndSourcePoolManifestError(f"cannot read solution {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise SndSourcePoolManifestError(f"{path}: unsupported solution schema")
    if payload.get("source_name") != expected_name:
        raise SndSourcePoolManifestError(f"{path}: source_name mismatch")
    exact = _load_solution(payload.get("exact_mcmcf_solution"), "exact solution")
    plan_raw = _mapping(payload.get("constructive_plan"), "constructive plan")
    plan_solution = _load_solution(plan_raw.get("solution"), "constructive solution")
    try:
        paths_raw = _mapping(plan_raw.get("demand_paths"), "demand_paths")
        copies_raw = _mapping(plan_raw.get("module_copies"), "module_copies")
        loads_raw = _mapping(plan_raw.get("link_loads"), "link_loads")
        plan = ConstructiveSndCapacityPlan(
            solution=plan_solution,
            demand_paths={
                str(key): tuple(_string_list(value, f"path {key!r}"))
                for key, value in paths_raw.items()
            },
            module_copies={
                str(key): tuple(_integer_list(value, f"module copies {key!r}"))
                for key, value in copies_raw.items()
            },
            link_loads={
                str(key): _nonnegative(value, f"link load {key!r}")
                for key, value in loads_raw.items()
            },
            default_routing_cost=_nonnegative(
                plan_raw.get("default_routing_cost"), "default_routing_cost"
            ),
            routing_cost=_nonnegative(plan_raw.get("routing_cost"), "routing_cost"),
            module_cost=_nonnegative(plan_raw.get("module_cost"), "module_cost"),
            setup_cost=_nonnegative(plan_raw.get("setup_cost"), "setup_cost"),
            total_design_cost=_nonnegative(
                plan_raw.get("total_design_cost"), "total_design_cost"
            ),
            method=str(plan_raw.get("method")),
            design_optimal=plan_raw.get("design_optimal"),
        )
    except (TypeError, ValueError) as error:
        raise SndSourcePoolManifestError(f"{path}: invalid constructive plan: {error}") from error
    try:
        validate_snd_solution(network, exact)
        validate_snd_solution(network, plan.solution)
    except (TypeError, ValueError) as error:
        raise SndSourcePoolManifestError(f"{path}: invalid cached solution: {error}") from error
    if dict(exact.installed_capacities) != dict(plan.solution.installed_capacities):
        raise SndSourcePoolManifestError(
            f"{path}: exact and constructive installed capacities differ"
        )
    return exact, plan


def _load_solution(value: object, label: str) -> SndSolution:
    raw = _mapping(value, label)
    capacities_raw = _mapping(raw.get("installed_capacities"), "installed_capacities")
    flows_raw = raw.get("flows")
    if not isinstance(flows_raw, list):
        raise SndSourcePoolManifestError(f"{label}.flows must be a list")
    flows: dict[tuple[str, str], float] = {}
    for value in flows_raw:
        row = _mapping(value, "flow row")
        demand_id = row.get("demand_id")
        link_id = row.get("link_id")
        if not isinstance(demand_id, str) or not isinstance(link_id, str):
            raise SndSourcePoolManifestError("flow identifiers must be strings")
        key = (demand_id, link_id)
        if key in flows:
            raise SndSourcePoolManifestError(f"duplicate flow row {key!r}")
        flows[key] = _finite(row.get("value"), f"flow {key!r}")
    try:
        return SndSolution(
            installed_capacities={
                str(key): _nonnegative(value, f"capacity {key!r}")
                for key, value in capacities_raw.items()
            },
            flows=flows,
        )
    except (TypeError, ValueError) as error:
        raise SndSourcePoolManifestError(f"invalid {label}: {error}") from error


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SndSourcePoolManifestError(f"{label} must be an object")
    return value


def _resolve(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SndSourcePoolManifestError(f"{label} must be a nonempty string")
    path = Path(value)
    resolved = path if path.is_absolute() else root / path
    if not resolved.is_file():
        raise SndSourcePoolManifestError(f"{label} does not exist: {resolved}")
    return resolved


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise SndSourcePoolManifestError(f"{label} must be finite")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise SndSourcePoolManifestError(f"{label} must be finite") from error
    if not math.isfinite(number):
        raise SndSourcePoolManifestError(f"{label} must be finite")
    return number


def _nonnegative(value: object, label: str) -> float:
    number = _finite(value, label)
    if number < 0.0:
        raise SndSourcePoolManifestError(f"{label} must be nonnegative")
    return number


def _string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SndSourcePoolManifestError(f"{label} must be a list of strings")
    return value


def _integer_list(value: object, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in value
        )
    ):
        raise SndSourcePoolManifestError(
            f"{label} must be a list of nonnegative integers"
        )
    return value
