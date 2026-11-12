"""Validated persistence for preprocessed PGLib source pools."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from planar_instance_generation.evaluation.source_pool import FeasibleSource
from planar_instance_generation.gates.base import DcOpfSolution
from planar_instance_generation.io.pglib import read_pglib


class SourcePoolManifestError(ValueError):
    """Raised when a source-pool manifest is incomplete or inconsistent."""


def load_pglib_source_pool(
    manifest_path: str | Path,
    *,
    path_root: str | Path | None = None,
    verify_hashes: bool = True,
) -> tuple[FeasibleSource, ...]:
    """Load feasible cases and cached DC solutions from preprocessing output."""

    path = Path(manifest_path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SourcePoolManifestError(f"cannot read source-pool manifest {path}: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise SourcePoolManifestError("unsupported source-pool manifest schema")

    root = Path.cwd() if path_root is None else Path(path_root)
    run = _mapping(manifest, "run")
    source_entries = run.get("sources")
    solutions = _mapping(manifest, "solutions")
    records = manifest.get("records")
    if not isinstance(source_entries, list) or not isinstance(records, list):
        raise SourcePoolManifestError("manifest run.sources and records must be lists")
    record_by_name: dict[str, dict[str, Any]] = {}
    for raw in records:
        if not isinstance(raw, dict) or not isinstance(raw.get("source_name"), str):
            raise SourcePoolManifestError("every source record needs a source_name")
        name = raw["source_name"]
        if name in record_by_name:
            raise SourcePoolManifestError(f"duplicate source record {name!r}")
        record_by_name[name] = raw

    loaded: list[FeasibleSource] = []
    seen_names: set[str] = set()
    for raw_entry in source_entries:
        entry = _entry(raw_entry, "source")
        source_path = _resolve(root, entry.get("path"), "source path")
        if verify_hashes:
            expected = entry.get("sha256")
            if not isinstance(expected, str) or sha256(source_path.read_bytes()).hexdigest() != expected:
                raise SourcePoolManifestError(f"source hash mismatch: {source_path}")
        case = read_pglib(source_path)
        if case.name in seen_names:
            raise SourcePoolManifestError(f"duplicate source case name {case.name!r}")
        seen_names.add(case.name)
        if case.name not in solutions:
            continue
        record = record_by_name.get(case.name)
        if (
            record is None
            or record.get("feasible") is not True
            or record.get("retained", True) is not True
        ):
            raise SourcePoolManifestError(
                f"{case.name}: cached solution has no retained feasible screening record"
            )
        solution_path = _resolve(root, solutions[case.name], "solution path")
        solution = _load_solution(solution_path)
        solution.validate_for(case)
        loaded.append(
            FeasibleSource(
                case=case,
                solution=solution,
                objective=_finite_float(record.get("objective"), "objective"),
                dc_opf_seconds=_nonnegative_float(
                    record.get("dc_opf_seconds"), "dc_opf_seconds"
                ),
                optimizer_seconds=(
                    None
                    if record.get("optimizer_seconds") is None
                    else _nonnegative_float(
                        record.get("optimizer_seconds"), "optimizer_seconds"
                    )
                ),
            )
        )
    declared = manifest.get("feasible")
    if declared != len(loaded):
        raise SourcePoolManifestError(
            f"manifest declares {declared!r} feasible sources but {len(loaded)} were loaded"
        )
    if not loaded:
        raise SourcePoolManifestError("source pool contains no feasible cases")
    return tuple(loaded)


def _load_solution(path: Path) -> DcOpfSolution:
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "bus_angles",
                "branch_from_flows",
                "branch_to_flows",
                "generator_dispatch",
            }
            if set(data.files) != required:
                raise SourcePoolManifestError(
                    f"{path}: expected solution vectors {sorted(required)}"
                )
            return DcOpfSolution(**{name: data[name] for name in required})
    except (OSError, ValueError) as error:
        if isinstance(error, SourcePoolManifestError):
            raise
        raise SourcePoolManifestError(f"cannot read cached solution {path}: {error}") from error


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise SourcePoolManifestError(f"manifest field {key!r} must be an object")
    return value


def _entry(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SourcePoolManifestError(f"every {label} entry must be an object")
    return value


def _resolve(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SourcePoolManifestError(f"{label} must be a nonempty string")
    path = Path(value)
    resolved = path if path.is_absolute() else root / path
    if not resolved.is_file():
        raise SourcePoolManifestError(f"{label} does not exist: {resolved}")
    return resolved


def _finite_float(value: object, label: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise SourcePoolManifestError(f"{label} must be finite") from error
    if not np.isfinite(number):
        raise SourcePoolManifestError(f"{label} must be finite")
    return number


def _nonnegative_float(value: object, label: str) -> float:
    number = _finite_float(value, label)
    if number < 0:
        raise SourcePoolManifestError(f"{label} must be nonnegative")
    return number
