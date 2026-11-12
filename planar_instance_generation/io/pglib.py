"""Strict, non-executing I/O for PGLib MATPOWER v2 case files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Final

import networkx as nx
import numpy as np
from numpy.typing import NDArray


FloatMatrix = NDArray[np.float64]

_REQUIRED_FIELDS: Final = {"version", "baseMVA", "bus", "gen", "branch", "gencost"}
_OPTIONAL_FIELDS: Final = {"areas"}
_MINIMUM_COLUMNS: Final = {"bus": 13, "gen": 10, "branch": 13, "gencost": 4, "areas": 2}
_IDENTIFIER: Final = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_NUMBER: Final = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?|Inf|NaN)$",
    re.IGNORECASE,
)


class PglibFormatError(ValueError):
    """Raised when a file is not in the supported PGLib/MATPOWER subset."""


@dataclass(slots=True)
class PglibCase:
    """A MATPOWER v2 case using PGLib's numeric fields.

    Arrays retain every input column. Topological algorithms should use
    :meth:`topology`, which deliberately projects parallel active branches to a
    simple, loop-free graph while recording their multiplicity.
    """

    name: str
    version: str
    base_mva: float
    bus: FloatMatrix
    gen: FloatMatrix
    branch: FloatMatrix
    gencost: FloatMatrix
    areas: FloatMatrix | None = None

    def __post_init__(self) -> None:
        self.bus = _as_matrix("bus", self.bus)
        self.gen = _as_matrix("gen", self.gen)
        self.branch = _as_matrix("branch", self.branch)
        self.gencost = _as_matrix("gencost", self.gencost)
        if self.areas is not None:
            self.areas = _as_matrix("areas", self.areas)
        self.validate()

    def validate(self) -> None:
        """Fail fast on inconsistencies that would make graph use ambiguous."""

        if not _IDENTIFIER.fullmatch(self.name):
            raise PglibFormatError(f"invalid MATLAB function name: {self.name!r}")
        if self.version != "2":
            raise PglibFormatError(f"only MATPOWER version '2' is supported, got {self.version!r}")
        if not np.isfinite(self.base_mva) or self.base_mva <= 0:
            raise PglibFormatError("baseMVA must be finite and positive")

        matrices = {
            "bus": self.bus,
            "gen": self.gen,
            "branch": self.branch,
            "gencost": self.gencost,
        }
        if self.areas is not None:
            matrices["areas"] = self.areas
        for field, matrix in matrices.items():
            minimum = _MINIMUM_COLUMNS[field]
            if matrix.shape[1] < minimum:
                raise PglibFormatError(
                    f"mpc.{field} has {matrix.shape[1]} columns; expected at least {minimum}"
                )
            if not np.isfinite(matrix).all():
                raise PglibFormatError(f"mpc.{field} contains a non-finite value")

        bus_ids = _integer_column("mpc.bus(:, 1)", self.bus[:, 0])
        if len(set(bus_ids)) != len(bus_ids):
            raise PglibFormatError("mpc.bus contains duplicate bus identifiers")
        known_buses = set(bus_ids)
        gen_buses = _integer_column("mpc.gen(:, 1)", self.gen[:, 0])
        branch_from = _integer_column("mpc.branch(:, 1)", self.branch[:, 0])
        branch_to = _integer_column("mpc.branch(:, 2)", self.branch[:, 1])
        unknown = (set(gen_buses) | set(branch_from) | set(branch_to)) - known_buses
        if unknown:
            preview = ", ".join(map(str, sorted(unknown)[:8]))
            raise PglibFormatError(f"generator or branch references unknown bus(es): {preview}")
        if self.gencost.shape[0] not in {self.gen.shape[0], 2 * self.gen.shape[0]}:
            raise PglibFormatError(
                "mpc.gencost must contain one active-cost row per generator, "
                "optionally followed by one reactive-cost row per generator"
            )

    def topology(self, *, active_only: bool = True) -> nx.Graph:
        """Return the simple, loop-free bus topology.

        Parallel physical branches are preserved through ``multiplicity`` and
        ``branch_indices`` edge attributes instead of parallel graph edges.
        """

        graph = nx.Graph(name=self.name)
        for index, row in enumerate(self.bus):
            bus_id = int(row[0])
            graph.add_node(bus_id, bus_index=index, bus_type=int(row[1]))

        for index, row in enumerate(self.branch):
            if active_only and row[10] <= 0:
                continue
            source, target = int(row[0]), int(row[1])
            if source == target:
                continue
            if graph.has_edge(source, target):
                edge = graph[source][target]
                edge["multiplicity"] += 1
                edge["branch_indices"] = (*edge["branch_indices"], index)
            else:
                graph.add_edge(
                    source,
                    target,
                    multiplicity=1,
                    branch_indices=(index,),
                )
        return graph


def read_pglib(path: str | Path) -> PglibCase:
    """Read a textual PGLib MATPOWER case without evaluating MATLAB code."""

    source_path = Path(path)
    try:
        text = source_path.read_text(encoding="utf-8")
    except OSError as error:
        raise PglibFormatError(f"cannot read {source_path}: {error}") from error

    clean = _strip_matlab_comments(text)
    function_match = re.search(
        r"\bfunction\s+mpc\s*=\s*([A-Za-z][A-Za-z0-9_]*)\b", clean
    )
    if function_match is None:
        raise PglibFormatError("missing 'function mpc = <name>' declaration")

    assigned = set(re.findall(r"\bmpc\.([A-Za-z][A-Za-z0-9_]*)\s*=", clean))
    missing = _REQUIRED_FIELDS - assigned
    unsupported = assigned - _REQUIRED_FIELDS - _OPTIONAL_FIELDS
    if missing:
        raise PglibFormatError(f"missing MATPOWER field(s): {', '.join(sorted(missing))}")
    if unsupported:
        raise PglibFormatError(f"unsupported MATPOWER field(s): {', '.join(sorted(unsupported))}")

    version_match = re.search(r"\bmpc\.version\s*=\s*'([^']+)'\s*;", clean)
    base_match = re.search(r"\bmpc\.baseMVA\s*=\s*([^;\s]+)\s*;", clean)
    if version_match is None or base_match is None:
        raise PglibFormatError("version or baseMVA assignment is malformed")

    return PglibCase(
        name=function_match.group(1),
        version=version_match.group(1),
        base_mva=_parse_number(base_match.group(1), "mpc.baseMVA"),
        bus=_parse_matrix(clean, "bus"),
        gen=_parse_matrix(clean, "gen"),
        branch=_parse_matrix(clean, "branch"),
        gencost=_parse_matrix(clean, "gencost"),
        areas=_parse_matrix(clean, "areas") if "areas" in assigned else None,
    )


def write_pglib(case: PglibCase, path: str | Path) -> Path:
    """Write a deterministic textual MATPOWER v2 case and return its path."""

    case.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        "%% Generated by planar-instance-generation; do not edit generated results.\n",
        f"function mpc = {case.name}\n",
        f"mpc.version = '{case.version}';\n",
        f"mpc.baseMVA = {_format_number(case.base_mva)};\n\n",
        _format_matrix("bus", case.bus),
        _format_matrix("gen", case.gen),
        _format_matrix("gencost", case.gencost),
        _format_matrix("branch", case.branch),
    ]
    if case.areas is not None:
        sections.append(_format_matrix("areas", case.areas))
    try:
        destination.write_text("".join(sections), encoding="utf-8")
    except OSError as error:
        raise PglibFormatError(f"cannot write {destination}: {error}") from error
    return destination


def _strip_matlab_comments(text: str) -> str:
    return "\n".join(line.split("%", 1)[0] for line in text.splitlines())


def _parse_matrix(text: str, field: str) -> FloatMatrix:
    match = re.search(rf"\bmpc\.{re.escape(field)}\s*=\s*\[(.*?)\]\s*;", text, re.DOTALL)
    if match is None:
        raise PglibFormatError(f"mpc.{field} matrix assignment is malformed")
    body = match.group(1).strip()
    rows: list[list[float]] = []
    for row_number, raw_row in enumerate(body.split(";"), start=1):
        raw_row = raw_row.strip()
        if not raw_row:
            continue
        tokens = raw_row.replace(",", " ").split()
        rows.append([_parse_number(token, f"mpc.{field} row {row_number}") for token in tokens])
    if not rows:
        raise PglibFormatError(f"mpc.{field} must not be empty")
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise PglibFormatError(f"mpc.{field} has non-rectangular rows")
    return np.asarray(rows, dtype=np.float64)


def _parse_number(token: str, context: str) -> float:
    if not _NUMBER.fullmatch(token):
        raise PglibFormatError(f"invalid numeric token {token!r} in {context}")
    return float(token)


def _as_matrix(field: str, values: FloatMatrix) -> FloatMatrix:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise PglibFormatError(f"mpc.{field} must be a two-dimensional matrix")
    return matrix


def _integer_column(context: str, values: FloatMatrix) -> list[int]:
    rounded = np.rint(values)
    if not np.array_equal(values, rounded):
        raise PglibFormatError(f"{context} must contain integer identifiers")
    return [int(value) for value in rounded]


def _format_number(value: float) -> str:
    return repr(float(value))


def _format_matrix(field: str, matrix: FloatMatrix) -> str:
    lines = [f"mpc.{field} = [\n"]
    lines.extend("\t" + "\t".join(map(_format_number, row)) + ";\n" for row in matrix)
    lines.append("];\n\n")
    return "".join(lines)
