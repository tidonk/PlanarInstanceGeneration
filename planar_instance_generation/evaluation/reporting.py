"""Tabular and plot-ready artifacts for the computational evaluation.

The writer deliberately stops at standalone LaTeX sources.  Compiling those
sources is a runner concern, which keeps this module deterministic and easy to
test on machines without a TeX installation.
"""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Hashable, Iterable, Mapping, Sequence, TypeAlias

import networkx as nx
import numpy as np

from planar_instance_generation.core.planarization import PlanarizationResult
from planar_instance_generation.evaluation.structural import (
    StructuralStatistics,
    measure_structure,
)
from planar_instance_generation.plt.boxplot import BoxplotSeries, write_boxplot
from planar_instance_generation.plt.tikz import PlotArtifacts


REPORT_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class EvaluationGraph:
    """A named graph and the nodes introduced or used by composition."""

    name: str
    graph: nx.Graph
    coupling_nodes: tuple[Hashable, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("graph name must be a nonempty string")


@dataclass(frozen=True, slots=True)
class SkewnessBounds:
    """Certified bounds independent of a particular optimizer implementation."""

    lower_bound: int
    upper_bound: int
    status: str | None = None
    runtime_seconds: float | None = None

    def __post_init__(self) -> None:
        lower = _nonnegative_integer(self.lower_bound, "lower_bound")
        upper = _nonnegative_integer(self.upper_bound, "upper_bound")
        if lower > upper:
            raise ValueError("lower_bound must not exceed upper_bound")
        if self.status is not None and not self.status:
            raise ValueError("status must be nonempty when provided")
        if self.runtime_seconds is not None and (
            not isinstance(self.runtime_seconds, Real)
            or isinstance(self.runtime_seconds, bool)
            or not math.isfinite(float(self.runtime_seconds))
            or self.runtime_seconds < 0
        ):
            raise ValueError("runtime_seconds must be finite and non-negative")
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        if self.runtime_seconds is not None:
            object.__setattr__(self, "runtime_seconds", float(self.runtime_seconds))

    @property
    def exact(self) -> bool:
        return self.lower_bound == self.upper_bound


SkewnessResult: TypeAlias = SkewnessBounds | PlanarizationResult


@dataclass(frozen=True, slots=True)
class SkewnessObservation:
    """Skewness result plus the data needed for normalized inflation.

    The source-floor interval is the sum of the constituent skewness
    intervals.  For a composite with ``coupling_edge_count > 0``, these data
    yield certified bounds on the paper's normalized inflation.
    """

    result: SkewnessResult
    source_floor_lower_bound: int | None = None
    source_floor_upper_bound: int | None = None
    coupling_edge_count: int | None = None
    # Supplied only for a connected composition with connected source pieces.
    # Omit for legacy inputs or general compositions without that certificate.
    source_piece_count: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.result, (SkewnessBounds, PlanarizationResult)):
            raise TypeError(
                "result must contain skewness bounds or a planarization result"
            )
        provided = (
            self.source_floor_lower_bound is not None,
            self.source_floor_upper_bound is not None,
            self.coupling_edge_count is not None,
        )
        if any(provided) and not all(provided):
            raise ValueError(
                "source-floor bounds and coupling_edge_count must be provided together"
            )
        if not any(provided):
            if self.source_piece_count is not None:
                raise ValueError("source_piece_count requires composition bounds")
            return
        lower = _nonnegative_integer(
            self.source_floor_lower_bound, "source_floor_lower_bound"
        )
        upper = _nonnegative_integer(
            self.source_floor_upper_bound, "source_floor_upper_bound"
        )
        coupling_count = _nonnegative_integer(
            self.coupling_edge_count, "coupling_edge_count"
        )
        if lower > upper:
            raise ValueError(
                "source_floor_lower_bound must not exceed source_floor_upper_bound"
            )
        object.__setattr__(self, "source_floor_lower_bound", lower)
        object.__setattr__(self, "source_floor_upper_bound", upper)
        object.__setattr__(self, "coupling_edge_count", coupling_count)
        if self.source_piece_count is not None:
            count = _nonnegative_integer(self.source_piece_count, "source_piece_count")
            if count < 1 or count - 1 > coupling_count:
                raise ValueError("a connected composition needs at least p - 1 coupling edges")
            object.__setattr__(self, "source_piece_count", count)


SkewnessInput: TypeAlias = SkewnessObservation | SkewnessResult
SkewnessKey: TypeAlias = tuple[str, str]
GraphGroup: TypeAlias = Sequence[EvaluationGraph] | Mapping[str, nx.Graph]


@dataclass(frozen=True, slots=True)
class EvaluationReportArtifacts:
    """Paths produced by :func:`write_evaluation_report`."""

    structural_csv: Path
    structural_json: Path
    degree_distribution_csv: Path
    coupling_betweenness_csv: Path
    skewness_csv: Path
    skewness_json: Path
    boxplot_manifest_json: Path
    boxplots: Mapping[str, PlotArtifacts]


@dataclass(frozen=True, slots=True)
class _MeasuredGraph:
    group: str
    sample: EvaluationGraph
    statistics: StructuralStatistics
    degrees: tuple[int, ...]
    betweenness_samples: int | None
    betweenness_seed: int | None


_STRUCTURAL_BOXPLOTS: tuple[tuple[str, str], ...] = (
    ("node_count", "Nodes"),
    ("edge_count", "Edges"),
    ("density", "Density"),
    ("mean_degree", "Mean degree"),
    ("maximum_degree", "Maximum degree"),
    ("average_clustering", "Average clustering coefficient"),
    ("degeneracy", "Degeneracy"),
    ("treewidth_upper_bound", "Treewidth upper bound"),
    ("diameter", "Diameter"),
    ("algebraic_connectivity", "Algebraic connectivity"),
    ("mean_betweenness", "Mean normalized betweenness"),
    ("betweenness_p95", "95th-percentile normalized betweenness"),
    ("degree_assortativity", "Degree assortativity"),
)
_GROUP_COLORS = (
    "0072B2",
    "E69F00",
    "009E73",
    "CC79A7",
    "56B4E9",
    "D55E00",
    "332288",
    "999999",
)


def write_evaluation_report(
    graph_groups: Mapping[str, GraphGroup],
    output_directory: str | Path,
    *,
    skewness: Mapping[SkewnessKey, SkewnessInput] | None = None,
    betweenness_samples: int | None = None,
    seed: int = 0,
    overwrite: bool = False,
) -> EvaluationReportArtifacts:
    """Measure named graph groups and write complete evaluation artifacts.

    A group may be a ``name -> graph`` mapping, or a sequence of
    :class:`EvaluationGraph` records when coupling nodes must be identified.
    ``skewness`` is keyed by ``(group_name, graph_name)``.  A bare
    :class:`PlanarizationResult` or :class:`SkewnessBounds` records skewness;
    wrap it in :class:`SkewnessObservation` to additionally compute normalized
    inflation for a composite.

    No PDF compiler is invoked.  Every boxplot consists of a TikZ fragment, a
    standalone ``.tex`` PDF input, and its numerical JSON sidecar.
    ``betweenness_samples`` is a per-graph cap; graphs no larger than that cap
    are measured exactly.
    """

    groups = _validate_groups(graph_groups)
    root_seed = _nonnegative_integer(seed, "seed")
    skewness_inputs = dict(skewness or {})
    known_keys = {
        (group, sample.name)
        for group, samples in groups
        for sample in samples
    }
    invalid_keys = [
        key
        for key in skewness_inputs
        if not isinstance(key, tuple)
        or len(key) != 2
        or not all(isinstance(part, str) for part in key)
    ]
    if invalid_keys:
        raise TypeError(
            "skewness keys must be (group_name, graph_name) string pairs"
        )
    unknown_skewness = set(skewness_inputs) - known_keys
    if unknown_skewness:
        raise ValueError(
            "skewness contains unknown graph key(s): "
            + ", ".join(map(repr, sorted(unknown_skewness)))
        )

    output = Path(output_directory)
    artifacts = _artifact_paths(output)
    tabular_paths = (
        artifacts.structural_csv,
        artifacts.structural_json,
        artifacts.degree_distribution_csv,
        artifacts.coupling_betweenness_csv,
        artifacts.skewness_csv,
        artifacts.skewness_json,
        artifacts.boxplot_manifest_json,
    )
    existing = [path for path in tabular_paths if path.exists()]
    boxplot_directory = output / "boxplots"
    if boxplot_directory.is_dir():
        existing.extend(
            path
            for path in boxplot_directory.iterdir()
            if path.suffix in {".tikz", ".tex", ".json"}
        )
    if existing and not overwrite:
        raise FileExistsError(f"evaluation artifact already exists: {existing[0]}")

    measured = _measure_groups(groups, betweenness_samples, root_seed)
    structural_rows = [_structural_row(item) for item in measured]
    degree_rows = _degree_rows(measured)
    coupling_rows = _coupling_rows(measured)
    skewness_rows = _skewness_rows(measured, skewness_inputs)

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(artifacts.structural_csv, structural_rows, _STRUCTURAL_FIELDS)
    _write_csv(
        artifacts.degree_distribution_csv,
        degree_rows,
        _DEGREE_DISTRIBUTION_FIELDS,
    )
    _write_csv(
        artifacts.coupling_betweenness_csv,
        coupling_rows,
        _COUPLING_BETWEENNESS_FIELDS,
    )
    _write_csv(artifacts.skewness_csv, skewness_rows, _SKEWNESS_FIELDS)
    _write_json(
        artifacts.structural_json,
        {
            "format_version": REPORT_FORMAT_VERSION,
            "betweenness_sample_cap": betweenness_samples,
            "seed": root_seed,
            "instances": [_structural_json_row(item) for item in measured],
        },
    )
    _write_json(
        artifacts.skewness_json,
        {
            "format_version": REPORT_FORMAT_VERSION,
            "instances": skewness_rows,
        },
    )

    boxplots, manifest = _write_boxplots(
        measured,
        skewness_rows,
        output / "boxplots",
        overwrite=overwrite,
    )
    _write_json(
        artifacts.boxplot_manifest_json,
        {
            "format_version": REPORT_FORMAT_VERSION,
            "compiler_invoked": False,
            "plots": manifest,
        },
    )
    return EvaluationReportArtifacts(
        structural_csv=artifacts.structural_csv,
        structural_json=artifacts.structural_json,
        degree_distribution_csv=artifacts.degree_distribution_csv,
        coupling_betweenness_csv=artifacts.coupling_betweenness_csv,
        skewness_csv=artifacts.skewness_csv,
        skewness_json=artifacts.skewness_json,
        boxplot_manifest_json=artifacts.boxplot_manifest_json,
        boxplots=boxplots,
    )


@dataclass(frozen=True, slots=True)
class _ArtifactPaths:
    structural_csv: Path
    structural_json: Path
    degree_distribution_csv: Path
    coupling_betweenness_csv: Path
    skewness_csv: Path
    skewness_json: Path
    boxplot_manifest_json: Path


def _artifact_paths(output: Path) -> _ArtifactPaths:
    return _ArtifactPaths(
        structural_csv=output / "structural.csv",
        structural_json=output / "structural.json",
        degree_distribution_csv=output / "degree_distribution.csv",
        coupling_betweenness_csv=output / "coupling_betweenness.csv",
        skewness_csv=output / "skewness.csv",
        skewness_json=output / "skewness.json",
        boxplot_manifest_json=output / "boxplots.json",
    )


def _validate_groups(
    graph_groups: Mapping[str, GraphGroup],
) -> tuple[tuple[str, tuple[EvaluationGraph, ...]], ...]:
    if not isinstance(graph_groups, Mapping) or not graph_groups:
        raise ValueError("graph_groups must be a nonempty mapping")
    result = []
    for group, samples_input in graph_groups.items():
        if not isinstance(group, str) or not group.strip():
            raise ValueError("group names must be nonempty strings")
        samples = (
            tuple(
                EvaluationGraph(name, graph)
                for name, graph in samples_input.items()
            )
            if isinstance(samples_input, Mapping)
            else tuple(samples_input)
        )
        if not samples:
            raise ValueError(f"group {group!r} must not be empty")
        if not all(isinstance(sample, EvaluationGraph) for sample in samples):
            raise TypeError("each graph group must contain EvaluationGraph records")
        names = [sample.name for sample in samples]
        if len(names) != len(set(names)):
            raise ValueError(f"graph names must be unique within group {group!r}")
        result.append((group, samples))
    return tuple(result)


def _measure_groups(
    groups: tuple[tuple[str, tuple[EvaluationGraph, ...]], ...],
    betweenness_samples: int | None,
    seed: int,
) -> tuple[_MeasuredGraph, ...]:
    measured = []
    for group, samples in groups:
        for sample in samples:
            node_count = sample.graph.number_of_nodes()
            effective_samples = (
                betweenness_samples
                if betweenness_samples is not None
                and betweenness_samples < node_count
                else None
            )
            statistics = measure_structure(
                sample.graph,
                coupling_nodes=sample.coupling_nodes,
                betweenness_samples=effective_samples,
                seed=seed,
            )
            measured.append(
                _MeasuredGraph(
                    group=group,
                    sample=sample,
                    statistics=statistics,
                    degrees=tuple(degree for _, degree in sample.graph.degree),
                    betweenness_samples=effective_samples,
                    betweenness_seed=seed if effective_samples is not None else None,
                )
            )
    return tuple(measured)


def _structural_row(item: _MeasuredGraph) -> dict[str, object]:
    statistics = item.statistics
    coupling_values = tuple(statistics.coupling_betweenness.values())
    return {
        "group": item.group,
        "instance": item.sample.name,
        "node_count": statistics.node_count,
        "edge_count": statistics.edge_count,
        "density": statistics.density,
        "mean_degree": statistics.mean_degree,
        "maximum_degree": statistics.maximum_degree,
        "average_clustering": statistics.average_clustering,
        "degeneracy": statistics.degeneracy,
        "treewidth_upper_bound": statistics.treewidth_upper_bound,
        "connected": statistics.connected,
        "diameter": statistics.diameter,
        "algebraic_connectivity": statistics.algebraic_connectivity,
        "mean_betweenness": statistics.mean_betweenness,
        "betweenness_p95": statistics.betweenness_p95,
        "betweenness_sample_count": statistics.betweenness_sample_count,
        "betweenness_exact": item.betweenness_samples is None,
        "betweenness_seed": item.betweenness_seed,
        "coupling_node_count": len(coupling_values),
        "coupling_betweenness_mean": _mean_or_none(coupling_values),
        "coupling_betweenness_p95": _quantile_or_none(coupling_values, 0.95),
        "coupling_betweenness_max": max(coupling_values, default=None),
        "degree_assortativity": statistics.degree_assortativity,
        "measurement_seconds": statistics.measurement_seconds,
    }


def _structural_json_row(item: _MeasuredGraph) -> dict[str, object]:
    row = _structural_row(item)
    degree_counts = Counter(item.degrees)
    row["degree_distribution"] = [
        {
            "degree": degree,
            "node_count": degree_counts[degree],
            "probability": probability,
        }
        for degree, probability in item.statistics.degree_distribution.items()
    ]
    row["coupling_betweenness"] = [
        {"node": repr(node), "value": value}
        for node, value in item.statistics.coupling_betweenness.items()
    ]
    return row


def _degree_rows(measured: Sequence[_MeasuredGraph]) -> list[dict[str, object]]:
    rows = []
    for item in measured:
        counts = Counter(item.degrees)
        for degree, probability in item.statistics.degree_distribution.items():
            rows.append(
                {
                    "group": item.group,
                    "instance": item.sample.name,
                    "degree": degree,
                    "node_count": counts[degree],
                    "probability": probability,
                }
            )
    return rows


def _coupling_rows(measured: Sequence[_MeasuredGraph]) -> list[dict[str, object]]:
    return [
        {
            "group": item.group,
            "instance": item.sample.name,
            "node": repr(node),
            "betweenness": value,
        }
        for item in measured
        for node, value in item.statistics.coupling_betweenness.items()
    ]


def _skewness_rows(
    measured: Sequence[_MeasuredGraph],
    inputs: Mapping[SkewnessKey, SkewnessInput],
) -> list[dict[str, object]]:
    rows = []
    for item in measured:
        key = (item.group, item.sample.name)
        if key not in inputs:
            continue
        observation = _as_observation(inputs[key])
        solver_bounds = _as_bounds(observation.result, item)
        if observation.source_piece_count is not None and not nx.is_connected(item.sample.graph):
            raise ValueError("source_piece_count requires a connected composite")
        bounds = propagate_composition_bounds(solver_bounds, observation)
        row: dict[str, object] = {
            "group": item.group,
            "instance": item.sample.name,
            "node_count": item.statistics.node_count,
            "edge_count": item.statistics.edge_count,
            "skewness_lower_bound": bounds.lower_bound,
            "skewness_upper_bound": bounds.upper_bound,
            "skewness_exact": bounds.exact,
            "skewness": bounds.lower_bound if bounds.exact else None,
            "absolute_gap": bounds.upper_bound - bounds.lower_bound,
            "relative_gap": (
                (bounds.upper_bound - bounds.lower_bound) / bounds.upper_bound
                if bounds.upper_bound
                else 0.0
            ),
            "status": bounds.status,
            "runtime_seconds": bounds.runtime_seconds,
            "source_floor_lower_bound": observation.source_floor_lower_bound,
            "source_floor_upper_bound": observation.source_floor_upper_bound,
            "coupling_edge_count": observation.coupling_edge_count,
            "source_piece_count": observation.source_piece_count,
            "coupling_forest_edge_count": (observation.source_piece_count - 1 if observation.source_piece_count is not None else 0),
            "solver_lower_bound": solver_bounds.lower_bound,
            "solver_upper_bound": solver_bounds.upper_bound,
            "inflation_lower_bound": None,
            "inflation_upper_bound": None,
            "inflation_exact": None,
            "inflation": None,
            "normalized_inflation_lower_bound": None,
            "normalized_inflation_upper_bound": None,
            "normalized_inflation_exact": None,
            "normalized_inflation": None,
        }
        _add_inflation(row, bounds, observation)
        rows.append(row)
    return rows


def _as_observation(value: SkewnessInput) -> SkewnessObservation:
    if isinstance(value, SkewnessObservation):
        return value
    if isinstance(value, (SkewnessBounds, PlanarizationResult)):
        return SkewnessObservation(value)
    raise TypeError("skewness values must be bounds, results, or observations")


def _as_bounds(result: SkewnessResult, item: _MeasuredGraph) -> SkewnessBounds:
    if isinstance(result, SkewnessBounds):
        bounds = result
    elif isinstance(result, PlanarizationResult):
        if (
            result.original_node_count != item.statistics.node_count
            or result.original_edge_count != item.statistics.edge_count
        ):
            raise ValueError(
                f"planarization dimensions do not match {(item.group, item.sample.name)!r}"
            )
        original_nodes = set(item.sample.graph)
        planar_nodes = set(result.planar_subgraph)
        original_edges = {
            frozenset(edge) for edge in item.sample.graph.edges
        }
        planar_edges = {
            frozenset(edge) for edge in result.planar_subgraph.edges
        }
        omitted_edges = {frozenset(edge) for edge in result.omitted_edges}
        if (
            planar_nodes != original_nodes
            or planar_edges & omitted_edges
            or planar_edges | omitted_edges != original_edges
        ):
            raise ValueError(
                f"planarization certificate does not match "
                f"{(item.group, item.sample.name)!r}"
            )
        bounds = SkewnessBounds(
            lower_bound=result.omitted_lower_bound,
            upper_bound=result.omitted_upper_bound,
            status=result.termination_status,
            runtime_seconds=result.total_runtime_seconds,
        )
    else:  # Defensive for callers bypassing type checking.
        raise TypeError("unsupported skewness result")
    if bounds.upper_bound > item.statistics.edge_count:
        raise ValueError(
            f"skewness upper bound exceeds edge count for {(item.group, item.sample.name)!r}"
        )
    return bounds


def propagate_composition_bounds(composite: SkewnessBounds, observation: SkewnessObservation) -> SkewnessBounds:
    """Intersect solver bounds with the source floor and constructive upper bound."""
    if observation.source_floor_lower_bound is None:
        return composite
    forest = observation.source_piece_count - 1 if observation.source_piece_count is not None else 0
    cap = int(observation.coupling_edge_count) - forest
    lower = max(composite.lower_bound, int(observation.source_floor_lower_bound))
    upper = min(composite.upper_bound, int(observation.source_floor_upper_bound) + cap)
    if lower > upper:
        raise ValueError("solver bounds contradict the certified composition interval")
    return SkewnessBounds(lower, upper, composite.status, composite.runtime_seconds)


def _add_inflation(
    row: dict[str, object],
    composite: SkewnessBounds,
    observation: SkewnessObservation,
) -> None:
    if observation.source_floor_lower_bound is None:
        return
    floor_lower = int(observation.source_floor_lower_bound)
    floor_upper = int(observation.source_floor_upper_bound)
    coupling_count = int(observation.coupling_edge_count)
    raw_lower = composite.lower_bound - floor_upper
    raw_upper = composite.upper_bound - floor_lower
    lower = max(0, raw_lower)
    forest = observation.source_piece_count - 1 if observation.source_piece_count is not None else 0
    upper = min(coupling_count - forest, raw_upper)
    if lower > upper:
        raise ValueError("skewness bounds contradict the composition inflation bounds")
    row["inflation_lower_bound"] = lower
    row["inflation_upper_bound"] = upper
    row["inflation_exact"] = lower == upper
    if lower == upper:
        row["inflation"] = lower
    if coupling_count == 0:
        return
    row["normalized_inflation_lower_bound"] = lower / coupling_count
    row["normalized_inflation_upper_bound"] = upper / coupling_count
    row["normalized_inflation_exact"] = lower == upper
    if lower == upper:
        row["normalized_inflation"] = lower / coupling_count


def _write_boxplots(
    measured: Sequence[_MeasuredGraph],
    skewness_rows: Sequence[Mapping[str, object]],
    output: Path,
    *,
    overwrite: bool,
) -> tuple[dict[str, PlotArtifacts], list[dict[str, object]]]:
    group_order = tuple(dict.fromkeys(item.group for item in measured))
    group_colors = {
        group: _GROUP_COLORS[index % len(_GROUP_COLORS)]
        for index, group in enumerate(group_order)
    }
    metric_values: dict[str, tuple[str, dict[str, list[float]]]] = {}
    for field, label in _STRUCTURAL_BOXPLOTS:
        metric_values[field] = (
            label,
            _group_values(
                group_order,
                (
                    (item.group, getattr(item.statistics, field))
                    for item in measured
                ),
            ),
        )
    metric_values["degree"] = (
        "Node degree",
        _group_values(
            group_order,
            (
                (item.group, degree)
                for item in measured
                for degree in item.degrees
            ),
        ),
    )
    metric_values["coupling_betweenness"] = (
        "Coupling-node normalized betweenness",
        _group_values(
            group_order,
            (
                (item.group, value)
                for item in measured
                for value in item.statistics.coupling_betweenness.values()
            ),
        ),
    )

    skewness_labels = {
        "skewness_lower_bound": "Skewness lower bound",
        "skewness_upper_bound": "Skewness upper bound",
        "normalized_inflation_lower_bound": "Normalized skewness inflation lower bound",
        "normalized_inflation_upper_bound": "Normalized skewness inflation upper bound",
        "normalized_inflation": "Normalized skewness inflation",
    }
    for field, label in skewness_labels.items():
        metric_values[field] = (
            label,
            _group_values(
                group_order,
                ((str(row["group"]), row[field]) for row in skewness_rows),
            ),
        )

    artifacts: dict[str, PlotArtifacts] = {}
    manifest = []
    for metric, (label, values_by_group) in metric_values.items():
        nonempty = {
            group: tuple(values)
            for group, values in values_by_group.items()
            if values
        }
        omitted_groups = [group for group in group_order if group not in nonempty]
        if nonempty:
            plot = write_boxplot(
                [
                    BoxplotSeries(group, values, color=group_colors[group])
                    for group, values in nonempty.items()
                ],
                output / metric,
                y_label=label,
                width_cm=16.0 if len(nonempty) > 4 else 8.4,
                height_cm=6.2 if len(nonempty) > 4 else 5.8,
                overwrite=overwrite,
            )
            artifacts[metric] = plot
            paths = {
                "tikz": str(Path("boxplots") / plot.tikz.name),
                "standalone_tex": str(
                    Path("boxplots") / plot.standalone_tex.name
                ),
                "data_json": str(Path("boxplots") / plot.coordinates_json.name),
            }
        else:
            if overwrite:
                for suffix in (".tikz", ".tex", ".json"):
                    (output / metric).with_suffix(suffix).unlink(missing_ok=True)
            paths = None
        manifest.append(
            {
                "metric": metric,
                "y_label": label,
                "value_counts": {
                    group: len(values_by_group[group]) for group in group_order
                },
                "omitted_groups": omitted_groups,
                "paths": paths,
            }
        )
    return artifacts, manifest


def _group_values(
    group_order: Sequence[str],
    observations: Iterable[tuple[str, object]],
) -> dict[str, list[float]]:
    values = {group: [] for group in group_order}
    for group, value in observations:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            values[group].append(number)
    return values


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _nonnegative_integer(value: object, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _quantile_or_none(values: Sequence[float], quantile: float) -> float | None:
    return float(np.quantile(values, quantile)) if values else None


_STRUCTURAL_FIELDS = (
    "group",
    "instance",
    "node_count",
    "edge_count",
    "density",
    "mean_degree",
    "maximum_degree",
    "average_clustering",
    "degeneracy",
    "treewidth_upper_bound",
    "connected",
    "diameter",
    "algebraic_connectivity",
    "mean_betweenness",
    "betweenness_p95",
    "betweenness_sample_count",
    "betweenness_exact",
    "betweenness_seed",
    "coupling_node_count",
    "coupling_betweenness_mean",
    "coupling_betweenness_p95",
    "coupling_betweenness_max",
    "degree_assortativity",
    "measurement_seconds",
)

_DEGREE_DISTRIBUTION_FIELDS = (
    "group",
    "instance",
    "degree",
    "node_count",
    "probability",
)

_COUPLING_BETWEENNESS_FIELDS = (
    "group",
    "instance",
    "node",
    "betweenness",
)

_SKEWNESS_FIELDS = (
    "group",
    "instance",
    "node_count",
    "edge_count",
    "skewness_lower_bound",
    "skewness_upper_bound",
    "skewness_exact",
    "skewness",
    "absolute_gap",
    "relative_gap",
    "status",
    "runtime_seconds",
    "source_floor_lower_bound",
    "source_floor_upper_bound",
    "coupling_edge_count",
    "source_piece_count",
    "coupling_forest_edge_count",
    "solver_lower_bound",
    "solver_upper_bound",
    "inflation_lower_bound",
    "inflation_upper_bound",
    "inflation_exact",
    "inflation",
    "normalized_inflation_lower_bound",
    "normalized_inflation_upper_bound",
    "normalized_inflation_exact",
    "normalized_inflation",
)
