"""Deterministic Kamada--Kawai layouts rendered as publication-ready TikZ."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Callable, Hashable, Mapping

import networkx as nx
import numpy as np

from planar_instance_generation.core import MergeResult, NamespacedNode, SourceId


Position = tuple[float, float]
TIKZ_FORMAT_VERSION = 3
_SOURCE_PALETTE = (
    "0072B2",  # blue
    "E69F00",  # orange
    "009E73",  # bluish green
    "CC79A7",  # reddish purple
    "56B4E9",  # sky blue
    "F0E442",  # yellow
    "000000",  # black
)
_HEX_COLOR = re.compile(r"^#?[0-9A-Fa-f]{6}$")


@dataclass(frozen=True, slots=True)
class SourceStyle:
    """One run-wide label and color for an original source graph."""

    label: str
    color: str

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("source label must not be empty")
        if not _HEX_COLOR.fullmatch(self.color):
            raise ValueError(f"invalid six-digit RGB color: {self.color!r}")

    @property
    def hex(self) -> str:
        return self.color.lstrip("#").upper()


@dataclass(frozen=True, slots=True)
class PlotConfig:
    """Dimensions and typography for one graph window."""

    width_cm: float = 8.4
    height_cm: float = 6.2
    source_gap_cm: float = 0.8
    panel_gap_cm: float = 0.18
    legend_width_cm: float = 4.2
    node_size_pt: float = 2.6
    show_labels: bool = False

    def __post_init__(self) -> None:
        if self.width_cm <= 0 or self.height_cm <= 0:
            raise ValueError("plot dimensions must be positive")
        if not 0 <= self.source_gap_cm < self.width_cm:
            raise ValueError("source gap must be nonnegative and smaller than plot width")
        if self.panel_gap_cm < 0 or self.legend_width_cm <= 0:
            raise ValueError("panel gap and legend width are invalid")
        if self.node_size_pt <= 0:
            raise ValueError("node size must be positive")


@dataclass(frozen=True, slots=True)
class PlotArtifacts:
    """Paper fragment, standalone preview, and machine-readable sidecar."""

    tikz: Path
    standalone_tex: Path
    coordinates_json: Path


@dataclass(frozen=True, slots=True)
class MergePlotArtifacts:
    """One pairwise merge figure and its separately reusable windows."""

    document: PlotArtifacts
    original: PlotArtifacts
    steps: tuple[PlotArtifacts, ...]
    final: PlotArtifacts

    @property
    def sources(self) -> PlotArtifacts:
        """Backward-compatible alias for the original two-graph window."""

        return self.original


def make_source_styles(labels: Mapping[SourceId, str]) -> dict[SourceId, SourceStyle]:
    """Assign the colorblind-safe run-wide palette in mapping order."""

    if len(labels) > len(_SOURCE_PALETTE):
        raise ValueError(f"at most {len(_SOURCE_PALETTE)} source colors are available")
    return {
        source_id: SourceStyle(label=label, color=_SOURCE_PALETTE[index])
        for index, (source_id, label) in enumerate(labels.items())
    }


def kamada_kawai_positions(
    graph: nx.Graph,
    *,
    width_cm: float,
    height_cm: float,
    center: Position = (0.0, 0.0),
) -> dict[Hashable, Position]:
    """Compute a stable-order Kamada--Kawai layout fitted to a rectangle."""

    _validate_graph(graph)
    if graph.number_of_nodes() == 0:
        raise ValueError("cannot embed an empty graph")
    if width_cm <= 0 or height_cm <= 0:
        raise ValueError("layout dimensions must be positive")

    ordered = nx.Graph()
    ordered.graph.update(graph.graph)
    ordered.add_nodes_from(
        (node, dict(graph.nodes[node])) for node in sorted(graph.nodes, key=_stable_key)
    )
    ordered.add_edges_from(
        (source, target, dict(attributes))
        for source, target, attributes in sorted(
            graph.edges(data=True),
            key=lambda edge: (_stable_key(edge[0]), _stable_key(edge[1])),
        )
    )
    raw = nx.kamada_kawai_layout(ordered, dim=2, scale=1.0)
    return _fit_positions(raw, width_cm=width_cm, height_cm=height_cm, center=center)


def write_graph_plot(
    graph: nx.Graph,
    output_base: str | Path,
    *,
    config: PlotConfig = PlotConfig(),
    positions: Mapping[Hashable, Position] | None = None,
    highlighted_edge: tuple[Hashable, Hashable] | None = None,
    source_styles: Mapping[SourceId, SourceStyle] | None = None,
    include_legend: bool = False,
    label: Callable[[Hashable], str] | None = None,
    overwrite: bool = False,
) -> PlotArtifacts:
    """Write a TikZ graph window, standalone wrapper, and coordinates."""

    _validate_graph(graph)
    if graph.number_of_nodes() == 0:
        raise ValueError("cannot plot an empty graph")
    base = Path(output_base)
    if base.suffix:
        raise ValueError("output_base must not have a file extension")
    artifacts = _artifacts(base)
    if _reuse_or_raise(artifacts, overwrite):
        return artifacts

    if positions is None:
        positions = kamada_kawai_positions(
            graph,
            width_cm=config.width_cm,
            height_cm=config.height_cm,
        )
    if set(positions) != set(graph):
        raise ValueError("positions must contain exactly the graph's nodes")
    if highlighted_edge is not None and not graph.has_edge(*highlighted_edge):
        raise ValueError("highlighted_edge is not present in the graph")

    styles = _resolve_source_styles(graph, source_styles)
    base.parent.mkdir(parents=True, exist_ok=True)
    node_order = sorted(graph.nodes, key=_stable_key)
    node_names = {node: f"n{index}" for index, node in enumerate(node_order)}
    graph_fragment = _tikz_fragment(
        graph,
        node_order,
        node_names,
        positions,
        config=config,
        highlighted_edge=highlighted_edge,
        source_styles=styles,
        label=label,
    )
    fragment = (
        _graph_with_legend(graph_fragment, styles, graph, config)
        if include_legend
        else graph_fragment
    )
    coordinate_data = {
        "format_version": TIKZ_FORMAT_VERSION,
        "layout": "kamada_kawai",
        "legend": include_legend,
        "config": asdict(config),
        "source_styles": _styles_json(styles),
        "nodes": [
            {
                "id": repr(node),
                "label": str(_default_label(node) if label is None else label(node)),
                "source": _json_source(_node_source(graph, node, styles)),
                "x_cm": float(positions[node][0]),
                "y_cm": float(positions[node][1]),
            }
            for node in node_order
        ],
        "edges": [
            {
                "source": repr(source),
                "target": repr(target),
                "source_graph": _json_source(_edge_source(graph, source, target, styles)),
                "merge_edge": bool(graph[source][target].get("merge_edge")),
            }
            for source, target in _ordered_edges(graph)
        ],
    }
    artifacts.tikz.write_text(fragment, encoding="utf-8")
    artifacts.standalone_tex.write_text(_standalone_wrapper(), encoding="utf-8")
    artifacts.coordinates_json.write_text(
        json.dumps(coordinate_data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifacts


def write_merge_process(
    result: MergeResult,
    output_directory: str | Path,
    *,
    stem: str = "merge",
    config: PlotConfig = PlotConfig(),
    source_styles: Mapping[SourceId, SourceStyle] | None = None,
    overwrite: bool = False,
) -> MergePlotArtifacts:
    """Write one side-by-side figure for one pairwise merge stage.

    The combined figure is ordered ``Original``, ``Step 1``, ..., ``Final``.
    Every window is embedded independently with Kamada--Kawai.
    """

    if not stem or Path(stem).name != stem:
        raise ValueError("stem must be a nonempty filename component")
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    styles = _resolve_source_styles(result.final_graph, source_styles)

    component_width = (config.width_cm - config.source_gap_cm) / 2.0
    horizontal_offset = (component_width + config.source_gap_cm) / 2.0
    left_positions = kamada_kawai_positions(
        result.source_graphs[0],
        width_cm=component_width,
        height_cm=config.height_cm,
        center=(-horizontal_offset, 0.0),
    )
    right_positions = kamada_kawai_positions(
        result.source_graphs[1],
        width_cm=component_width,
        height_cm=config.height_cm,
        center=(horizontal_offset, 0.0),
    )
    left_namespace, right_namespace = result.namespaces
    original_positions: dict[NamespacedNode, Position] = {
        **{(left_namespace, node): position for node, position in left_positions.items()},
        **{(right_namespace, node): position for node, position in right_positions.items()},
    }
    original = write_graph_plot(
        result.initial_graph,
        directory / f"{stem}_original",
        config=config,
        positions=original_positions,
        source_styles=styles,
        overwrite=overwrite,
    )

    step_artifacts = []
    for step, graph, newest_edge in result.iter_steps():
        positions = kamada_kawai_positions(
            graph,
            width_cm=config.width_cm,
            height_cm=config.height_cm,
        )
        step_artifacts.append(
            write_graph_plot(
                graph,
                directory / f"{stem}_step_{step:03d}",
                config=config,
                positions=positions,
                highlighted_edge=newest_edge,
                source_styles=styles,
                overwrite=overwrite,
            )
        )

    final_positions = kamada_kawai_positions(
        result.final_graph,
        width_cm=config.width_cm,
        height_cm=config.height_cm,
    )
    final = write_graph_plot(
        result.final_graph,
        directory / f"{stem}_final",
        config=config,
        positions=final_positions,
        source_styles=styles,
        overwrite=overwrite,
    )
    titles = ("Original", *(f"Step {step}" for step in range(1, len(step_artifacts) + 1)), "Final")
    windows = (original, *step_artifacts, final)
    document = _write_stage_document(
        result,
        directory / stem,
        windows=windows,
        titles=titles,
        source_styles=styles,
        config=config,
        overwrite=overwrite,
    )
    return MergePlotArtifacts(
        document=document,
        original=original,
        steps=tuple(step_artifacts),
        final=final,
    )


def _write_stage_document(
    result: MergeResult,
    base: Path,
    *,
    windows: tuple[PlotArtifacts, ...],
    titles: tuple[str, ...],
    source_styles: Mapping[SourceId, SourceStyle],
    config: PlotConfig,
    overwrite: bool,
) -> PlotArtifacts:
    artifacts = _artifacts(base)
    if _reuse_or_raise(artifacts, overwrite):
        return artifacts

    fragments = [window.tikz.read_text(encoding="utf-8") for window in windows]
    columns = [
        _legend_fragment(
            source_styles, result.final_graph, config, include_new_edge=True
        ),
        *(
            "\\begin{tabular}[t]{@{}c@{}}\n"
            f"{{\\small\\bfseries {_escape_latex(title)}}}\\\\[0.8mm]\n"
            f"{fragment}"
            "\\end{tabular}\n"
            for title, fragment in zip(titles, fragments, strict=True)
        ),
    ]
    separator = f"\n& \\hspace{{{config.panel_gap_cm:.3f}cm}}\n"
    stage_fragment = (
        "% One pairwise merge: Original, edge-addition steps, and Final.\n"
        "\\begingroup\n"
        "\\setlength{\\tabcolsep}{0pt}\n"
        "\\begin{tabular}{@{}" + "c" * len(columns) + "@{}}\n"
        + separator.join(columns)
        + "\\end{tabular}\n"
        "\\endgroup\n"
    )
    stage_data = {
        "format_version": TIKZ_FORMAT_VERSION,
        "kind": "pairwise_merge_figure",
        "stage_index": result.stage_index,
        "source_styles": _styles_json(source_styles),
        "panels": [
            {
                "title": title,
                "tikz": window.tikz.name,
                "coordinates_json": window.coordinates_json.name,
            }
            for title, window in zip(titles, windows, strict=True)
        ],
    }
    artifacts.tikz.write_text(stage_fragment, encoding="utf-8")
    artifacts.standalone_tex.write_text(_standalone_wrapper(), encoding="utf-8")
    artifacts.coordinates_json.write_text(
        json.dumps(stage_data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifacts


def _legend_fragment(
    styles: Mapping[SourceId, SourceStyle],
    graph: nx.Graph,
    config: PlotConfig,
    *,
    include_new_edge: bool = False,
) -> str:
    present = {
        graph.nodes[node].get("source_graph")
        for node in graph
        if graph.nodes[node].get("source_graph") is not None
    }
    entries = [(source_id, style) for source_id, style in styles.items() if source_id in present]
    height = max(config.height_cm, 1.15 + 0.58 * (len(entries) + 2))
    lines = [
        "\\begin{tikzpicture}[x=1cm,y=1cm,baseline=(current bounding box.north)]\n",
        *_color_definitions(styles),
        f"\\path[use as bounding box] (0,{-height:.3f}) rectangle ({config.legend_width_cm:.3f},0.35);\n",
        "\\node[anchor=north west,font=\\small\\bfseries] at (0,0.25) {Source graphs};\n",
    ]
    y = -0.55
    color_names = _source_color_names(styles)
    for source_id, style in entries:
        color_name = color_names[source_id]
        lines.extend(
            [
                f"\\draw[{color_name},line width=0.65pt] (0.05,{y:.3f}) -- (0.55,{y:.3f});\n",
                f"\\fill[{color_name}] (0.30,{y:.3f}) circle[radius=2.2pt];\n",
                f"\\node[anchor=west,font=\\footnotesize] at (0.68,{y:.3f}) "
                f"{{{_escape_latex(style.label)}}};\n",
            ]
        )
        y -= 0.58
    if any(attributes.get("merge_edge") for *_, attributes in graph.edges(data=True)):
        lines.extend(
            [
                f"\\draw[pigMerge,line width=0.9pt,densely dashed] (0.05,{y:.3f}) -- (0.55,{y:.3f});\n",
                f"\\node[anchor=west,font=\\footnotesize] at (0.68,{y:.3f}) {{Merge edge}};\n",
            ]
        )
        if include_new_edge:
            lines.extend(
                [
                    f"\\draw[pigNew,line width=1.25pt] (0.05,{y - 0.58:.3f}) -- (0.55,{y - 0.58:.3f});\n",
                    f"\\node[anchor=west,font=\\footnotesize] at (0.68,{y - 0.58:.3f}) {{Edge added in this step}};\n",
                ]
            )
    lines.append("\\end{tikzpicture}\n")
    return "".join(lines)


def _graph_with_legend(
    graph_fragment: str,
    styles: Mapping[SourceId, SourceStyle],
    graph: nx.Graph,
    config: PlotConfig,
) -> str:
    return (
        "% Graph and persistent source-color legend.\n"
        "\\begingroup\n"
        "\\setlength{\\tabcolsep}{0pt}\n"
        "\\begin{tabular}{@{}cc@{}}\n"
        f"{graph_fragment}\n"
        f"& \\hspace{{{config.panel_gap_cm:.3f}cm}}\n"
        f"{_legend_fragment(styles, graph, config)}"
        "\\end{tabular}\n"
        "\\endgroup\n"
    )


def _fit_positions(
    raw: Mapping[Hashable, np.ndarray],
    *,
    width_cm: float,
    height_cm: float,
    center: Position,
) -> dict[Hashable, Position]:
    nodes = list(raw)
    coordinates = np.asarray([raw[node] for node in nodes], dtype=float)
    minima = coordinates.min(axis=0)
    maxima = coordinates.max(axis=0)
    spans = maxima - minima
    nonzero = spans > 1e-12
    if not nonzero.any():
        return {node: center for node in nodes}
    scale_candidates = []
    if nonzero[0]:
        scale_candidates.append(width_cm / spans[0])
    if nonzero[1]:
        scale_candidates.append(height_cm / spans[1])
    scale = min(scale_candidates)
    midpoint = (minima + maxima) / 2.0
    fitted = (coordinates - midpoint) * scale + np.asarray(center)
    return {
        node: (float(position[0]), float(position[1]))
        for node, position in zip(nodes, fitted, strict=True)
    }


def _tikz_fragment(
    graph: nx.Graph,
    node_order: list[Hashable],
    node_names: Mapping[Hashable, str],
    positions: Mapping[Hashable, Position],
    *,
    config: PlotConfig,
    highlighted_edge: tuple[Hashable, Hashable] | None,
    source_styles: Mapping[SourceId, SourceStyle],
    label: Callable[[Hashable], str] | None,
) -> str:
    color_names = _source_color_names(source_styles)
    lines = [
        "% Generated by planar-instance-generation.\n",
        *_color_definitions(source_styles),
        "\\begin{tikzpicture}[x=1cm,y=1cm]\n",
        "\\tikzset{graph edge/.style={draw=pigNeutral,line width=0.28pt,opacity=0.65},"
        "merge edge/.style={draw=pigMerge,line width=0.70pt,densely dashed},"
        "new merge edge/.style={draw=pigNew,line width=1.15pt},"
        f"graph node/.style={{circle,draw=white,line width=0.2pt,minimum size={config.node_size_pt:.3f}pt,inner sep=0pt,fill=pigDefault}}}}\n",
    ]
    for node in node_order:
        x, y = positions[node]
        lines.append(f"\\coordinate ({node_names[node]}) at ({x:.8f},{y:.8f});\n")
    for source, target in _ordered_edges(graph):
        attributes = graph[source][target]
        edge_source = _edge_source(graph, source, target, source_styles)
        if attributes.get("merge_edge"):
            style = "merge edge"
        elif edge_source in color_names:
            style = f"draw={color_names[edge_source]},line width=0.38pt,opacity=0.72"
        else:
            style = "graph edge"
        if highlighted_edge is not None and _same_undirected_edge(
            (source, target), highlighted_edge
        ):
            style = "new merge edge"
        lines.append(f"\\draw[{style}] ({node_names[source]}) -- ({node_names[target]});\n")
    for node in node_order:
        source_id = _node_source(graph, node, source_styles)
        fill = color_names.get(source_id, "pigDefault")
        displayed = _default_label(node) if label is None else label(node)
        node_text = _escape_latex(str(displayed)) if config.show_labels else ""
        lines.append(
            f"\\node[graph node,fill={fill}] at ({node_names[node]}) {{{node_text}}};\n"
        )
    lines.append("\\end{tikzpicture}\n")
    return "".join(lines)


def _color_definitions(styles: Mapping[SourceId, SourceStyle]) -> list[str]:
    definitions = [
        "\\definecolor{pigDefault}{HTML}{0072B2}\n",
        "\\definecolor{pigNeutral}{HTML}{666666}\n",
        "\\definecolor{pigMerge}{HTML}{7F7F7F}\n",
        "\\definecolor{pigNew}{HTML}{D55E00}\n",
    ]
    definitions.extend(
        f"\\definecolor{{{name}}}{{HTML}}{{{styles[source_id].hex}}}\n"
        for source_id, name in _source_color_names(styles).items()
    )
    return definitions


def _resolve_source_styles(
    graph: nx.Graph,
    supplied: Mapping[SourceId, SourceStyle] | None,
) -> dict[SourceId, SourceStyle]:
    present = sorted(
        {
            graph.nodes[node].get("source_graph")
            for node in graph
            if graph.nodes[node].get("source_graph") is not None
        },
        key=_stable_key,
    )
    if supplied is None:
        return make_source_styles({source_id: str(source_id) for source_id in present})
    styles = dict(supplied)
    missing = set(present) - set(styles)
    if missing:
        raise ValueError(f"source styles missing for: {sorted(map(repr, missing))}")
    return styles


def _source_color_names(styles: Mapping[SourceId, SourceStyle]) -> dict[SourceId, str]:
    return {source_id: f"pigSrc{index + 1}" for index, source_id in enumerate(styles)}


def _node_source(
    graph: nx.Graph,
    node: Hashable,
    styles: Mapping[SourceId, SourceStyle],
) -> SourceId | None:
    source_id = graph.nodes[node].get("source_graph")
    if source_id is None and len(styles) == 1:
        return next(iter(styles))
    return source_id


def _edge_source(
    graph: nx.Graph,
    source: Hashable,
    target: Hashable,
    styles: Mapping[SourceId, SourceStyle],
) -> SourceId | None:
    source_id = graph[source][target].get("source_graph")
    if source_id is not None:
        return source_id
    first = _node_source(graph, source, styles)
    second = _node_source(graph, target, styles)
    return first if first == second else None


def _ordered_edges(graph: nx.Graph) -> list[tuple[Hashable, Hashable]]:
    normalized = []
    for source, target in graph.edges:
        if _stable_key(target) < _stable_key(source):
            source, target = target, source
        normalized.append((source, target))
    return sorted(normalized, key=lambda edge: (_stable_key(edge[0]), _stable_key(edge[1])))


def _same_undirected_edge(
    first: tuple[Hashable, Hashable], second: tuple[Hashable, Hashable]
) -> bool:
    return (first[0] == second[0] and first[1] == second[1]) or (
        first[0] == second[1] and first[1] == second[0]
    )


def _default_label(node: Hashable) -> Hashable:
    while isinstance(node, tuple) and len(node) == 2:
        node = node[1]
    return node


def _stable_key(value: Hashable) -> tuple[str, str]:
    kind = f"{type(value).__module__}.{type(value).__qualname__}"
    return kind, repr(value)


def _escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _artifacts(base: Path) -> PlotArtifacts:
    return PlotArtifacts(
        tikz=base.with_suffix(".tikz"),
        standalone_tex=base.with_suffix(".tex"),
        coordinates_json=base.with_suffix(".json"),
    )


def _reuse_or_raise(artifacts: PlotArtifacts, overwrite: bool) -> bool:
    existing = tuple(path.exists() for path in asdict(artifacts).values())
    if all(existing) and not overwrite:
        return True
    if any(existing) and not overwrite:
        raise FileExistsError(
            f"incomplete plot artifact set at {artifacts.tikz.with_suffix('')}; use overwrite=True"
        )
    return False


def _standalone_wrapper() -> str:
    return (
        "\\documentclass[border=1pt]{standalone}\n"
        "\\usepackage{tikz}\n"
        "\\begin{document}\n"
        "\\input{\\jobname.tikz}\n"
        "\\end{document}\n"
    )


def _styles_json(styles: Mapping[SourceId, SourceStyle]) -> list[dict[str, str]]:
    return [
        {"id": repr(source_id), "label": style.label, "color": f"#{style.hex}"}
        for source_id, style in styles.items()
    ]


def _json_source(source_id: SourceId | None) -> str | None:
    return None if source_id is None else str(source_id)


def _validate_graph(graph: nx.Graph) -> None:
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a NetworkX graph")
    if graph.is_directed() or graph.is_multigraph():
        raise TypeError("TikZ output requires a simple undirected graph")
