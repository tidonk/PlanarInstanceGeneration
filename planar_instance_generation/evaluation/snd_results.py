"""Persist one accepted SND candidate using the repository result contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from planar_instance_generation.evaluation.planar_certificates import generation_certificates
from pathlib import Path
import shutil
from typing import Mapping

import networkx as nx

from planar_instance_generation.core.planarization import ExactMergeNodeSelection
from planar_instance_generation.core.snd_merge import SndSourceMapping
from planar_instance_generation.core.snd_serialization import (
    SndlibSerialization,
    serialize_snd_instance,
)
from planar_instance_generation.evaluation.snd_generation import (
    SndGenerationCandidate,
    visual_snd_merge_stages,
)
from planar_instance_generation.io.artifacts import (
    InstanceArtifactPaths,
    allocate_instance_artifacts,
)
from planar_instance_generation.io.sndlib import SndlibNetwork, write_sndlib
from planar_instance_generation.plt import (
    PlotConfig,
    compile_plot_pdf,
    make_source_styles,
    write_graph_plot,
    write_merge_process,
)


@dataclass(frozen=True, slots=True)
class SavedSndCandidate:
    """Stable paths for one final network and all extracted constituents."""

    final: InstanceArtifactPaths
    pieces: tuple[InstanceArtifactPaths, ...]
    stage_pdfs: tuple[Path, ...]
    final_plot_pdf: Path | None
    metadata_json: Path


def save_snd_candidate(
    candidate: SndGenerationCandidate,
    run_directory: str | Path,
    *,
    compile_pdfs: bool = True,
    plot_config: PlotConfig = PlotConfig(),
    gate_log_path: str | Path | None = None,
    stem_suffix: str | None = None,
    shared_piece_paths: tuple[InstanceArtifactPaths, ...] | None = None,
) -> SavedSndCandidate:
    """Persist XML instances, colored TikZ figures, PDFs, and solver logs.

    ``stem_suffix`` is appended to every allocated artifact stem so that
    independently generated instances with the same node count remain
    distinguishable when their result directories are later consolidated.
    Pass ``shared_piece_paths`` to reuse one already-persisted extraction across
    merge strategies for the same requested output.
    """

    if candidate.feasible is not True:
        raise ValueError("only gate-feasible candidates can be saved")
    run = Path(run_directory)
    visual_stages = visual_snd_merge_stages(candidate)
    labels = {
        f"G_{index}": f"G_{index}: {piece.source_name}"
        for index, piece in enumerate(candidate.pieces, start=1)
    }
    styles = make_source_styles(labels)

    piece_paths = list(shared_piece_paths or ())
    if shared_piece_paths is not None:
        if len(piece_paths) != len(candidate.pieces):
            raise ValueError("shared_piece_paths must match candidate pieces")
        if any(not paths.instance_file.is_file() for paths in piece_paths):
            raise ValueError("shared_piece_paths contains a missing instance")
    piece_serializations: list[SndlibSerialization] = []
    for index, piece in enumerate(candidate.pieces, start=1):
        instance = piece.subinstance.instance
        serialization = serialize_snd_instance(instance)
        piece_serializations.append(serialization)
        if shared_piece_paths is not None:
            continue
        paths = allocate_instance_artifacts(
            run,
            len(instance.nodes),
            extension=".xml",
            suffix=stem_suffix,
            instance_tree="subinstances",
        )
        write_sndlib(serialization.network, paths.instance_file)
        graph = instance.topology()
        source_id = f"G_{index}"
        nx.set_node_attributes(graph, source_id, "source_graph")
        nx.set_edge_attributes(graph, source_id, "source_graph")
        plot = write_graph_plot(
            graph,
            paths.plot_directory / "full",
            config=plot_config,
            source_styles=styles,
            include_legend=True,
        )
        if compile_pdfs:
            compile_plot_pdf(
                plot,
                output_pdf=paths.plot_directory / "full.pdf",
                log_directory=paths.log_directory,
            )
        _write_json(
            paths.plot_directory / "metadata.json",
            {
                "kind": "extracted_snd_subinstance",
                "instance_tree": "subinstances",
                "plot_tree": "original",
                "artifact_stem": paths.stem,
                **piece.statistics(),
                "selected_nodes": list(piece.subinstance.selected_nodes),
                "retained_node_indices_zero_based": list(
                    piece.subinstance.retained_node_indices
                ),
                "internal_link_indices_zero_based": list(
                    piece.subinstance.internal_link_indices
                ),
                "cut_link_indices_zero_based": list(
                    piece.subinstance.cut_link_indices
                ),
                "compensation_by_demand": piece.subinstance.compensation_by_demand,
                "serialized_demand_provenance": [
                    asdict(value) for value in serialization.demand_provenance
                ],
            },
        )
        piece_paths.append(paths)

    final_paths = allocate_instance_artifacts(
        run,
        len(candidate.instance.nodes),
        extension=".xml",
        suffix=stem_suffix,
    )
    write_sndlib(candidate.serialization.network, final_paths.instance_file)
    resolved_gate_log = (
        gate_log_path
        if gate_log_path is not None
        else candidate.gate_result.log_path
    )
    copied_gate_log = _copy_gate_log(resolved_gate_log, final_paths)
    merge_plot_directory = final_paths.merge_plot_directory
    merge_log_directory = final_paths.merge_log_directory
    if merge_plot_directory is None or merge_log_directory is None:
        raise RuntimeError("final artifact paths need merged plot and log directories")
    selection_logs = _copy_exact_selection_logs(
        candidate,
        merge_log_directory,
    )

    final_visual = visual_stages[-1].final_graph
    final_plot = write_graph_plot(
        final_visual,
        final_paths.plot_directory / "full",
        config=plot_config,
        source_styles=styles,
        include_legend=True,
    )
    final_pdf = None
    if compile_pdfs:
        final_pdf = compile_plot_pdf(
            final_plot,
            output_pdf=final_paths.plot_directory / "full.pdf",
            log_directory=final_paths.log_directory,
        ).pdf

    stage_pdfs = []
    for stage in visual_stages:
        stage_directory = merge_plot_directory / f"stage_{stage.stage_index:03d}"
        plots = write_merge_process(
            stage,
            stage_directory,
            stem=f"merge_stage_{stage.stage_index:03d}",
            config=plot_config,
            source_styles=styles,
        )
        if compile_pdfs:
            stage_pdfs.append(
                compile_plot_pdf(
                    plots.document,
                    output_pdf=(
                        stage_directory / f"merge_stage_{stage.stage_index:03d}.pdf"
                    ),
                    log_directory=merge_log_directory,
                ).pdf
            )

    metadata_path = final_paths.plot_directory / "metadata.json"
    _write_json(
        metadata_path,
        {
            "kind": "final_snd_composite",
            "instance_tree": "instances",
            "plot_tree": "composite",
            "merge_plot_tree": "merged",
            "artifact_stem": final_paths.stem,
            "source_artifact_stems": [paths.stem for paths in piece_paths],
            "source_names": [piece.source_name for piece in candidate.pieces],
            "routing_unit": candidate.routing_unit,
            "source_stratum_size": candidate.source_stratum_size,
            "source_sampling": "unit-stratified uniform with replacement",
            "source_coordinate_types": [
                piece.subinstance.instance.coordinates_type
                for piece in candidate.pieces
            ],
            "output_coordinates_type": candidate.instance.coordinates_type,
            "coordinate_type_overridden": candidate.coordinate_type_overridden,
            "merge_config": asdict(candidate.merge_config),
            "statistics": candidate.statistics(),
            "stages": [stage.statistics() for stage in candidate.stages],
            "planarization_certificates": generation_certificates(candidate),
            "selection_log_files": selection_logs,
            "gate_log_file": copied_gate_log,
            "coupling_links": _coupling_link_records(candidate),
            "serialized_demand_provenance": [
                asdict(value)
                for value in candidate.serialization.demand_provenance
            ],
            "source_mappings": [
                _source_mapping_record(paths.stem, mapping)
                for paths, mapping in zip(
                    piece_paths,
                    candidate.final_merge.source_mappings,
                    strict=True,
                )
            ],
            "piece_serialized_demand_counts": [
                len(value.network.demands) for value in piece_serializations
            ],
        },
    )
    return SavedSndCandidate(
        final=final_paths,
        pieces=tuple(piece_paths),
        stage_pdfs=tuple(stage_pdfs),
        final_plot_pdf=final_pdf,
        metadata_json=metadata_path,
    )


def ensure_original_snd_plot(
    network: SndlibNetwork,
    source_name: str,
    originals_directory: str | Path,
    *,
    compile_pdf: bool = True,
    plot_config: PlotConfig = PlotConfig(),
) -> Path | None:
    """Create one cached full-source plot, reusing a complete prior result."""

    if not source_name or Path(source_name).name != source_name:
        raise ValueError("source_name must be a single filename component")
    root = Path(originals_directory) / source_name
    style = make_source_styles({"source": source_name})
    graph = network.topology()
    nx.set_node_attributes(graph, "source", "source_graph")
    nx.set_edge_attributes(graph, "source", "source_graph")
    plot = write_graph_plot(
        graph,
        root / "full",
        config=plot_config,
        source_styles=style,
        include_legend=True,
    )
    if not compile_pdf:
        return None
    return compile_plot_pdf(
        plot,
        output_pdf=root / "full.pdf",
        log_directory=root / "logs",
    ).pdf


def _copy_gate_log(
    gate_log_path: str | Path | None,
    final_paths: InstanceArtifactPaths,
) -> str | None:
    if gate_log_path is None:
        return None
    source = Path(gate_log_path)
    if not source.is_file():
        raise ValueError(f"gate log does not exist: {source}")
    destination = final_paths.gurobi_log
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite gate log: {destination}")
    shutil.copyfile(source, destination)
    return str(destination)


def _copy_exact_selection_logs(
    candidate: SndGenerationCandidate,
    log_directory: Path,
) -> list[str]:
    copied = []
    for stage in candidate.stages:
        if not isinstance(stage.selection, ExactMergeNodeSelection):
            continue
        for side, result in (
            ("left", stage.selection.left_planarization),
            ("right", stage.selection.right_planarization),
        ):
            if result.log_path is None:
                continue
            source = Path(result.log_path)
            if not source.is_file():
                raise ValueError(f"selection log does not exist: {source}")
            destination = (
                log_directory
                / "selection"
                / f"stage_{stage.stage_index:03d}"
                / f"{side}_planarization.log"
            )
            if destination.exists():
                raise FileExistsError(
                    f"refusing to overwrite selection log: {destination}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            copied.append(str(destination))
    return copied


def _coupling_link_records(
    candidate: SndGenerationCandidate,
) -> list[dict[str, object]]:
    capacities = candidate.serialization.reference_solution.installed_capacities
    records = []
    for index in candidate.final_merge.coupling_link_indices:
        link = candidate.instance.links[index]
        records.append(
            {
                "index_zero_based": index,
                "id": link.id,
                "source": link.source,
                "target": link.target,
                "installed_reference_capacity": capacities[link.id],
                "preinstalled_module": (
                    None
                    if link.preinstalled_module is None
                    else asdict(link.preinstalled_module)
                ),
                "additional_modules": [
                    asdict(module) for module in link.additional_modules
                ],
                "routing_cost": link.routing_cost,
                "setup_cost": link.setup_cost,
            }
        )
    return records


def _source_mapping_record(
    artifact_stem: str, mapping: SndSourceMapping
) -> dict[str, object]:
    return {
        "source_artifact_stem": artifact_stem,
        "source_name": mapping.source_name,
        "node_ids": dict(mapping.node_ids),
        "link_ids": dict(mapping.link_ids),
        "demand_ids": dict(mapping.demand_ids),
        "node_indices_zero_based": list(mapping.node_indices),
        "link_indices_zero_based": list(mapping.link_indices),
        "demand_indices_zero_based": list(mapping.demand_indices),
        "coordinates_type": mapping.coordinates_type,
        "metadata": (
            None
            if mapping.metadata is None
            else asdict(mapping.metadata)
        ),
    }


def _write_json(path: Path, data: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite metadata: {path}")
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
