"""Write one accepted power candidate using the repository result contract."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Mapping

import networkx as nx

from planar_instance_generation.core.planarization import ExactMergeNodeSelection
from planar_instance_generation.evaluation.planar_certificates import generation_certificates
from planar_instance_generation.evaluation.power_generation import (
    PowerGenerationCandidate,
    visual_merge_stages,
)
from planar_instance_generation.io.artifacts import (
    InstanceArtifactPaths,
    allocate_instance_artifacts,
)
from planar_instance_generation.io.pglib import PglibCase, write_pglib
from planar_instance_generation.plt import (
    LatexCompilationError,
    PlotConfig,
    compile_plot_pdf,
    make_source_styles,
    write_graph_plot,
    write_merge_process,
)


@dataclass(frozen=True, slots=True)
class SavedPowerCandidate:
    """Stable paths for one final case and all of its extracted constituents."""

    final: InstanceArtifactPaths
    pieces: tuple[InstanceArtifactPaths, ...]
    stage_pdfs: tuple[Path, ...]
    final_plot_pdf: Path | None
    metadata_json: Path


def save_power_candidate(
    candidate: PowerGenerationCandidate,
    run_directory: str | Path,
    *,
    compile_pdfs: bool = True,
    plot_config: PlotConfig = PlotConfig(),
    gate_log_path: str | Path | None = None,
    stem_suffix: str | None = None,
    shared_piece_paths: tuple[InstanceArtifactPaths, ...] | None = None,
) -> SavedPowerCandidate:
    """Persist instances, colored TikZ figures, PDFs, and solver logs.

    ``stem_suffix`` is appended to every allocated artifact stem so that
    independently generated instances with the same bus count remain
    distinguishable when their result directories are later consolidated.
    Pass ``shared_piece_paths`` to reuse one already-persisted extraction across
    merge strategies for the same requested output.
    """

    if not candidate.feasible:
        raise ValueError("only gate-feasible candidates can be saved")
    run = Path(run_directory)
    visual_stages = visual_merge_stages(candidate)
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
    for index, piece in enumerate(candidate.pieces, start=1):
        if shared_piece_paths is not None:
            continue
        case = piece.subcase.case
        paths = allocate_instance_artifacts(
            run,
            int(case.bus.shape[0]),
            extension=".m",
            suffix=stem_suffix,
            instance_tree="subinstances",
        )
        write_pglib(_renamed_case(case, f"pglib_subinstance_{paths.stem}"), paths.instance_file)
        graph = case.topology(active_only=True)
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
                "kind": "extracted_subinstance",
                "instance_tree": "subinstances",
                "plot_tree": "original",
                "artifact_stem": paths.stem,
                **piece.statistics(),
            },
        )
        piece_paths.append(paths)

    final_case = candidate.case
    final_paths = allocate_instance_artifacts(
        run, int(final_case.bus.shape[0]), extension=".m", suffix=stem_suffix
    )
    write_pglib(
        _renamed_case(final_case, f"pglib_generated_{final_paths.stem}"),
        final_paths.instance_file,
    )
    if gate_log_path is not None:
        source_log = Path(gate_log_path)
        if not source_log.is_file():
            raise ValueError(f"gate log does not exist: {source_log}")
        shutil.copyfile(source_log, final_paths.gurobi_log)

    merge_plot_directory = final_paths.merge_plot_directory
    merge_log_directory = final_paths.merge_log_directory
    if merge_plot_directory is None or merge_log_directory is None:
        raise RuntimeError("final artifact paths need merged plot and log directories")

    selection_logs = _copy_exact_selection_logs(candidate, merge_log_directory)

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
    stage_pdf_failures = []
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
            try:
                stage_pdfs.append(compile_plot_pdf(
                    plots.document,
                    output_pdf=stage_directory / f"merge_stage_{stage.stage_index:03d}.pdf",
                    log_directory=merge_log_directory,
                ).pdf)
            except LatexCompilationError as error:
                failure_path = merge_log_directory / (
                    f"merge_stage_{stage.stage_index:03d}.pdf-error.txt"
                )
                failure_path.write_text(f"{error}\n", encoding="utf-8")
                stage_pdf_failures.append(
                    {
                        "stage_index": stage.stage_index,
                        "error": str(error),
                        "error_file": str(failure_path),
                    }
                )

    metadata_path = final_paths.plot_directory / "metadata.json"
    _write_json(
        metadata_path,
        {
            "kind": "final_composite",
            "instance_tree": "instances",
            "plot_tree": "composite",
            "merge_plot_tree": "merged",
            "artifact_stem": final_paths.stem,
            "source_artifact_stems": [paths.stem for paths in piece_paths],
            "source_names": [piece.source_name for piece in candidate.pieces],
            "statistics": candidate.statistics(),
            "stages": [stage.statistics() for stage in candidate.stages],
            "planarization_certificates": generation_certificates(candidate),
            "selection_log_files": selection_logs,
            "stage_pdf_failures": stage_pdf_failures,
            "coupling_branch_rows_zero_based": list(
                candidate.final_merge.coupling_branch_rows
            ),
            "source_bus_mappings": [
                {
                    "source_artifact_stem": paths.stem,
                    "buses": [
                        {
                            "source_bus": int(source_bus),
                            "composite_bus": int(composite_bus),
                        }
                        for source_bus, composite_bus in mapping.bus_ids.items()
                    ],
                }
                for paths, mapping in zip(
                    piece_paths,
                    candidate.final_merge.source_mappings,
                    strict=True,
                )
            ],
        },
    )
    return SavedPowerCandidate(
        final=final_paths,
        pieces=tuple(piece_paths),
        stage_pdfs=tuple(stage_pdfs),
        final_plot_pdf=final_pdf,
        metadata_json=metadata_path,
    )


def ensure_original_power_plot(
    case: PglibCase,
    originals_directory: str | Path,
    *,
    compile_pdf: bool = True,
    plot_config: PlotConfig = PlotConfig(),
) -> Path | None:
    """Create one cached full-source plot, reusing a complete prior result."""

    root = Path(originals_directory) / case.name
    style = make_source_styles({"source": case.name})
    graph = case.topology(active_only=True)
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


def _copy_exact_selection_logs(
    candidate: PowerGenerationCandidate, log_directory: Path
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
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            copied.append(str(destination))
    return copied


def _renamed_case(case: PglibCase, name: str) -> PglibCase:
    return PglibCase(
        name=name,
        version=case.version,
        base_mva=case.base_mva,
        bus=case.bus.copy(),
        gen=case.gen.copy(),
        branch=case.branch.copy(),
        gencost=case.gencost.copy(),
        areas=None if case.areas is None else case.areas.copy(),
    )


def _write_json(path: Path, data: Mapping[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
