#!/usr/bin/env python3
"""Compute paper statistics and build the A4 review report for a PGLib run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Mapping

import networkx as nx

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from planar_instance_generation.core import (
    ogdf_planar_subgraph,
)
from planar_instance_generation.evaluation import (
    EmbeddedSourceGraph,
    EvaluationGraph,
    PreservationAudit,
    SkewnessBounds,
    SkewnessObservation,
    audit_preserved_properties,
    load_pglib_source_pool,
    write_evaluation_report,
    write_preservation_report,
)
from planar_instance_generation.evaluation.paired import (
    write_paired_strategy_report,
)
from planar_instance_generation.evaluation.planarization_batch import (
    PlanarizationTask,
    solve_partitioned_batch,
    source_mappings,
)
from planar_instance_generation.io import read_pglib
from planar_instance_generation.plt import (
    LatexCompilationError,
    PlotArtifacts,
    ReviewBoundRow,
    ReviewFigure,
    ReviewInstance,
    compile_plot_pdf,
    write_a4_review_report,
)


RESULTS_DIRECTORY = REPOSITORY_ROOT / "check/results"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_STRATEGY_LABELS = {
    "random": "Random",
    "high_degree": "High degree",
    "kamada_kawai": "Kamada-Kawai",
    "heuristic_outer_face": "Heuristic face",
    "exact_outer_face": "Exact face",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="generation result name below check/results")
    parser.add_argument("--source-pool", type=Path)
    parser.add_argument(
        "--cohort-checkpoints",
        type=Path,
        help=(
            "directory of distributed per-cohort exact-skewness JSON files"
        ),
    )
    parser.add_argument("--skewness-time-limit", type=float, default=3600.0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--skewness-workers",
        type=int,
        default=1,
        help="independent graph solves to run concurrently",
    )
    parser.add_argument(
        "--betweenness-samples",
        type=int,
        default=0,
        help="maximum sampled pivots per graph; use 0 for exact",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not _SAFE_NAME.fullmatch(args.run):
        raise SystemExit("run must be a single safe filename component")
    if (
        args.threads < 1
        or args.skewness_workers < 1
        or args.betweenness_samples < 0
    ):
        raise SystemExit(
            "--threads and --skewness-workers must be positive; "
            "--betweenness-samples must be nonnegative"
        )
    run = RESULTS_DIRECTORY / args.run
    evaluation_directory = run / "evaluation"
    if evaluation_directory.exists() and any(evaluation_directory.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"evaluation already exists: {evaluation_directory}; pass --overwrite to replace it"
        )
    generation_manifest_path = run / "manifest.json"
    if not generation_manifest_path.is_file():
        raise SystemExit(f"generation manifest does not exist: {generation_manifest_path}")
    generation_manifest = json.loads(
        generation_manifest_path.read_text(encoding="utf-8")
    )
    run_settings = generation_manifest.get("run", {})
    if not isinstance(run_settings, dict):
        raise RuntimeError("generation manifest run settings must be an object")
    min_source_nodes = run_settings.get("min_source_nodes", 1)
    if not isinstance(min_source_nodes, int) or min_source_nodes < 1:
        raise RuntimeError("generation manifest has an invalid min_source_nodes")
    source_manifest = _source_manifest(args.source_pool, generation_manifest)
    loaded_source_pool = load_pglib_source_pool(
        source_manifest, path_root=REPOSITORY_ROOT
    )
    originals = tuple(
        EvaluationGraph(source.case.name, source.case.topology(active_only=True))
        for source in loaded_source_pool
        if int(source.case.bus.shape[0]) >= min_source_nodes
    )

    subinstances: list[EvaluationGraph] = []
    finals: list[EvaluationGraph] = []
    finals_by_strategy: dict[str, list[EvaluationGraph]] = {}
    final_strategy_by_stem: dict[str, str] = {}
    metadata_by_stem: dict[str, dict[str, object]] = {}
    cases_by_stem = {}
    for metadata_path in _generation_metadata_paths(run):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        stem = str(metadata["artifact_stem"])
        kind = str(metadata["kind"])
        case_path = _generated_case_path(run, stem, kind)
        case = read_pglib(case_path)
        graph = case.topology(active_only=True)
        cases_by_stem[stem] = case
        metadata_by_stem[stem] = metadata
        if kind == "extracted_subinstance":
            subinstances.append(EvaluationGraph(stem, graph))
        elif kind == "final_composite":
            rows = tuple(int(row) for row in metadata["coupling_branch_rows_zero_based"])
            coupling_nodes = tuple(
                dict.fromkeys(
                    int(case.branch[row, endpoint])
                    for row in rows
                    for endpoint in (0, 1)
                )
            )
            final = EvaluationGraph(stem, graph, coupling_nodes)
            strategy = _metadata_strategy(metadata, stem)
            finals.append(final)
            finals_by_strategy.setdefault(strategy, []).append(final)
            final_strategy_by_stem[stem] = strategy
        else:
            raise SystemExit(f"unknown metadata kind in {metadata_path}")
    if not subinstances or not finals:
        raise SystemExit("generation run has no extracted or final instances")

    requested_strategies = _requested_strategies(
        generation_manifest, tuple(finals_by_strategy)
    )
    present_strategies = tuple(
        strategy
        for strategy in requested_strategies
        if strategy in finals_by_strategy
    ) + tuple(
        strategy
        for strategy in finals_by_strategy
        if strategy not in requested_strategies
    )
    strategy_groups = {
        strategy: f"Composite: {_strategy_label(strategy)}"
        for strategy in present_strategies
    }
    groups = {
        "Original sources": originals,
        "Extracted subinstances": tuple(subinstances),
    }
    groups.update(
        (strategy_groups[strategy], tuple(finals_by_strategy[strategy]))
        for strategy in present_strategies
    )
    preloaded_results = {}
    if args.cohort_checkpoints is not None:
        generated_bounds = _load_cohort_bounds(args.cohort_checkpoints.resolve())
        heuristic_rng = random.Random(args.seed)
        for group, samples in groups.items():
            if group == "Original sources":
                continue
            if group == "Extracted subinstances":
                for sample in samples:
                    if sample.name in generated_bounds:
                        preloaded_results[(group, sample.name)] = generated_bounds[sample.name]
                        continue
                    heuristic = ogdf_planar_subgraph(
                        sample.graph,
                        threads=args.threads,
                        seed=heuristic_rng.randrange(2**31),
                    )
                    preloaded_results[(group, sample.name)] = SkewnessBounds(
                        heuristic.omitted_lower_bound,
                        heuristic.omitted_upper_bound,
                        status=heuristic.termination_status,
                        runtime_seconds=heuristic.total_runtime_seconds,
                    )
                continue
            for sample in samples:
                try:
                    preloaded_results[(group, sample.name)] = generated_bounds[
                        sample.name
                    ]
                except KeyError as error:
                    raise SystemExit(
                        f"distributed final checkpoints do not cover {sample.name}"
                    ) from error
        for sample in originals:
            heuristic = ogdf_planar_subgraph(
                sample.graph,
                threads=args.threads,
                seed=heuristic_rng.randrange(2**31),
            )
            preloaded_results[("Original sources", sample.name)] = SkewnessBounds(
                heuristic.omitted_lower_bound,
                heuristic.omitted_upper_bound,
                status=heuristic.termination_status,
                runtime_seconds=heuristic.total_runtime_seconds,
            )
    rng = random.Random(args.seed)
    sample_records = []
    unique_fingerprints = []
    task_index_by_fingerprint = {}
    tasks = []
    skewness_logs = run / "logs/skewness"
    for group, samples in groups.items():
        for sample in samples:
            if (group, sample.name) in preloaded_results:
                sample_records.append((group, sample, None, False))
                continue
            fingerprint = _topology_fingerprint(sample.graph)
            if group.startswith("Composite:"):
                # A final solve also depends on its particular piece partition.
                fingerprint = (fingerprint, group, sample.name)
            reused = fingerprint in task_index_by_fingerprint
            if not reused:
                task_index_by_fingerprint[fingerprint] = len(tasks)
                unique_fingerprints.append(fingerprint)
                tasks.append(
                    PlanarizationTask(
                        graph=sample.graph,
                        seed=rng.randrange(2**31),
                        log_path=(
                            skewness_logs / _slug(group) / f"{sample.name}.log"
                        ),
                        cache_key=(
                            f"{len(tasks):06d}_{_slug(group)}_"
                            f"{_slug(sample.name)}"
                        ),
                    )
                )
            sample_records.append((group, sample, fingerprint, reused))

    print(
        f"Solving {len(tasks)} unique graph topologies with "
        f"{args.skewness_workers} worker process(es) and "
        f"{args.threads} OGDF heuristic thread(s) per solve.",
        flush=True,
    )
    task_by_sample = {
        (group, sample.name): tasks[task_index_by_fingerprint[fingerprint]].cache_key
        for group, sample, fingerprint, _reused in sample_records
        if fingerprint is not None
    }
    dependencies = {
        task_by_sample[(strategy_groups[final_strategy_by_stem[final.name]], final.name)]: [
            (task_by_sample[("Extracted subinstances", stem)], mapping)
            for stem, mapping in source_mappings(metadata_by_stem[final.name])
        ]
        for final in finals
        if (strategy_groups[final_strategy_by_stem[final.name]], final.name) in task_by_sample
    }
    results = solve_partitioned_batch(
        tasks,
        dependencies,
        generation_metadata=[
            (metadata_by_stem[final.name], [task_by_sample[("Extracted subinstances", stem)]
             for stem, _mapping in source_mappings(metadata_by_stem[final.name])])
            for final in finals
            if (strategy_groups[final_strategy_by_stem[final.name]], final.name) in task_by_sample
        ],
        time_limit_seconds=args.skewness_time_limit,

        solver_threads=args.threads,
        workers=args.skewness_workers,
        checkpoint_directory=evaluation_directory / "planarization_checkpoints",
    )
    planarization_cache = dict(zip(unique_fingerprints, results, strict=True))
    skewness = {}
    planarization_by_key = {}
    planarization_rows = []
    for group, sample, fingerprint, reused in sample_records:
        result = (
            preloaded_results[(group, sample.name)]
            if fingerprint is None
            else planarization_cache[fingerprint]
        )
        planarization_by_key[(group, sample.name)] = result
        skewness[(group, sample.name)] = result
        planarization_rows.append(
            {
                "group": group,
                "instance": sample.name,
                "topology_result_reused": reused,
                **_result_statistics(result),
            }
        )

    for final in finals:
        metadata = metadata_by_stem[final.name]
        final_group = strategy_groups[final_strategy_by_stem[final.name]]
        source_stems = tuple(str(value) for value in metadata["source_artifact_stems"])
        source_results = [
            planarization_by_key[("Extracted subinstances", stem)]
            for stem in source_stems
        ]
        final_result = planarization_by_key[(final_group, final.name)]
        skewness[(final_group, final.name)] = SkewnessObservation(
            final_result,
            source_piece_count=len(source_results),
            source_floor_lower_bound=sum(
                _result_bounds(result)[0] for result in source_results
            ),
            source_floor_upper_bound=sum(
                _result_bounds(result)[1] for result in source_results
            ),
            coupling_edge_count=len(
                metadata["coupling_branch_rows_zero_based"]
            ),
        )

    report_data = write_evaluation_report(
        groups,
        evaluation_directory,
        skewness=skewness,
        betweenness_samples=(
            None if args.betweenness_samples == 0 else args.betweenness_samples
        ),
        seed=args.seed,
        overwrite=args.overwrite,
    )
    _write_csv(evaluation_directory / "planarization.csv", planarization_rows)
    (evaluation_directory / "planarization.json").write_text(
        json.dumps(planarization_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    accepted_rows = _read_generation_table(
        generation_manifest, run, "accepted_instances", "accepted_instances.csv"
    )
    attempt_rows = _read_generation_table(
        generation_manifest, run, "attempts", "attempts.csv"
    )
    extraction_rows = _read_generation_table(
        generation_manifest, run, "extractions", "extractions.csv"
    )
    accepted_artifacts = {str(row.get("artifact_stem", "")) for row in accepted_rows}
    if accepted_artifacts != set(final_strategy_by_stem):
        missing_metadata = sorted(accepted_artifacts - set(final_strategy_by_stem))
        missing_table = sorted(set(final_strategy_by_stem) - accepted_artifacts)
        raise SystemExit(
            "accepted instances and final metadata disagree; "
            f"missing metadata={missing_metadata}, missing table rows={missing_table}"
        )
    structural_rows = _json_instance_rows(report_data.structural_json)
    skewness_rows = _json_instance_rows(report_data.skewness_json)
    paired_artifacts = write_paired_strategy_report(
        requested_strategies,
        accepted_rows,
        attempt_rows,
        extraction_rows,
        metadata_strategy_by_artifact=final_strategy_by_stem,
        strategy_groups=strategy_groups,
        structural_rows=structural_rows,
        skewness_rows=skewness_rows,
        output_directory=evaluation_directory,
        overwrite=args.overwrite,
    )

    preservation_audits = {}
    for final in finals:
        metadata = metadata_by_stem[final.name]
        mappings = metadata.get("source_bus_mappings")
        if not isinstance(mappings, list):
            raise SystemExit(
                f"{final.name}: generation metadata predates source bus mappings; regenerate it"
            )
        embedded = []
        for mapping in mappings:
            source_stem = str(mapping["source_artifact_stem"])
            source_case = cases_by_stem[source_stem]
            node_map = {
                int(item["source_bus"]): int(item["composite_bus"])
                for item in mapping["buses"]
            }
            embedded.append(
                EmbeddedSourceGraph(
                    source_stem,
                    source_case.topology(active_only=True),
                    node_map,
                )
            )
        final_case = cases_by_stem[final.name]
        coupling_edges = tuple(
            (
                int(final_case.branch[int(row), 0]),
                int(final_case.branch[int(row), 1]),
            )
            for row in metadata["coupling_branch_rows_zero_based"]
        )
        strategy = final_strategy_by_stem[final.name]
        preservation_audits[final.name] = audit_preserved_properties(
            embedded,
            final.graph,
            coupling_edges,
            composition_strategy=strategy,
            noncrossing_matching_certified=(
                True
                if strategy in {"exact_outer_face", "heuristic_outer_face"}
                else None
            ),
        )
    preservation_artifacts = write_preservation_report(
        preservation_audits,
        evaluation_directory,
        overwrite=args.overwrite,
    )
    summary_path = evaluation_directory / "manifest.json"
    summary = {
        "schema_version": 2,
        "source_pool": _display_path(source_manifest),
        "skewness_time_limit_seconds": args.skewness_time_limit,
        "skewness_warm_start_iterations": 10,
        "distributed_cohort_checkpoints": (
            None
            if args.cohort_checkpoints is None
            else _display_path(args.cohort_checkpoints.resolve())
        ),
        "skewness_workers": args.skewness_workers,
        "betweenness_samples": (
            "exact" if args.betweenness_samples == 0 else args.betweenness_samples
        ),
        "review_report": None,
        "comparison_pdfs": [],
        "stage_pdf_failures": [],
        "planarization_csv": _display_path(
            evaluation_directory / "planarization.csv"
        ),
        "preservation_properties_csv": _display_path(
            preservation_artifacts.properties_csv
        ),
        "requested_strategies": list(requested_strategies),
        "strategy_groups": strategy_groups,
        "paired_strategy_outcomes_csv": _display_path(
            paired_artifacts.outcomes_csv
        ),
        "paired_strategy_cohorts_csv": _display_path(
            paired_artifacts.cohorts_csv
        ),
        "paired_strategy_json": _display_path(paired_artifacts.report_json),
    }
    _write_json(summary_path, summary)
    if args.no_compile:
        print(f"Evaluation data: {_display_path(evaluation_directory)}")
        return

    report_logs = run / "logs/report"
    comparison_figures = []
    for metric, plot in _ordered_boxplots(report_data.boxplots):
        pdf = compile_plot_pdf(
            plot,
            output_pdf=evaluation_directory / "boxplots" / f"{metric}.pdf",
            log_directory=report_logs,
            overwrite=args.overwrite,
        ).pdf
        comparison_figures.append(ReviewFigure(_plot_label(plot), pdf))

    review_instances = []
    stage_pdf_failures = []
    for final in finals:
        stage_pdfs = []
        merge_plot_directory = run / "plots" / "merged" / final.name
        if not merge_plot_directory.is_dir():
            merge_plot_directory = run / "plots" / final.name
        for stage_directory in sorted(merge_plot_directory.glob("stage_*")):
            stem = stage_directory.name.replace("stage_", "merge_stage_")
            plot = PlotArtifacts(
                tikz=stage_directory / f"{stem}.tikz",
                standalone_tex=stage_directory / f"{stem}.tex",
                coordinates_json=stage_directory / f"{stem}.json",
            )
            stage_pdf, failure = _compile_review_stage_pdf(
                plot,
                output_pdf=stage_directory / f"{stem}.pdf",
                log_directory=run / "logs" / "merged" / final.name,
                overwrite=args.overwrite,
            )
            if stage_pdf is not None:
                stage_pdfs.append(stage_pdf)
            if failure is not None:
                stage_pdf_failures.append(
                    {
                        "instance": final.name,
                        "stage": stage_directory.name,
                        **failure,
                    }
                )
        metadata = metadata_by_stem[final.name]
        stats = metadata["statistics"]
        audit = preservation_audits[final.name]
        preservation_detail = (
            "noncrossing matching certified"
            if audit.noncrossing_matching_certified is True
            else "noncrossing matching not certified"
        )
        review_instances.append(
            ReviewInstance(
                stem=final.name,
                bus_count=final.graph.number_of_nodes(),
                stage_pdfs=tuple(stage_pdfs),
                detail=(
                    f"{stats['strategy']}; sources: "
                    + ", ".join(metadata["source_names"])
                    + f"; {preservation_detail}"
                ),
                preservation_rows=_review_bound_rows(audit),
            )
        )

    report = write_a4_review_report(
        run / "review_report",
        comparison_figures,
        review_instances,
        title=f"PGLib generation review: {args.run}",
        log_directory=report_logs,
        overwrite=args.overwrite,
    )
    summary.update(
        review_report=_display_path(report.pdf),
        comparison_pdfs=[
            _display_path(figure.pdf) for figure in comparison_figures
        ],
        stage_pdf_failures=stage_pdf_failures,
    )
    _write_json(summary_path, summary)
    print(f"Evaluation data: {_display_path(evaluation_directory)}")
    print(f"A4 review report: {_display_path(report.pdf)}")


def _compile_review_stage_pdf(
    plot: PlotArtifacts,
    *,
    output_pdf: Path,
    log_directory: Path,
    overwrite: bool,
) -> tuple[Path | None, dict[str, str] | None]:
    stdout_log = log_directory / f"{output_pdf.stem}.stdout.log"
    latex_log = log_directory / f"{output_pdf.stem}.latex.log"
    cached = output_pdf.is_file() and stdout_log.is_file() and latex_log.is_file()
    try:
        compiled = compile_plot_pdf(
            plot,
            output_pdf=output_pdf,
            log_directory=log_directory,
            overwrite=overwrite and not cached,
        )
    except LatexCompilationError as error:
        failure_path = log_directory / f"{output_pdf.stem}.pdf-error.txt"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(f"{error}\n", encoding="utf-8")
        return None, {
            "error": str(error),
            "error_file": _display_path(failure_path),
        }
    return compiled.pdf, None


def _load_cohort_bounds(directory: Path) -> dict[str, SkewnessBounds]:
    paths = sorted(
        (*directory.glob("i*.json"), *directory.glob("i*/*.json"))
    )
    if not paths:
        raise SystemExit(f"no distributed cohort checkpoints below {directory}")
    by_name = {}
    settings = set()
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"cannot read distributed checkpoint {path}: {error}")
        settings.add(json.dumps(payload.get("settings"), sort_keys=True))
        graph_results = payload.get("graph_results")
        if not isinstance(graph_results, dict):
            raise SystemExit(
                f"distributed checkpoint predates graph_results: {path}"
            )
        for key, raw in graph_results.items():
            if not isinstance(key, str) or not isinstance(raw, dict):
                raise SystemExit(f"invalid graph result in {path}")
            name = key.rsplit(":", 1)[-1]
            bounds = SkewnessBounds(
                int(raw["skewness_lower_bound"]),
                int(raw["skewness_upper_bound"]),
                status=str(raw["status"]),
                runtime_seconds=float(raw["runtime_seconds"]),
            )
            previous = by_name.setdefault(name, bounds)
            if previous != bounds:
                raise SystemExit(
                    f"inconsistent duplicate distributed result for {name}"
                )
    if len(settings) != 1:
        raise SystemExit("distributed checkpoints use inconsistent settings")
    return by_name


def _result_bounds(result: object) -> tuple[int, int]:
    if isinstance(result, SkewnessBounds):
        return result.lower_bound, result.upper_bound
    return int(result.omitted_lower_bound), int(result.omitted_upper_bound)


def _result_statistics(result: object) -> dict[str, object]:
    if not isinstance(result, SkewnessBounds):
        return result.statistics()
    lower, upper = _result_bounds(result)
    return {
        "method": "precomputed_skewness_bounds",
        "omitted_lower_bound": lower,
        "omitted_upper_bound": upper,
        "skewness": upper if lower == upper else None,
        "exact": lower == upper,
        "absolute_gap": upper - lower,
        "termination_status": result.status,
        "total_runtime_seconds": result.runtime_seconds,
    }


def _source_manifest(argument: Path | None, generation: dict[str, object]) -> Path:
    if argument is not None:
        return argument.resolve()
    run = generation.get("run")
    if not isinstance(run, dict) or not isinstance(run.get("source_pool"), str):
        raise SystemExit("generation manifest does not identify its source pool")
    path = Path(run["source_pool"])
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _generation_metadata_paths(run: Path) -> list[Path]:
    plot_root = run / "plots"
    nested = [
        *plot_root.joinpath("original").glob("*/metadata.json"),
        *plot_root.joinpath("composite").glob("*/metadata.json"),
    ]
    return sorted(nested or plot_root.glob("*/metadata.json"))


def _generated_case_path(run: Path, stem: str, kind: str) -> Path:
    if kind == "final_composite":
        path = run / "instances" / f"{stem}.m"
    elif kind == "extracted_subinstance":
        path = run / "subinstances" / f"{stem}.m"
        legacy_path = run / "instances" / f"{stem}.m"
        if not path.is_file() and legacy_path.is_file():
            path = legacy_path
    else:
        raise SystemExit(f"unknown artifact kind {kind!r} for {stem}")
    if not path.is_file():
        raise SystemExit(f"missing {kind} for {stem}: {path}")
    return path


def _metadata_strategy(metadata: Mapping[str, object], stem: str) -> str:
    statistics = metadata.get("statistics")
    if not isinstance(statistics, dict) or not isinstance(
        statistics.get("strategy"), str
    ):
        raise SystemExit(f"{stem}: metadata does not identify its strategy")
    strategy = str(statistics["strategy"])
    if not strategy:
        raise SystemExit(f"{stem}: metadata strategy must not be empty")
    return strategy


def _requested_strategies(
    generation: Mapping[str, object], present: tuple[str, ...]
) -> tuple[str, ...]:
    run = generation.get("run")
    configured = run.get("strategies") if isinstance(run, dict) else None
    if configured is None:
        return tuple(dict.fromkeys(present))
    if (
        not isinstance(configured, list)
        or not configured
        or any(not isinstance(value, str) or not value for value in configured)
        or len(configured) != len(set(configured))
    ):
        raise SystemExit("generation manifest run.strategies must be unique strings")
    return tuple(configured)


def _strategy_label(strategy: str) -> str:
    return _STRATEGY_LABELS.get(strategy, strategy.replace("_", " ").title())


def _read_generation_table(
    manifest: Mapping[str, object], run: Path, key: str, fallback_name: str
) -> list[dict[str, str]]:
    tables = manifest.get("tables")
    declared = tables.get(key) if isinstance(tables, dict) else None
    if declared is not None and not isinstance(declared, str):
        raise SystemExit(f"generation manifest tables.{key} must be a path string")
    path = Path(declared) if declared else run / "tables" / fallback_name
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    if not path.is_file():
        raise SystemExit(f"missing generation table {key}: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or any(not field for field in reader.fieldnames):
            raise SystemExit(f"generation table has an invalid header: {path}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise SystemExit(f"generation table has duplicate columns: {path}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise SystemExit(f"generation table has rows wider than its header: {path}")
    return rows


def _json_instance_rows(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("instances") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise SystemExit(f"evaluation table has invalid instances: {path}")
    return rows


def _topology_fingerprint(
    graph: nx.Graph,
) -> tuple[frozenset[object], frozenset[frozenset[object]]]:
    return (
        frozenset(graph.nodes),
        frozenset(frozenset((first, second)) for first, second in graph.edges),
    )


def _plot_label(plot: PlotArtifacts) -> str:
    data = json.loads(plot.coordinates_json.read_text(encoding="utf-8"))
    return str(data.get("y_label") or plot.tikz.stem.replace("_", " ").title())


def _ordered_boxplots(
    plots: Mapping[str, PlotArtifacts],
) -> tuple[tuple[str, PlotArtifacts], ...]:
    priority = (
        "normalized_inflation",
        "skewness_lower_bound",
        "skewness_upper_bound",
        "algebraic_connectivity",
        "diameter",
        "mean_betweenness",
        "betweenness_p95",
        "coupling_betweenness",
        "degree_assortativity",
        "degree",
        "maximum_degree",
        "average_clustering",
        "degeneracy",
        "treewidth_upper_bound",
        "node_count",
        "edge_count",
        "density",
        "mean_degree",
        "normalized_inflation_lower_bound",
        "normalized_inflation_upper_bound",
    )
    mapping = dict(plots)
    return tuple(
        (metric, mapping.pop(metric))
        for metric in priority
        if metric in mapping
    ) + tuple(mapping.items())


def _review_bound_rows(audit: PreservationAudit) -> tuple[ReviewBoundRow, ...]:
    properties = audit.properties
    labels = {
        "node_count": "Node count",
        "edge_count": "Edge count",
        "node_degree_identity_max_error": "Node degrees (max error)",
        "maximum_degree": "Maximum degree",
        "noncoupling_node_clustering_max_error": "Noncoupling clustering (max error)",
        "average_clustering_difference": "Average clustering (conditional)",
        "degeneracy": "Degeneracy (conditional)",
        "treewidth": "Treewidth (bracket)",
    }
    rows = []
    degree = max(audit.degree_distribution, key=lambda row: row.absolute_change)
    rows.append(
        ReviewBoundRow(
            "Degree distribution (max q)",
            f"change <= {_number(degree.paper_upper_bound)}",
            f"{_number(degree.absolute_change)} (q={degree.degree})",
            "pass" if degree.within_bound else "FAIL",
            slack=_number(degree.slack),
        )
    )
    for property_row in properties:
        if (
            property_row.property == "treewidth"
            and property_row.paper_upper_bound is None
        ):
            bound = "<= RHS " + _bound_text(
                property_row.paper_rhs_lower_estimate,
                property_row.paper_rhs_upper_estimate,
            )
        else:
            bound = _bound_text(
                property_row.paper_lower_bound, property_row.paper_upper_bound
            )
        observed = (
            _number(property_row.observed_value)
            if property_row.observed_value is not None
            else _bound_text(
                property_row.observed_lower_bound,
                property_row.observed_upper_bound,
            )
        )
        if property_row.paper_hypothesis_satisfied is False:
            check = (
                "hyp. false; num. pass"
                if property_row.numerically_within_bound is True
                else "hyp. false; num. FAIL"
            )
        elif property_row.paper_hypothesis_satisfied is None:
            check = (
                "not cert.; num. pass"
                if property_row.numerically_within_bound is True
                else "not cert.; num. FAIL"
            )
        else:
            check = (
                "pass"
                if property_row.empirically_within_bound is True
                else "FAIL"
                if property_row.empirically_within_bound is False
                else "inconclusive"
            )
        rows.append(
            ReviewBoundRow(
                labels[property_row.property],
                bound,
                observed,
                check,
                slack=_slack_text(property_row),
            )
        )
    return tuple(rows)


def _bound_text(lower: float | None, upper: float | None) -> str:
    if lower is not None and upper is not None:
        if math.isclose(lower, upper, rel_tol=0.0, abs_tol=1e-12):
            return f"= {_number(lower)}"
        return f"[{_number(lower)}, {_number(upper)}]"
    if upper is not None:
        return f"<= {_number(upper)}"
    if lower is not None:
        return f">= {_number(lower)}"
    return "n/a"


def _slack_text(row: object) -> str:
    lower = row.paper_lower_bound
    upper = row.paper_upper_bound
    if (
        row.property == "treewidth"
        and row.paper_rhs_lower_estimate is not None
        and row.paper_rhs_upper_estimate is not None
    ):
        if row.empirically_within_bound is True:
            return ">= " + _number(
                row.paper_rhs_lower_estimate - row.observed_upper_bound
            )
        if row.empirically_within_bound is False:
            return "<= " + _number(
                row.paper_rhs_upper_estimate - row.observed_lower_bound
            )
        return "not exact"
    if (
        lower is not None
        and upper is not None
        and math.isclose(lower, upper, rel_tol=0.0, abs_tol=1e-12)
        and row.observed_value is not None
    ):
        return "resid. " + _number(abs(row.observed_value - lower))
    if row.lower_slack is not None and row.upper_slack is not None:
        return f"L {_number(row.lower_slack)}; U {_number(row.upper_slack)}"
    if row.upper_slack is not None:
        return _number(row.upper_slack)
    return "not exact"


def _number(value: float | int) -> str:
    return format(float(value), ".5g")


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def _display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
