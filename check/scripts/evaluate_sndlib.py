#!/usr/bin/env python3
"""Compute paper statistics and build the A4 review report for an SNDlib run."""

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

from planar_instance_generation.core import PlanarizationResult
from planar_instance_generation.evaluation import (
    EvaluationGraph,
    GeneratedSndArtifacts,
    PreservationAudit,
    SkewnessBounds,
    SkewnessObservation,
    audit_snd_preservation,
    load_generated_snd_artifacts,
    load_snd_source_pool,
    write_evaluation_report,
    write_preservation_report,
)
from planar_instance_generation.evaluation.paired import (
    SNDLIB_PAIRED_SCHEMA,
    write_paired_strategy_report,
)
from planar_instance_generation.evaluation.planarization_batch import (
    PlanarizationTask,
    solve_partitioned_batch,
    source_mappings,
)
from planar_instance_generation.plt import (
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
    parser.add_argument("run", help="SNDlib generation result name below check/results")
    parser.add_argument("--source-pool", type=Path)
    parser.add_argument(
        "--cohort-checkpoints",
        type=Path,
        help="directory of distributed per-instance exact-skewness JSON files",
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
    _validate_args(args)
    run = RESULTS_DIRECTORY / args.run
    evaluation_directory = run / "evaluation"
    if (
        evaluation_directory.exists()
        and any(evaluation_directory.iterdir())
        and not args.overwrite
    ):
        raise SystemExit(
            f"evaluation already exists: {evaluation_directory}; "
            "pass --overwrite to replace it"
        )
    generation_manifest_path = run / "manifest.json"
    generation_manifest = _read_manifest(
        generation_manifest_path, "sndlib_generation"
    )
    source_manifest_path = _source_manifest(args.source_pool, generation_manifest)
    source_manifest = _read_manifest(source_manifest_path, "sndlib_source_pool")
    source_pool = load_snd_source_pool(
        source_manifest_path,
        path_root=REPOSITORY_ROOT,
    )
    generated = load_generated_snd_artifacts(run)

    originals = tuple(
        EvaluationGraph(source.source_name, source.network.topology())
        for source in source_pool
    )
    finals_by_strategy: dict[str, list[EvaluationGraph]] = {}
    final_strategy_by_stem: dict[str, str] = {}
    for final in generated.finals:
        finals_by_strategy.setdefault(final.strategy, []).append(
            final.evaluation_graph
        )
        final_strategy_by_stem[final.stem] = final.strategy

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
        "Extracted subinstances": tuple(
            artifact.evaluation_graph for artifact in generated.extracted
        ),
    }
    groups.update(
        (strategy_groups[strategy], tuple(finals_by_strategy[strategy]))
        for strategy in present_strategies
    )

    preloaded_results = (
        {}
        if args.cohort_checkpoints is None
        else _load_cohort_bounds(args.cohort_checkpoints.resolve(), groups)
    )
    skewness, planarization_rows = _compute_skewness(
        groups,
        generated,
        strategy_groups,
        run / "logs/skewness",
        preloaded_results=preloaded_results,
        time_limit=args.skewness_time_limit,

        threads=args.threads,
        workers=args.skewness_workers,
        seed=args.seed,
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
    _write_json(evaluation_directory / "planarization.json", planarization_rows)

    generation_tables = _generation_tables(generation_manifest, run)
    accepted_rows = _read_csv(generation_tables["accepted_instances"])
    attempt_rows = _read_csv(generation_tables["attempts"])
    extraction_rows = _read_csv(generation_tables["extractions"])
    accepted_artifacts = {
        str(row.get("artifact_stem", "")) for row in accepted_rows
    }
    if accepted_artifacts != set(final_strategy_by_stem):
        missing_metadata = sorted(accepted_artifacts - set(final_strategy_by_stem))
        missing_table = sorted(set(final_strategy_by_stem) - accepted_artifacts)
        raise SystemExit(
            "accepted instances and final metadata disagree; "
            f"missing metadata={missing_metadata}, missing table rows={missing_table}"
        )
    paired_artifacts = write_paired_strategy_report(
        requested_strategies,
        accepted_rows,
        attempt_rows,
        extraction_rows,
        metadata_strategy_by_artifact=final_strategy_by_stem,
        strategy_groups=strategy_groups,
        structural_rows=_json_instance_rows(report_data.structural_json),
        skewness_rows=_json_instance_rows(report_data.skewness_json),
        output_directory=evaluation_directory,
        schema=SNDLIB_PAIRED_SCHEMA,
        overwrite=args.overwrite,
    )

    preservation_audits = audit_snd_preservation(generated)
    preservation_artifacts = write_preservation_report(
        preservation_audits,
        evaluation_directory,
        overwrite=args.overwrite,
    )
    source_table = _source_screening_table(source_manifest)
    summary_path = evaluation_directory / "manifest.json"
    summary: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "sndlib_evaluation",
        "generation_manifest": _display_path(generation_manifest_path),
        "source_pool_manifest": _display_path(source_manifest_path),
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
        "requested_strategies": list(requested_strategies),
        "strategy_groups": strategy_groups,
        "review_report": None,
        "comparison_pdfs": [],
        "planarization_csv": _display_path(
            evaluation_directory / "planarization.csv"
        ),
        "preservation_properties_csv": _display_path(
            preservation_artifacts.properties_csv
        ),
        "paired_strategy_outcomes_csv": _display_path(
            paired_artifacts.outcomes_csv
        ),
        "paired_strategy_cohorts_csv": _display_path(
            paired_artifacts.cohorts_csv
        ),
        "paired_strategy_json": _display_path(paired_artifacts.report_json),
        "input_tables": {
            "source_screening": _display_path(source_table),
            **{
                f"generation_{name}": _display_path(path)
                for name, path in generation_tables.items()
            },
        },
        "input_table_roles": {
            "selection_runtimes": "generation_merge_stages",
            "acceptance_and_gate_outcomes": "generation_attempts",
            "accepted_instance_runtimes": "generation_accepted_instances",
            "source_preprocessing": "source_screening",
        },
        "theorem_certification": {
            "exact_outer_face": (
                "sequential noncrossing matching certified by construction"
            ),
            "heuristic_outer_face": (
                "sequential noncrossing matching certified by construction; "
                "planarization optimality is not required"
            ),
            "other_strategies": (
                "noncrossing hypothesis not certified; conditional numerical "
                "checks are descriptive"
            ),
        },
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
    for final in generated.finals:
        stage_pdfs = _compile_stage_pdfs(
            run,
            final.stem,
            overwrite=args.overwrite,
        )
        audit = preservation_audits[final.stem]
        preservation_detail = (
            "noncrossing matching certified"
            if audit.noncrossing_matching_certified is True
            else "noncrossing matching not certified"
        )
        source_names = final.metadata.get("source_names")
        if not isinstance(source_names, list) or any(
            not isinstance(value, str) for value in source_names
        ):
            raise SystemExit(f"{final.stem}: metadata source_names is invalid")
        routing_unit = final.metadata.get("routing_unit")
        unit_detail = "unitless" if routing_unit is None else str(routing_unit)
        review_instances.append(
            ReviewInstance(
                stem=final.stem,
                bus_count=final.graph.number_of_nodes(),
                stage_pdfs=stage_pdfs,
                detail=(
                    f"{final.strategy}; sources: {', '.join(source_names)}; "
                    f"routing unit: {unit_detail}; {preservation_detail}"
                ),
                preservation_rows=_review_bound_rows(audit),
                size_unit="nodes",
            )
        )

    report = write_a4_review_report(
        run / "review_report",
        comparison_figures,
        review_instances,
        title=f"SNDlib generation review: {args.run}",
        log_directory=report_logs,
        overwrite=args.overwrite,
    )
    summary.update(
        review_report=_display_path(report.pdf),
        comparison_pdfs=[
            _display_path(figure.pdf) for figure in comparison_figures
        ],
    )
    _write_json(summary_path, summary)
    print(f"Evaluation data: {_display_path(evaluation_directory)}")
    print(f"A4 review report: {_display_path(report.pdf)}")


def _validate_args(args: argparse.Namespace) -> None:
    if not _SAFE_NAME.fullmatch(args.run):
        raise SystemExit("run must be a single safe filename component")
    if (
        args.threads < 1
        or args.skewness_workers < 1
        or args.betweenness_samples < 0
        or args.seed < 0
    ):
        raise SystemExit(
            "--threads and --skewness-workers must be positive; "
            "--betweenness-samples and --seed must be nonnegative"
        )
    if (
        not math.isfinite(args.skewness_time_limit)
        or args.skewness_time_limit <= 0.0
    ):
        raise SystemExit("--skewness-time-limit must be finite and positive")


def _compute_skewness(
    groups: Mapping[str, tuple[EvaluationGraph, ...]],
    generated: GeneratedSndArtifacts,
    strategy_groups: Mapping[str, str],
    log_directory: Path,
    *,
    preloaded_results: Mapping[tuple[str, str], SkewnessBounds],
    time_limit: float,

    threads: int,
    workers: int,
    seed: int,
) -> tuple[
    dict[tuple[str, str], PlanarizationResult | SkewnessObservation],
    list[dict[str, object]],
]:
    rng = random.Random(seed)
    sample_records = []
    unique_fingerprints = []
    task_index_by_fingerprint = {}
    tasks = []
    for group, samples in groups.items():
        for sample in samples:
            if (group, sample.name) in preloaded_results:
                sample_records.append((group, sample, None, False))
                continue
            fingerprint = _topology_fingerprint(sample.graph)
            if group.startswith("Composite:"):
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
                            log_directory / _slug(group) / f"{sample.name}.log"
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
        f"{workers} worker process(es) and {threads} OGDF heuristic thread(s) per solve.",
        flush=True,
    )
    task_by_sample = {
        (group, sample.name): tasks[task_index_by_fingerprint[fingerprint]].cache_key
        for group, sample, fingerprint, _reused in sample_records
        if fingerprint is not None
    }
    dependencies = {
        task_by_sample[(strategy_groups[final.strategy], final.stem)]: [
            (task_by_sample[("Extracted subinstances", stem)], mapping)
            for stem, mapping in source_mappings(final.metadata)
        ]
        for final in generated.finals
        if (strategy_groups[final.strategy], final.stem) in task_by_sample
    }
    results = solve_partitioned_batch(
        tasks,
        dependencies,
        generation_metadata=[
            (final.metadata, [task_by_sample[("Extracted subinstances", stem)]
                              for stem, _mapping in source_mappings(final.metadata)])
            for final in generated.finals
            if (strategy_groups[final.strategy], final.stem) in task_by_sample
        ],
        time_limit_seconds=time_limit,

        solver_threads=threads,
        workers=workers,
        checkpoint_directory=(
            log_directory.parent.parent / "evaluation" / "planarization_checkpoints"
        ),
    )
    cache = dict(zip(unique_fingerprints, results, strict=True))
    by_key: dict[tuple[str, str], PlanarizationResult] = {}
    observations: dict[
        tuple[str, str], PlanarizationResult | SkewnessObservation
    ] = {}
    rows: list[dict[str, object]] = []
    for group, sample, fingerprint, reused in sample_records:
        result = (
            preloaded_results[(group, sample.name)]
            if fingerprint is None
            else cache[fingerprint]
        )
        by_key[(group, sample.name)] = result
        observations[(group, sample.name)] = result
        rows.append(
            {
                "group": group,
                "instance": sample.name,
                "topology_result_reused": reused,
                **_result_statistics(result),
            }
        )

    for final in generated.finals:
        final_group = strategy_groups[final.strategy]
        source_results = [
            by_key[("Extracted subinstances", stem)]
            for stem in final.source_stems
        ]
        observations[(final_group, final.stem)] = SkewnessObservation(
            by_key[(final_group, final.stem)],
            source_piece_count=len(source_results),
            source_floor_lower_bound=sum(
                _result_bounds(result)[0] for result in source_results
            ),
            source_floor_upper_bound=sum(
                _result_bounds(result)[1] for result in source_results
            ),
            coupling_edge_count=len(final.coupling_edges),
        )
    return observations, rows


def _load_cohort_bounds(
    directory: Path,
    groups: Mapping[str, tuple[EvaluationGraph, ...]],
) -> dict[tuple[str, str], SkewnessBounds]:
    paths = sorted((*directory.glob("i*.json"), *directory.glob("i*/*.json")))
    if not paths:
        raise SystemExit(f"no distributed SND checkpoints below {directory}")
    known = {
        sample.name: group
        for group, samples in groups.items()
        if group != "Original sources"
        for sample in samples
    }
    by_key: dict[tuple[str, str], SkewnessBounds] = {}
    settings = set()
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"cannot read distributed checkpoint {path}: {error}")
        settings.add(json.dumps(payload.get("settings"), sort_keys=True))
        graph_results = payload.get("graph_results")
        if not isinstance(graph_results, dict):
            raise SystemExit(f"distributed checkpoint predates graph_results: {path}")
        for raw in graph_results.values():
            if not isinstance(raw, dict) or not isinstance(raw.get("instance"), str):
                raise SystemExit(f"invalid graph result in {path}")
            name = raw["instance"]
            if name not in known:
                raise SystemExit(
                    f"distributed checkpoint names unknown graph {name}: {path}"
                )
            bounds = SkewnessBounds(
                int(raw["skewness_lower_bound"]),
                int(raw["skewness_upper_bound"]),
                status=str(raw["status"]),
                runtime_seconds=float(raw["runtime_seconds"]),
            )
            key = (known[name], name)
            previous = by_key.setdefault(key, bounds)
            if (
                previous.lower_bound != bounds.lower_bound
                or previous.upper_bound != bounds.upper_bound
            ):
                raise SystemExit(f"inconsistent distributed result for {name}")
    if len(settings) != 1:
        raise SystemExit("distributed SND checkpoints use inconsistent settings")
    missing = {
        (group, sample.name)
        for group, samples in groups.items()
        if group != "Original sources"
        for sample in samples
    } - set(by_key)
    if missing:
        sample = ", ".join(name for _group, name in sorted(missing)[:5])
        raise SystemExit(f"distributed SND checkpoints do not cover: {sample}")
    return by_key


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


def _compile_stage_pdfs(
    run: Path, stem: str, *, overwrite: bool
) -> tuple[Path, ...]:
    merge_plot_directory = run / "plots" / "merged" / stem
    if not merge_plot_directory.is_dir():
        merge_plot_directory = run / "plots" / stem
    stage_directories = sorted(merge_plot_directory.glob("stage_*"))
    if not stage_directories:
        raise SystemExit(f"{stem}: no pairwise merge stage plots found")
    pdfs = []
    for stage_directory in stage_directories:
        plot_stem = stage_directory.name.replace("stage_", "merge_stage_")
        plot = PlotArtifacts(
            tikz=stage_directory / f"{plot_stem}.tikz",
            standalone_tex=stage_directory / f"{plot_stem}.tex",
            coordinates_json=stage_directory / f"{plot_stem}.json",
        )
        pdfs.append(
            compile_plot_pdf(
                plot,
                output_pdf=stage_directory / f"{plot_stem}.pdf",
                log_directory=run / "logs" / "merged" / stem,
                overwrite=overwrite,
            ).pdf
        )
    return tuple(pdfs)


def _read_manifest(path: Path, artifact_type: str) -> dict[str, object]:
    if not path.is_file():
        raise SystemExit(f"manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"invalid JSON manifest {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("artifact_type") != artifact_type:
        raise SystemExit(f"{path}: expected artifact_type {artifact_type!r}")
    return payload


def _source_manifest(
    argument: Path | None, generation: Mapping[str, object]
) -> Path:
    if argument is not None:
        return argument.resolve()
    run = generation.get("run")
    if not isinstance(run, dict) or not isinstance(run.get("source_pool"), str):
        raise SystemExit("generation manifest does not identify its source pool")
    path = Path(run["source_pool"])
    return path if path.is_absolute() else REPOSITORY_ROOT / path


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


def _generation_tables(
    manifest: Mapping[str, object], run: Path
) -> dict[str, Path]:
    fallbacks = {
        "accepted_instances": "accepted_instances.csv",
        "attempts": "attempts.csv",
        "extractions": "extractions.csv",
        "merge_stages": "merge_stages.csv",
        "routing_unit_strata": "routing_unit_strata.csv",
    }
    tables = manifest.get("tables")
    if not isinstance(tables, dict):
        raise SystemExit("generation manifest tables must be an object")
    result = {}
    for key, fallback in fallbacks.items():
        declared = tables.get(key)
        if declared is not None and not isinstance(declared, str):
            raise SystemExit(f"generation manifest tables.{key} must be a path")
        path = Path(declared) if declared else run / "tables" / fallback
        if not path.is_absolute():
            path = REPOSITORY_ROOT / path
        if not path.is_file():
            raise SystemExit(f"missing generation table {key}: {path}")
        result[key] = path
    return result


def _source_screening_table(manifest: Mapping[str, object]) -> Path:
    value = manifest.get("records_csv")
    if not isinstance(value, str) or not value:
        raise SystemExit("source-pool manifest does not identify records_csv")
    path = Path(value)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    if not path.is_file():
        raise SystemExit(f"missing source screening table: {path}")
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or any(not field for field in reader.fieldnames):
            raise SystemExit(f"table has an invalid header: {path}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise SystemExit(f"table has duplicate columns: {path}")
        rows = list(reader)
    if not rows:
        raise SystemExit(f"table is empty: {path}")
    if any(None in row for row in rows):
        raise SystemExit(f"table has rows wider than its header: {path}")
    return rows


def _json_instance_rows(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("instances") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise SystemExit(f"evaluation table has invalid instances: {path}")
    return rows


def _review_bound_rows(audit: PreservationAudit) -> tuple[ReviewBoundRow, ...]:
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
    for property_row in audit.properties:
        if property_row.property == "treewidth" and property_row.paper_upper_bound is None:
            bound = "<= RHS " + _bound_text(
                property_row.paper_rhs_lower_estimate,
                property_row.paper_rhs_upper_estimate,
            )
        else:
            bound = _bound_text(
                property_row.paper_lower_bound,
                property_row.paper_upper_bound,
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
        (metric, mapping.pop(metric)) for metric in priority if metric in mapping
    ) + tuple(mapping.items())


def _topology_fingerprint(
    graph: nx.Graph,
) -> tuple[frozenset[object], frozenset[frozenset[object]]]:
    return (
        frozenset(graph.nodes),
        frozenset(frozenset((first, second)) for first, second in graph.edges),
    )


def _strategy_label(strategy: str) -> str:
    return _STRATEGY_LABELS.get(strategy, strategy.replace("_", " ").title())


def _plot_label(plot: PlotArtifacts) -> str:
    data = json.loads(plot.coordinates_json.read_text(encoding="utf-8"))
    return str(data.get("y_label") or plot.tikz.stem.replace("_", " ").title())


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
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
