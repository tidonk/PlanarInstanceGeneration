#!/usr/bin/env python3
"""Aggregate completed per-cohort PGLib skewness checkpoints and plot gallery."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import build_partial_pglib_report as partial
from planar_instance_generation.plt import (
    BoxplotSeries,
    EcdfSeries,
    PairedScatterPoint,
    compile_plot_pdf,
    write_boxplot,
    write_ecdf,
    write_paired_scatter,
)


RESULTS_DIRECTORY = REPOSITORY_ROOT / "check/results"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument(
        "--output-name",
        default=None,
        help="checkpoint/report directory (default: <experiment>_pglib_exact)",
    )
    parser.add_argument(
        "--gallery",
        choices=("composites", "all"),
        default="all",
    )
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_name = args.output_name or f"{args.experiment}_pglib_exact"
    if not _SAFE_NAME.fullmatch(args.experiment) or not _SAFE_NAME.fullmatch(
        output_name
    ):
        raise SystemExit("experiment and output names must be safe path components")
    output = RESULTS_DIRECTORY / output_name
    checkpoint_directory = output / "evaluation/cohort_checkpoints"
    checkpoint_paths = sorted(
        (
            *checkpoint_directory.glob("i*.json"),
            *checkpoint_directory.glob("i*/*.json"),
        )
    )
    if not checkpoint_paths:
        raise SystemExit(f"no method checkpoints found below {checkpoint_directory}")

    cohorts_by_index = {
        cohort.index: cohort for cohort in partial._discover_cohorts(args.experiment)
    }
    rows = []
    settings = set()
    completed_indices = set()
    method_strategies: dict[int, set[str]] = {}
    method_results: dict[int, dict[str, dict[str, object]]] = {}
    for path in checkpoint_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        index = int(payload["cohort"])
        if index not in cohorts_by_index:
            raise SystemExit(f"checkpoint has no complete generation cohort: {path}")
        settings.add(json.dumps(payload["settings"], sort_keys=True))
        if "rows" in payload:  # Legacy one-checkpoint-per-cohort format.
            completed_indices.add(index)
            rows.extend(payload["rows"])
            continue
        strategy = str(payload.get("strategy"))
        if strategy not in partial.STRATEGIES:
            raise SystemExit(f"invalid strategy in method checkpoint: {path}")
        seen = method_strategies.setdefault(index, set())
        if strategy in seen:
            raise SystemExit(f"duplicate {strategy} checkpoint for i{index:03d}")
        seen.add(strategy)
        combined = method_results.setdefault(index, {})
        for key, result in payload.get("graph_results", {}).items():
            if key in combined and combined[key] != result:
                raise SystemExit(f"inconsistent duplicate graph result in {path}")
            combined[key] = result
    if len(settings) != 1:
        raise SystemExit("cohort checkpoints use inconsistent solver settings")
    shared_settings = json.loads(next(iter(settings)))

    expected_strategies = set(partial.STRATEGIES)
    for index, strategies in method_strategies.items():
        if strategies != expected_strategies:
            missing = ", ".join(sorted(expected_strategies - strategies))
            raise SystemExit(f"i{index:03d} is missing method checkpoints: {missing}")
        cohort = cohorts_by_index[index]
        # New workers carry their shared source certificates. Legacy workers
        # solved finals only and still need the original heuristic source floor.
        all_specs = partial._task_specs(
            (cohort,), 10, int(shared_settings["seed"])
        )
        for stem in cohort.source_stems:
            key = f"source:{index}:{stem}"
            if shared_settings.get("shared_piece_policy", 0) >= 1:
                if key not in method_results[index]:
                    raise SystemExit(f"missing shared piece certificate: {key}")
            else:
                method_results[index][key] = partial._solve_one(all_specs[key])
        cohort_rows = partial._skewness_rows(
            (cohort,), method_results[index], iterations=10
        )
        for row in cohort_rows:
            row["method"] = "exact_branch_and_cut"
            row["source_floor_method"] = (
                "shared_exact_branch_and_cut"
                if shared_settings.get("shared_piece_policy", 0) >= 1
                else "deterministic_fixed_restart"
            )
            row["solver_time_limit_seconds"] = shared_settings["time_limit_seconds"]
            row["heuristic_runs"] = shared_settings.get("heuristic_runs", shared_settings.get("warm_start_iterations"))
            row.pop("heuristic_iterations", None)
        rows.extend(cohort_rows)
        completed_indices.add(index)

    cohorts = [cohorts_by_index[index] for index in sorted(completed_indices)]
    expected = {(cohort.index, strategy) for cohort in cohorts for strategy in partial.STRATEGIES}
    observed = {(int(row["cohort"]), str(row["strategy"])) for row in rows}
    if observed != expected:
        raise SystemExit("cohort checkpoints do not contain one row per strategy")

    evaluation = output / "evaluation"
    boxplots = evaluation / "boxplots"
    logs = output / "logs/report"
    for directory in (evaluation, boxplots, logs):
        directory.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: (int(row["cohort"]), partial.STRATEGIES.index(row["strategy"])))
    summaries = partial._summaries(rows)
    _write_union_csv(evaluation / "skewness.csv", rows)
    partial._write_csv(evaluation / "strategy_summary.csv", summaries)
    (evaluation / "strategy_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    figures = _write_boxplots(
        rows,
        boxplots,
        logs,
        overwrite=args.overwrite,
        compile_pdf=not args.no_compile,
    )
    figures += _write_added_skewness_ecdf(
        rows,
        boxplots,
        logs,
        overwrite=args.overwrite,
        compile_pdf=not args.no_compile,
    )
    figures += _write_outer_face_scatter(
        rows,
        boxplots,
        logs,
        overwrite=args.overwrite,
        compile_pdf=not args.no_compile,
    )
    gallery_tex = output / "plot_gallery.tex"
    gallery_pdf = output / "plot_gallery.pdf"
    tex = partial._gallery_tex(
        cohorts,
        figures,
        summaries,
        iterations=10,
        include_merge_stages=args.gallery == "all",
    ).replace(
        "Preliminary skewness uses 4 deterministic insertion orders, not the exact optimizer.",
        "Skewness intervals use the distributed exact branch-and-cut checkpoints available when this report was built.",
    )
    gallery_tex.write_text(tex, encoding="utf-8")
    if not args.no_compile:
        partial._compile_gallery(gallery_tex, gallery_pdf, logs)

    manifest = {
        "schema_version": 1,
        "experiment": args.experiment,
        "completed_exact_cohorts": len(cohorts),
        "requested_complete_cohorts": len(cohorts_by_index),
        "settings": shared_settings,
        "gallery_scope": args.gallery,
        "skewness_csv": str((evaluation / "skewness.csv").relative_to(REPOSITORY_ROOT)),
        "strategy_summary_csv": str(
            (evaluation / "strategy_summary.csv").relative_to(REPOSITORY_ROOT)
        ),
        "plot_gallery": (
            None if args.no_compile else str(gallery_pdf.relative_to(REPOSITORY_ROOT))
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Aggregated {len(cohorts)}/{len(cohorts_by_index)} exact cohorts.",
        flush=True,
    )
    if not args.no_compile:
        print(f"Scrollable gallery: {gallery_pdf.relative_to(REPOSITORY_ROOT)}")


def _write_union_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = tuple(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_added_skewness_ecdf(
    rows: list[dict[str, object]],
    output: Path,
    logs: Path,
    *,
    overwrite: bool,
    compile_pdf: bool,
) -> tuple[tuple[str, Path], ...]:
    plot = write_ecdf(
        [
            EcdfSeries(
                partial.STRATEGY_LABELS[strategy].replace("--", "-"),
                tuple(
                    float(row["inflation_upper_bound"])
                    for row in rows
                    if row["strategy"] == strategy
                    and row["inflation_upper_bound"] is not None
                ),
                partial.STRATEGY_COLORS[strategy],
            )
            for strategy in partial.STRATEGIES
        ],
        output / "added_skewness_upper_bound_ecdf",
        x_label="Added skewness upper bound above the source floor",
        width_cm=12.0,
        height_cm=7.6,
        overwrite=overwrite,
    )
    if not compile_pdf:
        return ()
    pdf = compile_plot_pdf(
        plot,
        output_pdf=output / "added_skewness_upper_bound_ecdf.pdf",
        log_directory=logs,
        overwrite=overwrite,
    ).pdf
    return (
        ("Empirical distribution of added skewness upper bounds", pdf),
    )



def _write_outer_face_scatter(
    rows: list[dict[str, object]],
    output: Path,
    logs: Path,
    *,
    overwrite: bool,
    compile_pdf: bool,
) -> tuple[tuple[str, Path], ...]:
    by_pair = {
        (int(row["cohort"]), str(row["strategy"])): row
        for row in rows
    }
    cohorts = sorted({int(row["cohort"]) for row in rows})
    points = []
    for cohort in cohorts:
        heuristic = by_pair.get((cohort, "heuristic_outer_face"))
        exact = by_pair.get((cohort, "exact_outer_face"))
        if heuristic is None or exact is None:
            raise ValueError(f"missing paired outer face row for i{cohort:03d}")
        heuristic_value = heuristic.get("skewness_upper_bound")
        exact_value = exact.get("skewness_upper_bound")
        if heuristic_value is None or exact_value is None:
            continue
        points.append(
            PairedScatterPoint(
                label=f"i{cohort:03d}",
                x=float(exact_value),
                y=float(heuristic_value),
            )
        )
    plot = write_paired_scatter(
        tuple(points),
        output / "outer_face_skewness_upper_bound_paired",
        x_label="Exact outer face skewness upper bound",
        y_label="Heuristic outer face skewness upper bound",
        width_cm=10.2,
        height_cm=8.4,
        overwrite=overwrite,
    )
    if not compile_pdf:
        return ()
    pdf = compile_plot_pdf(
        plot,
        output_pdf=output / "outer_face_skewness_upper_bound_paired.pdf",
        log_directory=logs,
        overwrite=overwrite,
    ).pdf
    return (("Paired outer face final skewness upper bounds", pdf),)


def _write_boxplots(
    rows: list[dict[str, object]],
    output: Path,
    logs: Path,
    *,
    overwrite: bool,
    compile_pdf: bool,
) -> tuple[tuple[str, Path], ...]:
    metrics = (
        ("skewness_lower_bound", "Certified skewness lower bound"),
        ("skewness_upper_bound", "Certified skewness upper bound"),
        (
            "normalized_inflation_upper_bound",
            "Normalized added skewness upper bound",
        ),
    )
    figures = []
    for metric, label in metrics:
        plot = write_boxplot(
            [
                BoxplotSeries(
                    partial.STRATEGY_LABELS[strategy].replace("--", "-"),
                    tuple(
                        float(row[metric])
                        for row in rows
                        if row["strategy"] == strategy and row[metric] is not None
                    ),
                    partial.STRATEGY_COLORS[strategy],
                )
                for strategy in partial.STRATEGIES
            ],
            output / metric,
            y_label=label,
            width_cm=16.0,
            height_cm=7.2,
            overwrite=overwrite,
        )
        if compile_pdf:
            pdf = compile_plot_pdf(
                plot,
                output_pdf=output / f"{metric}.pdf",
                log_directory=logs,
                overwrite=overwrite,
            ).pdf
            figures.append((label, pdf))
    return tuple(figures)


if __name__ == "__main__":
    main()
