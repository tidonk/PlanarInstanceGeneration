#!/usr/bin/env python3
"""Build a partial PGLib report with default OGDF heuristic bounds."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
from typing import Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from planar_instance_generation.core import ogdf_planar_subgraph
from planar_instance_generation.io import read_pglib
from planar_instance_generation.plt import (
    BoxplotSeries,
    compile_plot_pdf,
    write_boxplot,
)


RESULTS_DIRECTORY = REPOSITORY_ROOT / "check/results"
STRATEGIES = (
    "random",
    "high_degree",
    "kamada_kawai",
    "heuristic_outer_face",
    "exact_outer_face",
)
STRATEGY_LABELS = {
    "random": "Random",
    "high_degree": "High degree",
    "kamada_kawai": "Kamada--Kawai",
    "heuristic_outer_face": "Heuristic face",
    "exact_outer_face": "Exact face",
}
STRATEGY_COLORS = {
    "random": "0072B2",
    "high_degree": "E69F00",
    "kamada_kawai": "009E73",
    "heuristic_outer_face": "CC79A7",
    "exact_outer_face": "D55E00",
}
_JOB_SUFFIX = re.compile(r"_job_i(\d+)$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="generation experiment tag")
    parser.add_argument(
        "--output-name",
        default=None,
        help="result directory name (default: <experiment>_pglib_partial)",
    )
    parser.add_argument("--iterations", type=int, choices=(10,), default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cohorts", type=int, default=None)
    parser.add_argument(
        "--gallery",
        choices=("composites", "all"),
        default="all",
        help="include only cohort/source/final pages or every merge-stage plot",
    )
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_name = args.output_name or f"{args.experiment}_pglib_partial"
    if not _SAFE_NAME.fullmatch(args.experiment) or not _SAFE_NAME.fullmatch(
        output_name
    ):
        raise SystemExit("experiment and output names must be safe path components")
    if args.iterations < 1 or args.workers < 1:
        raise SystemExit("--iterations and --workers must be positive")
    if args.max_cohorts is not None and args.max_cohorts < 1:
        raise SystemExit("--max-cohorts must be positive")

    cohorts = _discover_cohorts(args.experiment)
    if args.max_cohorts is not None:
        cohorts = cohorts[: args.max_cohorts]
    if not cohorts:
        raise SystemExit("no complete five-strategy PGLib cohorts were found")

    output = RESULTS_DIRECTORY / output_name
    output.mkdir(parents=True, exist_ok=True)
    evaluation = output / "evaluation"
    boxplots = evaluation / "boxplots"
    logs = output / "logs/report"
    for directory in (evaluation, boxplots, logs):
        directory.mkdir(parents=True, exist_ok=True)

    task_specs = _task_specs(cohorts, args.iterations, args.seed)
    print(
        f"Computing default OGDF Fast10 bounds for "
        f"{len(task_specs)} graphs from {len(cohorts)} complete cohorts with "
        f"{args.workers} workers.",
        flush=True,
    )
    results = _solve(task_specs, args.workers)
    rows = _skewness_rows(cohorts, results, args.iterations)
    summaries = _summaries(rows)

    _write_csv(evaluation / "heuristic_skewness.csv", rows)
    _write_csv(evaluation / "strategy_summary.csv", summaries)
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
    gallery_tex = output / "plot_gallery.tex"
    gallery_pdf = output / "plot_gallery.pdf"
    if gallery_tex.exists() and not args.overwrite:
        raise SystemExit(f"gallery already exists: {gallery_tex}; pass --overwrite")
    gallery_tex.write_text(
        _gallery_tex(
            cohorts,
            figures,
            summaries,
            iterations=args.iterations,
            include_merge_stages=args.gallery == "all",
        ),
        encoding="utf-8",
    )
    if not args.no_compile:
        _compile_gallery(gallery_tex, gallery_pdf, logs)

    manifest = {
        "schema_version": 1,
        "experiment": args.experiment,
        "complete_cohorts": len(cohorts),
        "strategies": list(STRATEGIES),
        "skewness_method": "ogdf_default_fast10",
        "heuristic_iterations": args.iterations,
        "seed": args.seed,
        "certification": {
            "lower_bound": "componentwise Euler/nonplanarity bound",
            "upper_bound": "explicit planar-subgraph deletion certificate",
            "exact_solver_used": False,
        },
        "gallery_scope": args.gallery,
        "heuristic_skewness_csv": str(
            (evaluation / "heuristic_skewness.csv").relative_to(REPOSITORY_ROOT)
        ),
        "strategy_summary_csv": str(
            (evaluation / "strategy_summary.csv").relative_to(REPOSITORY_ROOT)
        ),
        "plot_gallery": (
            None
            if args.no_compile
            else str(gallery_pdf.relative_to(REPOSITORY_ROOT))
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Partial data: {evaluation.relative_to(REPOSITORY_ROOT)}", flush=True)
    if not args.no_compile:
        print(f"Scrollable gallery: {gallery_pdf.relative_to(REPOSITORY_ROOT)}")


class Cohort:
    __slots__ = ("index", "directory", "rows", "target_size", "source_stems")

    def __init__(
        self,
        index: int,
        directory: Path,
        rows: tuple[dict[str, str], ...],
    ) -> None:
        self.index = index
        self.directory = directory
        self.rows = rows
        self.target_size = int(rows[0]["target_size"])
        self.source_stems = tuple(
            value for value in rows[0]["source_artifact_stems"].split(";") if value
        )


def _discover_cohorts(
    experiment: str, *, index: int | None = None,
    use_configured_strategies: bool = False,
) -> list[Cohort]:
    cohorts_by_index: dict[int, Cohort] = {}
    prefix = f"{experiment}_pglib_job_i"
    archive = RESULTS_DIRECTORY / "_cohorts" / f"{experiment}_pglib"
    for root in (RESULTS_DIRECTORY, archive):
        directories = root.glob(f"{prefix}*") if index is None else (root / f"{prefix}{index:03d}",)
        for directory in directories:
            match = _JOB_SUFFIX.search(directory.name)
            accepted = directory / "tables/accepted_instances.csv"
            if match is None or not accepted.is_file():
                continue
            with accepted.open(encoding="utf-8", newline="") as stream:
                rows = tuple(csv.DictReader(stream))
            by_strategy = {row.get("strategy"): row for row in rows}
            strategies = STRATEGIES
            if use_configured_strategies:
                settings_path = directory / "manifest.json"
                if not settings_path.is_file():
                    continue
                settings = json.loads(settings_path.read_text(encoding="utf-8"))
                configured = settings.get("run", {}).get("strategies")
                if (not isinstance(configured, list) or not configured
                        or len(set(configured)) != len(configured)
                        or not set(configured).issubset(STRATEGIES)):
                    continue
                strategies = tuple(s for s in STRATEGIES if s in configured)
            if len(rows) != len(strategies) or set(by_strategy) != set(strategies):
                continue
            ordered = tuple(by_strategy[strategy] for strategy in strategies)
            if not all(_instance_path(row, directory).is_file() for row in ordered):
                continue
            found_index = int(match.group(1))
            previous = cohorts_by_index.get(found_index)
            if previous is not None and previous.directory != directory:
                raise SystemExit(
                    f"duplicate active and archived PGLib cohort i{found_index:03d}"
                )
            cohorts_by_index[found_index] = Cohort(found_index, directory, ordered)
    return [cohorts_by_index[index] for index in sorted(cohorts_by_index)]


def _instance_path(row: dict[str, str], directory: Path) -> Path:
    recorded = Path(row["instance_file"])
    candidate = recorded if recorded.is_absolute() else REPOSITORY_ROOT / recorded
    if candidate.is_file():
        return candidate
    return directory / "instances" / f"{row['artifact_stem']}.m"


def _task_specs(
    cohorts: Iterable[Cohort], iterations: int, base_seed: int
) -> dict[str, tuple[str, int, int]]:
    specs: dict[str, tuple[str, int, int]] = {}
    for cohort in cohorts:
        cohort_seed = _stable_seed(base_seed, cohort.index)
        for row in cohort.rows:
            stem = row["artifact_stem"]
            specs[f"final:{cohort.index}:{stem}"] = (
                str(_instance_path(row, cohort.directory)),
                iterations,
                cohort_seed,
            )
        for stem in cohort.source_stems:
            path = cohort.directory / "subinstances" / f"{stem}.m"
            if not path.is_file():
                raise SystemExit(f"missing extracted subinstance: {path}")
            specs[f"source:{cohort.index}:{stem}"] = (
                str(path),
                iterations,
                cohort_seed,
            )
    return specs


def _stable_seed(base_seed: int, cohort: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{cohort}".encode("ascii")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _solve_one(spec: tuple[str, int, int]) -> dict[str, object]:
    path, iterations, seed = spec
    graph = read_pglib(Path(path)).topology(active_only=True)
    result = ogdf_planar_subgraph(
        graph,
        threads=16,
        seed=seed,
    )
    return {
        "node_count": result.original_node_count,
        "edge_count": result.original_edge_count,
        "skewness_lower_bound": result.omitted_lower_bound,
        "skewness_upper_bound": result.omitted_upper_bound,
        "skewness_exact": result.exact,
        "absolute_gap": result.omitted_upper_bound - result.omitted_lower_bound,
        "status": result.termination_status,
        "runtime_seconds": result.total_runtime_seconds,
        "seed": result.seed,
    }


def _solve(
    specs: dict[str, tuple[str, int, int]], workers: int
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = {
            executor.submit(_solve_one, spec): key for key, spec in specs.items()
        }
        for completed, future in enumerate(as_completed(pending), start=1):
            key = pending[future]
            results[key] = future.result()
            if completed % 50 == 0 or completed == len(pending):
                print(f"  completed {completed}/{len(pending)} bounds", flush=True)
    return results


def _skewness_rows(
    cohorts: Iterable[Cohort],
    results: dict[str, dict[str, object]],
    iterations: int,
) -> list[dict[str, object]]:
    rows = []
    for cohort in cohorts:
        source_results = [
            results[f"source:{cohort.index}:{stem}"] for stem in cohort.source_stems
        ]
        floor_lower = sum(int(item["skewness_lower_bound"]) for item in source_results)
        floor_upper = sum(int(item["skewness_upper_bound"]) for item in source_results)
        for strategy, accepted in zip(STRATEGIES, cohort.rows, strict=True):
            stem = accepted["artifact_stem"]
            result = results[f"final:{cohort.index}:{stem}"]
            lower = int(result["skewness_lower_bound"])
            upper = int(result["skewness_upper_bound"])
            coupling = int(accepted["coupling_branch_count"])
            forest_edges = len(source_results) - 1
            if coupling < forest_edges:
                raise ValueError("connected composition has too few coupling edges")
            lower = max(lower, floor_lower)
            upper = min(upper, floor_upper + coupling - forest_edges)
            inflation_lower = max(0, lower - floor_upper)
            inflation_upper = min(coupling - forest_edges, upper - floor_lower)
            if inflation_lower > inflation_upper:
                raise RuntimeError(
                    f"inconsistent inflation bounds for cohort {cohort.index} {strategy}"
                )
            rows.append(
                {
                    "cohort": cohort.index,
                    "strategy": strategy,
                    "strategy_label": STRATEGY_LABELS[strategy].replace("--", "-"),
                    "instance": stem,
                    "target_size": cohort.target_size,
                    **result,
                    "solver_lower_bound": int(result["skewness_lower_bound"]),
                    "solver_upper_bound": int(result["skewness_upper_bound"]),
                    "skewness_lower_bound": lower,
                    "skewness_upper_bound": upper,
                    "skewness_exact": lower == upper,
                    "absolute_gap": upper - lower,
                    "source_piece_count": len(source_results),
                    "coupling_forest_edge_count": forest_edges,
                    "source_floor_lower_bound": floor_lower,
                    "source_floor_upper_bound": floor_upper,
                    "coupling_edge_count": coupling,
                    "inflation_lower_bound": inflation_lower,
                    "inflation_upper_bound": inflation_upper,
                    "normalized_inflation_lower_bound": (
                        inflation_lower / coupling if coupling else None
                    ),
                    "normalized_inflation_upper_bound": (
                        inflation_upper / coupling if coupling else None
                    ),
                    "heuristic_iterations": iterations,
                }
            )
    return rows


def _summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries = []
    for strategy in STRATEGIES:
        selected = [row for row in rows if row["strategy"] == strategy]
        lower = [float(row["skewness_lower_bound"]) for row in selected]
        upper = [float(row["skewness_upper_bound"]) for row in selected]
        normalized_upper = [
            float(row["normalized_inflation_upper_bound"]) for row in selected
        ]
        summaries.append(
            {
                "strategy": strategy,
                "strategy_label": STRATEGY_LABELS[strategy].replace("--", "-"),
                "count": len(selected),
                "exact_count": sum(bool(row["skewness_exact"]) for row in selected),
                "mean_lower_bound": statistics.mean(lower),
                "median_lower_bound": statistics.median(lower),
                "mean_upper_bound": statistics.mean(upper),
                "median_upper_bound": statistics.median(upper),
                "mean_absolute_gap": statistics.mean(
                    float(row["absolute_gap"]) for row in selected
                ),
                "mean_normalized_inflation_upper_bound": statistics.mean(
                    normalized_upper
                ),
                "mean_runtime_seconds": statistics.mean(
                    float(row["runtime_seconds"]) for row in selected
                ),
            }
        )
    return summaries


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
                    STRATEGY_LABELS[strategy].replace("--", "-"),
                    tuple(
                        float(row[metric])
                        for row in rows
                        if row["strategy"] == strategy and row[metric] is not None
                    ),
                    STRATEGY_COLORS[strategy],
                )
                for strategy in STRATEGIES
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


def _gallery_tex(
    cohorts: Iterable[Cohort],
    figures: tuple[tuple[str, Path], ...],
    summaries: list[dict[str, object]],
    *,
    iterations: int,
    include_merge_stages: bool,
) -> str:
    lines = [
        "\\documentclass[9pt,a4paper]{article}\n",
        "\\usepackage[T1]{fontenc}\n",
        "\\usepackage{lmodern}\n",
        "\\usepackage[margin=10mm,headheight=13pt]{geometry}\n",
        "\\usepackage{graphicx,booktabs,fancyhdr}\n",
        "\\pagestyle{fancy}\\fancyhf{}\\lhead{Fresh PGLib partial report}",
        "\\rhead{\\thepage}\n",
        "\\setlength{\\parindent}{0pt}\n",
        "\\begin{document}\n",
        "\\begin{center}\\LARGE\\bfseries Fresh PGLib partial report\\end{center}\n",
        f"Complete paired cohorts: {len(tuple(cohorts))}. Preliminary skewness uses "
        f"default OGDF Fast with ten runs.\\par\n",
        "\\vspace{3mm}\n",
        "\\begin{center}\\small\n",
        "\\begin{tabular}{lrrrr}\\toprule\n",
        "Strategy & Mean lower & Mean upper & Exact & Mean norm. infl. upper \\\\\n",
        "\\midrule\n",
    ]
    for row in summaries:
        lines.append(
            f"{_tex_escape(str(row['strategy_label']))} & "
            f"{float(row['mean_lower_bound']):.2f} & "
            f"{float(row['mean_upper_bound']):.2f} & "
            f"{int(row['exact_count'])}/{int(row['count'])} & "
            f"{float(row['mean_normalized_inflation_upper_bound']):.3f} \\\\\n"
        )
    lines.extend(["\\bottomrule\\end{tabular}\n", "\\end{center}\n"])
    for label, pdf in figures:
        lines.extend(
            [
                "\\begin{center}\n",
                f"\\includegraphics[width=.95\\textwidth,height=.24\\textheight,keepaspectratio]"
                f"{{{_pdf_path(pdf)}}}\\par\n",
                f"\\small\\bfseries {_tex_escape(label)}\n",
                "\\end{center}\n",
            ]
        )

    for cohort in cohorts:
        lines.extend(
            [
                "\\clearpage\n",
                f"\\section*{{Cohort i{cohort.index:03d}: extracted sources "
                f"(target {cohort.target_size} buses)}}\n",
            ]
        )
        source_pdfs = [
            (
                stem,
                cohort.directory / "plots/original" / stem / "full.pdf",
            )
            for stem in cohort.source_stems
        ]
        lines.extend(_image_grid(source_pdfs, height=0.32))

        lines.extend(
            [
                "\\clearpage\n",
                f"\\section*{{Cohort i{cohort.index:03d}: five final composites "
                f"({cohort.target_size} buses)}}\n",
            ]
        )
        final_pdfs = [
            (
                STRATEGY_LABELS[strategy],
                cohort.directory
                / "plots/composite"
                / row["artifact_stem"]
                / "full.pdf",
            )
            for strategy, row in zip(STRATEGIES, cohort.rows, strict=True)
        ]
        lines.extend(_image_grid(final_pdfs, height=0.25))

        if include_merge_stages:
            for strategy, row in zip(STRATEGIES, cohort.rows, strict=True):
                stage_root = (
                    cohort.directory / "plots/merged" / row["artifact_stem"]
                )
                stage_pdfs = [
                    (f"Merge stage {index}", path)
                    for index, path in enumerate(
                        sorted(stage_root.glob("stage_*/*.pdf")), start=1
                    )
                ]
                if not stage_pdfs:
                    continue
                lines.extend(
                    [
                        "\\clearpage\n",
                        f"\\section*{{Cohort i{cohort.index:03d}: "
                        f"{_tex_escape(STRATEGY_LABELS[strategy])} merge stages}}\n",
                    ]
                )
                lines.extend(_image_grid(stage_pdfs, height=0.34))
    lines.append("\\end{document}\n")
    return "".join(lines)


def _image_grid(images: Iterable[tuple[str, Path]], *, height: float) -> list[str]:
    items = tuple(images)
    lines = []
    for index, (label, pdf) in enumerate(items, start=1):
        if not pdf.is_file():
            raise SystemExit(f"missing worker-generated PDF: {pdf}")
        lines.extend(
            [
                "\\begin{minipage}[t]{.49\\textwidth}\\centering\n",
                f"\\includegraphics[width=.98\\linewidth,height={height:.3f}\\textheight,keepaspectratio]"
                f"{{{_pdf_path(pdf)}}}\\par\n",
                f"\\small\\bfseries {_tex_escape(label)}\n",
                "\\end{minipage}\n",
            ]
        )
        lines.append("\\hfill\n" if index % 2 else "\\par\\vspace{2mm}\n")
    return lines


def _compile_gallery(tex: Path, pdf: Path, logs: Path) -> None:
    executable = shutil.which("pdflatex")
    if executable is None:
        raise SystemExit("pdflatex is required to compile the gallery")
    process = subprocess.run(
        [
            executable,
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={logs}",
            tex.name,
        ],
        cwd=tex.parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    stdout = logs / "plot_gallery.stdout.log"
    stdout.write_text(process.stdout, encoding="utf-8")
    compiled = logs / pdf.name
    if process.returncode != 0 or not compiled.is_file():
        raise SystemExit(f"gallery compilation failed; see {stdout}")
    compiled.replace(pdf)


def _pdf_path(path: Path) -> str:
    return "\\detokenize{" + str(path.resolve()) + "}"


def _tex_escape(value: str) -> str:
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
    }
    return "".join(replacements.get(character, character) for character in value)


if __name__ == "__main__":
    main()
