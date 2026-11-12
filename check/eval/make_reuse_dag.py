#!/usr/bin/env python3
"""Prepare a fresh, isolated 600/3600-second full experiment (no submission)."""

import argparse
import json
from pathlib import Path
import re

try:
    from .make_full_dag import STRATEGIES
except ImportError:  # Direct script execution.
    from make_full_dag import STRATEGIES


def source_jobs(experiment, source_files=None):
    root = Path(__file__).resolve().parents[2]
    paths = (
        tuple(sorted((root / "check/data/pglib").glob("*.m")))
        if source_files is None
        else tuple(source_files)
    )
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("source files must be nonempty and distinct")
    jobs = []
    for index, path in enumerate(paths, 1):
        if any(c in str(path) for c in '\n\r" '):
            raise ValueError("source paths must not contain whitespace or quotes")
        jobs.append(
            {
                "source": str(path),
                "result_name": f"{experiment}_pglib_source_{index:03d}",
            }
        )
    return jobs


def build_dag(
    experiment,
    settings,
    instances=100,
    datasets=("pglib", "snd"),
    prepare_sources=False,
    source_files=None,
):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", experiment):
        raise ValueError("unsafe experiment name")
    datasets = tuple(datasets)
    if (
        not datasets
        or len(set(datasets)) != len(datasets)
        or any(dataset not in {"pglib", "snd"} for dataset in datasets)
    ):
        raise ValueError("datasets must contain distinct supported data sets")
    if any(c in str(settings) for c in '\n\r" '):
        raise ValueError("settings path must not contain whitespace or quotes")
    lines = [
        "# 600 s merge planarization; 3600 s final and shared-piece exact solves.",
        "# All five rules share saved generation certificates and exact piece results.",
    ]
    if prepare_sources and "pglib" in datasets:
        lines += [
            "JOB sources_pglib check/eval/pglib_source_pool.sub",
            f'VARS sources_pglib experiment="{experiment}" source_jobs="check/eval/{experiment}_sources.json"',
            "PRIORITY sources_pglib 50",
        ]
        for index, job in enumerate(source_jobs(experiment, source_files), 1):
            node = f"source_pglib_{index:03d}"
            lines += [
                f"JOB {node} check/eval/pglib_sources.sub",
                f'VARS {node} experiment="{experiment}" settings_file="{settings}" source_id="{index:03d}" source="{job["source"]}" result_name="{job["result_name"]}"',
                f"PRIORITY {node} 50",
                f"PARENT {node} CHILD sources_pglib",
            ]
    for dataset in datasets:
        data, summary = f"data_{dataset}", f"summary_{dataset}"
        lines += [
            f"JOB {data} check/eval/report.sub",
            f'VARS {data} experiment="{experiment}" dataset="{dataset}" settings_file="{settings}" report_cpus="16" max_runtime="1209600" data_only="1"',
            f"JOB {summary} check/eval/application_summary.sub",
            f'VARS {summary} experiment="{experiment}" dataset="{dataset}"',
            # Assemble the complete input index after all individual arrays finish.
            f"SCRIPT PRE {summary} /usr/bin/env PIG_SETTINGS_FILE={settings} check/eval/prepare_application.sh {dataset} {experiment}",
            f"PARENT {data} CHILD {summary}",
        ]
        for index in range(1, instances + 1):
            generation = f"gen_{dataset}_i{index:03d}"
            lines += [
                f"JOB {generation} check/eval/pglib_one.sub",
                f'VARS {generation} experiment="{experiment}" dataset="{dataset}" instance="{index}" seed="{index}" sampling="consecutive" cpus="16" settings_file="{settings}" max_runtime="172800"',
            ]
            if prepare_sources and dataset == "pglib":
                lines.append(f"PARENT sources_pglib CHILD {generation}")
            application = f"app_{dataset}_i{index:03d}"
            jobs_file = (
                f"check/eval/application_jobs/{experiment}/{dataset}_i{index:03d}.jobs"
            )
            lines += [
                f"JOB {application} check/eval/{dataset}_application.sub",
                f'VARS {application} experiment="{experiment}" settings_file="{settings}" jobs_file="{jobs_file}" application_time_limit="10800" topological_action_limit="1" max_runtime="28800" job_priority="0"',
                f"SCRIPT PRE {application} /usr/bin/env PIG_SETTINGS_FILE={settings} check/eval/prepare_application.sh {dataset} {experiment} {index}",
                f"PARENT {generation} CHILD {application}",
                f"PARENT {application} CHILD {summary}",
            ]
            skewness_submit = (
                "pglib_cohort_skewness.sub"
                if dataset == "pglib"
                else "snd_cohort_skewness.sub"
            )
            for strategy in STRATEGIES:
                skew = f"skew_{dataset}_i{index:03d}_{strategy}"
                lines += [
                    f"JOB {skew} check/eval/{skewness_submit}",
                    f'VARS {skew} experiment="{experiment}" output_name="{experiment}_{dataset}_exact" instance="{index}" seed="0" strategy="{strategy}" time_limit="3600" threads="16" cpus="16" max_runtime="86400" hold_margin="3600" job_priority="100"',
                    f"PRIORITY {skew} 100",
                    f"PARENT {generation} CHILD {skew}",
                    f"PARENT {skew} CHILD {data}",
                ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument(
        "--dag-name",
        help="different workflow filename when replacing an earlier submission",
    )
    parser.add_argument(
        "--dataset",
        choices=("both", "pglib", "snd"),
        default="both",
        help="prepare both experiment branches or only the selected data set",
    )
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument(
        "--prepare-sources",
        action="store_true",
        help="screen fresh InnOpTEM source dispatches before PGLib generation",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    settings = (
        args.settings.resolve()
        if args.settings
        else root / "check/eval/ogdf_600_3600.env"
    )
    datasets = ("pglib", "snd") if args.dataset == "both" else (args.dataset,)
    instances = args.instances
    sources = tuple(sorted((root / "check/data/pglib").glob("*.m")))
    text = build_dag(
        args.experiment,
        settings,
        instances=instances,
        datasets=datasets,
        prepare_sources=args.prepare_sources,
        source_files=sources,
    )
    dag_name = args.dag_name or args.experiment
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", dag_name):
        raise SystemExit("unsafe workflow name")
    output = root / "check/eval" / f"{dag_name}.dag"
    if output.exists() or any(
        (root / "check/results" / f"{args.experiment}_{dataset}_job_i001").exists()
        for dataset in datasets
    ):
        raise SystemExit("choose a fresh experiment name; existing run is preserved")
    for dataset in datasets:
        for relative in ("", "skewness", "application", "sources"):
            (root / "check/eval/logs" / args.experiment / dataset / relative).mkdir(
                parents=True, exist_ok=True
            )
    if args.prepare_sources and "pglib" in datasets:
        source_list = root / "check/eval" / f"{args.experiment}_sources.json"
        if source_list.exists():
            raise SystemExit(
                "source job list already exists; choose a fresh experiment name"
            )
        source_list.write_text(
            json.dumps(source_jobs(args.experiment, sources), indent=2) + "\n"
        )
    output.write_text(text, encoding="utf-8")
    generation_jobs = instances * len(datasets)
    exact_jobs = instances * len(STRATEGIES) * len(datasets)
    branch_count = len(datasets)
    print(
        f"Prepared {output}: {generation_jobs} generation jobs, {exact_jobs} "
        f"distributed exact jobs, {branch_count} data processing jobs "
        f"and {generation_jobs} application arrays (one per generated set)."
    )


if __name__ == "__main__":
    main()
