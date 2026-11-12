#!/usr/bin/env python3
"""Generate and plot fixed-capacity-MCMCF-feasible SNDlib composites."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
import re
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from planar_instance_generation.core import (
    SndCouplingLinkConfig,
    SndInterNetworkDemandConfig,
    SndMergeConfig,
)
from planar_instance_generation.evaluation import (
    ensure_original_snd_plot,
    load_snd_source_pool,
    run_snd_experiment,
    save_snd_candidate,
)
from planar_instance_generation.gates import (
    DEFAULT_MAX_VARIABLES,
    GurobiMcmcfGate,
)


RESULTS_DIRECTORY = REPOSITORY_ROOT / "check/results"
DEFAULT_POOL = RESULTS_DIRECTORY / "sndlib_source_pool/manifest.json"
STRATEGIES = (
    "random",
    "high_degree",
    "kamada_kawai",
    "heuristic_outer_face",
    "exact_outer_face",
)
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--result-name", default="snd_generation")
    parser.add_argument(
        "--suffix",
        default=None,
        help=(
            "append this tag to every artifact stem so instances with the same "
            "node count from independent jobs stay distinct after consolidation"
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="accepted outputs requested per strategy",
    )
    parser.add_argument("--num-subinstances", type=int, nargs=2, default=[2, 5])
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=None,
        help=(
            "inclusive composite-node range; by default use the smallest "
            "loaded SND source size through 100 nodes"
        ),
    )
    parser.add_argument("--size-range", type=float, nargs=2, default=[0.15, 0.85])
    parser.add_argument("--max-attempts", type=int, default=25)
    parser.add_argument(
        "--coupling-count-rule", choices=("log2", "sqrt"), default="log2",
        help="merge pairs: min(n_min, max(3, ceil(rule(n_min))))",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=STRATEGIES,
        default=list(STRATEGIES),
    )
    parser.add_argument("--mcmcf-time-limit", type=float)
    parser.add_argument("--max-mcmcf-variables", type=int, default=DEFAULT_MAX_VARIABLES)
    parser.add_argument("--default-routing-cost", type=float, default=1.0)
    parser.add_argument("--exact-time-limit", type=float, default=600.0)
    parser.add_argument(
        "--outer-face-sampling",
        choices=("equidistant", "consecutive", "adaptive_skip_one"),
        default="consecutive",
        help=(
            "select face endpoints evenly, consecutively, or with as many "
            "balanced one-node gaps as boundary capacity permits"
        ),
    )
    parser.add_argument(
        "--outer-face-prefer-non-leaf",
        action="store_true",
        help=(
            "use degree-one face vertices only when non-leaf candidates are exhausted"
        ),
    )
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--application-threads", type=int, default=1)
    parser.add_argument("--coupling-module-capacity", type=float)
    parser.add_argument("--coupling-module-cost", type=float, default=0.0)
    parser.add_argument("--coupling-routing-cost", type=float, default=0.0)
    parser.add_argument("--coupling-setup-cost", type=float, default=0.0)
    parser.add_argument(
        "--inter-network-demand",
        type=float,
        help="optional demand on every coupling link; default: none",
    )
    parser.add_argument(
        "--mixed-coordinates-type",
        default="pixel",
        help="recorded output coordinate type when source metadata differs",
    )
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-original-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    run_directory = RESULTS_DIRECTORY / args.result_name
    if run_directory.exists():
        raise SystemExit(
            f"result run already exists: {run_directory}; choose another --result-name"
        )
    source_pool = load_snd_source_pool(
        args.source_pool.resolve(),
        path_root=REPOSITORY_ROOT,
    )
    if args.target_size is None:
        minimum_source_size = min(
            len(source.network.nodes) for source in source_pool
        )
        if minimum_source_size > 100:
            raise SystemExit("smallest SND source exceeds the default maximum of 100")
        args.target_size = [minimum_source_size, 100]

    run_directory.mkdir(parents=True)

    compile_pdfs = not args.no_compile
    if not args.no_original_plots:
        originals = RESULTS_DIRECTORY / "originals/sndlib"
        for source in source_pool:
            ensure_original_snd_plot(
                source.network,
                source.source_name,
                originals,
                compile_pdf=compile_pdfs,
            )

    attempt_root = run_directory / "logs/attempts"

    def attempt_directory(strategy: str, output: int, attempt: int) -> Path:
        return (
            attempt_root
            / strategy
            / f"output_{output:04d}"
            / f"attempt_{attempt:04d}"
        )

    def gate_factory(strategy: str, output: int, attempt: int) -> GurobiMcmcfGate:
        return GurobiMcmcfGate(
            max_variables=args.max_mcmcf_variables,
            time_limit=args.mcmcf_time_limit,
            threads=args.application_threads,
            seed=args.seed,
            default_routing_cost=args.default_routing_cost,
            verbose=args.verbose_solver,
            log_directory=attempt_directory(strategy, output, attempt) / "mcmcf",
        )

    coupling = SndCouplingLinkConfig(
        module_capacity=args.coupling_module_capacity,
        module_cost=args.coupling_module_cost,
        routing_cost=args.coupling_routing_cost,
        setup_cost=args.coupling_setup_cost,
    )
    inter_demand = (
        None
        if args.inter_network_demand is None
        else SndInterNetworkDemandConfig(value=args.inter_network_demand)
    )
    merge_config = SndMergeConfig(
        coupling=coupling,
        inter_network_demands=inter_demand,
    )
    started_at = datetime.now(timezone.utc)
    experiment = run_snd_experiment(
        source_pool,
        gate_factory,
        strategies=tuple(args.strategies),
        requested_per_strategy=args.count,
        num_subinstances=tuple(args.num_subinstances),
        target_size=tuple(args.target_size),
        size_range=tuple(args.size_range),
        max_attempts_per_output=args.max_attempts,
        seed=args.seed,
        merge_config=merge_config,
        mixed_coordinates_type=args.mixed_coordinates_type,
        exact_time_limit_seconds=args.exact_time_limit,
        exact_threads=args.threads,

        coupling_count_rule=args.coupling_count_rule,
        outer_face_sampling=args.outer_face_sampling,
        outer_face_prefer_non_leaf=args.outer_face_prefer_non_leaf,
        selection_log_directory=lambda strategy, output, attempt: str(
            attempt_directory(strategy, output, attempt) / "selection"
        ),
    )

    accepted_rows = []
    stage_rows = []
    extraction_rows = []
    shared_piece_paths_by_output = {}
    for accepted in experiment.accepted:
        final_node_count = len(accepted.candidate.instance.nodes)
        allocated_total = sum(
            piece.target_nodes for piece in accepted.candidate.pieces
        )
        if final_node_count != allocated_total:
            raise RuntimeError("accepted SND candidate violates its exact target")
        if not args.target_size[0] <= final_node_count <= args.target_size[1]:
            raise RuntimeError("accepted SND candidate violates target-size range")
        shared_piece_paths = shared_piece_paths_by_output.get(accepted.output_index)
        saved = save_snd_candidate(
            accepted.candidate,
            run_directory,
            compile_pdfs=compile_pdfs,
            stem_suffix=args.suffix,
            shared_piece_paths=shared_piece_paths,
        )
        shared_piece_paths_by_output.setdefault(
            accepted.output_index, saved.pieces
        )
        accepted_rows.append(
            {
                "output_index": accepted.output_index,
                "attempt_index": accepted.attempt_index,
                "artifact_stem": saved.final.stem,
                "instance_file": _display_path(saved.final.instance_file),
                "plot_directory": _display_path(saved.final.plot_directory),
                "log_directory": _display_path(saved.final.log_directory),
                "source_artifact_stems": ";".join(
                    path.stem for path in saved.pieces
                ),
                **accepted.candidate.statistics(),
            }
        )
        for piece, paths in zip(
            accepted.candidate.pieces,
            saved.pieces,
            strict=True,
        ):
            extraction_rows.append(
                {
                    "final_artifact_stem": saved.final.stem,
                    "subinstance_artifact_stem": paths.stem,
                    "subinstance_file": _display_path(paths.instance_file),
                    **piece.statistics(),
                }
            )
        for stage in accepted.candidate.stages:
            stage_rows.append(
                {
                    "final_artifact_stem": saved.final.stem,
                    **stage.statistics(),
                }
            )

    attempt_rows = []
    for record in experiment.attempts:
        row = asdict(record)
        statistics = row.pop("candidate_statistics")
        row["source_names"] = ";".join(row["source_names"])
        row["target_nodes"] = ";".join(str(value) for value in row["target_nodes"])
        row["realized_fractions"] = ";".join(
            format(value, ".17g") for value in row["realized_fractions"]
        )
        if statistics is not None:
            row.update(statistics)
        attempt_rows.append(row)
    stratum_rows = [
        {
            "routing_unit": stratum.routing_unit,
            "source_count": stratum.source_count,
            "source_names": ";".join(stratum.source_names),
            "sampling": "uniform with replacement",
        }
        for stratum in experiment.routing_unit_strata
    ]

    tables = run_directory / "tables"
    tables.mkdir()
    _write_csv(tables / "attempts.csv", attempt_rows)
    _write_csv(tables / "accepted_instances.csv", accepted_rows)
    _write_csv(tables / "merge_stages.csv", stage_rows)
    _write_csv(tables / "extractions.csv", extraction_rows)
    _write_csv(tables / "routing_unit_strata.csv", stratum_rows)
    from planar_instance_generation.core.ogdf import build_identity
    manifest = {
        "planarization": build_identity(),
        "schema_version": 1,
        "artifact_type": "sndlib_generation",
        "domain": "SNDlib fixed-capacity MCMCF",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": _run_configuration(args),
        "coupling": {
            **asdict(merge_config),
            "installed_reference_capacity_without_inter_demand": 0.0,
        },
        "source_sampling": {
            "scheme": "routing-unit-stratified uniform with replacement",
            "stratum_probability": "proportional to retained source count",
            "repeated_source_occurrences_are_distinct": True,
            "strata": stratum_rows,
        },
        "backend": {
            "gate": "Gurobi fixed-capacity MCMCF",
            "formulation": "continuous signed flow plus absolute-flow epigraph",
            "gurobipy_version": _package_version("gurobipy"),
            "networkx_version": _package_version("networkx"),
            "python_version": platform.python_version(),
        },
        "source_pool_size": len(source_pool),
        "accepted": accepted_rows,
        "failures": [
            {"strategy": strategy, "output_index": output}
            for strategy, output in experiment.failures
        ],
        "acceptance_rates": {
            strategy: experiment.acceptance_rate(strategy)
            for strategy in args.strategies
        },
        "tables": {
            "attempts": _display_path(tables / "attempts.csv"),
            "accepted_instances": _display_path(
                tables / "accepted_instances.csv"
            ),
            "merge_stages": _display_path(tables / "merge_stages.csv"),
            "extractions": _display_path(tables / "extractions.csv"),
            "routing_unit_strata": _display_path(
                tables / "routing_unit_strata.csv"
            ),
        },
    }
    manifest_path = run_directory / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
    print(
        f"Accepted {len(experiment.accepted)} composites; "
        f"unmet requests: {len(experiment.failures)}."
    )
    print(f"Results: {_display_path(run_directory)}")
    if experiment.failures:
        raise SystemExit(2)


def _validate_args(args: argparse.Namespace) -> None:
    if not _SAFE_NAME.fullmatch(args.result_name):
        raise SystemExit("--result-name must be a single safe filename component")
    if args.suffix is not None and not _SAFE_NAME.fullmatch(args.suffix):
        raise SystemExit("--suffix must be a single safe filename component")
    for name in (
        "count",
        "max_attempts",
        "threads",
        "max_mcmcf_variables",
    ):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if not 2 <= args.num_subinstances[0] <= args.num_subinstances[1] <= 7:
        raise SystemExit(
            "--num-subinstances must satisfy 2 <= minimum <= maximum <= 7"
        )
    if args.target_size is not None and not (
        1 <= args.target_size[0] <= args.target_size[1]
    ):
        raise SystemExit(
            "--target-size must contain positive nondecreasing bounds"
        )
    if args.seed < 0:
        raise SystemExit("--seed must be nonnegative")
    if len(args.strategies) != len(set(args.strategies)):
        raise SystemExit("--strategies must not contain duplicates")
    if (
        any(not math.isfinite(value) for value in args.size_range)
        or not 0.0 < args.size_range[0] <= args.size_range[1] <= 1.0
    ):
        raise SystemExit(
            "--size-range must satisfy 0 < minimum <= maximum <= 1"
        )
    if not math.isfinite(args.exact_time_limit) or args.exact_time_limit <= 0.0:
        raise SystemExit("--exact-time-limit must be finite and positive")
    if args.mcmcf_time_limit is not None and (
        not math.isfinite(args.mcmcf_time_limit)
        or args.mcmcf_time_limit <= 0.0
    ):
        raise SystemExit("--mcmcf-time-limit must be finite and positive")
    for name in (
        "coupling_module_cost",
        "coupling_routing_cost",
        "coupling_setup_cost",
        "default_routing_cost",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.coupling_module_capacity is not None and (
        not math.isfinite(args.coupling_module_capacity)
        or args.coupling_module_capacity <= 0.0
    ):
        raise SystemExit("--coupling-module-capacity must be finite and positive")
    if args.inter_network_demand is not None and (
        not math.isfinite(args.inter_network_demand)
        or args.inter_network_demand <= 0.0
    ):
        raise SystemExit("--inter-network-demand must be finite and positive")
    if not args.mixed_coordinates_type.strip():
        raise SystemExit("--mixed-coordinates-type must not be empty")


def _run_configuration(args: argparse.Namespace) -> dict[str, object]:
    return {
        "command": [str(Path(sys.argv[0]).resolve()), *sys.argv[1:]],
        "source_pool": _display_path(args.source_pool),
        "result_name": args.result_name,
        "suffix": args.suffix,
        "requested_per_strategy": args.count,
        "num_subinstances": args.num_subinstances,
        "target_size": args.target_size,
        "size_range": args.size_range,
        "max_attempts_per_output": args.max_attempts,
        "seed": args.seed,
        "strategies": args.strategies,
        "mcmcf_time_limit": args.mcmcf_time_limit,
        "max_mcmcf_variables": args.max_mcmcf_variables,
        "default_routing_cost": args.default_routing_cost,
        "exact_time_limit": args.exact_time_limit,
        "heuristic_iterations": 10,
        "coupling_count_rule": args.coupling_count_rule,
        "outer_face_sampling": args.outer_face_sampling,
        "outer_face_prefer_non_leaf": args.outer_face_prefer_non_leaf,
        "threads": args.threads,
        "mixed_coordinates_type": args.mixed_coordinates_type,
        "compile_pdfs": not args.no_compile,
        "original_plots": not args.no_original_plots,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        if not rows:
            return
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    main()
