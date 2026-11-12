#!/usr/bin/env python3
"""Generate and plot DC-feasible PGLib composites from a cached source pool."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import re
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from planar_instance_generation.core import (
    CouplingBranchConfig,
)
from planar_instance_generation.evaluation import (
    ensure_original_power_plot,
    load_pglib_source_pool,
    run_power_experiment,
    save_power_candidate,
)
from planar_instance_generation.gates import InnoptemDcOpfGate
from planar_instance_generation.io import read_pglib


RESULTS_DIRECTORY = REPOSITORY_ROOT / "check/results"
DEFAULT_POOL = RESULTS_DIRECTORY / "pglib_source_pool_innoptem/manifest.json"
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
    parser.add_argument("--result-name", default="pglib_generation")
    parser.add_argument(
        "--suffix",
        default=None,
        help=(
            "append this tag to every artifact stem so instances with the same "
            "bus count from independent jobs stay distinct after consolidation"
        ),
    )
    parser.add_argument("--count", type=int, default=10, help="accepted outputs per strategy")
    parser.add_argument("--num-subinstances", type=int, nargs=2, default=[2, 5])
    parser.add_argument(
        "--min-source-nodes",
        type=int,
        default=1,
        help="discard PGLib inputs below this bus count before source selection",
    )
    parser.add_argument("--target-size", type=int, nargs=2, default=[100, 1_000])
    parser.add_argument("--size-range", type=float, nargs=2, default=[0.15, 0.85])
    parser.add_argument("--max-attempts", type=int, default=25)
    parser.add_argument(
        "--coupling-count-rule", choices=("log2", "sqrt"), default="log2",
        help="merge pairs: min(n_min, max(3, ceil(rule(n_min))))",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--opf-time-limit", type=float)
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
    parser.add_argument("--threads", type=int, default=16, help="OGDF heuristic thread limit")
    parser.add_argument("--opf-threads", type=int, default=1)
    parser.add_argument("--coupling-reactance", type=float, default=1e-4)
    parser.add_argument("--coupling-rate", type=float)
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-original-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    run_directory = RESULTS_DIRECTORY / args.result_name
    manifest_path = run_directory / "manifest.json"
    if run_directory.exists():
        raise SystemExit(
            f"result run already exists: {run_directory}; choose another --result-name"
        )
    from planar_instance_generation.gates.innoptem import BACKEND
    source_record = json.loads(args.source_pool.read_text())
    if source_record.get("backend", {}).get("name") != BACKEND:
        raise SystemExit("Source dispatches must be recomputed with preprocess_pglib.py and InnOpTEM DC_OPF.")
    if source_record["backend"] != InnoptemDcOpfGate().identity():
        raise SystemExit("Source dispatches use a different OPF implementation. Recompute the source pool.")
    loaded_source_pool = load_pglib_source_pool(
        args.source_pool.resolve(), path_root=REPOSITORY_ROOT
    )
    source_pool = tuple(
        source
        for source in loaded_source_pool
        if int(source.case.bus.shape[0]) >= args.min_source_nodes
    )
    if not source_pool:
        raise SystemExit(
            "--min-source-nodes removed every source from the PGLib pool"
        )
    run_directory.mkdir(parents=True)

    compile_pdfs = not args.no_compile
    if not args.no_original_plots:
        originals = RESULTS_DIRECTORY / "originals/pglib"
        for source in source_pool:
            ensure_original_power_plot(
                source.case,
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

    def gate_factory(strategy: str, output: int, attempt: int) -> InnoptemDcOpfGate:
        return InnoptemDcOpfGate(
            time_limit=args.opf_time_limit,
            solver_params={"Threads": args.opf_threads},
            verbose=args.verbose_solver,
            log_path=attempt_directory(strategy, output, attempt) / "gurobi.log",
        )

    # Initialize Julia and compile the common OPF path before timing methods.
    # Keep the cached source dispatches; this solve is only runtime setup.
    setup = InnoptemDcOpfGate(
        time_limit=args.opf_time_limit,
        solver_params={"Threads": args.opf_threads},
        log_path=run_directory / "logs/opf_initialization.log",
    ).solve(source_pool[0].case)
    opf_initialization = {
        "source": source_pool[0].case.name,
        "runtime_seconds": setup.runtime_seconds,
        "optimizer_seconds": setup.optimizer_seconds,
        "status": setup.status,
        "feasible": setup.feasible,
        "included_in_method_times": False,
    }

    experiment = run_power_experiment(
        source_pool,
        gate_factory,
        strategies=tuple(args.strategies),
        requested_per_strategy=args.count,
        num_subinstances=tuple(args.num_subinstances),
        target_size=tuple(args.target_size),
        size_range=tuple(args.size_range),
        max_attempts_per_output=args.max_attempts,
        seed=args.seed,
        coupling=CouplingBranchConfig(
            reactance_pu=args.coupling_reactance,
            rate_mva=args.coupling_rate,
        ),
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
        final_bus_count = int(accepted.candidate.case.bus.shape[0])
        allocated_total = sum(
            piece.target_nodes for piece in accepted.candidate.pieces
        )
        if final_bus_count != allocated_total:
            raise RuntimeError("accepted PGLib candidate violates its exact target")
        if not args.target_size[0] <= final_bus_count <= args.target_size[1]:
            raise RuntimeError("accepted PGLib candidate violates target-size range")
        strategy = accepted.candidate.strategy
        gate_log = (
            attempt_directory(strategy, accepted.output_index, accepted.attempt_index)
            / "gurobi.log"
        )
        shared_piece_paths = shared_piece_paths_by_output.get(accepted.output_index)
        saved = save_power_candidate(
            accepted.candidate,
            run_directory,
            compile_pdfs=compile_pdfs,
            gate_log_path=gate_log,
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
                "source_artifact_stems": ";".join(path.stem for path in saved.pieces),
                **accepted.candidate.statistics(),
            }
        )
        for piece, paths in zip(
            accepted.candidate.pieces, saved.pieces, strict=True
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

    tables = run_directory / "tables"
    tables.mkdir(exist_ok=True)
    _write_csv(tables / "attempts.csv", attempt_rows)
    _write_csv(tables / "accepted_instances.csv", accepted_rows)
    _write_csv(tables / "merge_stages.csv", stage_rows)
    _write_csv(tables / "extractions.csv", extraction_rows)
    _validate_final_instance_tree(
        run_directory,
        accepted_rows,
        target_size=tuple(args.target_size),
    )
    from planar_instance_generation.core.ogdf import build_identity
    manifest = {
        "opf_initialization": opf_initialization,
        "planarization": build_identity(),
        "schema_version": 1,
        "domain": "PGLib DC power flow",
        "run": vars(args) | {"source_pool": _display_path(args.source_pool)},
        "backend": {
            "gate": "InnOpTEM DC_OPF",
            "opf": InnoptemDcOpfGate().identity(),
            "branch_switching": False,
            "gurobipy_version": _package_version("gurobipy"),
        },
        "source_pool_size": len(source_pool),
        "unfiltered_source_pool_size": len(loaded_source_pool),
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
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(
        f"Accepted {len(experiment.accepted)} composites; "
        f"unmet requests: {len(experiment.failures)}."
    )
    print(f"Results: {_display_path(run_directory)}")
    if experiment.failures:
        raise SystemExit(2)


def _validate_final_instance_tree(
    run_directory: Path,
    accepted_rows: list[dict[str, object]],
    *,
    target_size: tuple[int, int],
) -> None:
    """Require ``instances/`` to contain exactly the in-range final cases."""

    instance_directory = run_directory / "instances"
    files = sorted(instance_directory.glob("*.m"))
    expected_stems = {str(row["artifact_stem"]) for row in accepted_rows}
    actual_stems = {path.stem for path in files}
    if actual_stems != expected_stems:
        raise RuntimeError(
            "PGLib instances/ must contain final composites only; "
            f"expected={sorted(expected_stems)}, actual={sorted(actual_stems)}"
        )
    minimum, maximum = target_size
    for path in files:
        bus_count = int(read_pglib(path).bus.shape[0])
        if not minimum <= bus_count <= maximum:
            raise RuntimeError(
                f"PGLib final artifact {path} has {bus_count} buses; "
                f"expected {minimum}..{maximum}"
            )


def _validate_args(args: argparse.Namespace) -> None:
    if not _SAFE_NAME.fullmatch(args.result_name):
        raise SystemExit("--result-name must be a single safe filename component")
    if args.suffix is not None and not _SAFE_NAME.fullmatch(args.suffix):
        raise SystemExit("--suffix must be a single safe filename component")
    for name in (
        "count",
        "max_attempts",
        "threads",
    ):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if not 2 <= args.num_subinstances[0] <= args.num_subinstances[1] <= 7:
        raise SystemExit(
            "--num-subinstances must satisfy 2 <= minimum <= maximum <= 7"
        )
    if not 1 <= args.target_size[0] <= args.target_size[1]:
        raise SystemExit(
            "--target-size must contain positive nondecreasing bounds"
        )
    if args.min_source_nodes < 1:
        raise SystemExit("--min-source-nodes must be positive")
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
    if not math.isfinite(args.exact_time_limit) or args.exact_time_limit <= 0:
        raise SystemExit("--exact-time-limit must be finite and positive")
    if args.opf_time_limit is not None and (
        not math.isfinite(args.opf_time_limit) or args.opf_time_limit <= 0
    ):
        raise SystemExit("--opf-time-limit must be finite and positive")
    if not math.isfinite(args.coupling_reactance) or args.coupling_reactance <= 0:
        raise SystemExit("--coupling-reactance must be finite and positive")
    if args.coupling_rate is not None and (
        not math.isfinite(args.coupling_rate) or args.coupling_rate <= 0
    ):
        raise SystemExit("--coupling-rate must be finite and positive")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
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
