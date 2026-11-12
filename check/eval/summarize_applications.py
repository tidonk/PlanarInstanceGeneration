#!/usr/bin/env python3
"""Compile post-generation application summaries into one CSV and JSON report."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIRECTORY = REPOSITORY_ROOT / "check/results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("pglib", "snd"))
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run-name", help="override the application result directory name")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIRECTORY)
    parser.add_argument(
        "--snd-method",
        choices=("standard", "adaptive_ml"),
        default="standard",
        help="select the standard or paired adaptive SND run",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="include manifest entries whose worker summary is not available yet",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suffix = "scopf" if args.dataset == "pglib" else "benders"
    run_name = args.run_name or f"{args.experiment}_{args.dataset}_{suffix}"
    if args.dataset == "snd" and args.snd_method == "adaptive_ml":
        run_name += "_adaptive_ml"
    root = args.results_dir / run_name
    manifest_path = root / "jobs.csv"
    if not manifest_path.is_file():
        raise SystemExit(f"missing application manifest: {manifest_path}")
    with manifest_path.open(encoding="utf-8", newline="") as stream:
        jobs = list(csv.DictReader(stream))

    rows: list[dict[str, object]] = []
    for job in jobs:
        summary_path = root / "logs" / job["artifact_stem"] / "summary.json"
        if not summary_path.is_file():
            if not args.allow_incomplete:
                raise SystemExit(f"missing application summary: {summary_path}")
            summary: dict[str, object] = {"state": "missing"}
        else:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        result = _mapping(summary.get("result"))
        time_data = _mapping(result.get("time"))
        callback = _mapping(result.get("callback"))
        network = _mapping(result.get("network"))
        contingencies = _mapping(result.get("contingencies"))
        scopf_feasible = _scopf_feasible(result)
        coupling = _mapping(result.get("rating_normalization_validation"))
        coupling_flow_multiplier = _coupling_flow_multiplier_to_mw(result, coupling)
        row: dict[str, object] = {
            **job,
            "instance_sha256": summary.get("instance_sha256"),
            "benders_network_design_version": summary.get(
                "benders_network_design_version"
            ),
            "benders_network_design_git_revision": summary.get(
                "benders_network_design_git_revision"
            ),
            "state": summary.get("state"),
            "started_at": summary.get("started_at"),
            "process_wall_seconds": summary.get("wall_seconds"),
            "termination_status": result.get("termination_status"),
            "primal_status": result.get("primal_status"),
            "has_finite_objective": result.get("has_finite_objective"),
            "scopf_feasible": scopf_feasible,
            "scopf_feasible_without_overload": result.get(
                "scopf_feasible_without_overload"
            ),
            "overload_cost_all_contingencies": result.get(
                "overload_cost_all_contingencies"
            ),
            "overload_cost_base_case": result.get("overload_cost_base_case"),
            "objective_value": result.get("objective_value"),
            "objective_bound": _finite_solver_number(result.get("objective_bound")),
            "relative_gap": _finite_solver_number(result.get("relative_gap")),
            "benders_iterations": result.get("iterations"),
            "solver_node_count": result.get("node_count"),
            "adapter_wall_seconds": result.get("wall_seconds"),
            "solver_total_seconds": _first_non_none(
                time_data.get("total"), result.get("solver_seconds")
            ),
            "solver_build_seconds": time_data.get("build"),
            "solver_solve_seconds": time_data.get("solve"),
            "solver_warmstart_seconds": time_data.get("warmstart", 0),
            "solver_solve_with_warmstart_seconds": (
                time_data["solve"] + time_data.get("warmstart", 0)
                if isinstance(time_data.get("solve"), (int, float))
                and isinstance(time_data.get("warmstart", 0), (int, float)) else None
            ),
            "master_problem_seconds": _first_non_none(
                time_data.get("MP"), result.get("master_seconds")
            ),
            "subproblem_seconds": _first_non_none(
                time_data.get("SP"), result.get("subproblem_seconds")
            ),
            "callback_seconds": result.get("callback_seconds"),
            "callback_count": callback.get("n_callbacks"),
            "subproblems_solved": _first_non_none(
                callback.get("n_solved_subs_total"),
                result.get("total_subproblems_solved"),
            ),
            "benders_cuts_found": _first_non_none(
                callback.get("n_benders_cuts_found"), result.get("total_cuts_found")
            ),
            "benders_cuts_added": _first_non_none(
                callback.get("n_benders_cuts_added"), result.get("total_cuts_added")
            ),
            "node_count": _first_non_none(
                network.get("buses"), result.get("network_nodes")
            ),
            "branch_count": _first_non_none(
                network.get("branches"), result.get("network_links")
            ),
            "generator_count": network.get("generators"),
            "demand_count": result.get("network_demands"),
            "outage_candidates": _first_non_none(
                contingencies.get("single_branch_candidates"),
                result.get("outage_candidates"),
            ),
            "outage_scenarios_retained": _first_non_none(
                contingencies.get("retained_single_branch_failures"),
                result.get("outage_scenarios_retained"),
            ),
            "outage_scenarios_dropped": _first_non_none(
                contingencies.get("critical_or_disconnecting_failures_dropped"),
                result.get("disconnecting_outages_dropped"),
            ),
            "base_case_subproblem_entries": contingencies.get(
                "base_case_subproblem_entries"
            ),
            "base_case_in_master": result.get("base_case_in_master"),
            "scenario_generation_seconds": result.get("scenario_generation_seconds"),
            "time_limit_verified": result.get("time_limit_verified"),
            "expected_coupling_branch_count": coupling.get(
                "coupling_branch_count"
            ),
            "coupling_branch_count": coupling.get("observed_coupling_branch_count"),
            "coupling_branch_observation_fraction": coupling.get(
                "observed_coupling_branch_fraction"
            ),
            "positive_flow_branch_count": coupling.get(
                "positive_flow_branch_count"
            ),
            "positive_flow_branch_fraction": coupling.get(
                "positive_flow_branch_fraction"
            ),
            "coupling_mean_absolute_flow_mw": _scaled_number(
                coupling.get("mean_absolute_flow_mw"), coupling_flow_multiplier
            ),
            "coupling_median_absolute_flow_mw": _scaled_number(
                coupling.get("median_absolute_flow_mw"), coupling_flow_multiplier
            ),
            "coupling_p95_absolute_flow_mw": _scaled_number(
                coupling.get("p95_absolute_flow_mw"), coupling_flow_multiplier
            ),
            "coupling_maximum_absolute_flow_mw": _scaled_number(
                coupling.get("maximum_absolute_flow_mw"), coupling_flow_multiplier
            ),
            "expected_noncoupling_branch_count": coupling.get(
                "noncoupling_branch_count"
            ),
            "noncoupling_branch_count": coupling.get(
                "observed_noncoupling_branch_count"
            ),
            "noncoupling_branch_observation_fraction": coupling.get(
                "observed_noncoupling_branch_fraction"
            ),
            "noncoupling_mean_absolute_flow_mw": _scaled_number(
                coupling.get("noncoupling_mean_absolute_flow_mw"),
                coupling_flow_multiplier,
            ),
            "coupling_maximum_utilization": _finite_number(
                coupling.get("maximum_utilization")
            ),
            "coupling_rating_empirically_nonbinding": coupling.get(
                "empirically_nonbinding"
            ),
            "time_limit_seconds": _mapping(summary.get("configuration")).get(
                "time_limit_seconds"
            ),
            "summary_file": str(summary_path.relative_to(REPOSITORY_ROOT)),
        }
        rows.append(row)

    fieldnames = tuple(rows[0])
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    state_counts = Counter(str(row["state"]) for row in rows)
    method_counts = Counter(str(row["strategy"]) for row in rows)
    complete_count = sum(row["state"] == "complete" for row in rows)
    finite_incumbent_count = sum(
        row["state"] == "complete" and row["has_finite_objective"] is True
        for row in rows
    )
    scopf_feasible_count = sum(
        row["state"] == "complete" and row["scopf_feasible"] is True
        for row in rows
    )
    scopf_feasible_without_overload_count = sum(
        row["state"] == "complete" and row["scopf_feasible_without_overload"] is True
        for row in rows
    )
    application_feasible_count = (
        scopf_feasible_count if args.dataset == "pglib" else finite_incumbent_count
    )
    feasibility_definition = (
        "certified finite SCOPF incumbent (overload slack permitted)"
        if args.dataset == "pglib"
        else "finite incumbent feasible for all retained N-1 outages"
    )
    aggregate = {
        "dataset": args.dataset,
        "experiment": args.experiment,
        "application_run": root.name,
        "evaluation_only": True,
        "job_count": len(rows),
        "complete_count": complete_count,
        "application_feasibility_definition": feasibility_definition,
        "application_feasible_count": application_feasible_count,
        "application_feasible_fraction_of_complete": (
            application_feasible_count / complete_count if complete_count else None
        ),
        "application_feasible_fraction_of_submitted": (
            application_feasible_count / len(rows) if rows else None
        ),
        "finite_incumbent_count": finite_incumbent_count,
        "finite_incumbent_fraction_of_complete": (
            finite_incumbent_count / complete_count if complete_count else None
        ),
        "scopf_feasible_count": scopf_feasible_count,
        "scopf_feasible_fraction_of_complete": (
            scopf_feasible_count / complete_count if complete_count else None
        ),
        "scopf_feasible_without_overload_count": scopf_feasible_without_overload_count,
        "scopf_feasible_without_overload_fraction_of_complete": (
            scopf_feasible_without_overload_count / complete_count
            if complete_count
            else None
        ),
        "per_strategy": _strategy_statistics(rows),
        "state_counts": dict(sorted(state_counts.items())),
        "strategy_counts": dict(sorted(method_counts.items())),
        "summary_csv": str((root / "summary.csv").relative_to(REPOSITORY_ROOT)),
    }
    (root / "summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _first_non_none(*values: object) -> object:
    return next((value for value in values if value is not None), None)


def _scopf_feasible(result: dict[str, object]) -> bool | None:
    explicit = result.get("scopf_feasible")
    if isinstance(explicit, bool):
        return explicit
    if result.get("application") != "DC_SCOTS_TIME_SLACK_BENDERS":
        return None
    # Schema versions before v4 did not expose this field. A finite objective
    # with FEASIBLE_POINT is the conservative legacy equivalent. In particular,
    # callback CUT_ERROR runs are not treated as certified feasible incumbents.
    return (
        result.get("has_finite_objective") is True
        and result.get("primal_status") == "FEASIBLE_POINT"
    )


def _finite_solver_number(value: object) -> float | None:
    number = _finite_number(value)
    if number is None or abs(number) >= 1.0e90:
        return None
    return number


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _coupling_flow_multiplier_to_mw(
    result: dict[str, object], coupling: dict[str, object]
) -> float | None:
    unit = coupling.get("flow_unit")
    if unit == "MW":
        return 1.0
    legacy_scopf_per_unit = (
        unit == "per_unit"
        or (
            unit is None
            and result.get("application") == "DC_SCOTS_TIME_SLACK_BENDERS"
            and result.get("schema_version") == 2
        )
    )
    if not legacy_scopf_per_unit:
        return 1.0
    normalization = _mapping(result.get("rating_normalization"))
    system_scale = _mapping(normalization.get("system_scale_components_mva"))
    return _finite_number(system_scale.get("base_mva"))


def _scaled_number(value: object, multiplier: float | None) -> float | None:
    number = _finite_number(value)
    if number is None or multiplier is None:
        return None
    return number * multiplier


def _percentile(ordered: list[float], probability: float) -> float:
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _describe(values: list[object]) -> dict[str, float | int] | None:
    numbers = sorted(
        number for value in values if (number := _finite_number(value)) is not None
    )
    if not numbers:
        return None
    positive = [number for number in numbers if number > 0.0]
    description: dict[str, float | int] = {
        "count": len(numbers),
        "minimum": numbers[0],
        "q25": _percentile(numbers, 0.25),
        "median": _percentile(numbers, 0.5),
        "q75": _percentile(numbers, 0.75),
        "maximum": numbers[-1],
        "mean": sum(numbers) / len(numbers),
    }
    if all(number >= 0.0 for number in numbers):
        description["shifted_geometric_mean"] = math.exp(
            sum(math.log1p(number) for number in numbers) / len(numbers)
        ) - 1.0
    if len(positive) == len(numbers):
        description["geometric_mean"] = math.exp(
            sum(math.log(number) for number in positive) / len(positive)
        )
    return description


def _strategy_statistics(rows: list[dict[str, object]]) -> dict[str, object]:
    statistics: dict[str, object] = {}
    for strategy in sorted({str(row["strategy"]) for row in rows}):
        selected = [row for row in rows if row["strategy"] == strategy]
        completed = [row for row in selected if row["state"] == "complete"]
        finite_incumbents = [
            row for row in completed if row["has_finite_objective"] is True
        ]
        scopf_feasible = [
            row for row in completed if row["scopf_feasible"] is True
        ]
        scopf_feasible_without_overload = [
            row
            for row in completed
            if row["scopf_feasible_without_overload"] is True
        ]
        coupling_rows = [
            row
            for row in completed
            if _finite_number(row["coupling_branch_count"]) is not None
            and _finite_number(row["positive_flow_branch_count"]) is not None
            and row["scopf_feasible"] is True
        ]
        coupling_branch_count = sum(
            int(row["coupling_branch_count"]) for row in coupling_rows
        )
        positive_flow_branch_count = sum(
            int(row["positive_flow_branch_count"]) for row in coupling_rows
        )
        statistics[strategy] = {
            "job_count": len(selected),
            "complete_count": len(completed),
            "finite_incumbent_count": len(finite_incumbents),
            "finite_incumbent_fraction_of_complete": (
                len(finite_incumbents) / len(completed) if completed else None
            ),
            "finite_incumbent_fraction_of_submitted": (
                len(finite_incumbents) / len(selected) if selected else None
            ),
            "scopf_feasible_count": len(scopf_feasible),
            "scopf_feasible_fraction_of_complete": (
                len(scopf_feasible) / len(completed) if completed else None
            ),
            "scopf_feasible_fraction_of_submitted": (
                len(scopf_feasible) / len(selected) if selected else None
            ),
            "scopf_feasible_without_overload_count": len(
                scopf_feasible_without_overload
            ),
            "scopf_feasible_without_overload_fraction_of_complete": (
                len(scopf_feasible_without_overload) / len(completed)
                if completed
                else None
            ),
            "termination_status_counts": dict(
                sorted(Counter(str(row["termination_status"]) for row in completed).items())
            ),
            "process_wall_seconds": _describe(
                [row["process_wall_seconds"] for row in completed]
            ),
            "adapter_wall_seconds": _describe(
                [row["adapter_wall_seconds"] for row in completed]
            ),
            "solver_total_seconds": _describe(
                [row["solver_total_seconds"] for row in completed]
            ),
            "solver_solve_seconds": _describe(
                [row["solver_solve_seconds"] for row in completed]
            ),
            "objective_bound": _describe(
                [row["objective_bound"] for row in completed]
            ),
            "relative_gap": _describe(
                [row["relative_gap"] for row in completed]
            ),
            "benders_iterations": _describe(
                [row["benders_iterations"] for row in completed]
            ),
            "master_problem_seconds": _describe(
                [row["master_problem_seconds"] for row in completed]
            ),
            "subproblem_seconds": _describe(
                [row["subproblem_seconds"] for row in completed]
            ),
            "callback_seconds": _describe(
                [row["callback_seconds"] for row in completed]
            ),
            "subproblems_solved": _describe(
                [row["subproblems_solved"] for row in completed]
            ),
            "benders_cuts_found": _describe(
                [row["benders_cuts_found"] for row in completed]
            ),
            "benders_cuts_added": _describe(
                [row["benders_cuts_added"] for row in completed]
            ),
            "outage_scenarios_retained": _describe(
                [row["outage_scenarios_retained"] for row in completed]
            ),
            "outage_scenarios_dropped": _describe(
                [row["outage_scenarios_dropped"] for row in completed]
            ),
            "coupling_branches_observed": coupling_branch_count,
            "coupling_branches_with_positive_flow": positive_flow_branch_count,
            "positive_coupling_flow_fraction": (
                positive_flow_branch_count / coupling_branch_count
                if coupling_branch_count
                else None
            ),
            "coupling_mean_absolute_flow_mw": _describe(
                [row["coupling_mean_absolute_flow_mw"] for row in completed]
            ),
            "coupling_p95_absolute_flow_mw": _describe(
                [row["coupling_p95_absolute_flow_mw"] for row in completed]
            ),
            "coupling_maximum_absolute_flow_mw": _describe(
                [row["coupling_maximum_absolute_flow_mw"] for row in completed]
            ),
            "noncoupling_mean_absolute_flow_mw": _describe(
                [row["noncoupling_mean_absolute_flow_mw"] for row in completed]
            ),
            "coupling_maximum_utilization": _describe(
                [row["coupling_maximum_utilization"] for row in completed]
            ),
        }
    return statistics


if __name__ == "__main__":
    main()
