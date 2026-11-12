#!/usr/bin/env python3
"""Export completed OF-B sampling comparisons and their input audit."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "check/results"
BASELINE = "merge_count_600_3600_20260911_log2"
PREFIX = "sampling_600_3600_20260927_log2"
POLICIES = {
    "consecutive": "Consecutive",
    "equidistant": "Equidistant",
    "adaptive_skip_one": "Adaptive skip-one",
    "prefer_non_leaf": r"Consecutive, prefer degree $>1$",
}


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_input(task):
    policy, index, accepted, directory = task
    settings = json.loads((directory / "manifest.json").read_text())["run"]
    assert settings["exact_time_limit"] == 600 and settings["coupling_count_rule"] == "log2"
    assert settings["outer_face_sampling"] == ("consecutive" if policy == "prefer_non_leaf" else policy)
    assert settings["outer_face_prefer_non_leaf"] == (policy == "prefer_non_leaf")
    metadata = json.loads((directory / "plots/composite" / accepted["artifact_stem"] / "metadata.json").read_text())
    hashes = [hashlib.sha256((directory / "subinstances" / (stem + ".m")).read_bytes()).hexdigest()
              for stem in accepted["source_artifact_stems"].split(";")]
    fingerprint = (accepted["seed"], accepted["target_size"], accepted["target_nodes"],
                   tuple(metadata["source_names"]), tuple(hashes))
    experiment = BASELINE if policy == "consecutive" else f"{PREFIX}_{policy}"
    checkpoint = RESULTS / f"{experiment}_pglib_exact/evaluation/cohort_checkpoints/i{index:03d}/exact_outer_face.json"
    data = json.loads(checkpoint.read_text())
    assert data["settings"]["time_limit_seconds"] == 3600
    assert data["strategy"] == "exact_outer_face"
    graphs = data["graph_results"]
    final = graphs[f"final:{index}:{accepted['artifact_stem']}"]
    pieces = [value for key, value in graphs.items() if key.startswith("source:")]
    assert len(pieces) == int(accepted["source_count"])
    floor_lo = sum(x["skewness_lower_bound"] for x in pieces)
    floor_hi = sum(x["skewness_upper_bound"] for x in pieces)
    coupling = int(accepted["coupling_branch_count"])
    forest = len(pieces) - 1
    lower = max(final["skewness_lower_bound"], floor_lo)
    upper = min(final["skewness_upper_bound"], floor_hi + coupling - forest)
    added_lo = max(0, lower - floor_hi)
    added_hi = min(coupling - forest, upper - floor_lo)
    assert 0 <= lower <= upper and 0 <= added_lo <= added_hi <= coupling
    return policy, index, fingerprint, {
        "upper": upper, "added_upper": added_hi, "normalized_upper": added_hi / coupling,
        "zero_added": added_hi == 0, "exact": lower == upper,
    }


def shifted_mean(values):
    return math.expm1(math.fsum(math.log1p(x) for x in values) / len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    global BASELINE, PREFIX, RESULTS
    parser.add_argument("paper", type=Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--baseline-application-summary", type=Path)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--audit-output", type=Path, default=ROOT / "check/results/publication_checks/sampling.json")
    args = parser.parse_args()
    BASELINE, PREFIX, RESULTS = args.baseline, args.experiment, args.results
    baseline_root = RESULTS / f"{BASELINE}_pglib"
    baseline = {int(row["output_index"]): row for row in rows(baseline_root / "tables/accepted_instances.csv")
                if row["strategy"] == "exact_outer_face"}
    tasks = [("consecutive", i, row, baseline_root) for i, row in baseline.items()]
    for policy in list(POLICIES)[1:]:
        for i in range(1, 101):
            directory = RESULTS / f"{PREFIX}_{policy}_pglib_job_i{i:03d}"
            accepted = [row for row in rows(directory / "tables/accepted_instances.csv")
                        if row["strategy"] == "exact_outer_face"]
            assert len(accepted) == 1 and accepted[0]["coupling_count_rule"] == "log2"
            tasks.append((policy, i, accepted[0], directory))
    fingerprints = {policy: {} for policy in POLICIES}
    statistics = {policy: {} for policy in POLICIES}
    with ThreadPoolExecutor(max_workers=8) as executor:
        for policy, index, fingerprint, metrics in executor.map(load_input, tasks):
            fingerprints[policy][index] = fingerprint
            if metrics is not None:
                statistics[policy][index] = metrics
    paired = baseline_root / "evaluation/paired_strategy_outcomes_exact.csv"
    if not paired.is_file(): paired = baseline_root / "evaluation/paired_strategy_outcomes.csv"
    for row in rows(paired):
        if row["strategy"] != "exact_outer_face":
            continue
        published = {
            "upper": float(row["skewness_upper_bound"]),
            "added_upper": float(row["skewness_inflation_upper_bound"]),
            "normalized_upper": float(row["skewness_normalized_inflation_upper_bound"]),
            "zero_added": float(row["skewness_inflation_upper_bound"]) == 0,
            "exact": row["skewness_exact"] == "True",
        }
        assert statistics["consecutive"][int(row["output_index"])] == published
    expected = set(range(1, 101))
    assert all(set(fingerprints[p]) == set(statistics[p]) == expected for p in POLICIES)
    shared = [i for i in sorted(expected) if len({fingerprints[p][i] for p in POLICIES}) == 1]
    assert shared, "no common inputs"
    report = {"baseline": BASELINE, "experiment": PREFIX, "matched_inputs": shared,
              "matching": "seed, target sizes, source names, and SHA256 of extracted subinstance files",
              "policies": {}}
    skew_table = [r"\begin{tabular}{lrrrrr}", r"  \toprule",
                 r"  Sampling & $\bar s_C$ & $\bar a$ & $\bar\eta$ & $N_{\bar a=0}$ & $N_{\underline s_C=\bar s_C}$ \\",
                 r"  \midrule"]
    scopf_table = [r"\begin{tabular}{lrrrr}", r"  \toprule",
                  r"  Sampling configuration & Feasible & Infeasible & No solution & Feasible (matched) \\", r"  \midrule"]
    for policy, label in POLICIES.items():
        experiment = BASELINE if policy == "consecutive" else f"{PREFIX}_{policy}"
        summary = (args.baseline_application_summary if policy == "consecutive" and args.baseline_application_summary
                   else RESULTS / f"{experiment}_pglib_scopf/summary.csv")
        application = [row for row in rows(summary)
                       if row["strategy"] == "exact_outer_face"]
        assert len(application) == 100 and {int(x["instance_index"]) for x in application} == expected
        assert all(row["state"] == "complete" for row in application)
        accepted_stems = {index: accepted["artifact_stem"] for task_policy, index, accepted, _ in tasks if task_policy == policy}
        assert all(row["artifact_stem"] == accepted_stems[int(row["instance_index"])] for row in application)
        def outcome(row):
            if row["termination_status"].endswith("ERROR"):
                return "unknown"
            if row["scopf_feasible"] == "True" and math.isfinite(float(row["objective_value"])):
                return "feasible"
            return "infeasible" if row["termination_status"] == "INFEASIBLE" else "unknown"
        counts = Counter(outcome(row) for row in application)
        matched_counts = Counter(outcome(row) for row in application if int(row["instance_index"]) in shared)
        selected = [statistics[policy][i] for i in shared]
        summary = {key: shifted_mean([x[key] for x in selected])
                   for key in ("upper", "added_upper", "normalized_upper")}
        summary.update({key: sum(x[key] for x in selected) for key in ("zero_added", "exact")})
        report["policies"][policy] = {
            "matched_with_baseline": [i for i in sorted(expected) if fingerprints[policy][i] == fingerprints["consecutive"][i]],
            "skewness_matched": summary, "scopf_all": dict(counts), "scopf_matched": dict(matched_counts),
            "termination_status": dict(Counter(x["termination_status"] for x in application)),
        }
        skew_table.append(f"  {label} & {summary['upper']:.3f} & {summary['added_upper']:.3f} & "
                          f"{summary['normalized_upper']:.3f} & {summary['zero_added']} & {summary['exact']} " + r"\\")
        scopf_table.append(f"  {label} & {counts['feasible']} & {counts['infeasible']} & {counts['unknown']} & {matched_counts['feasible']} " + r"\\")
    for filename, lines in (("tab-app-spacing.tex", skew_table), ("tab-app-scopf-sampling.tex", scopf_table)):
        lines.extend([r"  \bottomrule", r"\end{tabular}", ""])
        (args.paper / "resources/evaluation" / filename).write_text("\n".join(lines))
    output = args.audit_output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
