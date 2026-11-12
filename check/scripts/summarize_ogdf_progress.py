#!/usr/bin/env python3
"""Summarize completed rerun results without solving or changing experiment data."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from check.eval.postprocess import STRATEGIES
from planar_instance_generation.evaluation.reporting import (
    SkewnessBounds,
    SkewnessObservation,
    propagate_composition_bounds,
    _add_inflation,
)

LABELS = dict(zip(STRATEGIES, ("R", "HD", "KK", "OF-H", "OF-B")))
H, B = STRATEGIES[-2:]


def result_row(payload, accepted):
    """Use the same composition bounds as the full evaluation report."""
    index = payload["cohort"]
    stem = accepted["artifact_stem"]
    sources = accepted["source_artifact_stems"].split(";")
    results = payload["graph_results"]
    final_key = f"final:{index}:{stem}"
    expected = {final_key, *(f"source:{index}:{s}" for s in sources)}
    if set(results) != expected or len(set(sources)) != len(sources):
        raise ValueError(
            "checkpoint does not match the accepted graph and constituents"
        )
    bounds = {
        key: SkewnessBounds(
            r["skewness_lower_bound"],
            r["skewness_upper_bound"],
            r.get("status"),
            r.get("runtime_seconds"),
        )
        for key, r in results.items()
    }
    for key, value in bounds.items():
        if value.upper_bound > results[key]["edge_count"]:
            raise ValueError("bound exceeds the graph's edge count")
    floor = [bounds[f"source:{index}:{s}"] for s in sources]
    observation = SkewnessObservation(
        bounds[final_key],
        sum(x.lower_bound for x in floor),
        sum(x.upper_bound for x in floor),
        int(accepted.get("coupling_branch_count") or accepted["coupling_link_count"]),
        len(sources),
    )
    final = propagate_composition_bounds(bounds[final_key], observation)
    row = {
        "input": index,
        "strategy": payload["strategy"],
        "instance": stem,
        "lower": final.lower_bound,
        "upper": final.upper_bound,
        "exact": final.exact,
        "status": final.status,
        "selection_seconds": float(accepted["merge_node_selection_seconds"]),
        "final_seconds": final.runtime_seconds,
    }
    _add_inflation(row, final, observation)
    for field in ("selection_seconds", "final_seconds"):
        if row[field] is None or not math.isfinite(row[field]) or row[field] < 0:
            raise ValueError(f"invalid {field}")
    signature = [(s, x.lower_bound, x.upper_bound) for s, x in zip(sources, floor)]
    return row, signature


def summarize(rows):
    """Compare only methods evaluated on the same accepted input."""
    by_input = {}
    for row in rows:
        by_input.setdefault(row["input"], {})[row["strategy"]] = row
    complete = {i: r for i, r in by_input.items() if set(r) == set(STRATEGIES)}
    pairs = {i: r for i, r in by_input.items() if H in r and B in r}
    methods = {}
    for strategy in STRATEGIES:
        selected = [r[strategy] for r in complete.values()]
        means = {}
        for field in (
            "normalized_inflation_upper_bound",
            "selection_seconds",
            "final_seconds",
        ):
            values = [r[field] for r in selected]
            means[field] = (
                math.expm1(math.fsum(math.log1p(v) for v in values) / len(values))
                if values
                else None
            )
        methods[strategy] = {
            "completed": sum(r["strategy"] == strategy for r in rows),
            "matched": len(selected),
            "exact": sum(r["exact"] for r in selected),
            "zero_added": sum(r["inflation_upper_bound"] == 0 for r in selected),
            "strictly_better_than_all_baselines": [
                i
                for i, r in complete.items()
                if all(r[strategy]["upper"] < r[s]["lower"] for s in STRATEGIES[:3])
            ]
            if strategy in (H, B)
            else [],
            "shifted_geometric_means": means,
        }
    # The constituent skewness is the same in a pair, so it cancels when
    # comparing added skewness. Comparing final intervals also works when
    # the common constituent skewness is not known exactly.
    def select(condition):
        return [i for i, r in pairs.items() if condition(r[H], r[B])]

    return {
        "complete_inputs": sorted(complete),
        "outer_face_pairs": sorted(pairs),
        "methods": methods,
        "b_proven_better": select(lambda h, b: b["upper"] < h["lower"]),
        "h_proven_better": select(lambda h, b: h["upper"] < b["lower"]),
        "proven_equal": select(
            lambda h, b: h["exact"] and b["exact"] and h["lower"] == b["lower"]
        ),
        "b_smaller_upper_bound": select(lambda h, b: b["upper"] < h["upper"]),
    }


def collect(results, experiment, instances):
    report = {
        "experiment": experiment,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "expected_inputs_per_dataset": instances,
        "datasets": {},
        "errors": [],
    }
    all_rows = []
    for dataset in ("pglib", "snd"):
        generated, rows, settings = [], [], None
        for index in range(1, instances + 1):
            name = f"{experiment}_{dataset}_job_i{index:03d}"
            locations = (
                results / name,
                results / "_cohorts" / f"{experiment}_{dataset}" / name,
            )
            existing = [p for p in locations if p.is_dir()]
            if not existing:
                continue
            if len(existing) != 1:
                report["errors"].append(
                    f"{dataset} i{index:03d}: duplicate result directories"
                )
                continue
            run = existing[0]
            if not (run / "manifest.json").is_file():
                continue
            try:
                with (run / "tables/accepted_instances.csv").open(newline="") as f:
                    accepted = list(csv.DictReader(f))
                by_method = {r["strategy"]: r for r in accepted}
                if len(accepted) != 5 or set(by_method) != set(STRATEGIES):
                    raise ValueError("expected exactly one accepted graph per method")
                signatures = {
                    (r["source_artifact_stems"], r["attempt_index"], r["seed"])
                    for r in accepted
                }
                if len(signatures) != 1 or any(
                    r["gate_feasible"] != "True" for r in accepted
                ):
                    raise ValueError("methods do not share one accepted extraction")
                generated.append(index)
            except (OSError, ValueError, KeyError) as error:
                report["errors"].append(f"{dataset} i{index:03d}: {error}")
                continue
            input_rows, source_signature = [], None
            for strategy in STRATEGIES:
                path = (
                    results
                    / f"{experiment}_{dataset}_exact/evaluation/cohort_checkpoints"
                    / f"i{index:03d}"
                    / f"{strategy}.json"
                )
                if not path.is_file():
                    continue
                try:
                    payload = json.loads(path.read_text())
                    if (
                        payload["experiment"],
                        payload["cohort"],
                        payload["strategy"],
                    ) != (experiment, index, strategy):
                        raise ValueError("checkpoint identifies a different run")
                    row, signature = result_row(payload, by_method[strategy])
                    if settings is not None and payload["settings"] != settings:
                        raise ValueError("inconsistent solver settings")
                    settings = payload["settings"]
                    if source_signature is not None and signature != source_signature:
                        raise ValueError("inconsistent shared constituent bounds")
                    source_signature = signature
                    input_rows.append({"dataset": dataset, **row})
                except (OSError, ValueError, KeyError, TypeError) as error:
                    report["errors"].append(
                        f"{dataset} i{index:03d} {strategy}: {error}"
                    )
                    # Exclude this input from comparisons if its data disagree.
                    input_rows = []
                    break
            rows.extend(input_rows)
        suffix = "scopf" if dataset == "pglib" else "benders"
        app = results / f"{experiment}_{dataset}_{suffix}"
        states = Counter()
        if (app / "jobs.csv").is_file():
            with (app / "jobs.csv").open(newline="") as f:
                for job in csv.DictReader(f):
                    path = app / "logs" / job["artifact_stem"] / "summary.json"
                    try:
                        state = (
                            json.loads(path.read_text()).get("state", "unknown")
                            if path.is_file()
                            else "pending"
                        )
                        states[state] += 1
                    except (OSError, ValueError, TypeError) as error:
                        report["errors"].append(f"{path}: {error}")
                        states["unreadable"] += 1
        report["datasets"][dataset] = {
            "generated_inputs": generated,
            "settings": settings,
            "applications": dict(states),
            **summarize(rows),
        }
        all_rows.extend(rows)
    return report, all_rows


def html_report(report):
    def table(headers, rows):
        return (
            "<table><tr>"
            + "".join(f"<th>{escape(str(h))}</th>" for h in headers)
            + "</tr>"
            + "".join(
                "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>"
                for row in rows
            )
            + "</table>"
        )

    def number(value):
        return "—" if value is None else f"{value:.3g}"

    parts = [
        f'<h1>{escape(report["experiment"])}: experiment progress</h1>',
        f'<p>Updated {escape(report["updated_utc"])}. <a href="results.csv">Individual results</a> · <a href="summary.json">Complete summary</a></p>',
        "<p>Partial results can favor easier inputs. Only complete five-method input sets enter the method comparisons. OF-H versus OF-B uses the completed pairs, with its own denominator. Missing results are not counted as failures or ties.</p>",
    ]
    for dataset, data in report["datasets"].items():
        parts += [
            f"<h2>{escape(dataset.upper())}</h2>",
            f'<p>Generated: {len(data["generated_inputs"])}/{report["expected_inputs_per_dataset"]}. All five final solves completed: {len(data["complete_inputs"])}. OF-H/OF-B pairs completed: {len(data["outer_face_pairs"])}.</p>',
        ]
        methods = []
        for strategy, m in data["methods"].items():
            means = m["shifted_geometric_means"]
            methods.append(
                [
                    LABELS[strategy],
                    m["completed"],
                    m["matched"],
                    m["exact"],
                    m["zero_added"],
                    len(m["strictly_better_than_all_baselines"]),
                    number(means["normalized_inflation_upper_bound"]),
                    number(means["selection_seconds"]),
                    number(means["final_seconds"]),
                ]
            )
        parts.append(
            table(
                [
                    "Method",
                    "Final solves",
                    "Matched inputs",
                    "Exact final skewness",
                    "Zero added skewness",
                    "Proven better than R/HD/KK",
                    "Added skewness / coupling edges (upper bound)",
                    "Selection seconds",
                    "Final solve seconds",
                ],
                methods,
            )
        )
        parts.append(
            "<p>The last three columns show shifted geometric means on the matched inputs. Zero added skewness means the final graph is proven to preserve the sum of constituent skewnesses. Exact final skewness is a separate property.</p>"
        )
        comparisons = [
            ("b_proven_better", "OF-B proven better than OF-H"),
            ("h_proven_better", "OF-H proven better than OF-B"),
            ("proven_equal", "Proven equal"),
            (
                "b_smaller_upper_bound",
                "OF-B has a smaller upper bound (not necessarily a proof of improvement)",
            ),
        ]
        parts.append(
            table(
                ["OF-H versus OF-B", "Count", "Input IDs"],
                [
                    (label, len(data[key]), ", ".join(f"i{i:03d}" for i in data[key]))
                    for key, label in comparisons
                ],
            )
        )
        decisive = (
            len(data["b_proven_better"])
            + len(data["h_proven_better"])
            + len(data["proven_equal"])
        )
        parts.append(
            f'<p>Comparison unresolved on {len(data["outer_face_pairs"]) - decisive} completed pairs. A strict improvement is certified when one final upper bound is smaller than the other final lower bound. Both methods use the same constituents, so this also proves smaller added skewness.</p>'
        )
        parts.append(
            f'<p>Application job states: {escape(str(data["applications"])) if data["applications"] else "not prepared yet"}. These jobs start after the dataset report finishes. Application feasibility and runtime conclusions require their separate summaries.</p>'
        )
    parts.append(
        "<h2>Interpretation</h2><p>Check whether the outer face methods frequently certify zero added skewness and beat the three alternatives, and how often OF-B improves on OF-H at additional runtime. No strict OF-B improvement would require revising that empirical claim; it would not invalidate the construction or its theoretical guarantees. This report does not establish a runtime benefit from Euler preprocessing. Full-source and supplementary studies are separate.</p>"
    )
    parts.append(
        "<h2>Data issues</h2>"
        + (
            "<ul>"
            + "".join(f"<li>{escape(e)}</li>" for e in report["errors"])
            + "</ul>"
            if report["errors"]
            else "<p>None detected in the available results.</p>"
        )
    )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Experiment progress</title><style>body{font:16px system-ui;max-width:1250px;margin:36px auto;padding:0 20px;line-height:1.5;color:#17212b}table{border-collapse:collapse;width:100%;margin:20px 0;font-size:14px}th,td{padding:8px;border-bottom:1px solid #ccd3db;text-align:left}th{background:#eef2f7}h2{margin-top:40px}</style><body>'
        + "".join(parts)
        + "</body></html>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "check/results")
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        metavar="SECONDS",
        help="refresh repeatedly; 0 writes one snapshot",
    )
    args = parser.parse_args()
    if args.watch and args.watch < 30:
        parser.error("--watch must be 0 or at least 30 seconds")
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.experiment)
        or args.instances < 1
    ):
        parser.error("use a safe experiment name and a positive input count")
    while True:
        report, rows = collect(args.results_dir, args.experiment, args.instances)
        output = args.results_dir / f"{args.experiment}_progress"
        output.mkdir(parents=True, exist_ok=True)
        for name, content in [
            ("summary.json", json.dumps(report, indent=2) + "\n"),
            ("index.html", html_report(report)),
        ]:
            temporary = output / (name + ".tmp")
            temporary.write_text(content)
            temporary.replace(output / name)
        columns = sorted({key for row in rows for key in row}) or [
            "dataset",
            "input",
            "strategy",
        ]
        temporary = output / "results.csv.tmp"
        with temporary.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output / "results.csv")
        print(output / "index.html")
        for dataset, data in report["datasets"].items():
            print(
                f'{dataset}: {len(data["generated_inputs"])} generated, {len(data["complete_inputs"])} complete comparisons, {len(data["b_proven_better"])} proven OF-B improvements'
            )
        if report["errors"]:
            print(
                f'{len(report["errors"])} data issue(s); see summary.json',
                file=sys.stderr,
            )
            if not args.watch:
                raise SystemExit(1)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
