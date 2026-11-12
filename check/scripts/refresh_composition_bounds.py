#!/usr/bin/env python3
"""Refresh derived composition bounds, preserving solver outputs and backups.

For connected pipeline outputs only. This does not run an optimizer and must
not be used to claim performance of the new objective restriction.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from planar_instance_generation.evaluation.reporting import (
    SkewnessBounds, SkewnessObservation, propagate_composition_bounds, _add_inflation,
)


def refresh_row(row, piece_count, *, paired=False):
    output = dict(row)
    def key(field):
        return "skewness_" + field if paired and not field.startswith("skewness") else field
    def value(field):
        return output.get(key(field))
    lower = int(float(value("solver_lower_bound") if value("solver_lower_bound") not in (None, "") else value("skewness_lower_bound")))
    upper = int(float(value("solver_upper_bound") if value("solver_upper_bound") not in (None, "") else value("skewness_upper_bound")))
    raw = SkewnessBounds(lower, upper, str(value("status")))
    observation = SkewnessObservation(raw,
        source_floor_lower_bound=int(float(value("source_floor_lower_bound"))),
        source_floor_upper_bound=int(float(value("source_floor_upper_bound"))),
        coupling_edge_count=int(float(value("coupling_edge_count"))),
        source_piece_count=piece_count)
    bounds = propagate_composition_bounds(raw, observation)
    changes = {
        "solver_lower_bound": lower, "solver_upper_bound": upper,
        "source_piece_count": piece_count, "coupling_forest_edge_count": piece_count - 1,
        "skewness_lower_bound": bounds.lower_bound, "skewness_upper_bound": bounds.upper_bound,
        "skewness_exact": bounds.exact, "skewness": bounds.lower_bound if bounds.exact else None,
        "absolute_gap": bounds.upper_bound - bounds.lower_bound,
        "relative_gap": (bounds.upper_bound - bounds.lower_bound) / bounds.upper_bound if bounds.upper_bound else 0.0,
        "inflation": None, "normalized_inflation": None,
        "normalized_inflation_lower_bound": None, "normalized_inflation_upper_bound": None,
        "normalized_inflation_exact": None,
    }
    _add_inflation(changes, bounds, observation)
    output.update({key(k): v for k, v in changes.items()})
    return output


def csv_content(records):
    fields = list(dict.fromkeys(k for row in records for k in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)
    return stream.getvalue()


def refresh_directory(directory):
    main = directory / "paired_strategy_outcomes.csv"
    paired = list(csv.DictReader(io.StringIO(main.read_text())))
    counts = {}
    for row in paired:
        count = int(row["source_count"])
        if str(row["structural_connected"]).lower() != "true":
            raise ValueError("only connected pipeline outputs can use p - 1")
        if len(row["source_artifact_stems"].split(";")) != count:
            raise ValueError("source count does not match the extracted-piece occurrences")
        counts[row["artifact_stem"]] = count
    pending = {}
    for name in ("paired_strategy_outcomes.csv", "paired_strategy_outcomes_exact.csv", "skewness.csv"):
        path = directory / name
        if not path.is_file():
            continue
        records = list(csv.DictReader(io.StringIO(path.read_text())))
        is_paired = name.startswith("paired_")
        refreshed = []
        for row in records:
            stem = row["artifact_stem" if is_paired else "instance"]
            refreshed.append(refresh_row(row, counts[stem], paired=is_paired) if stem in counts else row)
        pending[path] = csv_content(refreshed)
    for name, records_key, stem_key, is_paired in (
        ("skewness.json", "instances", "instance", False),
        ("paired_strategy.json", "outcomes", "artifact_stem", True),
    ):
        path = directory / name
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        data[records_key] = [refresh_row(r, counts[r[stem_key]], paired=is_paired)
                            if r[stem_key] in counts else r for r in data[records_key]]
        data["bound_postprocessing"] = "coupling forest: cap |M|-(p-1); no solver rerun; solver bounds and times retained"
        pending[path] = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    backup = directory / "before_coupling_forest_bound"
    backup.mkdir(exist_ok=True)
    for path, content in pending.items():
        saved = backup / path.name
        if not saved.exists():
            saved.write_bytes(path.read_bytes())
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content)
        temporary.replace(path)
    print(f"Refreshed {len(counts)} composites in {directory}; original reports in {backup}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", nargs="+", type=Path)
    for directory in parser.parse_args().evaluation:
        refresh_directory(directory)
