"""Auditable paired comparisons across generation strategies."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
from typing import Mapping, Sequence


PAIRING_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class PairedStrategyArtifacts:
    """Machine-readable paired outcomes and the cohort inclusion audit."""

    outcomes_csv: Path
    cohorts_csv: Path
    report_json: Path


@dataclass(frozen=True, slots=True)
class PairedStrategySchema:
    """Domain-specific columns used in one paired strategy comparison.

    Pairing itself is domain independent.  The schema only identifies the
    extraction fields that prove two strategies used the same constituents
    and the generation fields copied into each complete-case outcome.
    """

    name: str
    extraction_fields: tuple[str, ...]
    extraction_signature_fields: tuple[str, ...]
    extraction_integer_fields: frozenset[str]
    extraction_decimal_fields: frozenset[str]
    generation_fields: tuple[str, ...]
    extraction_nullable_fields: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("paired strategy schema name must be nonempty")
        if not self.extraction_fields or self.extraction_fields[0] != "final_artifact_stem":
            raise ValueError(
                "extraction_fields must start with final_artifact_stem"
            )
        if "source_position" not in self.extraction_signature_fields:
            raise ValueError(
                "extraction_signature_fields must include source_position"
            )
        for label, fields in (
            ("extraction_fields", self.extraction_fields),
            ("extraction_signature_fields", self.extraction_signature_fields),
            ("generation_fields", self.generation_fields),
        ):
            if any(not isinstance(field, str) or not field for field in fields):
                raise ValueError(f"{label} must contain nonempty strings")
            if len(fields) != len(set(fields)):
                raise ValueError(f"{label} must not contain duplicates")
        known = set(self.extraction_fields)
        if not set(self.extraction_signature_fields) <= known:
            raise ValueError(
                "extraction_signature_fields must occur in extraction_fields"
            )
        if "final_artifact_stem" in self.extraction_signature_fields:
            raise ValueError(
                "the extraction signature must not include final_artifact_stem"
            )
        typed = self.extraction_integer_fields | self.extraction_decimal_fields
        if not typed <= set(self.extraction_signature_fields):
            raise ValueError(
                "typed extraction fields must occur in extraction_signature_fields"
            )
        if self.extraction_integer_fields & self.extraction_decimal_fields:
            raise ValueError("an extraction field cannot have two numeric types")
        if not self.extraction_nullable_fields <= set(
            self.extraction_signature_fields
        ):
            raise ValueError(
                "nullable extraction fields must occur in extraction_signature_fields"
            )


_ACCEPTED_REQUIRED = (
    "output_index",
    "attempt_index",
    "artifact_stem",
    "strategy",
    "merge_node_selection_seconds",
    "composition_seconds",
    "merge_seconds",
    "generation_and_gate_seconds",
    "gate_feasible",
    "gate_status",
    "objective",
    "gate_seconds",
    "gate_optimizer_seconds",
)
_ATTEMPT_REQUIRED = (
    "output_index",
    "attempt_index",
    "strategy",
    "status",
    "draw_seed",
    "generation_seed",
    "source_names",
)
_PGLIB_EXTRACTION_FIELDS = (
    "final_artifact_stem",
    "source_position",
    "source_name",
    "source_node_count",
    "target_nodes",
    "realized_fraction",
    "seed",
    "bus_count",
    "generator_count",
    "branch_count",
    "cut_branch_count",
    "compensated_bus_count",
    "compensation_generator_count",
)
_PGLIB_GENERATION_FIELDS = (
    "instance_file",
    "plot_directory",
    "log_directory",
    "source_artifact_stems",
    "seed",
    "source_count",
    "bus_count",
    "branch_count",
    "merge_node_selection_seconds",
    "composition_seconds",
    "merge_seconds",
    "generation_and_gate_seconds",
    "gate_feasible",
    "gate_status",
    "objective",
    "gate_seconds",
    "gate_optimizer_seconds",
    "coupling_branch_count",
    "absolute_flow_mean_mw",
    "absolute_flow_max_mw",
    "finite_rating_count",
    "utilization_mean",
    "utilization_max",
)
_SNDLIB_EXTRACTION_FIELDS = (
    "final_artifact_stem",
    "source_position",
    "source_name",
    "routing_unit",
    "source_coordinates_type",
    "source_node_count",
    "target_nodes",
    "realized_fraction",
    "seed",
    "extraction_seconds",
    "node_count",
    "link_count",
    "commodity_count",
    "cut_link_count",
    "compensated_node_count",
    "compensated_commodity_count",
    "reference_flow_nonzero_count",
)
_SNDLIB_GENERATION_FIELDS = (
    "instance_file",
    "plot_directory",
    "log_directory",
    "source_artifact_stems",
    "seed",
    "routing_unit",
    "source_stratum_size",
    "source_count",
    "node_count",
    "link_count",
    "commodity_count",
    "serialized_demand_count",
    "coordinates_type",
    "coordinate_type_overridden",
    "merge_node_selection_seconds",
    "composition_seconds",
    "merge_seconds",
    "serialization_seconds",
    "generation_and_gate_seconds",
    "gate_feasible",
    "gate_status",
    "gate_reason",
    "gate_resource_skipped",
    "objective",
    "gate_seconds",
    "gate_optimizer_seconds",
    "coupling_link_count",
    "positive_capacity_count",
    "absolute_flow_mean",
    "absolute_flow_max",
    "utilization_mean",
    "utilization_max",
)


PGLIB_PAIRED_SCHEMA = PairedStrategySchema(
    name="pglib",
    extraction_fields=_PGLIB_EXTRACTION_FIELDS,
    extraction_signature_fields=_PGLIB_EXTRACTION_FIELDS[1:],
    extraction_integer_fields=frozenset(
        {
            "source_position",
            "source_node_count",
            "target_nodes",
            "seed",
            "bus_count",
            "generator_count",
            "branch_count",
            "cut_branch_count",
            "compensated_bus_count",
            "compensation_generator_count",
        }
    ),
    extraction_decimal_fields=frozenset({"realized_fraction"}),
    generation_fields=_PGLIB_GENERATION_FIELDS,
)
SNDLIB_PAIRED_SCHEMA = PairedStrategySchema(
    name="sndlib",
    extraction_fields=_SNDLIB_EXTRACTION_FIELDS,
    extraction_signature_fields=tuple(
        field
        for field in _SNDLIB_EXTRACTION_FIELDS[1:]
        if field != "extraction_seconds"
    ),
    extraction_integer_fields=frozenset(
        {
            "source_position",
            "source_node_count",
            "target_nodes",
            "seed",
            "node_count",
            "link_count",
            "commodity_count",
            "cut_link_count",
            "compensated_node_count",
            "compensated_commodity_count",
            "reference_flow_nonzero_count",
        }
    ),
    extraction_decimal_fields=frozenset({"realized_fraction"}),
    generation_fields=_SNDLIB_GENERATION_FIELDS,
    extraction_nullable_fields=frozenset({"routing_unit"}),
)
_STRUCTURAL_FIELDS = (
    "node_count",
    "edge_count",
    "density",
    "mean_degree",
    "maximum_degree",
    "average_clustering",
    "degeneracy",
    "treewidth_upper_bound",
    "connected",
    "diameter",
    "algebraic_connectivity",
    "mean_betweenness",
    "betweenness_p95",
    "betweenness_sample_count",
    "betweenness_exact",
    "betweenness_seed",
    "coupling_node_count",
    "coupling_betweenness_mean",
    "coupling_betweenness_p95",
    "coupling_betweenness_max",
    "degree_assortativity",
    "measurement_seconds",
)
_SKEWNESS_FIELDS = (
    "skewness_lower_bound",
    "skewness_upper_bound",
    "skewness_exact",
    "skewness",
    "absolute_gap",
    "relative_gap",
    "status",
    "runtime_seconds",
    "source_floor_lower_bound",
    "source_floor_upper_bound",
    "coupling_edge_count",
    "source_piece_count",
    "coupling_forest_edge_count",
    "solver_lower_bound",
    "solver_upper_bound",
    "inflation_lower_bound",
    "inflation_upper_bound",
    "inflation_exact",
    "inflation",
    "normalized_inflation_lower_bound",
    "normalized_inflation_upper_bound",
    "normalized_inflation_exact",
    "normalized_inflation",
)
_COHORT_FIELDS = (
    "output_index",
    "attempt_index",
    "status",
    "reason",
    "requested_strategies",
    "accepted_strategies",
    "missing_strategies",
    "artifact_stems",
    "draw_seeds",
    "generation_seeds",
    "source_names",
    "shared_extraction_verified",
)
_OUTCOME_PREFIX_FIELDS = (
    "output_index",
    "attempt_index",
    "artifact_stem",
    "strategy",
    "strategy_group",
    "source_names",
    "shared_extraction_verified",
)


def _outcome_fields(schema: PairedStrategySchema) -> tuple[str, ...]:
    return (
        *_OUTCOME_PREFIX_FIELDS,
        *schema.generation_fields,
        *(f"structural_{field}" for field in _STRUCTURAL_FIELDS),
        *(
            field if field.startswith("skewness") else f"skewness_{field}"
            for field in _SKEWNESS_FIELDS
        ),
    )


def write_paired_strategy_report(
    requested_strategies: Sequence[str],
    accepted_rows: Sequence[Mapping[str, object]],
    attempt_rows: Sequence[Mapping[str, object]],
    extraction_rows: Sequence[Mapping[str, object]],
    *,
    metadata_strategy_by_artifact: Mapping[str, str],
    strategy_groups: Mapping[str, str],
    structural_rows: Sequence[Mapping[str, object]],
    skewness_rows: Sequence[Mapping[str, object]],
    output_directory: str | Path,
    schema: PairedStrategySchema = PGLIB_PAIRED_SCHEMA,
    overwrite: bool = False,
) -> PairedStrategyArtifacts:
    """Write complete-case strategy outcomes without manufacturing pairings.

    A cohort is included only when every requested strategy was accepted for
    the same ``(output_index, attempt_index)`` and its draw, generation seed,
    source list, and extraction signature agree across strategies.  All other
    cohorts remain visible in the companion audit table.
    """

    if not isinstance(schema, PairedStrategySchema):
        raise TypeError("schema must be a PairedStrategySchema")
    strategies = _validate_strategies(requested_strategies)
    _require_columns(accepted_rows, _ACCEPTED_REQUIRED, "accepted instances")
    _require_columns(attempt_rows, _ATTEMPT_REQUIRED, "attempts")
    _require_columns(extraction_rows, schema.extraction_fields, "extractions")

    accepted = _unique_index(
        accepted_rows,
        lambda row: (*_cohort_key(row), _text(row, "strategy")),
        "accepted instance",
    )
    attempts = _unique_index(
        attempt_rows,
        lambda row: (*_cohort_key(row), _text(row, "strategy")),
        "attempt",
    )
    extractions_by_artifact = _extractions_by_artifact(extraction_rows)
    structural = _evaluation_index(structural_rows, "structural")
    skewness = _evaluation_index(skewness_rows, "skewness")

    cohort_keys = sorted(
        {key[:2] for key in accepted} | {key[:2] for key in attempts}
    )
    cohort_rows: list[dict[str, object]] = []
    outcome_rows: list[dict[str, object]] = []
    for output_index, attempt_index in cohort_keys:
        key = (output_index, attempt_index)
        accepted_strategies = [
            strategy
            for strategy in strategies
            if (*key, strategy) in accepted
            and _text(attempts.get((*key, strategy), {}), "status", required=False)
            .strip()
            .lower()
            == "accepted"
        ]
        missing = [strategy for strategy in strategies if strategy not in accepted_strategies]
        inconsistencies = _acceptance_inconsistencies(
            key, strategies, accepted, attempts
        )
        reason = ""
        shared = False
        if inconsistencies:
            reason = "acceptance_table_mismatch: " + "; ".join(inconsistencies)
        elif missing:
            reason = "incomplete_requested_strategies"
        else:
            records = [accepted[(*key, strategy)] for strategy in strategies]
            mismatch = _shared_extraction_mismatch(
                strategies,
                records,
                attempts,
                extractions_by_artifact,
                key,
                schema,
            )
            if mismatch:
                reason = mismatch
            else:
                shared = True
                for strategy, record in zip(strategies, records, strict=True):
                    artifact = _text(record, "artifact_stem")
                    metadata_strategy = metadata_strategy_by_artifact.get(artifact)
                    if metadata_strategy != strategy:
                        raise ValueError(
                            f"{artifact}: metadata strategy {metadata_strategy!r} "
                            f"does not match accepted strategy {strategy!r}"
                        )
                    group = strategy_groups.get(strategy)
                    if group is None:
                        raise ValueError(f"no evaluation group for strategy {strategy!r}")
                    structural_row = structural.get((group, artifact))
                    skewness_row = skewness.get((group, artifact))
                    if structural_row is None or skewness_row is None:
                        raise ValueError(
                            f"missing evaluated outcome for {(group, artifact)!r}"
                        )
                    outcome_rows.append(
                        _outcome_row(
                            key,
                            strategy,
                            group,
                            record,
                            attempts[(*key, strategy)],
                            structural_row,
                            skewness_row,
                            schema.generation_fields,
                        )
                    )

        cohort_attempts = [
            attempts[(*key, strategy)]
            for strategy in strategies
            if (*key, strategy) in attempts
        ]
        cohort_accepted = [
            accepted[(*key, strategy)]
            for strategy in strategies
            if (*key, strategy) in accepted
        ]
        cohort_rows.append(
            {
                "output_index": output_index,
                "attempt_index": attempt_index,
                "status": "included" if shared else "excluded",
                "reason": "complete_shared_extraction" if shared else reason,
                "requested_strategies": ";".join(strategies),
                "accepted_strategies": ";".join(accepted_strategies),
                "missing_strategies": ";".join(missing),
                "artifact_stems": ";".join(
                    _text(row, "artifact_stem") for row in cohort_accepted
                ),
                "draw_seeds": _joined_unique(cohort_attempts, "draw_seed"),
                "generation_seeds": _joined_unique(
                    cohort_attempts, "generation_seed"
                ),
                "source_names": _joined_unique(cohort_attempts, "source_names"),
                "shared_extraction_verified": shared,
            }
        )

    output = Path(output_directory)
    artifacts = PairedStrategyArtifacts(
        outcomes_csv=output / "paired_strategy_outcomes.csv",
        cohorts_csv=output / "paired_strategy_cohorts.csv",
        report_json=output / "paired_strategy.json",
    )
    existing = [path for path in _artifact_paths(artifacts) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"paired-strategy artifact already exists: {existing[0]}")
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(artifacts.outcomes_csv, outcome_rows, _outcome_fields(schema))
    _write_csv(artifacts.cohorts_csv, cohort_rows, _COHORT_FIELDS)
    artifacts.report_json.write_text(
        json.dumps(
            {
                "format_version": PAIRING_FORMAT_VERSION,
                "pairing_definition": (
                    "All requested strategies accepted the same output/attempt "
                    "with identical draw, generation seed, sources, and extraction signature."
                ),
                "domain_schema": schema.name,
                "requested_strategies": list(strategies),
                "included_cohort_count": sum(
                    row["status"] == "included" for row in cohort_rows
                ),
                "excluded_cohort_count": sum(
                    row["status"] == "excluded" for row in cohort_rows
                ),
                "cohorts": cohort_rows,
                "outcomes": outcome_rows,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return artifacts


def _validate_strategies(values: Sequence[str]) -> tuple[str, ...]:
    strategies = tuple(values)
    if not strategies or any(
        not isinstance(value, str) or not value.strip() for value in strategies
    ):
        raise ValueError("requested_strategies must contain nonempty strings")
    if len(strategies) != len(set(strategies)):
        raise ValueError("requested_strategies must be unique")
    return strategies


def _require_columns(
    rows: Sequence[Mapping[str, object]], required: Sequence[str], label: str
) -> None:
    if not rows:
        raise ValueError(f"{label} table must contain at least one row")
    missing = [field for field in required if any(field not in row for row in rows)]
    if missing:
        raise ValueError(f"{label} table is missing columns: {', '.join(missing)}")


def _cohort_key(row: Mapping[str, object]) -> tuple[int, int]:
    return (
        _positive_integer(row, "output_index"),
        _positive_integer(row, "attempt_index"),
    )


def _unique_index(rows, key, label):
    result = {}
    for row in rows:
        item_key = key(row)
        if item_key in result:
            raise ValueError(f"duplicate {label} row for {item_key!r}")
        result[item_key] = row
    return result


def _acceptance_inconsistencies(key, strategies, accepted, attempts) -> list[str]:
    issues = []
    for strategy in strategies:
        accepted_present = (*key, strategy) in accepted
        attempt = attempts.get((*key, strategy))
        attempt_accepted = (
            attempt is not None and _text(attempt, "status").strip().lower() == "accepted"
        )
        if accepted_present != attempt_accepted:
            issues.append(
                f"{strategy} accepted_csv={accepted_present} attempts_csv={attempt_accepted}"
            )
    return issues


def _extractions_by_artifact(rows):
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(_text(row, "final_artifact_stem"), []).append(row)
    result = {}
    for artifact, items in grouped.items():
        positions = [_positive_integer(row, "source_position") for row in items]
        if len(positions) != len(set(positions)):
            raise ValueError(f"{artifact}: duplicate extraction source_position")
        result[artifact] = tuple(
            sorted(items, key=lambda row: _positive_integer(row, "source_position"))
        )
    return result


def _shared_extraction_mismatch(
    strategies,
    records,
    attempts,
    extractions_by_artifact,
    key,
    schema: PairedStrategySchema,
) -> str:
    attempt_signatures = {
        (
            _text(attempts[(*key, strategy)], "draw_seed"),
            _text(attempts[(*key, strategy)], "generation_seed"),
            _text(attempts[(*key, strategy)], "source_names"),
        )
        for strategy in strategies
    }
    if len(attempt_signatures) != 1:
        return "shared_draw_mismatch"
    extraction_signatures = set()
    for record in records:
        artifact = _text(record, "artifact_stem")
        rows = extractions_by_artifact.get(artifact)
        if not rows:
            return f"missing_extraction_rows:{artifact}"
        extraction_signatures.add(
            tuple(
                tuple(
                    _canonical(row, field, schema)
                    for field in schema.extraction_signature_fields
                )
                for row in rows
            )
        )
    if len(extraction_signatures) != 1:
        return "shared_extraction_mismatch"
    return ""


def _evaluation_index(rows, label):
    return _unique_index(
        rows,
        lambda row: (_text(row, "group"), _text(row, "instance")),
        label,
    )


def _outcome_row(
    key,
    strategy,
    group,
    accepted,
    attempt,
    structural,
    skewness,
    generation_fields: Sequence[str],
) -> dict[str, object]:
    row = {
        "output_index": key[0],
        "attempt_index": key[1],
        "artifact_stem": _text(accepted, "artifact_stem"),
        "strategy": strategy,
        "strategy_group": group,
        "source_names": _text(attempt, "source_names"),
        "shared_extraction_verified": True,
    }
    row.update(
        (field, _csv_scalar(accepted.get(field)))
        for field in generation_fields
    )
    row.update(
        (f"structural_{field}", structural.get(field))
        for field in _STRUCTURAL_FIELDS
    )
    row.update(
        (
            field if field.startswith("skewness") else f"skewness_{field}",
            skewness.get(field),
        )
        for field in _SKEWNESS_FIELDS
    )
    return row


def _canonical(
    row: Mapping[str, object], field: str, schema: PairedStrategySchema
) -> object:
    value = _text(
        row,
        field,
        required=field not in schema.extraction_nullable_fields,
    )
    if field in schema.extraction_integer_fields:
        try:
            return int(value)
        except ValueError as error:
            raise ValueError(f"{field} must be an integer, got {value!r}") from error
    if field in schema.extraction_decimal_fields:
        try:
            return Decimal(value)
        except InvalidOperation as error:
            raise ValueError(f"{field} must be numeric, got {value!r}") from error
    return value


def _positive_integer(row: Mapping[str, object], field: str) -> int:
    value = _text(row, field)
    try:
        number = int(value)
    except ValueError as error:
        raise ValueError(f"{field} must be a positive integer, got {value!r}") from error
    if number < 1:
        raise ValueError(f"{field} must be a positive integer, got {value!r}")
    return number


def _text(
    row: Mapping[str, object], field: str, *, required: bool = True
) -> str:
    if field not in row:
        if not required:
            return ""
        raise ValueError(f"missing field {field!r}")
    value = row[field]
    if value is None:
        value = ""
    text = str(value)
    if required and not text:
        raise ValueError(f"field {field!r} must not be empty")
    return text


def _csv_scalar(value: object) -> object:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        return int(text)
    except ValueError:
        try:
            number = float(text)
        except ValueError:
            return value
        return number


def _joined_unique(rows, field: str) -> str:
    return ";".join(dict.fromkeys(_text(row, field) for row in rows))


def _artifact_paths(artifacts: PairedStrategyArtifacts) -> tuple[Path, ...]:
    return artifacts.outcomes_csv, artifacts.cohorts_csv, artifacts.report_json


def _write_csv(path: Path, rows, fields) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
