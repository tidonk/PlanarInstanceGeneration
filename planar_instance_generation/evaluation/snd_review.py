"""Validated SNDlib artifacts used by structural and preservation evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import networkx as nx

from planar_instance_generation.evaluation.preservation import (
    EmbeddedSourceGraph,
    PreservationAudit,
    audit_preserved_properties,
)
from planar_instance_generation.evaluation.reporting import EvaluationGraph
from planar_instance_generation.io.sndlib import SndlibNetwork, read_sndlib


class SndEvaluationInputError(ValueError):
    """Raised when a saved SND generation run has inconsistent provenance."""


@dataclass(frozen=True, slots=True)
class ExtractedSndArtifact:
    """One serialized extracted constituent and its generation metadata."""

    stem: str
    network: SndlibNetwork
    graph: nx.Graph
    metadata: Mapping[str, object]

    @property
    def evaluation_graph(self) -> EvaluationGraph:
        return EvaluationGraph(self.stem, self.graph)


@dataclass(frozen=True, slots=True)
class FinalSndArtifact:
    """One serialized composite with validated source and coupling provenance."""

    stem: str
    strategy: str
    noncrossing_matching_certified: bool | None
    network: SndlibNetwork
    graph: nx.Graph
    metadata: Mapping[str, object]
    source_stems: tuple[str, ...]
    coupling_link_indices: tuple[int, ...]
    coupling_edges: tuple[tuple[str, str], ...]

    @property
    def evaluation_graph(self) -> EvaluationGraph:
        coupling_nodes = tuple(
            dict.fromkeys(node for edge in self.coupling_edges for node in edge)
        )
        return EvaluationGraph(self.stem, self.graph, coupling_nodes)


@dataclass(frozen=True, slots=True)
class GeneratedSndArtifacts:
    """All extracted and final XML artifacts from one generation run."""

    extracted: tuple[ExtractedSndArtifact, ...]
    finals: tuple[FinalSndArtifact, ...]

    @property
    def networks_by_stem(self) -> dict[str, SndlibNetwork]:
        return {
            artifact.stem: artifact.network
            for artifact in (*self.extracted, *self.finals)
        }


def load_generated_snd_artifacts(
    run_directory: str | Path,
) -> GeneratedSndArtifacts:
    """Load saved XML files and fail on stale or inconsistent metadata."""

    run = Path(run_directory)
    plot_root = run / "plots"
    nested_metadata = [
        *plot_root.joinpath("original").glob("*/metadata.json"),
        *plot_root.joinpath("composite").glob("*/metadata.json"),
    ]
    metadata_paths = sorted(
        nested_metadata or plot_root.glob("*/metadata.json")
    )
    if not metadata_paths:
        raise SndEvaluationInputError(
            f"generation run has no plot metadata below {plot_root}"
        )

    extracted: list[ExtractedSndArtifact] = []
    pending_finals: list[tuple[str, SndlibNetwork, nx.Graph, Mapping[str, object]]] = []
    seen: set[str] = set()
    for metadata_path in metadata_paths:
        metadata = _read_object(metadata_path)
        stem = _required_string(metadata, "artifact_stem", str(metadata_path))
        if stem != metadata_path.parent.name:
            raise SndEvaluationInputError(
                f"{metadata_path}: artifact_stem does not match its directory"
            )
        if stem in seen:
            raise SndEvaluationInputError(f"duplicate artifact stem {stem!r}")
        seen.add(stem)
        kind = _required_string(metadata, "kind", str(metadata_path))
        instance_path = _generated_instance_path(
            run, stem, kind, metadata_path
        )
        network = read_sndlib(instance_path)
        graph = network.topology()
        if kind == "extracted_snd_subinstance":
            extracted.append(ExtractedSndArtifact(stem, network, graph, metadata))
        elif kind == "final_snd_composite":
            pending_finals.append((stem, network, graph, metadata))
        else:
            raise SndEvaluationInputError(
                f"{metadata_path}: unsupported artifact kind {kind!r}"
            )

    if not extracted or not pending_finals:
        raise SndEvaluationInputError(
            "generation run must contain extracted and final SND artifacts"
        )
    extracted_by_stem = {artifact.stem: artifact for artifact in extracted}
    finals = tuple(
        _load_final(stem, network, graph, metadata, extracted_by_stem)
        for stem, network, graph, metadata in pending_finals
    )
    return GeneratedSndArtifacts(tuple(extracted), finals)


def audit_snd_preservation(
    artifacts: GeneratedSndArtifacts,
) -> dict[str, PreservationAudit]:
    """Audit paper bounds solely from serialized XML and saved node maps."""

    extracted = {artifact.stem: artifact for artifact in artifacts.extracted}
    audits: dict[str, PreservationAudit] = {}
    for final in artifacts.finals:
        raw_mappings = _required_list(
            final.metadata, "source_mappings", f"metadata for {final.stem}"
        )
        if len(raw_mappings) != len(final.source_stems):
            raise SndEvaluationInputError(
                f"{final.stem}: source_mappings and source_artifact_stems disagree"
            )
        embedded: list[EmbeddedSourceGraph] = []
        for expected_stem, raw_mapping in zip(
            final.source_stems, raw_mappings, strict=True
        ):
            mapping = _as_object(raw_mapping, f"{final.stem} source mapping")
            source_stem = _required_string(
                mapping, "source_artifact_stem", f"metadata for {final.stem}"
            )
            if source_stem != expected_stem:
                raise SndEvaluationInputError(
                    f"{final.stem}: source mapping order disagrees with source stems"
                )
            source = extracted[source_stem]
            raw_node_map = _as_object(
                mapping.get("node_ids"), f"{final.stem}/{source_stem} node_ids"
            )
            if any(
                not isinstance(source_node, str)
                or not isinstance(composite_node, str)
                for source_node, composite_node in raw_node_map.items()
            ):
                raise SndEvaluationInputError(
                    f"{final.stem}/{source_stem}: node_ids must map strings to strings"
                )
            node_map = dict(raw_node_map)
            if set(node_map) != set(source.graph):
                raise SndEvaluationInputError(
                    f"{final.stem}/{source_stem}: node map does not cover the source XML"
                )
            if len(set(node_map.values())) != len(node_map):
                raise SndEvaluationInputError(
                    f"{final.stem}/{source_stem}: node map is not injective"
                )
            if not set(node_map.values()) <= set(final.graph):
                raise SndEvaluationInputError(
                    f"{final.stem}/{source_stem}: node map leaves the final XML"
                )
            embedded.append(
                EmbeddedSourceGraph(source_stem, source.graph, node_map)
            )

        audits[final.stem] = audit_preserved_properties(
            embedded,
            final.graph,
            final.coupling_edges,
            composition_strategy=final.strategy,
            noncrossing_matching_certified=final.noncrossing_matching_certified,
        )
    return audits


def _generated_instance_path(
    run: Path, stem: str, kind: str, metadata_path: Path
) -> Path:
    if kind == "final_snd_composite":
        path = run / "instances" / f"{stem}.xml"
    elif kind == "extracted_snd_subinstance":
        path = run / "subinstances" / f"{stem}.xml"
        legacy_path = run / "instances" / f"{stem}.xml"
        if not path.is_file() and legacy_path.is_file():
            path = legacy_path
    else:
        raise SndEvaluationInputError(
            f"{metadata_path}: unsupported artifact kind {kind!r}"
        )
    if not path.is_file():
        raise SndEvaluationInputError(
            f"{metadata_path}: missing serialized instance {path}"
        )
    return path


def _load_final(
    stem: str,
    network: SndlibNetwork,
    graph: nx.Graph,
    metadata: Mapping[str, object],
    extracted: Mapping[str, ExtractedSndArtifact],
) -> FinalSndArtifact:
    statistics = _as_object(metadata.get("statistics"), f"metadata for {stem}")
    strategy = _required_string(statistics, "strategy", f"metadata for {stem}")
    source_stems = tuple(
        _string_value(value, f"{stem} source_artifact_stems")
        for value in _required_list(
            metadata, "source_artifact_stems", f"metadata for {stem}"
        )
    )
    if len(source_stems) < 2:
        raise SndEvaluationInputError(f"{stem}: a final needs at least two sources")
    missing = [source for source in source_stems if source not in extracted]
    if missing:
        raise SndEvaluationInputError(
            f"{stem}: unknown extracted source artifact(s): {', '.join(missing)}"
        )

    coupling_records = _required_list(
        metadata, "coupling_links", f"metadata for {stem}"
    )
    if not coupling_records:
        raise SndEvaluationInputError(f"{stem}: final has no coupling links")
    indices: list[int] = []
    edges: list[tuple[str, str]] = []
    for raw_record in coupling_records:
        record = _as_object(raw_record, f"{stem} coupling link")
        index = _nonnegative_integer(
            record.get("index_zero_based"), f"{stem} coupling link index"
        )
        if index >= len(network.links):
            raise SndEvaluationInputError(
                f"{stem}: coupling link index {index} is outside the XML"
            )
        link = network.links[index]
        declared = (
            _required_string(record, "id", f"metadata for {stem}"),
            _required_string(record, "source", f"metadata for {stem}"),
            _required_string(record, "target", f"metadata for {stem}"),
        )
        if declared != (link.id, link.source, link.target):
            raise SndEvaluationInputError(
                f"{stem}: coupling record {index} disagrees with serialized XML"
            )
        indices.append(index)
        edges.append((link.source, link.target))
    if len(indices) != len(set(indices)):
        raise SndEvaluationInputError(f"{stem}: duplicate coupling link indices")
    canonical_edges = {frozenset(edge) for edge in edges}
    if len(canonical_edges) != len(edges):
        raise SndEvaluationInputError(
            f"{stem}: coupling links are not distinct in the simple topology"
        )
    certificate = _stage_certificate(
        stem,
        strategy,
        metadata,
        expected_stage_count=len(source_stems) - 1,
        coupling_link_count=len(indices),
    )
    return FinalSndArtifact(
        stem=stem,
        strategy=strategy,
        noncrossing_matching_certified=certificate,
        network=network,
        graph=graph,
        metadata=metadata,
        source_stems=source_stems,
        coupling_link_indices=tuple(indices),
        coupling_edges=tuple(edges),
    )


def _stage_certificate(
    stem: str,
    strategy: str,
    metadata: Mapping[str, object],
    *,
    expected_stage_count: int,
    coupling_link_count: int,
) -> bool | None:
    stages = _required_list(metadata, "stages", f"metadata for {stem}")
    if len(stages) != expected_stage_count:
        raise SndEvaluationInputError(
            f"{stem}: expected {expected_stage_count} merge-stage records"
        )
    recorded_coupling_links = 0
    for expected_index, raw_stage in enumerate(stages, start=1):
        stage = _as_object(raw_stage, f"{stem} merge stage {expected_index}")
        stage_index = _positive_integer(
            stage.get("stage_index"), f"{stem} merge stage index"
        )
        if stage_index != expected_index:
            raise SndEvaluationInputError(
                f"{stem}: merge-stage indices are not consecutive"
            )
        if _required_string(
            stage, "strategy", f"metadata for {stem}"
        ) != strategy:
            raise SndEvaluationInputError(
                f"{stem}: merge-stage strategy disagrees with final strategy"
            )
        pair_count = _positive_integer(
            stage.get("pair_count"), f"{stem} merge-stage pair count"
        )
        new_links = _positive_integer(
            stage.get("new_coupling_link_count"),
            f"{stem} merge-stage coupling count",
        )
        if pair_count != new_links:
            raise SndEvaluationInputError(
                f"{stem}: pair and coupling-link counts disagree"
            )
        recorded_coupling_links += new_links
        if (
            strategy == "heuristic_outer_face"
            and stage.get("planarity_certificate") is not True
        ):
            raise SndEvaluationInputError(
                f"{stem}: heuristic face stage lacks its planarity certificate"
            )
    if recorded_coupling_links != coupling_link_count:
        raise SndEvaluationInputError(
            f"{stem}: stage and final coupling-link counts disagree"
        )
    if strategy in {"exact_outer_face", "heuristic_outer_face"}:
        return True
    return None


def _read_object(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SndEvaluationInputError(f"cannot read {path}: {error}") from error
    return _as_object(value, str(path))


def _as_object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise SndEvaluationInputError(f"{label} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise SndEvaluationInputError(f"{label} must have string keys")
    return value


def _required_list(
    value: Mapping[str, object], key: str, label: str
) -> list[object]:
    result = value.get(key)
    if not isinstance(result, list):
        raise SndEvaluationInputError(f"{label}: {key} must be a JSON array")
    return result


def _required_string(
    value: Mapping[str, object], key: str, label: str
) -> str:
    return _string_value(value.get(key), f"{label}: {key}")


def _string_value(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SndEvaluationInputError(f"{label} must be a nonempty string")
    return value


def _nonnegative_integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SndEvaluationInputError(f"{label} must be a nonnegative integer")
    return value


def _positive_integer(value: object, label: str) -> int:
    number = _nonnegative_integer(value, label)
    if number == 0:
        raise SndEvaluationInputError(f"{label} must be positive")
    return number
