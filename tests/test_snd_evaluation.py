from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import networkx as nx

from planar_instance_generation.evaluation import EvaluationGraph
from planar_instance_generation.evaluation.snd_generation import (
    extract_snd_pieces,
    merge_snd_pieces,
)
from planar_instance_generation.evaluation.snd_results import save_snd_candidate
from planar_instance_generation.evaluation.snd_review import (
    SndEvaluationInputError,
    audit_snd_preservation,
    load_generated_snd_artifacts,
)
from tests.test_snd_generation import _FeasibleGate, _source


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "evaluate_sndlib_for_tests", ROOT / "check/scripts/evaluate_sndlib.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_EVALUATE_SNDLIB = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EVALUATE_SNDLIB)


class SndEvaluationTests(unittest.TestCase):
    def test_loads_distributed_skewness_bounds_by_instance_name(self) -> None:
        groups = {
            "Original sources": (EvaluationGraph("source", nx.path_graph(3)),),
            "Extracted subinstances": (EvaluationGraph("piece", nx.path_graph(3)),),
            "Composite: Random": (EvaluationGraph("composite", nx.complete_graph(5)),),
        }
        settings = {
            "mode": "exact_branch_and_cut",
            "time_limit_seconds": 3600.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoints = Path(directory) / "i001"
            checkpoints.mkdir()
            for strategy, rows in (
                ("random", (("piece", 0, 0), ("composite", 1, 2))),
                ("high_degree", (("piece", 0, 0),)),
            ):
                graph_results = {
                    f"result:{name}": {
                        "instance": name,
                        "skewness_lower_bound": lower,
                        "skewness_upper_bound": upper,
                        # Duplicate source certificates can have identical
                        # bounds but different run metadata across workers.
                        "status": strategy,
                        "runtime_seconds": 1.0 if strategy == "random" else 2.0,
                    }
                    for name, lower, upper in rows
                }
                (checkpoints / f"{strategy}.json").write_text(
                    json.dumps({"settings": settings, "graph_results": graph_results}),
                    encoding="utf-8",
                )

            loaded = _EVALUATE_SNDLIB._load_cohort_bounds(Path(directory), groups)

        self.assertEqual(loaded[("Extracted subinstances", "piece")].upper_bound, 0)
        self.assertEqual(loaded[("Composite: Random", "composite")].lower_bound, 1)
        self.assertEqual(loaded[("Composite: Random", "composite")].upper_bound, 2)

    def test_loads_serialized_provenance_and_certifies_face_strategy(self) -> None:
        pieces = extract_snd_pieces((_source("one"), _source("two")), 1.0, seed=2)
        candidate = merge_snd_pieces(
            pieces,
            _FeasibleGate(),
            strategy="heuristic_outer_face",
            seed=3,

        )
        with tempfile.TemporaryDirectory() as directory:
            saved = save_snd_candidate(candidate, directory, compile_pdfs=False)
            artifacts = load_generated_snd_artifacts(directory)
            audits = audit_snd_preservation(artifacts)

            self.assertEqual(len(artifacts.extracted), 2)
            self.assertEqual(len(artifacts.finals), 1)
            final = artifacts.finals[0]
            self.assertEqual(final.stem, saved.final.stem)
            self.assertEqual(len(final.coupling_edges), 3)
            audit = audits[final.stem]
            self.assertTrue(audit.sequential_matching)
            self.assertTrue(audit.noncrossing_matching_certified)
            self.assertTrue(
                all(
                    row.numerically_within_bound is not False
                    for row in audit.properties
                )
            )

    def test_rejects_coupling_metadata_that_disagrees_with_xml(self) -> None:
        pieces = extract_snd_pieces((_source("one"), _source("two")), 1.0, seed=2)
        candidate = merge_snd_pieces(
            pieces,
            _FeasibleGate(),
            strategy="random",
            seed=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            saved = save_snd_candidate(candidate, directory, compile_pdfs=False)
            metadata = json.loads(saved.metadata_json.read_text(encoding="utf-8"))
            metadata["coupling_links"][0]["source"] = "not-in-the-xml"
            saved.metadata_json.write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                SndEvaluationInputError, "disagrees with serialized XML"
            ):
                load_generated_snd_artifacts(Path(directory))


if __name__ == "__main__":
    unittest.main()
