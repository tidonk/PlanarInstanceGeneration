from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tests.test_snd_generation import _FeasibleGate, _source

from planar_instance_generation.evaluation.snd_generation import (
    extract_snd_pieces,
    merge_snd_pieces,
)
from planar_instance_generation.evaluation.snd_results import (
    ensure_original_snd_plot,
    save_snd_candidate,
)
from planar_instance_generation.io import read_sndlib


class SndResultTests(unittest.TestCase):
    def test_saves_all_xml_instances_and_composite_left_stage_plots(self) -> None:
        sources = (_source("one"), _source("two"), _source("three"))
        pieces = extract_snd_pieces(sources, 1.0, seed=2)
        candidate = merge_snd_pieces(
            pieces,
            _FeasibleGate(),
            strategy="random",
            seed=3,
            source_stratum_size=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "attempt.log"
            log.write_text("gurobi test log\n", encoding="utf-8")
            saved = save_snd_candidate(
                candidate,
                directory,
                compile_pdfs=False,
                gate_log_path=log,
            )

            self.assertEqual(
                [path.stem for path in saved.pieces],
                ["3", "3_2", "3_3"],
            )
            self.assertEqual(saved.final.stem, "9")
            self.assertEqual(len(read_sndlib(saved.final.instance_file).nodes), 9)
            self.assertEqual(
                saved.final.instance_file.parent, Path(directory) / "instances"
            )
            self.assertEqual(
                {path.instance_file.parent for path in saved.pieces},
                {Path(directory) / "subinstances"},
            )
            self.assertEqual(
                {path.stem for path in Path(directory, "instances").glob("*.xml")},
                {saved.final.stem},
            )
            self.assertEqual(
                saved.final.gurobi_log.read_text(encoding="utf-8"),
                "gurobi test log\n",
            )
            metadata = json.loads(saved.metadata_json.read_text(encoding="utf-8"))
            self.assertEqual(metadata["routing_unit"], "MBITPERSEC")
            self.assertEqual(
                [entry["source_artifact_stem"] for entry in metadata["source_mappings"]],
                ["3", "3_2", "3_3"],
            )
            self.assertTrue(metadata["serialized_demand_provenance"])
            for stage in (1, 2):
                plot = (
                    saved.final.merge_plot_directory
                    / f"stage_{stage:03d}"
                    / f"merge_stage_{stage:03d}.tikz"
                )
                text = plot.read_text(encoding="utf-8")
                self.assertIn("Source graphs", text)
                self.assertIn("Original", text)
                self.assertIn("Step 1", text)
                self.assertIn("Final", text)
                expected_labels = stage + 1
                for source_index in range(1, expected_labels + 1):
                    self.assertIn(f"G\\_{source_index}:", text)

            reused = save_snd_candidate(
                candidate,
                directory,
                compile_pdfs=False,
                gate_log_path=log,
                shared_piece_paths=saved.pieces,
            )
            self.assertEqual(reused.pieces, saved.pieces)
            self.assertEqual(
                len(list(Path(directory, "subinstances").glob("*.xml"))), 3
            )

    def test_original_plot_is_cached_once(self) -> None:
        source = _source("source_once")
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(
                ensure_original_snd_plot(
                    source.network,
                    source.source_name,
                    directory,
                    compile_pdf=False,
                )
            )
            self.assertIsNone(
                ensure_original_snd_plot(
                    source.network,
                    source.source_name,
                    directory,
                    compile_pdf=False,
                )
            )
            text = Path(directory, source.source_name, "full.tikz").read_text(
                encoding="utf-8"
            )
        self.assertIn("source\\_once", text)


if __name__ == "__main__":
    unittest.main()
