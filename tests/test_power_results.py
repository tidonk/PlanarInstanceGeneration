from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tests.test_power_generation import _FeasibleGate, _source

from planar_instance_generation.evaluation import (
    ensure_original_power_plot,
    extract_power_pieces,
    merge_power_pieces,
    save_power_candidate,
)
from planar_instance_generation.plt import LatexCompilationError


class PowerResultTests(unittest.TestCase):
    def test_stage_pdf_failure_is_recorded_but_final_pdf_is_retained(self) -> None:
        sources = (_source("source_one"), _source("source_two"), _source("source_three"))
        pieces = extract_power_pieces(sources, 1.0, seed=2)
        candidate = merge_power_pieces(pieces, _FeasibleGate(), strategy="random", seed=3)
        with tempfile.TemporaryDirectory() as directory:
            compiled = SimpleNamespace(pdf=Path(directory) / "compiled.pdf")
            with patch(
                "planar_instance_generation.evaluation.power_results.compile_plot_pdf",
                side_effect=[
                    compiled,
                    compiled,
                    compiled,
                    compiled,
                    LatexCompilationError("stage too large"),
                    compiled,
                ],
            ):
                saved = save_power_candidate(candidate, directory, compile_pdfs=True)

            metadata = json.loads(saved.metadata_json.read_text(encoding="utf-8"))
            self.assertEqual(saved.final_plot_pdf, compiled.pdf)
            self.assertEqual(len(saved.stage_pdfs), 1)
            self.assertEqual(metadata["stage_pdf_failures"][0]["stage_index"], 1)
            failure = Path(metadata["stage_pdf_failures"][0]["error_file"])
            self.assertTrue(failure.is_file())
            self.assertIn("stage too large", failure.read_text(encoding="utf-8"))

    def test_saves_every_subinstance_final_and_stage_plot(self) -> None:
        sources = (_source("source_one"), _source("source_two"), _source("source_three"))
        pieces = extract_power_pieces(sources, 1.0, seed=2)
        candidate = merge_power_pieces(
            pieces,
            _FeasibleGate(),
            strategy="random",
            seed=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            saved = save_power_candidate(candidate, directory, compile_pdfs=False)

            self.assertEqual([path.stem for path in saved.pieces], ["3", "3_2", "3_3"])
            self.assertEqual(saved.final.stem, "9")
            self.assertTrue(saved.final.instance_file.is_file())
            self.assertEqual(
                saved.final.instance_file.parent, Path(directory) / "instances"
            )
            self.assertEqual(
                {path.instance_file.parent for path in saved.pieces},
                {Path(directory) / "subinstances"},
            )
            self.assertEqual(
                {path.stem for path in Path(directory, "instances").glob("*.m")},
                {saved.final.stem},
            )
            self.assertTrue(saved.metadata_json.is_file())
            metadata = json.loads(saved.metadata_json.read_text(encoding="utf-8"))
            self.assertEqual(
                [entry["source_artifact_stem"] for entry in metadata["source_bus_mappings"]],
                ["3", "3_2", "3_3"],
            )
            self.assertEqual(
                {bus["composite_bus"] for entry in metadata["source_bus_mappings"] for bus in entry["buses"]},
                set(range(1, 10)),
            )
            self.assertEqual(len(saved.stage_pdfs), 0)
            for path in (*saved.pieces, saved.final):
                self.assertTrue(path.plot_directory.joinpath("full.tikz").is_file())
                self.assertTrue(path.plot_directory.joinpath("metadata.json").is_file())
            for stage in (1, 2):
                plot = saved.final.merge_plot_directory / f"stage_{stage:03d}" / f"merge_stage_{stage:03d}.tikz"
                self.assertTrue(plot.is_file())
                text = plot.read_text(encoding="utf-8")
                self.assertIn("Source graphs", text)
                self.assertIn("Original", text)
                self.assertIn("Final", text)

            reused = save_power_candidate(
                candidate,
                directory,
                compile_pdfs=False,
                shared_piece_paths=saved.pieces,
            )
            self.assertEqual(reused.pieces, saved.pieces)
            self.assertEqual(
                len(list(Path(directory, "subinstances").glob("*.m"))), 3
            )

    def test_original_plot_is_cached_by_case_name(self) -> None:
        case = _source("source_once").case
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(
                ensure_original_power_plot(case, directory, compile_pdf=False)
            )
            self.assertIsNone(
                ensure_original_power_plot(case, directory, compile_pdf=False)
            )
            text = Path(directory, case.name, "full.tikz").read_text(encoding="utf-8")
        self.assertIn("source\\_once", text)


if __name__ == "__main__":
    unittest.main()
