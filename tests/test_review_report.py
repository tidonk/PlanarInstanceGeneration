from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from planar_instance_generation.plt import (
    ReviewBoundRow,
    ReviewFigure,
    ReviewInstance,
    write_a4_review_report,
)


class ReviewReportTests(unittest.TestCase):
    def test_orders_comparisons_before_one_page_per_instance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comparison = root / "max_degree.pdf"
            stage_one = root / "stage_001.pdf"
            stage_two = root / "stage_002.pdf"
            for path in (comparison, stage_one, stage_two):
                path.touch()

            artifacts = write_a4_review_report(
                root / "review_report",
                [ReviewFigure("Maximum degree", comparison)],
                [
                    ReviewInstance(
                        "120_2",
                        120,
                        (stage_one, stage_two),
                        detail="Three extracted sources",
                        preservation_rows=(
                            ReviewBoundRow("Nodes", "= 120", "120", "pass"),
                        ),
                    )
                ],
                compile_pdf=False,
            )

            source = artifacts.tex.read_text(encoding="utf-8")
            self.assertLess(source.index("Structural properties"), source.index("120\\_2"))
            self.assertLess(source.index("stage_001.pdf"), source.index("stage_002.pdf"))
            self.assertIn("a4paper", source)
            self.assertIn("Paper bound", source)
            self.assertIn("Slack", source)
            self.assertIn("Nodes & = 120 & 120 & n/a & pass", source)
            self.assertIsNone(artifacts.pdf)

    def test_requires_existing_pdf_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "existing PDF"):
                write_a4_review_report(
                    root / "report",
                    [ReviewFigure("Missing", root / "missing.pdf")],
                    [ReviewInstance("4", 4, (root / "also_missing.pdf",))],
                    compile_pdf=False,
                )

    def test_supports_domain_specific_size_units_without_changing_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comparison = root / "comparison.pdf"
            stage = root / "stage.pdf"
            comparison.touch()
            stage.touch()
            artifacts = write_a4_review_report(
                root / "report",
                [ReviewFigure("Density", comparison)],
                [ReviewInstance("12", 12, (stage,), size_unit="nodes")],
                compile_pdf=False,
            )
            source = artifacts.tex.read_text(encoding="utf-8")
            self.assertIn("Instance 12 (12 nodes)", source)
            self.assertNotIn("12 buses", source)


if __name__ == "__main__":
    unittest.main()
