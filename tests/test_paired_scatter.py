from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from planar_instance_generation.plt import PairedScatterPoint, write_paired_scatter


class PairedScatterTests(unittest.TestCase):
    def test_writes_identity_line_and_highlight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = write_paired_scatter(
                (
                    PairedScatterPoint("i001", 2.0, 2.0),
                    PairedScatterPoint("i070", 4.0, 3.0, highlighted=True),
                ),
                Path(directory) / "paired",
                x_label="Exact",
                y_label="Heuristic",
            )
            fragment = artifacts.tikz.read_text(encoding="utf-8")
            data = json.loads(artifacts.coordinates_json.read_text(encoding="utf-8"))
            self.assertIn("(0,0)", fragment)
            self.assertIn("diamond*", fragment)
            self.assertIn("i070", fragment)
            self.assertEqual(data["kind"], "paired_scatter")
            self.assertTrue(data["identity_line"])
            self.assertTrue(data["points"][1]["highlighted"])

    def test_rejects_empty_or_nonfinite_points(self) -> None:
        with self.assertRaises(ValueError):
            write_paired_scatter(
                (),
                Path("unused"),
                x_label="x",
                y_label="y",
            )
        with self.assertRaises(ValueError):
            PairedScatterPoint("bad", float("nan"), 1.0)


if __name__ == "__main__":
    unittest.main()
