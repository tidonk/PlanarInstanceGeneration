from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from planar_instance_generation.plt import BoxplotSeries, write_boxplot


class BoxplotTests(unittest.TestCase):
    def test_writes_tukey_statistics_and_publication_fragment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = write_boxplot(
                [
                    BoxplotSeries("Sources", (1.0, 2.0, 3.0, 4.0, 100.0)),
                    BoxplotSeries("Merged_1", (2.0, 3.0, 4.0)),
                ],
                Path(directory) / "degree",
                y_label="Maximum degree",
                title="Structural comparison",
            )

            fragment = artifacts.tikz.read_text(encoding="utf-8")
            data = json.loads(artifacts.coordinates_json.read_text(encoding="utf-8"))
            self.assertIn("boxplot prepared", fragment)
            self.assertIn("Merged\\_1", fragment)
            self.assertEqual(data["whisker_rule"], "1.5_iqr")
            self.assertEqual(data["series"][0]["summary"]["outliers"], [100.0])
            self.assertTrue(artifacts.standalone_tex.is_file())

    def test_rejects_empty_or_nonfinite_series(self) -> None:
        with self.assertRaises(ValueError):
            BoxplotSeries("empty", ())
        with self.assertRaises(ValueError):
            BoxplotSeries("bad", (float("nan"),))

    def test_seven_groups_receive_distinct_colors_and_steeper_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = write_boxplot(
                [
                    BoxplotSeries(f"Group {index}", (float(index),))
                    for index in range(7)
                ],
                Path(directory) / "strategies",
                y_label="Value",
                width_cm=16.0,
            )
            data = json.loads(artifacts.coordinates_json.read_text(encoding="utf-8"))
            self.assertEqual(len({row["color"] for row in data["series"]}), 7)
            fragment = artifacts.tikz.read_text(encoding="utf-8")
            self.assertIn("width=16.000cm", fragment)
            self.assertIn("rotate=32", fragment)


if __name__ == "__main__":
    unittest.main()
