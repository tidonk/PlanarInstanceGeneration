from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from planar_instance_generation.plt import EcdfSeries, write_ecdf


class EcdfTests(unittest.TestCase):
    def test_writes_empirical_steps_and_all_series(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = write_ecdf(
                (
                    EcdfSeries("Random", (1.0, 2.0, 2.0), "0072B2"),
                    EcdfSeries("Exact face", (0.0, 1.0, 1.0), "#D55E00"),
                ),
                Path(directory) / "added",
                x_label="Added skewness",
            )
            fragment = artifacts.tikz.read_text(encoding="utf-8")
            data = json.loads(artifacts.coordinates_json.read_text(encoding="utf-8"))
            self.assertIn("Fraction of instances", fragment)
            self.assertIn("\\addlegendentry{Exact face}", fragment)
            self.assertEqual(data["kind"], "ecdf")
            self.assertEqual(
                data["series"][0]["step_coordinates"],
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1 / 3], [2.0, 1 / 3], [2.0, 1.0], [2.0, 1.0]],
            )

    def test_rejects_empty_or_nonfinite_series(self) -> None:
        with self.assertRaises(ValueError):
            write_ecdf((), Path("unused"), x_label="x")
        with self.assertRaises(ValueError):
            EcdfSeries("bad", (float("nan"),), "0072B2")
        with self.assertRaises(ValueError):
            EcdfSeries("bad", (1.0,), "not-a-color")


if __name__ == "__main__":
    unittest.main()
