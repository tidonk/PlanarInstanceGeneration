from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from planar_instance_generation.io import PglibFormatError, read_pglib, write_pglib


ROOT = Path(__file__).resolve().parents[1]
CASE14 = ROOT / "check/data/pglib/pglib_opf_case14_ieee.m"


class PglibIoTests(unittest.TestCase):
    def test_reads_downloaded_case(self) -> None:
        case = read_pglib(CASE14)
        self.assertEqual(case.name, "pglib_opf_case14_ieee")
        self.assertEqual(case.bus.shape, (14, 13))
        self.assertEqual(case.branch.shape[1], 13)
        self.assertEqual(case.base_mva, 100.0)

    def test_round_trip_preserves_all_numeric_tables(self) -> None:
        original = read_pglib(CASE14)
        with tempfile.TemporaryDirectory() as directory:
            reread = read_pglib(write_pglib(original, Path(directory) / "case.m"))
        self.assertEqual(reread.name, original.name)
        self.assertEqual(reread.version, original.version)
        self.assertEqual(reread.base_mva, original.base_mva)
        for field in ("bus", "gen", "branch", "gencost"):
            np.testing.assert_array_equal(getattr(reread, field), getattr(original, field))

    def test_topology_tracks_parallel_branches_and_ignores_inactive_and_loops(self) -> None:
        text = CASE14.read_text(encoding="utf-8")
        case = read_pglib(CASE14)
        branch = np.vstack(
            [
                case.branch,
                case.branch[0],
                np.array([1, 3, 0, 1, 0, 1, 1, 1, 0, 0, 0, -30, 30], dtype=float),
                np.array([1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, -30, 30], dtype=float),
            ]
        )
        case.branch = branch
        graph = case.topology()
        self.assertEqual(graph[1][2]["multiplicity"], 2)
        self.assertFalse(graph.has_edge(1, 3))
        self.assertNotIn((1, 1), graph.edges)
        self.assertIn("function mpc", text)

    def test_rejects_code_in_numeric_matrix(self) -> None:
        malicious = CASE14.read_text(encoding="utf-8").replace("\t1\t 3", "\tsystem('oops')\t 3", 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.m"
            path.write_text(malicious, encoding="utf-8")
            with self.assertRaises(PglibFormatError):
                read_pglib(path)

    def test_rejects_unknown_matpower_fields(self) -> None:
        text = CASE14.read_text(encoding="utf-8") + "\nmpc.dcline = [1 2 3];\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.m"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(PglibFormatError, "unsupported"):
                read_pglib(path)


if __name__ == "__main__":
    unittest.main()
