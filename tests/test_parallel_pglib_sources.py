from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from check.eval.merge_pglib_source_pools import merge_source_pools
from planar_instance_generation.evaluation import load_pglib_source_pool
from planar_instance_generation.io import read_pglib

ROOT = Path(__file__).resolve().parents[1]
CASE14 = ROOT / "check/data/pglib/pglib_opf_case14_ieee.m"
CASE3 = ROOT / "check/data/pglib/pglib_opf_case3_lmbd.m"


class ParallelPglibSourcesTests(unittest.TestCase):
    def make_part(self, root, source, result_name, feasible):
        case = read_pglib(source)
        output = root / "check/results" / result_name
        output.mkdir(parents=True)
        solution = output / "solution.npz"
        if feasible:
            np.savez_compressed(
                solution,
                bus_angles=np.zeros(len(case.bus)),
                branch_from_flows=np.zeros(len(case.branch)),
                branch_to_flows=np.zeros(len(case.branch)),
                generator_dispatch=np.zeros(len(case.gen)),
            )
        part = {
            "schema_version": 1,
            "run": {
                "sources": [
                    {
                        "path": str(source),
                        "sha256": sha256(source.read_bytes()).hexdigest(),
                    }
                ],
                "max_buses": 2000,
                "time_limit": 300,
                "threads": 1,
                "exclude_euler_bound_violations": True,
            },
            "backend": {"revision": "test"},
            "attempted": 1,
            "feasible": int(feasible),
            "excluded_euler_bound_violations": 0,
            "total_dc_opf_wall_seconds": 0.2,
            "excluded_over_bus_limit": [],
            "records": [
                {
                    "source_name": case.name,
                    "feasible": feasible,
                    "retained": feasible,
                    "objective": 1.25 if feasible else None,
                    "dc_opf_seconds": 0.2,
                    "optimizer_seconds": 0.1,
                }
            ],
            "solutions": {case.name: str(solution)} if feasible else {},
        }
        (output / "manifest.json").write_text(json.dumps(part))
        return {"source": str(source), "result_name": result_name}, part

    def test_merges_feasible_and_infeasible_sources_without_losing_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = [
                self.make_part(root, CASE14, "first", True)[0],
                self.make_part(root, CASE3, "second", False)[0],
            ]
            output = root / "merged"
            result = merge_source_pools(jobs, output, root=root)
            self.assertEqual(result["attempted"], 2)
            self.assertEqual(result["feasible"], 1)
            self.assertEqual(len(result["records"]), 2)
            self.assertEqual(
                len(load_pglib_source_pool(output / "manifest.json", path_root=root)), 1
            )

    def test_incompatible_backend_or_duplicate_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, _ = self.make_part(root, CASE14, "first", True)
            second, part = self.make_part(root, CASE3, "second", False)
            changed = deepcopy(part)
            changed["backend"]["revision"] = "different"
            (root / "check/results/second/manifest.json").write_text(
                json.dumps(changed)
            )
            with self.assertRaisesRegex(ValueError, "inconsistent settings"):
                merge_source_pools([first, second], root / "merged", root=root)
            with self.assertRaisesRegex(ValueError, "duplicate source"):
                merge_source_pools([first, first], root / "duplicate", root=root)

    def test_missing_source_cannot_release_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, _ = self.make_part(root, CASE14, "first", True)
            second = {"source": str(CASE3), "result_name": "missing"}
            with self.assertRaises(FileNotFoundError):
                merge_source_pools([first, second], root / "merged", root=root)
            self.assertFalse((root / "merged/manifest.json").exists())

    def test_invalid_solution_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, part = self.make_part(root, CASE14, "first", True)
            Path(next(iter(part["solutions"].values()))).write_bytes(b"invalid")
            with self.assertRaises(ValueError):
                merge_source_pools([first], root / "merged", root=root)
            self.assertFalse((root / "merged/manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
