from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from planar_instance_generation.evaluation import (
    SourcePoolManifestError,
    load_pglib_source_pool,
)
from planar_instance_generation.io import read_pglib


ROOT = Path(__file__).resolve().parents[1]
CASE14 = ROOT / "check/data/pglib/pglib_opf_case14_ieee.m"


class SourcePoolIoTests(unittest.TestCase):
    def test_loads_and_validates_cached_solution(self) -> None:
        case = read_pglib(CASE14)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            solution = root / "solution.npz"
            np.savez_compressed(
                solution,
                bus_angles=np.zeros(case.bus.shape[0]),
                branch_from_flows=np.zeros(case.branch.shape[0]),
                branch_to_flows=np.zeros(case.branch.shape[0]),
                generator_dispatch=np.zeros(case.gen.shape[0]),
            )
            manifest = self._manifest(solution)
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")

            loaded = load_pglib_source_pool(path, path_root=ROOT)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].case.name, case.name)
        self.assertEqual(loaded[0].objective, 1.25)

    def test_rejects_hash_and_vector_mismatch(self) -> None:
        case = read_pglib(CASE14)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            solution = root / "solution.npz"
            np.savez_compressed(
                solution,
                bus_angles=np.zeros(1),
                branch_from_flows=np.zeros(case.branch.shape[0]),
                branch_to_flows=np.zeros(case.branch.shape[0]),
                generator_dispatch=np.zeros(case.gen.shape[0]),
            )
            manifest = self._manifest(solution)
            path = root / "manifest.json"
            bad_hash = json.loads(json.dumps(manifest))
            bad_hash["run"]["sources"][0]["sha256"] = "bad"
            path.write_text(json.dumps(bad_hash), encoding="utf-8")
            with self.assertRaisesRegex(SourcePoolManifestError, "hash mismatch"):
                load_pglib_source_pool(path, path_root=ROOT)

            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "length 1"):
                load_pglib_source_pool(path, path_root=ROOT)

    def _manifest(self, solution: Path) -> dict[str, object]:
        case = read_pglib(CASE14)
        return {
            "schema_version": 1,
            "run": {
                "sources": [
                    {
                        "path": str(CASE14.relative_to(ROOT)),
                        "sha256": sha256(CASE14.read_bytes()).hexdigest(),
                    }
                ]
            },
            "feasible": 1,
            "records": [
                {
                    "source_name": case.name,
                    "feasible": True,
                    "objective": 1.25,
                    "dc_opf_seconds": 0.2,
                    "optimizer_seconds": 0.1,
                }
            ],
            "solutions": {case.name: str(solution)},
        }


if __name__ == "__main__":
    unittest.main()
