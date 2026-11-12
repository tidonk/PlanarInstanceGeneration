from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import networkx as nx

from planar_instance_generation.evaluation.planarization_batch import (
    PlanarizationTask,
    solve_planarization_batch,
)


class PlanarizationBatchTests(unittest.TestCase):
    def test_transient_nfs_lock_creation_race_is_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = PlanarizationTask(
                nx.cycle_graph(5), 7, root / "cycle.log", "cycle"
            )
            original_open = Path.open
            attempts = 0

            def flaky_open(path, *args, **kwargs):
                nonlocal attempts
                if path.name == "cycle.lock" and attempts < 2:
                    attempts += 1
                    raise FileExistsError(17, "simulated NFS race", str(path))
                return original_open(path, *args, **kwargs)

            with mock.patch.object(Path, "open", flaky_open), mock.patch(
                "planar_instance_generation.evaluation.planarization_batch.time.sleep"
            ):
                result = solve_planarization_batch(
                    (task,), time_limit_seconds=0.1,
                     solver_threads=1, workers=1,
                    checkpoint_directory=root / "checkpoints",
                )
            self.assertEqual(attempts, 2)
            self.assertTrue(result[0].exact)

    def test_parallel_fast_paths_preserve_task_order_and_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tasks = tuple(
                PlanarizationTask(
                    nx.cycle_graph(size),
                    seed=seed,
                    log_path=Path(temporary) / f"{size}.log",
                    cache_key=f"cycle_{size}",
                )
                for size, seed in ((7, 31), (5, 29), (6, 23))
            )
            results = solve_planarization_batch(
                tasks,
                time_limit_seconds=0.1,

                solver_threads=1,
                workers=2,
                checkpoint_directory=Path(temporary) / "checkpoints",
            )
            checkpoint_count = len(
                list((Path(temporary) / "checkpoints").glob("*.json"))
            )

        self.assertEqual(checkpoint_count, 3)
        self.assertEqual(
            tuple(result.original_node_count for result in results), (7, 5, 6)
        )
        self.assertEqual(tuple(result.seed for result in results), (31, 29, 23))
        self.assertTrue(all(result.exact for result in results))

    def test_completed_checkpoint_is_reused_without_resolving(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = PlanarizationTask(
                nx.cycle_graph(8),
                seed=41,
                log_path=root / "cycle.log",
                cache_key="cycle_8",
            )
            arguments = {
                "time_limit_seconds": 0.1,
                "solver_threads": 1,
                "workers": 1,
                "checkpoint_directory": root / "checkpoints",
            }
            first = solve_planarization_batch((task,), **arguments)
            with mock.patch(
                "planar_instance_generation.evaluation.planarization_batch."
                "maximum_planar_subgraph",
                side_effect=AssertionError("checkpoint was not reused"),
            ):
                second = solve_planarization_batch((task,), **arguments)

        self.assertEqual(first[0].statistics(), second[0].statistics())

    def test_rejects_invalid_worker_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "workers"):
            solve_planarization_batch(
                (),
                time_limit_seconds=1.0,

                solver_threads=1,
                workers=0,
            )


if __name__ == "__main__":
    unittest.main()
