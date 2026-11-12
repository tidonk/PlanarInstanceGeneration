from __future__ import annotations

import unittest

import numpy as np

from tests.test_power_generation import _FeasibleGate, _source

from planar_instance_generation.evaluation import FeasibleSource, run_power_experiment
from planar_instance_generation.gates import DcOpfResult, DcOpfSolution
from planar_instance_generation.io import PglibCase


class _InfeasibleGate:
    def solve(self, case: PglibCase, /) -> DcOpfResult:
        return DcOpfResult(False, "INFEASIBLE", None, 0.1)


def _symmetric_source(name: str) -> FeasibleSource:
    bus = np.asarray(
        [
            [1, 3, 0, 0, 0, 0, 1, 1, 0, 100, 1, 1.1, 0.9],
            [2, 1, 0, 0, 0, 0, 1, 1, 0, 100, 1, 1.1, 0.9],
            [3, 1, 0, 0, 0, 0, 1, 1, 0, 100, 1, 1.1, 0.9],
        ],
        dtype=float,
    )
    gen = np.asarray([[1, 0, 0, 1, -1, 1, 100, 1, 1, 0]], dtype=float)
    branch = np.asarray(
        [
            [1, 2, 0, 0.1, 0, 10, 10, 10, 0, 0, 1, -360, 360],
            [1, 3, 0, 0.1, 0, 10, 10, 10, 0, 0, 1, -360, 360],
        ],
        dtype=float,
    )
    gencost = np.asarray([[2, 0, 0, 2, 1, 0]], dtype=float)
    case = PglibCase(name, "2", 100, bus, gen, branch, gencost)
    solution = DcOpfSolution(np.zeros(3), np.zeros(2), np.zeros(2), np.zeros(1))
    return FeasibleSource(case, solution, 0.0, 0.0, 0.0)

def _triangulated_source(name: str, node_count: int = 12) -> FeasibleSource:
    bus = np.asarray(
        [
            [node + 1, 3 if node == 0 else 1, 0, 0, 0, 0, 1, 1, 0, 100, 1, 1.1, 0.9]
            for node in range(node_count)
        ],
        dtype=float,
    )
    gen = np.asarray([[1, 0, 0, 1, -1, 1, 100, 1, 1, 0]], dtype=float)
    edges = {
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    }
    previous = 2
    for node in range(4, node_count):
        edges.update({(0, node), (1, node), (previous, node)})
        previous = node
    branch = np.asarray(
        [
            [first + 1, second + 1, 0, 0.1, 0, 10, 10, 10, 0, 0, 1, -360, 360]
            for first, second in sorted(edges)
        ],
        dtype=float,
    )
    gencost = np.asarray([[2, 0, 0, 2, 1, 0]], dtype=float)
    case = PglibCase(name, "2", 100, bus, gen, branch, gencost)
    solution = DcOpfSolution(
        np.zeros(node_count),
        np.zeros(len(branch)),
        np.zeros(len(branch)),
        np.zeros(1),
    )
    return FeasibleSource(case, solution, 0.0, 0.0, 0.0)


class PowerExperimentTests(unittest.TestCase):
    def test_shared_draws_retry_and_acceptance_statistics(self) -> None:
        sources = (_source("one"), _source("two"))

        def gate_factory(strategy: str, output: int, attempt: int) -> object:
            if strategy == "random" and attempt == 1:
                return _InfeasibleGate()
            return _FeasibleGate()

        result = run_power_experiment(
            sources,
            gate_factory,  # type: ignore[arg-type]
            strategies=("random", "high_degree"),
            requested_per_strategy=1,
            num_subinstances=(2, 2),
            target_size=(6, 6),
            size_range=(1.0, 1.0),
            max_attempts_per_output=3,
            seed=12,
        )

        self.assertEqual(len(result.accepted), 2)
        self.assertFalse(result.failures)
        first_attempt = [record for record in result.attempts if record.attempt_index == 1]
        self.assertEqual({record.target_size for record in result.attempts}, {first_attempt[0].target_size})
        self.assertEqual({record.source_names for record in first_attempt}, {first_attempt[0].source_names})
        self.assertEqual(first_attempt[0].status, "gate_infeasible")
        self.assertEqual(first_attempt[1].status, "paired_retry")
        self.assertEqual(first_attempt[1].individual_status, "not_run")
        self.assertEqual(first_attempt[1].retry_trigger_strategy, "random")
        self.assertEqual(first_attempt[1].retry_trigger_status, "gate_infeasible")
        self.assertTrue(all(item.attempt_index == 2 for item in result.accepted))
        self.assertEqual(sum(result.attempts[-1].target_nodes), result.attempts[-1].target_size)

    def test_repeated_isomorphic_outputs_are_accepted(self) -> None:
        source = _symmetric_source("same")
        result = run_power_experiment(
            (source,),
            lambda *_: _FeasibleGate(),
            strategies=("high_degree",),
            requested_per_strategy=2,
            num_subinstances=(2, 2),
            target_size=(6, 6),
            size_range=(1.0, 1.0),
            max_attempts_per_output=2,
            seed=0,
        )

        self.assertEqual(len(result.accepted), 2)
        self.assertFalse(result.failures)
        self.assertTrue(all(record.status == "accepted" for record in result.attempts))

    def test_size_limit_and_validation_fail_cleanly(self) -> None:
        source = _source("three")
        result = run_power_experiment(
            (source,),
            lambda *_: _FeasibleGate(),
            strategies=("random",),
            requested_per_strategy=1,
            num_subinstances=(2, 2),
            target_size=(5, 5),
            size_range=(1.0, 1.0),
            max_attempts_per_output=1,
        )
        self.assertEqual(result.attempts[0].status, "size_limit")
        self.assertEqual(result.attempts[0].target_size, 5)
        self.assertEqual(result.failures, (("random", 1),))
        with self.assertRaisesRegex(ValueError, "at most seven"):
            run_power_experiment(
                (source,),
                lambda *_: _FeasibleGate(),
                strategies=("random",),
                requested_per_strategy=1,
                num_subinstances=(8, 8),
                target_size=(6, 6),
                size_range=(1.0, 1.0),
                max_attempts_per_output=1,
            )

    def test_exact_target_is_inclusive_and_ranges_are_validated(self) -> None:
        source = _source("three")
        boundary = run_power_experiment(
            (source,),
            lambda *_: _FeasibleGate(),
            strategies=("random",),
            requested_per_strategy=1,
            num_subinstances=(2, 2),
            target_size=(6, 6),
            size_range=(1.0, 1.0),
            max_attempts_per_output=1,
        )
        self.assertEqual(len(boundary.accepted), 1)
        self.assertEqual(boundary.accepted[0].candidate.case.bus.shape[0], 6)

        with self.assertRaisesRegex(ValueError, "must not exceed"):
            run_power_experiment(
                (source,),
                lambda *_: _FeasibleGate(),
                strategies=("random",),
                requested_per_strategy=1,
                num_subinstances=(2, 2),
                target_size=(7, 6),
                size_range=(1.0, 1.0),
                max_attempts_per_output=1,
            )

    def test_outer_face_shortage_retries_the_whole_paired_attempt(self) -> None:
        result = run_power_experiment(
            (_triangulated_source("triangulated"),),
            lambda *_: _FeasibleGate(),
            strategies=("random", "exact_outer_face"),
            requested_per_strategy=1,
            num_subinstances=(2, 2),
            target_size=(24, 24),
            size_range=(1.0, 1.0),
            max_attempts_per_output=1,
            seed=5,
        )

        self.assertEqual(
            result.failures,
            (("random", 1), ("exact_outer_face", 1)),
        )
        records = {record.strategy: record for record in result.attempts}
        self.assertEqual(records["random"].status, "paired_retry")
        self.assertEqual(records["exact_outer_face"].status, "selection_failed")
        self.assertEqual(records["random"].individual_status, "accepted")
        self.assertEqual(records["random"].retry_trigger_strategy, "exact_outer_face")
        self.assertEqual(records["random"].retry_trigger_status, "selection_failed")
        self.assertEqual(
            records["random"].candidate_statistics["coupling_pair_counts"],
            (4,),
        )
        self.assertIn(
            "requested 4 distinct boundary nodes",
            records["exact_outer_face"].reason or "",
        )
        self.assertIn("found 3 and 3", records["exact_outer_face"].reason or "")

    def test_heuristic_strategy_flows_through_experiment_configuration(self) -> None:
        result = run_power_experiment(
            (_source("one"), _source("two")),
            lambda *_: _FeasibleGate(),
            strategies=("heuristic_outer_face",),
            requested_per_strategy=1,
            num_subinstances=(2, 2),
            target_size=(6, 6),
            size_range=(1.0, 1.0),
            max_attempts_per_output=1,

            seed=4,
        )

        self.assertEqual(len(result.accepted), 1)
        self.assertEqual(
            result.accepted[0].candidate.strategy, "heuristic_outer_face"
        )
        stage_statistics = result.accepted[0].candidate.stages[0].statistics()
        self.assertEqual(stage_statistics["iterations_per_graph"], 10)


if __name__ == "__main__":
    unittest.main()
