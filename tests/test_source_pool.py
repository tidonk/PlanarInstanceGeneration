from __future__ import annotations

import unittest
import warnings

import numpy as np

from planar_instance_generation.evaluation import (
    EulerPlanarityWarning,
    screen_source_cases,
    select_cases_by_bus_limit,
)
from planar_instance_generation.gates import DcOpfResult, DcOpfSolution
from planar_instance_generation.io.pglib import PglibCase


def _case(name: str, bus_count: int, edges: list[tuple[int, int, int]]) -> PglibCase:
    bus = np.zeros((bus_count, 13))
    bus[:, 0] = np.arange(1, bus_count + 1)
    bus[:, 1] = 1
    bus[0, 1] = 3
    bus[:, 9] = 110
    bus[:, 11] = 1.1
    bus[:, 12] = 0.9

    gen = np.zeros((1, 10))
    gen[0, [0, 5, 6, 7, 8, 9]] = [1, 1, 100, 1, 100, 0]

    branch = np.zeros((len(edges), 13))
    for row, (source, target, status) in zip(branch, edges, strict=True):
        row[[0, 1, 3, 5, 6, 7, 10, 11, 12]] = [
            source, target, 0.1, 100, 100, 100, status, -360, 360
        ]

    return PglibCase(
        name=name,
        version="2",
        base_mva=100,
        bus=bus,
        gen=gen,
        branch=branch,
        gencost=np.array([[2, 0, 0, 3, 0.01, 1, 0]], dtype=float),
    )


def _solution(case: PglibCase) -> DcOpfSolution:
    return DcOpfSolution(
        bus_angles=np.zeros(case.bus.shape[0]),
        branch_from_flows=np.zeros(case.branch.shape[0]),
        branch_to_flows=np.zeros(case.branch.shape[0]),
        generator_dispatch=np.zeros(case.gen.shape[0]),
    )


class _Gate:
    def __init__(self, infeasible: set[str] = set()) -> None:
        self.infeasible = infeasible
        self.calls: list[str] = []

    def solve(self, case: PglibCase) -> DcOpfResult:
        self.calls.append(case.name)
        if case.name in self.infeasible:
            return DcOpfResult(False, "INFEASIBLE", None, 0.25)
        return DcOpfResult(True, "OPTIMAL", 12.5, 0.5, _solution(case))


class SourcePoolTests(unittest.TestCase):
    def test_solves_once_and_retains_only_feasible_sources(self) -> None:
        cases = [
            _case("feasible_case", 3, [(1, 2, 1), (2, 3, 1)]),
            _case("infeasible_case", 3, [(1, 2, 1)]),
        ]
        gate = _Gate({"infeasible_case"})

        pool = screen_source_cases(cases, gate)

        self.assertEqual(gate.calls, ["feasible_case", "infeasible_case"])
        self.assertEqual([source.case.name for source in pool.sources], ["feasible_case"])
        self.assertEqual(pool.attempted_count, 2)
        self.assertEqual(pool.feasible_count, 1)
        self.assertEqual(pool.total_dc_opf_seconds, 0.75)
        self.assertEqual(pool.records[0].objective, 12.5)

    def test_euler_filter_is_default_on_and_uses_active_simple_edges(self) -> None:
        complete_five = [(i, j, 1) for i in range(1, 6) for j in range(i + 1, 6)]
        dense = _case("dense", 5, complete_five)
        parallel = _case(
            "parallel",
            3,
            [(1, 2, 1)] * 10 + [(2, 3, 1), (1, 3, 0)],
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gate = _Gate()
            pool = screen_source_cases([dense, parallel], gate)

        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, EulerPlanarityWarning)
        self.assertEqual(gate.calls, ["parallel"])
        self.assertEqual([source.case.name for source in pool.sources], ["parallel"])
        self.assertEqual(pool.euler_excluded_count, 1)
        self.assertTrue(pool.records[0].euler_bound_exceeded)
        self.assertTrue(pool.records[0].excluded_by_euler_bound)
        self.assertFalse(pool.records[0].retained)
        self.assertIsNone(pool.records[0].feasible)
        self.assertEqual(pool.records[0].status, "EULER_BOUND_EXCEEDED")
        self.assertEqual(pool.records[0].active_simple_edge_count, 10)
        self.assertFalse(pool.records[1].euler_bound_exceeded)
        self.assertEqual(pool.records[1].active_branch_count, 11)
        self.assertEqual(pool.records[1].active_simple_edge_count, 2)

        retained = screen_source_cases(
            [dense, parallel],
            _Gate(),
            exclude_euler_bound_violations=False,
            warn_on_euler_bound=False,
        )
        self.assertEqual(
            [source.case.name for source in retained.sources], ["dense", "parallel"]
        )
        self.assertEqual(retained.euler_excluded_count, 0)
        self.assertTrue(retained.records[0].retained)

    def test_euler_filter_option_requires_a_bool(self) -> None:
        with self.assertRaisesRegex(
            TypeError, "exclude_euler_bound_violations must be a bool"
        ):
            screen_source_cases(
                [],
                _Gate(),
                exclude_euler_bound_violations=1,  # type: ignore[arg-type]
            )

    def test_bus_limit_is_explicit_and_inclusive(self) -> None:
        small = _case("small", 2, [(1, 2, 1)])
        exact = _case("exact", 3, [(1, 2, 1)])
        large = _case("large", 4, [(1, 2, 1)])

        selection = select_cases_by_bus_limit([small, exact, large], max_buses=3)

        self.assertEqual([case.name for case in selection.included], ["small", "exact"])
        self.assertEqual([case.name for case in selection.excluded], ["large"])
        with self.assertRaises(ValueError):
            select_cases_by_bus_limit([], max_buses=0)

    def test_rejects_solution_vectors_for_a_different_case(self) -> None:
        case = _case("case_three", 3, [(1, 2, 1), (2, 3, 1)])

        class WrongGate:
            def solve(self, _: PglibCase) -> DcOpfResult:
                wrong = _solution(_case("case_two", 2, [(1, 2, 1)]))
                return DcOpfResult(True, "OPTIMAL", 0.0, 0.1, wrong)

        with self.assertRaisesRegex(ValueError, "bus_angles has length 2; expected 3"):
            screen_source_cases([case], WrongGate())


if __name__ == "__main__":
    unittest.main()
