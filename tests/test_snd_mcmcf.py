from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from time import perf_counter
import unittest

from planar_instance_generation.core.snd import SndSolution
from planar_instance_generation.evaluation.snd_source_pool import (
    CONSTRUCTIVE_METHOD,
    ConstructiveCapacityPlanError,
    ConstructiveSndCapacityPlan,
    NamedSndSource,
    construct_snd_capacity_plan,
    screen_snd_source_cases,
)
from planar_instance_generation.evaluation.snd_source_pool_io import (
    SndSourcePoolManifestError,
    load_snd_source_pool,
    write_snd_source_solution,
)
from planar_instance_generation.gates.mcmcf import GurobiMcmcfGate
from planar_instance_generation.gates.snd_base import (
    McmcfModelEstimate,
    McmcfResult,
)
from planar_instance_generation.io.sndlib import (
    SndlibCapacityModule,
    SndlibDemand,
    SndlibLink,
    SndlibMetadata,
    SndlibNetwork,
    SndlibNode,
    write_sndlib,
)


def _module(capacity: float, cost: float = 1.0) -> SndlibCapacityModule:
    return SndlibCapacityModule(capacity, cost)


def _network(*, restricted: bool = False) -> SndlibNetwork:
    return SndlibNetwork(
        "1.0",
        "pixel",
        (
            SndlibNode("A", 0, 0),
            SndlibNode("B", 1, 0),
            SndlibNode("C", 2, 0),
        ),
        (
            SndlibLink(
                "l_ab",
                "A",
                "B",
                preinstalled_module=_module(1, 9),
                additional_modules=(_module(5, 2),),
                routing_cost=1,
                setup_cost=3,
            ),
            # Reverse XML orientation exercises signed undirected flow.
            SndlibLink(
                "l_cb",
                "C",
                "B",
                additional_modules=(_module(4, 1),),
                routing_cost=1,
            ),
            SndlibLink(
                "l_ac",
                "A",
                "C",
                additional_modules=(_module(10, 10),),
                routing_cost=10,
            ),
        ),
        (
            SndlibDemand(
                "d",
                "A",
                "C",
                6,
                max_path_length=4 if restricted else None,
            ),
        ),
        SndlibMetadata("1", "1", "Mbps", "test"),
    )


def _dense_network() -> SndlibNetwork:
    node_ids = tuple("ABCDE")
    return SndlibNetwork(
        "1.0",
        "pixel",
        tuple(
            SndlibNode(node_id, index, 0)
            for index, node_id in enumerate(node_ids)
        ),
        tuple(
            SndlibLink(
                f"l_{source.lower()}{target.lower()}",
                source,
                target,
                additional_modules=(_module(10),),
            )
            for source_index, source in enumerate(node_ids)
            for target in node_ids[source_index + 1 :]
        ),
        (SndlibDemand("d", "A", "E", 1),),
    )


class _ExactPlanGate:
    max_variables = 10_000
    default_routing_cost = 1.0

    def estimate(self, network: SndlibNetwork, /) -> McmcfModelEstimate:
        return McmcfModelEstimate.for_network(network)

    def solve(
        self,
        network: SndlibNetwork,
        installed_capacities: dict[str, float],
        /,
        *,
        source_name: str | None = None,
    ) -> McmcfResult:
        started = perf_counter()
        plan = construct_snd_capacity_plan(network, source_name=source_name)
        solution = SndSolution(installed_capacities, plan.solution.flows)
        return McmcfResult(
            feasible=True,
            status="OPTIMAL",
            objective=plan.routing_cost,
            runtime_seconds=perf_counter() - started,
            optimizer_seconds=0.0,
            solution=solution,
            estimate=self.estimate(network),
        )


class ConstructivePlanTests(unittest.TestCase):
    def test_signed_flow_and_module_rounding(self) -> None:
        plan = construct_snd_capacity_plan(_network(), source_name="tiny")
        self.assertEqual(plan.demand_paths["d"], ("l_ab", "l_cb"))
        self.assertEqual(plan.solution.flow("d", "l_ab"), 6)
        self.assertEqual(plan.solution.flow("d", "l_cb"), -6)
        self.assertEqual(plan.solution.installed_capacities["l_ab"], 6)
        self.assertEqual(plan.solution.installed_capacities["l_cb"], 8)
        self.assertEqual(plan.module_copies["l_ab"], (1,))
        self.assertEqual(plan.module_copies["l_cb"], (2,))
        self.assertEqual(plan.additional_module_count, 3)
        self.assertEqual(plan.routing_cost, 12)
        self.assertEqual(plan.total_design_cost, 7)
        self.assertFalse(plan.design_optimal)

    def test_fixed_only_link_uses_residual_capacity(self) -> None:
        network = SndlibNetwork(
            "1.0",
            "pixel",
            tuple(SndlibNode(node, index, 0) for index, node in enumerate("ABC")),
            (
                SndlibLink("short", "A", "C", preinstalled_module=_module(5)),
                SndlibLink("ab", "A", "B", additional_modules=(_module(10),)),
                SndlibLink("bc", "B", "C", additional_modules=(_module(10),)),
            ),
            (SndlibDemand("d", "A", "C", 6),),
        )
        plan = construct_snd_capacity_plan(network)
        self.assertEqual(plan.demand_paths["d"], ("ab", "bc"))
        self.assertEqual(plan.solution.installed_capacities["short"], 5)

    def test_restrictions_are_rejected_without_semantic_relaxation(self) -> None:
        with self.assertRaisesRegex(
            ConstructiveCapacityPlanError, "routing restrictions"
        ):
            construct_snd_capacity_plan(_network(restricted=True))

    def test_plan_metadata_is_strict(self) -> None:
        plan = construct_snd_capacity_plan(_network())
        values = {
            field: getattr(plan, field)
            for field in plan.__dataclass_fields__
            if field not in {"method", "design_optimal"}
        }
        with self.assertRaisesRegex(ValueError, "unsupported constructive method"):
            ConstructiveSndCapacityPlan(**values, method="unknown")
        with self.assertRaisesRegex(ValueError, "exactly False"):
            ConstructiveSndCapacityPlan(
                **values, method=CONSTRUCTIVE_METHOD, design_optimal=None  # type: ignore[arg-type]
            )


class McmcfGateTests(unittest.TestCase):
    def test_size_estimate_and_resource_skip_are_not_infeasible(self) -> None:
        network = _network()
        estimate = McmcfModelEstimate.for_network(network)
        self.assertEqual(estimate.total_variable_count, 6)
        self.assertEqual(estimate.total_constraint_count, 12)
        capacities = construct_snd_capacity_plan(network).solution.installed_capacities
        result = GurobiMcmcfGate(max_variables=5).solve(network, capacities)
        self.assertIsNone(result.feasible)
        self.assertFalse(result.infeasible)
        self.assertTrue(result.resource_skipped)
        self.assertEqual(result.status, "RESOURCE_SKIPPED_VARIABLE_LIMIT")

    def test_gate_rejects_restrictions_before_solver_import(self) -> None:
        network = _network(restricted=True)
        capacities = {link.id: 100 for link in network.links}
        result = GurobiMcmcfGate().solve(network, capacities)
        self.assertIsNone(result.feasible)
        self.assertFalse(result.resource_skipped)
        self.assertEqual(result.status, "UNSUPPORTED_ROUTING_RESTRICTIONS")

    def test_capacity_keys_and_preinstalled_floor_fail_fast(self) -> None:
        network = _network()
        with self.assertRaisesRegex(ValueError, "missing links"):
            GurobiMcmcfGate(max_variables=1).solve(network, {})
        capacities = {link.id: 0 for link in network.links}
        with self.assertRaisesRegex(ValueError, "below preinstalled"):
            GurobiMcmcfGate(max_variables=1).solve(network, capacities)

    def test_screening_records_resource_skip_before_construction(self) -> None:
        source = NamedSndSource("tiny", _network())
        pool = screen_snd_source_cases(
            (source,), GurobiMcmcfGate(max_variables=5)
        )
        self.assertEqual(pool.feasible_count, 0)
        self.assertEqual(pool.resource_skipped_count, 1)
        record = pool.records[0]
        self.assertIsNone(record.feasible)
        self.assertEqual(record.constructive_seconds, 0)
        self.assertGreaterEqual(record.preprocessing_seconds, 0)
        self.assertEqual(record.routing_unit, "Mbps")

    def test_screening_excludes_euler_violations_by_default(self) -> None:
        source = NamedSndSource("dense", _dense_network())

        excluded = screen_snd_source_cases((source,), _ExactPlanGate())

        self.assertEqual(excluded.feasible_count, 0)
        self.assertEqual(excluded.euler_excluded_count, 1)
        record = excluded.records[0]
        self.assertEqual(record.simple_edge_count, 10)
        self.assertEqual(record.euler_edge_limit, 9)
        self.assertTrue(record.euler_bound_exceeded)
        self.assertTrue(record.excluded_by_euler_bound)
        self.assertIsNone(record.feasible)
        self.assertEqual(record.status, "EULER_BOUND_EXCEEDED")

        retained = screen_snd_source_cases(
            (source,),
            _ExactPlanGate(),
            exclude_euler_bound_violations=False,
        )
        self.assertEqual(retained.feasible_count, 1)
        self.assertEqual(retained.euler_excluded_count, 0)
        self.assertTrue(retained.records[0].euler_bound_exceeded)
        self.assertFalse(retained.records[0].excluded_by_euler_bound)

    def test_screening_euler_filter_option_requires_a_bool(self) -> None:
        with self.assertRaisesRegex(
            TypeError, "exclude_euler_bound_violations must be a bool"
        ):
            screen_snd_source_cases(
                (),
                _ExactPlanGate(),
                exclude_euler_bound_violations=1,  # type: ignore[arg-type]
            )


class SndSourcePoolIoTests(unittest.TestCase):
    def test_round_trip_revalidates_exact_and_constructive_solutions(self) -> None:
        network = _network()
        pool = screen_snd_source_cases(
            (NamedSndSource("tiny", network),), _ExactPlanGate()
        )
        source = pool.sources[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            xml = write_sndlib(network, root / "tiny.xml")
            log = root / "tiny.log"
            log.write_text("solver log\n", encoding="utf-8")
            solution = write_snd_source_solution(root / "tiny.json", source)
            record = asdict(pool.records[0])
            record["log_path"] = "tiny.log"
            manifest = {
                "schema_version": 1,
                "artifact_type": "sndlib_source_pool",
                "run": {
                    "sources": [
                        {
                            "source_name": "tiny",
                            "path": "tiny.xml",
                            "sha256": sha256(xml.read_bytes()).hexdigest(),
                        }
                    ]
                },
                "retained": 1,
                "records": [record],
                "solutions": {"tiny": solution.name},
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded = load_snd_source_pool(manifest_path, path_root=root)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].source_name, "tiny")
        self.assertEqual(dict(loaded[0].solution.flows), dict(source.solution.flows))
        self.assertEqual(loaded[0].constructive_plan.method, CONSTRUCTIVE_METHOD)

    def test_solution_writer_refuses_overwrite(self) -> None:
        source = screen_snd_source_cases(
            (NamedSndSource("tiny", _network()),), _ExactPlanGate()
        ).sources[0]
        with tempfile.TemporaryDirectory() as directory:
            path = write_snd_source_solution(Path(directory) / "solution.json", source)
            with self.assertRaises(FileExistsError):
                write_snd_source_solution(path, source)

    def test_loader_rejects_missing_manifest_kind(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text('{"schema_version": 1}', encoding="utf-8")
            with self.assertRaisesRegex(
                SndSourcePoolManifestError, "unsupported"
            ):
                load_snd_source_pool(path)


if __name__ == "__main__":
    unittest.main()
