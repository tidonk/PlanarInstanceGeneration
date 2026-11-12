from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import unittest

from planar_instance_generation.evaluation.scopf_application import (
    ScopfBendersConfiguration,
    execute_scopf_benders_run,
    prepare_scopf_benders_run,
    render_scopf_benders_settings,
)
from check.eval.summarize_applications import (
    _coupling_flow_multiplier_to_mw,
    _scopf_feasible,
    _scaled_number,
)

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "check/scripts/run_scopf_benders.jl"


def _write_generated_case(path: Path) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        """function mpc = case42
mpc.version = '2';
mpc.baseMVA = 100;
mpc.bus = [
1 3 0 0 0 0 1 1 0 1 1 1.1 0.9;
2 1 10 0 0 0 1 1 0 1 1 1.1 0.9;
];
mpc.gen = [1 10 0 10 -10 1 100 1 20 0;];
mpc.gencost = [2 0 0 2 1 0;];
mpc.branch = [1 2 0 0.1 0 0 0 0 0 0 1 -30 30;];
""",
        encoding="utf-8",
    )
    metadata = path.parent.parent / "plots" / "composite" / path.stem / "metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        json.dumps(
            {
                "artifact_stem": path.stem,
                "kind": "final_composite",
                "coupling_branch_rows_zero_based": [0],
            }
        ),
        encoding="utf-8",
    )


class ScopfBendersRunnerTests(unittest.TestCase):
    def test_unlimited_solver_budget_renders_valid_toml(self):
        import math
        import tomllib
        config = ScopfBendersConfiguration(time_limit_seconds=0)
        settings = tomllib.loads(render_scopf_benders_settings(config))
        self.assertEqual(settings["SOLVER"]["SOLVING_TIME_LIMIT"], math.inf)
        for invalid in (-1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                ScopfBendersConfiguration(time_limit_seconds=invalid)

    def test_active_unlimited_run_is_not_stopped_by_a_time_budget(self):
        from planar_instance_generation.evaluation import scopf_application as module
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "process.log"
            log.touch()
            run = SimpleNamespace(command=["julia"], summary={"innoptem_repository": directory},
                                  paths=SimpleNamespace(process_log=log, adapter_result=root / "result.json"))
            from unittest.mock import Mock
            process = Mock(pid=12345)
            process.poll.side_effect = [None, 0]
            stat = "12345 (julia) R " + " ".join(["0"] * 10 + ["1000", "10"])
            with patch.object(module.subprocess, "Popen", return_value=process), \
                 patch.object(module.time, "monotonic", side_effect=[0, 100000]), \
                 patch.object(module.time, "sleep"), \
                 patch.object(module.Path, "read_text", return_value=stat), \
                 patch.object(module.os, "killpg") as kill:
                self.assertEqual(module._wait_for_julia(run, {}, None, ScopfBendersConfiguration(time_limit_seconds=0)), (0, None))
                kill.assert_not_called()

    def test_startup_stall_is_stopped_without_waiting_for_kernel_io(self):
        from planar_instance_generation.evaluation import scopf_application as module
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "process.log"
            log.touch()
            run = SimpleNamespace(command=["julia"], summary={"innoptem_repository": directory},
                                  paths=SimpleNamespace(process_log=log, adapter_result=root / "result.json"))
            process = SimpleNamespace(pid=12345, poll=lambda: None)
            with patch.object(module.subprocess, "Popen", return_value=process), \
                 patch.object(module.time, "monotonic", side_effect=[0, 901]), \
                 patch.object(module.os, "killpg") as kill:
                with self.assertRaisesRegex(TimeoutError, "startup"):
                    module._wait_for_julia(run, {}, None, ScopfBendersConfiguration())
                kill.assert_called_once_with(12345, module.signal.SIGKILL)

    def test_complete_result_is_recovered_when_julia_cannot_exit(self):
        from planar_instance_generation.evaluation import scopf_application as module
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "result.json"
            result.write_text(json.dumps({"scopf_feasible": True, "termination_status": "OPTIMAL",
                                          "time": {"solve": 2}, "rating_normalization_validation": {}}))
            module.os.utime(result, (0, 0))
            run = SimpleNamespace(command=["julia"], summary={"innoptem_repository": directory},
                                  paths=SimpleNamespace(adapter_result=result))
            process = SimpleNamespace(pid=12345, poll=lambda: None)
            with patch.object(module.subprocess, "Popen", return_value=process), \
                 patch.object(module.os, "killpg") as kill:
                code, reason = module._wait_for_julia(run, {}, None, ScopfBendersConfiguration())
                self.assertIsNone(code)
                self.assertIn("recovered", reason)
                kill.assert_called_once_with(12345, module.signal.SIGKILL)

    def test_exported_source_revision_does_not_use_enclosing_repository(self):
        from planar_instance_generation.evaluation.scopf_application import (
            _git_revision,
        )

        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            revision = "f502511dd59ad30608f1e3a12a34324d44aab7ee"
            with patch("subprocess.run") as run:
                self.assertIsNone(_git_revision(repository))
                (repository / "source.json").write_text(
                    json.dumps({"revision": revision})
                )
                self.assertEqual(_git_revision(repository), revision)
                run.assert_not_called()

    def test_paper_scots_budget_and_warmstart_are_separate(self):
        import tomllib

        settings = tomllib.loads(
            render_scopf_benders_settings(
                ScopfBendersConfiguration(
                    topological_action_limit=1,
                    scopf_warmstart=True,
                    benders_settings=str(ROOT / "check/eval/scots_benders.toml"),
                )
            )
        )
        self.assertEqual(settings["SOLVER"]["SOLVING_TIME_LIMIT"], 10800)
        self.assertFalse(settings["SOLVER"]["INCLUDE_WARMSTART_TIME_LIMIT"])
        self.assertEqual(settings["WARMSTARTING"]["SCOPF_TIME_LIMIT"], 10800)
        self.assertEqual(settings["WARMSTARTING"]["WARMSTART_METHOD"], ":scopf")
        self.assertTrue(settings["WARMSTARTING"]["ENABLED"])
        self.assertEqual(settings["LIMITS"]["BRANCH"]["TAL_TIMESTEP"], 1)
        self.assertTrue(settings["BENDERS"]["PARTIAL_DECOMP"]["INCLUDE_BASE_PF"])
        self.assertEqual(
            settings["BENDERS"]["PARTIAL_DECOMP"]["N_MAX_CONTINGENCIES"], 0
        )

    def test_adapter_reports_branch_flows_in_mw(self) -> None:
        text = ADAPTER.read_text(encoding="utf-8")
        self.assertIn('"schema_version" => 5', text)
        self.assertIn('"flow_unit" => "MW"', text)
        self.assertIn("flow = abs(Float64(value)) * base_mva", text)
        self.assertIn('"noncoupling_mean_absolute_flow_mw"', text)
        self.assertIn('"scopf_feasible" => scopf_feasible', text)

    def test_infers_scopf_feasibility_for_legacy_results(self) -> None:
        legacy = {
            "application": "DC_SCOTS_TIME_SLACK_BENDERS",
            "has_finite_objective": True,
            "primal_status": "FEASIBLE_POINT",
        }
        self.assertTrue(_scopf_feasible(legacy))
        self.assertFalse(
            _scopf_feasible(
                {
                    **legacy,
                    "primal_status": "CUT_ERROR",
                }
            )
        )
        self.assertIsNone(
            _scopf_feasible({"application": "CAPACITATED_NETWORK_DESIGN_BENDERS"})
        )
        self.assertFalse(_scopf_feasible({**legacy, "scopf_feasible": False}))

    def test_recovers_known_no_incumbent_postprocessing_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "check/results"
            instance = results / "generation/instances/42.m"
            _write_generated_case(instance)
            external = root / "TopologyOptimizer"
            (external / "src").mkdir(parents=True)
            (external / "Project.toml").write_text(
                'name = "TopologyOptimizer"\nversion = "1.2.3"\n', encoding="utf-8"
            )
            (external / "src/TopologyOptimizer.jl").write_text(
                "module TopologyOptimizer\nend\n", encoding="utf-8"
            )
            prepared = prepare_scopf_benders_run(
                instance,
                run_name="application",
                results_directory=results,
                innoptem_repository=external,
                adapter=ADAPTER,
                julia="julia-custom",
                use_manifest_julia=False,
            )
            log_text = """Solution count 0
KeyError: key \"nw\" not found
postprocess_results!
calculate_power_flow!
"""

            def failed_adapter(*args, **kwargs):
                args[2].write(log_text)
                args[2].flush()
                return 1, None

            with patch(
                "planar_instance_generation.evaluation.scopf_application._wait_for_julia",
                side_effect=failed_adapter,
            ):
                summary = execute_scopf_benders_run(prepared)

            self.assertEqual(summary["state"], "complete")
            self.assertEqual(summary["adapter_exit_code"], 1)
            self.assertFalse(summary["result"]["scopf_feasible"])
            self.assertEqual(
                summary["result"]["scopf_feasibility_status"],
                "no_feasible_incumbent_found",
            )
            self.assertIsNone(
                summary["result"]["rating_normalization_validation"][
                    "positive_flow_branch_count"
                ]
            )

    def test_schema_two_flow_metrics_are_converted_from_per_unit(self) -> None:
        result = {
            "application": "DC_SCOTS_TIME_SLACK_BENDERS",
            "schema_version": 2,
            "rating_normalization": {
                "system_scale_components_mva": {"base_mva": 100.0}
            },
        }
        multiplier = _coupling_flow_multiplier_to_mw(result, {})
        self.assertEqual(multiplier, 100.0)
        self.assertEqual(_scaled_number(1.25, multiplier), 125.0)

    def test_explicit_mw_flow_metrics_are_not_rescaled(self) -> None:
        multiplier = _coupling_flow_multiplier_to_mw({}, {"flow_unit": "MW"})
        self.assertEqual(multiplier, 1.0)
        self.assertEqual(_scaled_number(1.25, multiplier), 1.25)

    def test_configuration_is_dc_benders_without_switching(self) -> None:
        text = render_scopf_benders_settings(
            ScopfBendersConfiguration(time_limit_seconds=37.5, threads=3, seed=11)
        )
        self.assertIn('APPLICATION = "DC_SCOTS_TIME_SLACK_BENDERS"', text)
        self.assertIn("SOLVING_TIME_LIMIT = 37.5", text)
        self.assertIn("N_THREADS_BENDERS = 3", text)
        self.assertIn("TAL_TIMESTEP = 0", text)
        self.assertIn("BRANCH_SWITCHING = true", text)
        self.assertIn("ESTIMATE_LOSSES = false", text)
        self.assertIn("[POSTPROCESSING.AC_FEAS]\nCHECK = false", text)

    def test_configuration_allows_one_topological_action(self) -> None:
        config = ScopfBendersConfiguration(topological_action_limit=1)
        text = render_scopf_benders_settings(config)
        self.assertIn("TAL_TIMESTEP = 1", text)

    def test_dry_run_is_reproducible_and_non_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "check/results"
            instance = results / "generation/instances/42.m"
            _write_generated_case(instance)
            external = root / "TopologyOptimizer"
            (external / "src").mkdir(parents=True)
            (external / "Project.toml").write_text(
                'name = "TopologyOptimizer"\nversion = "1.2.3"\n', encoding="utf-8"
            )
            (external / "src/TopologyOptimizer.jl").write_text(
                "module TopologyOptimizer\nend\n", encoding="utf-8"
            )
            (external / "Manifest.toml").write_text(
                'julia_version = "1.12.6"\nmanifest_format = "2.0"\n',
                encoding="utf-8",
            )

            prepared = prepare_scopf_benders_run(
                instance,
                run_name="application",
                results_directory=results,
                innoptem_repository=external,
                adapter=ADAPTER,
                julia="julia-custom",
                config=ScopfBendersConfiguration(threads=2),
            )
            summary = execute_scopf_benders_run(prepared, dry_run=True)

            expected = results / "application/logs/42"
            self.assertEqual(prepared.paths.log_directory, expected)
            self.assertEqual(summary["state"], "dry_run")
            self.assertTrue(prepared.paths.settings.is_file())
            self.assertTrue(prepared.paths.invocation.is_file())
            self.assertTrue(prepared.paths.summary.is_file())
            self.assertTrue(prepared.paths.innoptem_instance.is_file())
            self.assertTrue(prepared.paths.normalization.is_file())
            self.assertFalse(prepared.paths.process_log.exists())
            stored = json.loads(prepared.paths.summary.read_text(encoding="utf-8"))
            self.assertEqual(stored["application"], "DC_SCOTS_TIME_SLACK_BENDERS")
            self.assertEqual(stored["model_invariants"]["topological_action_limit"], 0)
            self.assertFalse(stored["model_invariants"]["switching_actions_allowed"])
            self.assertTrue(
                stored["model_invariants"]["branch_switching_variable_layer"]
            )
            self.assertEqual(stored["rating_normalization"]["changed_branch_count"], 1)
            self.assertEqual(
                stored["rating_normalization"]["replacement_rating_mva"], 200.0
            )
            self.assertEqual(prepared.command[0:2], ("julia-custom", "+1.12.6"))
            self.assertIn("--threads=2", prepared.command)
            with self.assertRaises(FileExistsError):
                execute_scopf_benders_run(prepared, dry_run=True)

    def test_explicit_julia_channel_overrides_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "check/results"
            instance = results / "generation/instances/42.m"
            _write_generated_case(instance)
            external = root / "TopologyOptimizer"
            (external / "src").mkdir(parents=True)
            (external / "Project.toml").write_text(
                'name = "TopologyOptimizer"\n', encoding="utf-8"
            )
            (external / "src/TopologyOptimizer.jl").write_text("", encoding="utf-8")
            (external / "Manifest.toml").write_text(
                'julia_version = "1.12.6"\n', encoding="utf-8"
            )
            prepared = prepare_scopf_benders_run(
                instance,
                run_name="application",
                results_directory=results,
                innoptem_repository=external,
                adapter=ADAPTER,
                julia_channel="1.12.4",
            )
            self.assertEqual(prepared.command[0:2], ("julia", "+1.12.4"))
            self.assertEqual(prepared.summary["manifest_julia_version"], "1.12.6")
            self.assertEqual(prepared.summary["selected_julia_channel"], "1.12.4")

    def test_rejects_raw_or_external_instance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "check/results"
            raw = root / "check/data/pglib/raw.m"
            raw.parent.mkdir(parents=True)
            raw.write_text("function mpc = raw\nend\n", encoding="utf-8")
            external = root / "TopologyOptimizer"
            (external / "src").mkdir(parents=True)
            (external / "Project.toml").write_text("", encoding="utf-8")
            (external / "src/TopologyOptimizer.jl").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "generated below"):
                prepare_scopf_benders_run(
                    raw,
                    run_name="application",
                    results_directory=results,
                    innoptem_repository=external,
                    adapter=ADAPTER,
                )

    def test_rejects_invalid_resources(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            ScopfBendersConfiguration(time_limit_seconds=-1)
        with self.assertRaisesRegex(ValueError, "positive"):
            ScopfBendersConfiguration(threads=0)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            ScopfBendersConfiguration(seed=-1)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            ScopfBendersConfiguration(topological_action_limit=-1)

    def test_refuses_to_normalize_a_source_branch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "check/results"
            instance = results / "generation/instances/42.m"
            _write_generated_case(instance)
            metadata = results / "generation/plots/composite/42/metadata.json"
            metadata.write_text(
                json.dumps(
                    {
                        "artifact_stem": "42",
                        "coupling_branch_rows_zero_based": [],
                    }
                ),
                encoding="utf-8",
            )
            external = root / "TopologyOptimizer"
            (external / "src").mkdir(parents=True)
            (external / "Project.toml").write_text(
                'name = "TopologyOptimizer"\n', encoding="utf-8"
            )
            (external / "src/TopologyOptimizer.jl").write_text("", encoding="utf-8")
            prepared = prepare_scopf_benders_run(
                instance,
                run_name="application",
                results_directory=results,
                innoptem_repository=external,
                adapter=ADAPTER,
            )
            with self.assertRaisesRegex(ValueError, "non-coupling"):
                execute_scopf_benders_run(prepared, dry_run=True)


if __name__ == "__main__":
    unittest.main()
