from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from planar_instance_generation.evaluation import (
    SndBendersConfiguration,
    execute_snd_benders_run,
    prepare_snd_benders_run,
    render_snd_benders_settings,
)


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "check/scripts/run_snd_benders.jl"


class SndBendersRunnerTests(unittest.TestCase):
    def test_kernel_io_stall_cannot_make_cleanup_wait_forever(self):
        from planar_instance_generation.evaluation import snd_application as module
        process = Mock(pid=12345)
        process.poll.return_value = None
        process.wait.side_effect = subprocess.TimeoutExpired("julia", 5)
        with patch.object(module.os, "killpg") as kill:
            module._stop_process_group(process)
        self.assertEqual(kill.call_count, 2)
        self.assertEqual(process.wait.call_count, 2)
        for call in process.wait.call_args_list:
            self.assertEqual(call.kwargs["timeout"], 5.0)

    def test_saved_result_survives_stalled_julia_shutdown(self):
        from planar_instance_generation.evaluation import snd_application as module
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "adapter_result.toml"
            result.write_text('termination_status = "OPTIMAL"\ntime_limit_verified = true\n')
            module.os.utime(result, (0, 0))
            process = Mock()
            process.poll.return_value = None
            with patch.object(module.subprocess, "Popen", return_value=process), \
                 patch.object(module, "_stop_process_group") as stop:
                code = module._run_adapter_process(("julia",), cwd=root, env={}, stdout=None, timeout=60)
            self.assertIsNone(code)
            stop.assert_called_once_with(process)

    def test_adapter_refuses_unsupported_routing_restrictions(self) -> None:
        text = ADAPTER.read_text(encoding="utf-8")
        self.assertIn("demand.max_path_length !== nothing", text)
        self.assertIn("!isempty(demand.admissible_paths)", text)
        self.assertIn("refusing restricted demand", text)

    def test_adapter_installs_and_verifies_time_limit_override(self) -> None:
        text = ADAPTER.read_text(encoding="utf-8")
        self.assertLess(
            text.index("@eval BendersNetworkDesign"),
            text.index("function main"),
        )
        self.assertIn('"TimeLimit" => settings.time_limit', text)
        self.assertIn("JuMP.time_limit_sec(result.model)", text)
        self.assertIn('"time_limit_verified" => true', text)

    def test_configuration_is_full_n_minus_one_benders(self) -> None:
        text = render_snd_benders_settings(
            SndBendersConfiguration(time_limit_seconds=37.5, threads=3, seed=11)
        )
        self.assertIn('model_type = "benders"', text)
        self.assertIn("time_limit = 37.5", text)
        self.assertIn("num_outage_scenarios = -1", text)
        self.assertIn("contingency_k = 1", text)
        self.assertIn('ordering = "none"', text)
        self.assertIn('strategy = "none"', text)

    def test_adaptive_configuration_matches_generated_instance_setting(self) -> None:
        text = render_snd_benders_settings(
            SndBendersConfiguration(method="adaptive_ml")
        )
        self.assertIn('ordering = "score"', text)
        self.assertIn('strategy = "static"', text)
        self.assertIn("min_score_threshold = 0.5", text)
        self.assertIn("absolute = 5", text)
        self.assertIn("stabilization_frequency = 20", text)
        self.assertIn("root_node_stabilization = 5", text)

        with self.assertRaisesRegex(ValueError, "method"):
            SndBendersConfiguration(method="unknown")

    def test_dry_run_writes_only_below_instance_log_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "check/results"
            instance = results / "generation/instances/42.xml"
            instance.parent.mkdir(parents=True)
            instance.write_text("<network/>\n", encoding="utf-8")
            external = root / "BendersNetworkDesign.jl"
            (external / "src").mkdir(parents=True)
            (external / "Project.toml").write_text(
                'name = "BendersNetworkDesign"\n', encoding="utf-8"
            )
            (external / "src/BendersNetworkDesign.jl").write_text(
                "module BendersNetworkDesign\nend\n", encoding="utf-8"
            )
            (external / "Manifest.toml").write_text(
                'julia_version = "1.12.4"\nmanifest_format = "2.0"\n',
                encoding="utf-8",
            )

            prepared = prepare_snd_benders_run(
                instance,
                run_name="application",
                results_directory=results,
                benders_repository=external,
                adapter=ADAPTER,
                julia="julia-custom",
                config=SndBendersConfiguration(threads=2),
            )
            summary = execute_snd_benders_run(prepared, dry_run=True)

            expected = results / "application/logs/42"
            self.assertEqual(prepared.paths.log_directory, expected)
            self.assertEqual(summary["state"], "dry_run")
            self.assertTrue(prepared.paths.settings.is_file())
            self.assertTrue(prepared.paths.invocation.is_file())
            self.assertTrue(prepared.paths.summary.is_file())
            self.assertFalse(prepared.paths.solver_log.exists())
            stored = json.loads(prepared.paths.summary.read_text(encoding="utf-8"))
            self.assertEqual(stored["contingencies"]["level"], "N-1")
            self.assertEqual(stored["configuration"]["threads"], 2)
            self.assertEqual(stored["wall_clock_limit_seconds"], 10_860.0)
            self.assertEqual(prepared.command[0], "julia-custom")
            self.assertEqual(prepared.command[1], "+1.12.4")
            with self.assertRaises(FileExistsError):
                execute_snd_benders_run(prepared, dry_run=True)

    def test_rejects_raw_or_external_instance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "check/results"
            raw = root / "check/data/sndlib/raw.xml"
            raw.parent.mkdir(parents=True)
            raw.write_text("<network/>\n", encoding="utf-8")
            external = root / "BendersNetworkDesign.jl"
            (external / "src").mkdir(parents=True)
            (external / "Project.toml").write_text("", encoding="utf-8")
            (external / "src/BendersNetworkDesign.jl").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "generated below"):
                prepare_snd_benders_run(
                    raw,
                    run_name="application",
                    results_directory=results,
                    benders_repository=external,
                    adapter=ADAPTER,
                )

    def test_rejects_path_components_and_invalid_limits(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            SndBendersConfiguration(time_limit_seconds=0)
        with self.assertRaisesRegex(ValueError, "positive"):
            SndBendersConfiguration(threads=0)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            SndBendersConfiguration(seed=-1)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            SndBendersConfiguration(wall_clock_grace_seconds=-1)

    def test_wall_clock_guard_records_an_explicit_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prepared = self._prepared_run(
                Path(directory),
                config=SndBendersConfiguration(
                    time_limit_seconds=12.0,
                    wall_clock_grace_seconds=3.0,
                ),
            )
            expired = subprocess.TimeoutExpired(prepared.command, timeout=15.0)
            with patch(
                "planar_instance_generation.evaluation.snd_application."
                "_run_adapter_process",
                side_effect=expired,
            ) as mocked:
                summary = execute_snd_benders_run(prepared)

        self.assertEqual(summary["state"], "wall_time_limit")
        self.assertEqual(summary["wall_clock_limit_seconds"], 15.0)
        self.assertIn("wall-clock grace", summary["error"])
        self.assertEqual(mocked.call_args.kwargs["timeout"], 15.0)

    def test_success_requires_verified_effective_time_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prepared = self._prepared_run(
                Path(directory),
                config=SndBendersConfiguration(time_limit_seconds=120.0),
            )

            def completed_with_wrong_limit(*args, **kwargs):
                prepared.paths.adapter_result.write_text(
                    "requested_time_limit_seconds = 120.0\n"
                    "effective_time_limit_seconds = 10800.0\n"
                    "time_limit_verified = true\n",
                    encoding="utf-8",
                )
                return 0

            with patch(
                "planar_instance_generation.evaluation.snd_application."
                "_run_adapter_process",
                side_effect=completed_with_wrong_limit,
            ):
                summary = execute_snd_benders_run(prepared)

        self.assertEqual(summary["state"], "adapter_error")
        self.assertIn("10800.0", summary["error"])
        self.assertIn("does not match requested 120.0", summary["error"])

    def _prepared_run(
        self,
        root: Path,
        *,
        config: SndBendersConfiguration,
    ):
        results = root / "check/results"
        instance = results / "generation/instances/42.xml"
        instance.parent.mkdir(parents=True)
        instance.write_text("<network/>\n", encoding="utf-8")
        external = root / "BendersNetworkDesign.jl"
        (external / "src").mkdir(parents=True)
        (external / "Project.toml").write_text(
            'name = "BendersNetworkDesign"\n', encoding="utf-8"
        )
        (external / "src/BendersNetworkDesign.jl").write_text(
            "module BendersNetworkDesign\nend\n", encoding="utf-8"
        )
        return prepare_snd_benders_run(
            instance,
            run_name="application",
            results_directory=results,
            benders_repository=external,
            adapter=ADAPTER,
            config=config,
        )


if __name__ == "__main__":
    unittest.main()
