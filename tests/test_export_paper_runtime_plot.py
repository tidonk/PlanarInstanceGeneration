from importlib.util import module_from_spec, spec_from_file_location
import csv
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "check/scripts/export_paper_runtime_plot.py"
SPEC = spec_from_file_location("export_paper_runtime_plot", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PaperRuntimePlotTests(unittest.TestCase):
    def test_components_are_aggregated_from_per_instance_measurements(self) -> None:
        generated, evaluated = [], []
        for method, _ in MODULE.METHODS:
            for index in range(100):
                selection = 1 if index % 2 == 0 else 17
                feasibility_check = 3 if index % 2 == 0 else 15
                generated.append({
                    "strategy": method,
                    "merge_node_selection_seconds": selection,
                    "gate_optimizer_seconds": 2,
                    "gate_seconds": feasibility_check,
                    "generation_and_gate_seconds": selection + feasibility_check + 3,
                })
                evaluated.append({
                    "strategy": method,
                    "skewness_exact": index % 2 == 0,
                    "skewness_runtime_seconds": 7 if index % 2 == 0 else 3600,
                    "skewness_status": "OPTIMAL" if index % 2 == 0 else "TIME_LIMIT",
                })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tables").mkdir()
            evaluation = root / "evaluation.csv"
            for path, records in (
                (root / "tables/accepted_instances.csv", generated),
                (evaluation, evaluated),
            ):
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                    writer.writeheader()
                    writer.writerows(records)
            # Use the complete check time, including work outside optimization.
            # Subtract it per instance; subtracting means changes the remainder.
            result = MODULE.summarize(evaluation, root)
        for method, _ in MODULE.METHODS:
            for actual, expected in zip(result[method], (5, 7, 3, 7, 50)):
                self.assertAlmostEqual(actual, expected)

    def test_hatched_components_use_a_line_pattern(self) -> None:
        summary = {
            method: (1.0, 2.0, 3.0, 4.0, 0)
            for method, _label in MODULE.METHODS
        }
        output = MODULE.component_plots(summary, 1, 0, hatched=True)
        self.assertEqual(output.count("pattern=north east lines"), 5)
        self.assertEqual(output.count("pattern color=pigMethod"), 5)
        self.assertNotIn("fill=", output)


if __name__ == "__main__":
    unittest.main()
