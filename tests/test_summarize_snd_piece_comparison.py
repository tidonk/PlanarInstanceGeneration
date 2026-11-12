from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "check/scripts/summarize_snd_piece_comparison.py"
)
SPEC = spec_from_file_location("summarize_snd_piece_comparison", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SummaryTests(unittest.TestCase):
    def test_summary_and_table_report_direction_and_median(self) -> None:
        rows = [
            {
                "strategy": "random",
                "composite_minus_standalone": -1.0,
                "relative_delta_percent": -10.0,
            },
            {
                "strategy": "random",
                "composite_minus_standalone": 0.0,
                "relative_delta_percent": 0.0,
            },
            {
                "strategy": "random",
                "composite_minus_standalone": 2.0,
                "relative_delta_percent": 20.0,
            },
        ]
        summaries = MODULE.summary_rows(rows)
        random = summaries[0]
        self.assertEqual(
            (random["count"], random["lower"], random["equal"], random["higher"]),
            (3, 1, 1, 1),
        )
        self.assertEqual(random["median_relative_delta_percent"], 0.0)
        table = MODULE.paper_table(summaries)
        self.assertIn("R     & 3 & 1 & 1 & 1 & 0.0", table)
        self.assertIn("All   & 3 & 1 & 1 & 1 & 0.0", table)


if __name__ == "__main__":
    unittest.main()
