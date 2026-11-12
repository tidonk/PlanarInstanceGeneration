from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "check/scripts/export_paper_skewness_tables.py"
SPEC = spec_from_file_location("export_paper_skewness_tables", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PaperSkewnessTableTests(unittest.TestCase):
    def test_only_strict_interval_winner_is_bold(self) -> None:
        intervals = {
            "random": (2.0, 18.0),
            "high_degree": (2.0, 16.0),
            "kamada_kawai": (2.0, 14.0),
            "heuristic_outer_face": (2.0, 2.0),
            "exact_outer_face": (0.0, 0.0),
        }
        self.assertEqual(MODULE._provable_winners(intervals), {"exact_outer_face"})
        table = MODULE._table("PGLib", {96: intervals})
        self.assertIn(r"\textbf{0}", table)
        self.assertEqual(table.count(r"\textbf{"), 1)

    def test_tied_or_overlapping_intervals_are_not_bold(self) -> None:
        tied = {strategy: (0.0, 0.0) for strategy in MODULE.STRATEGIES}
        self.assertEqual(MODULE._provable_winners(tied), set())
        overlapping = {
            strategy: (0.0, float(index + 1))
            for index, strategy in enumerate(MODULE.STRATEGIES)
        }
        self.assertEqual(MODULE._provable_winners(overlapping), set())


if __name__ == "__main__":
    unittest.main()
