from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "check/scripts/compare_scopf_normalization.py"
)
SPEC = spec_from_file_location("compare_scopf_normalization", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PaperTableTests(unittest.TestCase):
    def test_table_uses_strategy_order_and_computed_totals(self) -> None:
        baseline = [51, 65, 43, 52, 52]
        normalized = [45, 59, 35, 47, 50]
        summaries = [
            {
                "scope": strategy,
                "baseline_feasible": before,
                "normalized_feasible": after,
            }
            for strategy, before, after in zip(
                MODULE.STRATEGIES, baseline, normalized, strict=True
            )
        ]

        table = MODULE._paper_table(summaries)

        self.assertIn("& R & HD & KK & OF-H & OF-B & Total", table)
        self.assertIn("& 51 & 65 & 43 & 52 & 52 & 263", table)
        self.assertIn("& 45 & 59 & 35 & 47 & 50 & 236", table)


if __name__ == "__main__":
    unittest.main()
