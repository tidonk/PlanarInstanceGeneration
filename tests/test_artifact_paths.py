from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from planar_instance_generation.io import allocate_instance_artifacts


class ArtifactPathTests(unittest.TestCase):
    def test_uses_bus_count_and_one_based_collision_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = allocate_instance_artifacts(root, 120, extension=".m")
            first.instance_file.touch()
            second = allocate_instance_artifacts(root, 120, extension=".m")

            self.assertEqual(first.stem, "120")
            self.assertEqual(second.stem, "120_2")
            self.assertEqual(second.instance_file, root / "instances/120_2.m")
            self.assertEqual(second.plot_directory, root / "plots/composite/120_2")
            self.assertEqual(second.gurobi_log, root / "logs/composite/120_2/gurobi.log")

            piece = allocate_instance_artifacts(
                root, 12, extension=".m", instance_tree="subinstances"
            )
            self.assertEqual(piece.instance_file, root / "subinstances/12.m")
            self.assertEqual(
                piece.plot_directory, root / "plots/original/12"
            )
            self.assertIsNone(piece.merge_plot_directory)
            self.assertEqual(
                second.merge_plot_directory, root / "plots/merged/120_2"
            )

    def test_suffix_disambiguates_shared_node_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = allocate_instance_artifacts(
                root, 42, extension=".m", suffix="i007_random"
            )
            first.instance_file.touch()
            again = allocate_instance_artifacts(
                root, 42, extension=".m", suffix="i007_random"
            )
            other = allocate_instance_artifacts(
                root, 42, extension=".m", suffix="i008_random"
            )

            self.assertEqual(first.stem, "42_i007_random")
            self.assertEqual(again.stem, "42_i007_random_2")
            self.assertEqual(other.stem, "42_i008_random")
            self.assertEqual(other.instance_file, root / "instances/42_i008_random.m")

    def test_rejects_unknown_instance_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "instance_tree"):
                allocate_instance_artifacts(
                    directory, 10, extension=".m", instance_tree="pieces"
                )

    def test_rejects_unsafe_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for bad in ("bad/suffix", "_leading", "has space", ""):
                with self.assertRaises(ValueError):
                    allocate_instance_artifacts(
                        root, 10, extension=".m", suffix=bad
                    )

    def test_partial_artifacts_also_prevent_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plots/composite/14").mkdir(parents=True)

            paths = allocate_instance_artifacts(root, 14, extension=".xml")

            self.assertEqual(paths.stem, "14_2")


if __name__ == "__main__":
    unittest.main()
