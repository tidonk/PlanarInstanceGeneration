from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from check.scripts import build_ogdf, package_evaluation
from check.scripts import prepare_topology_optimizer as topology


class SourceDependencyTests(unittest.TestCase):
    def archive_source(self, root):
        source = root / "source"
        for name in topology.FILES:
            if name == "src":
                p = source / "src/TopologyOptimizer.jl"
            elif name == "LICENSES":
                p = source / "LICENSES/PowerModels.txt"
            else:
                p = source / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("upstream source\n")
        (source / "source.json").write_text(json.dumps({"revision": "abc123"}))
        defaults = root / "evaluation.toml"
        defaults.write_text("[BENDERS]\nFIX_FORCED_ON_BRANCHES = false\n")
        return source, defaults

    def test_archive_setup_keeps_source_and_uses_evaluation_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, defaults = self.archive_source(root)
            target = root / "build/runtime"
            topology.prepare(source, target, defaults, {"revision": "abc123"})
            self.assertEqual((source / "settings/default.toml").read_text(), "upstream source\n")
            self.assertEqual((target / "settings/default.toml").read_bytes(), defaults.read_bytes())
            self.assertEqual((target / "src/TopologyOptimizer.jl").read_bytes(), (source / "src/TopologyOptimizer.jl").read_bytes())
            case = "check/instances/test/pglib_opf_case118_ieee/pglib_opf_case118_ieee_01.m"
            self.assertEqual((target / case).read_bytes(), (source / case).read_bytes())
            topology.prepare(source, target, defaults, {"revision": "abc123"})
            (target / "settings/default.toml").write_text("changed\n")
            with self.assertRaisesRegex(RuntimeError, "defaults have changed"):
                topology.prepare(source, target, defaults, {"revision": "abc123"})

    def test_archive_setup_rejects_a_different_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, defaults = self.archive_source(root)
            target = root / "runtime"
            with self.assertRaisesRegex(RuntimeError, "pinned dependency record"):
                topology.prepare(source, target, defaults, {"revision": "different"})
            self.assertFalse(target.exists())

    def test_git_setup_rejects_a_different_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, defaults = self.archive_source(root)
            (source / ".git").mkdir()
            with patch.object(topology.subprocess, "check_output", return_value="different\n"):
                with self.assertRaisesRegex(RuntimeError, "differs from the pinned revision"):
                    topology.prepare(source, root / "runtime", defaults, {"revision": "abc123"})

    def test_missing_ogdf_has_initialization_instructions(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(build_ogdf, "ROOT", Path(directory)), patch("sys.argv", ["build_ogdf.py"]):
                with self.assertRaisesRegex(RuntimeError, "git submodule update"):
                    build_ogdf.main()

    def test_ogdf_requires_the_recorded_submodule_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "third_party/ogdf").mkdir(parents=True)
            (root / "third_party/ogdf/CMakeLists.txt").write_text("unused\n")
            with patch.object(build_ogdf, "ROOT", root), patch("sys.argv", ["build_ogdf.py"]):
                with patch.object(build_ogdf.subprocess, "check_output", return_value="different\n"):
                    with self.assertRaisesRegex(RuntimeError, "differs from the pinned revision"):
                        build_ogdf.main()

    def test_package_includes_submodule_files_and_rejects_missing_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child, parent = root / "child", root / "parent"
            child.mkdir(); parent.mkdir()
            def git(where, *args):
                return subprocess.run(["git", "-C", str(where), *args], check=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            git(child, "init", "-q")
            (child / "library.py").write_text("value = 1\n")
            git(child, "add", "library.py")
            git(child, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "test source")
            git(parent, "init", "-q")
            git(parent, "-c", "protocol.file.allow=always", "submodule", "add", str(child), "third_party/library")
            (parent / "README.md").write_text("source package\n")
            self.assertIn("third_party/library/library.py", package_evaluation.git_files(parent))
            self.assertIn("README.md", package_evaluation.git_files(parent))
            (parent / "third_party/library/.git").unlink()
            with self.assertRaisesRegex(RuntimeError, "Initialize submodules"):
                package_evaluation.git_files(parent)
