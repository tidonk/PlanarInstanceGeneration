#!/usr/bin/env python3
"""Prepare the pinned TopologyOptimizer source with the evaluation defaults."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
FILES = ("src", "settings/default.toml", "Project.toml", "Manifest.toml", "LICENSE.md",
         "THIRD_PARTY_NOTICES.md", "LICENSES", "CONTRIBUTORS.md",
         "check/instances/README.md", "check/instances/SOURCES.tsv",
         "check/instances/test/pglib_opf_case118_ieee/pglib_opf_case118_ieee_01.m")


def prepare(source: Path, target: Path, defaults: Path, dependency: dict) -> None:
    expected = dict(revision=dependency["revision"], source=str(source.resolve()),
                    evaluation_defaults_sha256=hashlib.sha256(defaults.read_bytes()).hexdigest(),
                    julia="1.12.6")
    record = target / "source.json"
    if target.exists():
        if not record.is_file() or json.loads(record.read_text()) != expected:
            raise RuntimeError(f"Existing runtime differs from the pinned setup; use a fresh directory: {target}")
        if (target / "settings/default.toml").read_bytes() != defaults.read_bytes():
            raise RuntimeError("Prepared evaluation defaults have changed")
        return
    if not (source / "src/TopologyOptimizer.jl").is_file():
        raise RuntimeError("Initialize TopologyOptimizer with: git submodule update --init --recursive")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix="topology-setup-") as staging:
        work = Path(staging) / "runtime"
        work.mkdir()
        if (source / ".git").exists():
            revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
            if revision != dependency["revision"]:
                raise RuntimeError("TopologyOptimizer differs from the pinned revision")
            with tempfile.TemporaryFile() as stream:
                subprocess.run(["git", "-C", str(source), "archive", revision, *FILES], stdout=stream, check=True)
                stream.seek(0)
                with tarfile.open(fileobj=stream) as archive:
                    for entry in archive.getmembers():
                        if entry.name.startswith("/") or ".." in Path(entry.name).parts:
                            raise RuntimeError("unsafe source archive path")
                    archive.extractall(work)
        else:
            source_record = source / "source.json"
            if not source_record.is_file() or json.loads(source_record.read_text())["revision"] != dependency["revision"]:
                raise RuntimeError("Source archive lacks the pinned dependency record")
            for name in FILES:
                p = source / name
                if p.is_dir():
                    shutil.copytree(p, work / name)
                else:
                    (work / name).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, work / name)
        shutil.copy2(defaults, work / "settings/default.toml")
        (work / "source.json").write_text(json.dumps(expected, indent=2) + "\n")
        work.rename(target)


def main():
    dependency = json.loads((ROOT / "reproduction/dependencies.json").read_text())["TopologyOptimizer"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / dependency["path"])
    parser.add_argument("--instantiate", action="store_true")
    parser.add_argument("--julia", default=os.environ.get("PIG_JULIA", shutil.which("julia") or "julia"))
    args = parser.parse_args()
    target = ROOT / "build/TopologyOptimizer"
    prepare(args.source, target, ROOT / "reproduction/settings/topology-default.toml", dependency)
    if args.instantiate:
        environment = dict(os.environ, JULIA_DEPOT_PATH=os.environ.get("JULIA_DEPOT_PATH", str(ROOT / "build/julia-depot")),
                           JULIA_PKG_PRECOMPILE_AUTO="0")
        program = """
using Pkg
VERSION == v"1.12.6" || error("Use Julia 1.12.6 for the pinned environment")
Pkg.instantiate(; allow_autoprecomp=false)
for info in values(Pkg.dependencies())
    info.source === nothing && continue
    artifacts = joinpath(info.source, "Artifacts.toml")
    isfile(artifacts) || continue
    Pkg.Artifacts.ensure_all_artifacts_installed(artifacts; include_lazy=true)
end
using TopologyOptimizer
println("TopologyOptimizer runtime dependencies loaded successfully")
"""
        subprocess.run([args.julia, "--startup-file=no", "--compiled-modules=existing",
                        "--project=" + str(target), "-e", program], env=environment, check=True)
    print("Prepared TopologyOptimizer with the recorded evaluation defaults:", target)


if __name__ == "__main__":
    main()
