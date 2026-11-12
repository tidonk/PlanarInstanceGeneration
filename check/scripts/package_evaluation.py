#!/usr/bin/env python3
"""Collect the final evaluation source, inputs, results, and manuscript in a ZIP."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("random", "high_degree", "kamada_kawai", "heuristic_outer_face", "exact_outer_face")


def git_files(root):
    """Collect source files, including initialized submodules at their pinned commits."""
    root = Path(root)
    names = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-co", "--exclude-standard", "-z"]
    ).decode().split("\0")
    entries = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "--stage", "-z"]
    ).decode().split("\0")
    links = {}
    for entry in entries:
        if entry:
            metadata, name = entry.split("\t", 1)
            mode, revision, _ = metadata.split()
            if mode == "160000":
                links[name] = revision
    collected = {name for name in names if name and (root / name).is_file()
                 and Path(name).name != "AGENTS.md" and not name.endswith((".orig", ".pyc"))}
    for name, expected in links.items():
        submodule = root / name
        if not (submodule / ".git").exists():
            raise RuntimeError(f"Initialize submodules before packaging: {name}")
        actual = subprocess.check_output(
            ["git", "-C", str(submodule), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != expected:
            raise RuntimeError(f"Submodule differs from its recorded revision: {name}")
        collected.update(str(Path(name) / child) for child in git_files(submodule))
    return sorted(collected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", required=True, type=Path)
    parser.add_argument("--paper", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--zip", type=Path)
    args = parser.parse_args()
    production, paper, output = args.production.resolve(), args.paper.resolve(), args.output.resolve()
    if output.exists():
        parser.error("choose a new output directory")
    output.mkdir(parents=True)
    files = {}

    def add(src, dst):
        src = Path(src)
        if src.is_file():
            files[str(dst)] = src

    def tree(src, dst, suffixes=None):
        src = Path(src)
        if not src.is_dir():
            return
        for directory, directories, names in os.walk(src):
            directories[:] = sorted(d for d in directories
                                    if d not in (".git", "__pycache__", ".venv", "attempts", "solver_logs",
                                                 "initial_run", "cpu_recovery", "scheduler_logs")
                                    and not d.endswith(".ogdf") and not d.startswith("attempts-before"))
            for name in sorted(names):
                path = Path(directory) / name
                if name == "AGENTS.md" or path.suffix in (".pyc", ".orig"):
                    continue
                if suffixes is None or path.suffix in suffixes or name.upper().startswith("LICENSE"):
                    add(path, Path(dst) / path.relative_to(src))

    print("collecting source files", flush=True)
    add(ROOT / "reproduction/PACKAGE_README.md", "README.md")
    for name in git_files(ROOT):
        if not name.startswith(("check/results/", "check/data/", "build/")):
            add(ROOT / name, Path("code") / name)
    for name in git_files(paper):
        if not name.startswith((".vscode/", ".claude/")) and Path(name).suffix not in (".aux", ".log", ".out", ".bbl", ".blg", ".gz"):
            add(paper / name, Path("paper") / name)
    for name in ("planar-instance-gen.pdf", "supplementary-material.pdf"):
        add(paper / name, Path("paper") / name)
    for name in git_files(production):
        if name.startswith(("native/", "planar_instance_generation/", "check/scripts/", "check/eval/")) or name == "pyproject.toml":
            add(production / name, Path("provenance/production-source") / name)
    for name in ("ogdf_20261104_portable.dag", "ogdf_supplement_20261106.dag", "ogdf_sources_20261106.dag"):
        add(production / "check/eval" / name, Path("provenance/final-workflows") / name)
    build_record = json.loads((production / "build/ogdf/build.json").read_text())
    for name, expected in build_record["sha256"].items():
        if hashlib.sha256((production / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"native production source or executable changed: {name}")
    tree(production / "check/data", "code/check/data", {".m", ".xml", ".json", ".csv", ".txt", ".md"})
    # Include the precise native upstream source and the executables used, without caches.
    native_source = ROOT / "build/ogdf/source"
    if not native_source.is_dir():
        native_source = production / "build/ogdf/source"
    else:
        local_build = json.loads((ROOT / "build/ogdf/build.json").read_text())
        if local_build["ogdf_commit"] != build_record["ogdf_commit"]:
            raise ValueError("local OGDF sources have a different revision")
    print("collecting pinned OGDF source", flush=True)
    tree(native_source, "code/build/ogdf/source")
    for name in ("exact", "heuristic", "build.json", "MaxCPlanarMaster.cpp"):
        add(production / "build/ogdf" / name, Path("code/build/ogdf") / name)

    results = production / "check/results"
    roots = ["ogdf_20261104_pglib", "ogdf_20261104_snd", "ogdf_20261104_pglib_exact", "ogdf_20261104_snd_exact",
             "ogdf_20261104_pglib_scopf", "ogdf_20261104_snd_benders", "ogdf_20261104_pglib_scopf_tal0",
             "ogdf_20261104_normalized_pglib_scopf", "ogdf_20261104_piece_comparison_snd_pieces_benders",
             "ogdf_20261106_pglib_source_scots", "snd_source_benders_10800", "pglib_source_pool_innoptem", "sndlib_source_pool",
             "ogdf_sources_20261106_pglib", "ogdf_sources_20261106_voltage", "ogdf_sources_20261106_snd",
             "ogdf_20261104_submission"]
    for policy in ("equidistant", "adaptive_skip_one", "prefer_non_leaf"):
        roots += [f"ogdf_sampling_20261106_{policy}_pglib_exact", f"ogdf_sampling_20261106_{policy}_pglib_scopf"]
        roots += [f"ogdf_sampling_20261106_{policy}_pglib_job_i{i:03d}" for i in range(1, 101)]
    text_data = {".json", ".csv", ".m", ".xml", ".toml", ".env", ".jobs", ".dag", ".sub", ".txt"}
    print("collecting final result records", flush=True)
    def collect_result(item):
        number, name = item
        src = results / name
        if not src.is_dir():
            raise FileNotFoundError(src)
        dst = Path("code/check/results") / name
        # Only final inputs and numerical records are needed, not redundant plot PDFs or solver traces.
        if name in ("ogdf_20261104_pglib", "ogdf_20261104_snd"):
            for item in ("manifest.json",):
                add(src / item, dst / item)
            for sub in ("tables", "evaluation", "instances", "subinstances"):
                tree(src / sub, dst / sub, text_data)
            accepted = list(csv.DictReader((src / "tables/accepted_instances.csv").open()))
            for row in accepted:
                stem = row["artifact_stem"]
                add(src / "plots/composite" / stem / "metadata.json", dst / "plots/composite" / stem / "metadata.json")
                # Each stage needs its original and final graph for vertex-degree statistics.
                for stage in (src / "plots/merged" / stem).glob("stage_*"):
                    for suffix in ("original", "final"):
                        p = stage / f"merge_{stage.name}_{suffix}.json"
                        add(p, dst / p.relative_to(src))
                if name.endswith("pglib") and int(row["output_index"]) == 70:
                    tree(src / "plots/composite" / stem, dst / "plots/composite" / stem, {".json", ".tikz"})
                    tree(src / "plots/merged" / stem, dst / "plots/merged" / stem, {".json", ".tikz"})
        elif "_pglib_job_i" in name:
            for sub in ("tables", "subinstances", "instances"):
                tree(src / sub, dst / sub, text_data)
            add(src / "manifest.json", dst / "manifest.json")
            for metadata in (src / "plots/composite").glob("*/metadata.json"):
                add(metadata, dst / metadata.relative_to(src))
        elif name.endswith(("_pglib_exact", "_snd_exact")):
            tree(src / "evaluation", dst / "evaluation", text_data)
            for record in src.glob("*.json"):
                add(record, dst / record.name)
        else:
            tree(src, dst, text_data)
        if number < 25 or number % 25 == 0:
            print(f"selected {number + 1}/{len(roots)} result directories; {len(files)} files", flush=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(collect_result, enumerate(roots)))
    # Historical SND summaries remain as evidence for results reused after exact input checks.
    reused = set()
    for name in ("ogdf_20261104_snd_benders", "ogdf_20261104_piece_comparison_snd_pieces_benders"):
        for summary in (results / name / "logs").glob("*/summary.json"):
            data = json.loads(summary.read_text())
            reuse = data.get("reuse") or {}
            for key in ("source_summary", "source_instance"):
                if reuse.get(key):
                    p = Path(reuse[key])
                    if not p.is_file():
                        raise FileNotFoundError(p)
                    if key == "source_summary" and hashlib.sha256(p.read_bytes()).hexdigest() != reuse["source_summary_sha256"]:
                        raise ValueError(f"reused source summary changed: {p}")
                    relative = p.relative_to(production)
                    add(p, Path("code") / relative)
                    reused.add(str(relative))
    source_records = {}

    def copy(item):
        dst, src = item
        target = output / dst
        target.parent.mkdir(parents=True, exist_ok=True)
        data = src.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        target.write_bytes(data)
        shutil.copymode(src, target)
        return dst, digest, len(data)

    with ThreadPoolExecutor(max_workers=8) as executor:
        for i, (name, digest, size) in enumerate(executor.map(copy, sorted(files.items())), 1):
            source_records[name] = {"sha256": digest, "bytes": size}
            if i % 1000 == 0:
                print(f"copied {i}/{len(files)} files", flush=True)
    dependencies = json.loads((ROOT / "reproduction/dependencies.json").read_text())
    for dependency in dependencies.values():
        name = str(Path("code") / dependency["path"] / "source.json")
        data = (json.dumps(dependency, indent=2) + "\n").encode()
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        source_records[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    provenance = output / "provenance"
    provenance.mkdir(exist_ok=True)
    (provenance / "source-files.json").write_text(json.dumps(source_records, indent=2) + "\n")
    info = {"production_root": str(production), "paper_root": str(paper), "results": roots,
            "reused_original_records": sorted(reused), "copied_files": len(source_records), "dependencies": dependencies,
            "source_revisions": {str(r): subprocess.check_output(["git", "-C", str(r), "rev-parse", "HEAD"], text=True).strip()
                                 for r in (ROOT, production, paper)},
            "excluded": ["Julia/Python caches", "Gurobi binaries and licenses", "redundant plot renderings", "verbose solver traces", "unsuccessful attempts"]}
    (provenance / "package.json").write_text(json.dumps(info, indent=2) + "\n")
    if args.zip:
        with zipfile.ZipFile(args.zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    archive.write(path, output.name + "/" + path.relative_to(output).as_posix())
        print(f"ZIP: {args.zip} ({args.zip.stat().st_size} bytes)", flush=True)
    print(f"Package: {output}", flush=True)


if __name__ == "__main__":
    main()
