#!/usr/bin/env python3
"""Build the pinned OGDF executables with portable compiler settings."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "5b6795655399b9d8e2921afec9d97bab9107d5ee"


def run(args):
    subprocess.run(list(map(str, args)), check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        help="local OGDF Git checkout containing the pinned commit",
    )
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()
    target = ROOT / "build/ogdf"
    target.mkdir(parents=True, exist_ok=True)
    source = target / "source"
    if not (source / "CMakeLists.txt").is_file():
        checkout = args.source
        if checkout is None:
            checkout = ROOT / "third_party/ogdf"
            if not (checkout / "CMakeLists.txt").is_file():
                raise RuntimeError("Initialize OGDF with: git submodule update --init --recursive")
            revision = subprocess.check_output(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
            ).strip()
            if revision != COMMIT:
                raise RuntimeError("OGDF differs from the pinned revision; run git submodule update --init")
        # Export precisely the recorded revision, ignoring modifications in the checkout.
        with tempfile.TemporaryFile() as stream:
            subprocess.run(
                ["git", "-C", str(checkout), "archive", COMMIT],
                stdout=stream,
                check=True,
            )
            stream.seek(0)
            source.mkdir(exist_ok=True)
            with tarfile.open(fileobj=stream) as archive:

                for entry in archive.getmembers():
                    if entry.name.startswith("/") or ".." in Path(entry.name).parts:
                        raise RuntimeError("unsafe source archive path")
                archive.extractall(source)
    build = target / "library"
    run(
        [
            "cmake",
            "-S",
            source,
            "-B",
            build,
            "-DCMAKE_BUILD_TYPE=Release",
            # OGDF defaults to -march=native, which is unsafe on older workers.
            "-DOGDF_ARCH=x86-64",
            # Recheck SIMD support if an existing build used a different CPU.
            "-Uhas_sse3_*",
            "-DOGDF_MEMORY_MANAGER=POOL_TS",
            "-DOGDF_USE_ASSERT_EXCEPTIONS=ON",
            "-DOGDF_INCLUDE_CGAL=OFF",
            "-DOGDF_INCLUDE_COIN=ON",
        ]
    )
    run(["cmake", "--build", build, "--target", "OGDF", "COIN", "-j", args.jobs])
    # Supply the single resource-control change without editing upstream files.
    master = (source / "src/ogdf/cluster/internal/MaxCPlanarMaster.cpp").read_text()
    marker = "fastHeur.runs(m_fastHeuristicRuns);"
    if master.count(marker) != 1:
        raise RuntimeError("pinned native initializer no longer matches")
    master = "#include <cstdlib>\n" + master.replace(
        marker,
        marker
        + '\n\t\tfastHeur.maxThreads(std::getenv("PIG_OGDF_THREADS") ? std::atoi(std::getenv("PIG_OGDF_THREADS")) : 16);',
    )
    patched = target / "MaxCPlanarMaster.cpp"
    patched.write_text(master)
    flags = [
        "g++",
        "-O3",
        "-march=x86-64",
        "-mtune=generic",
        "-DNDEBUG",
        "-std=c++17",
        "-I" + str(build / "include/ogdf-release"),
        "-I" + str(source / "include"),
        "-I" + str(source / "include/coin"),
    ]
    obj = target / "MaxCPlanarMaster.o"
    run(flags + ["-c", patched, "-o", obj])
    commands = []
    for name in ("exact", "heuristic"):
        command = flags + [ROOT / "native/ogdf" / (name + ".cpp")]
        if name == "exact":
            command += [obj]
        command += [
            build / "libOGDF.a",
            build / "libCOIN.a",
            "-lpthread",
            "-o",
            target / name,
        ]
        run(command)
        commands.append(list(map(str, command)))
    paths = [ROOT / "native/ogdf" / (name + ".cpp") for name in ("exact", "heuristic")]
    paths += [target / name for name in ("exact", "heuristic")]
    (target / "build.json").write_text(
        json.dumps(
            dict(
                ogdf_commit=COMMIT,
                target_architecture="x86-64",
                commands=commands,
                sha256={
                    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in paths
                },
                native_changes="thread cap for native Fast initialization only",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
