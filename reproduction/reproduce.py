#!/usr/bin/env python3
"""Verify the packaged inputs, rebuild the manuscript, or export numerical results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

CODE = Path(__file__).resolve().parents[1]
PACKAGE = CODE.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "build-paper", "export"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "verify":
        records = json.loads((PACKAGE / "provenance/source-files.json").read_text())
        for name, record in records.items():
            path = PACKAGE / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise SystemExit(f"checksum mismatch: {name}")
        print(f"Verified {len(records)} copied files")
    elif args.action == "build-paper":
        for name in ("planar-instance-gen.tex", "supplementary-material.tex"):
            subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", name],
                           cwd=PACKAGE / "paper", check=True)
    else:
        if args.output is None:
            parser.error("export requires --output pointing to a new directory")
        subprocess.run([sys.executable, str(CODE / "check/scripts/export_paper.py"),
                        "--config", str(CODE / "reproduction/paper.json"),
                        "--output", str(args.output.resolve())], cwd=CODE, check=True)


if __name__ == "__main__":
    main()
