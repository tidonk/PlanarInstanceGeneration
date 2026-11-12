# Final evaluation package

This package contains the code, benchmark inputs, recorded evaluation data,
and manuscript build inputs used for the completed November 2026 evaluation.
It does not launch optimization jobs automatically.

## Contents

- `code/`: portable evaluation source and plotting tools, pinned third-party
  source, original benchmark inputs, generated instances, final numerical
  records, and the OGDF executables and upstream source used in the experiments.
- `paper/`: manuscript sources and the exact figure and table inputs at packaging
  time, including the larger legend in the composite example.
- `provenance/production-source/`: a separate copy of the production source and
  settings before portability changes. The production checkout contains local
  changes, so its Git revision alone does not identify the experiment code.
- `provenance/source-files.json`: SHA-256 values for every copied file.
- `provenance/package.json`: source revisions, selected result directories,
  reused original SND records, and exclusions.

## Verify and rebuild the exact paper figures

Run from the extracted package directory:

```bash
python3 code/reproduction/reproduce.py verify
python3 code/reproduction/reproduce.py build-paper
```

The second command requires LaTeX, BibTeX, latexmk, TikZ, PGFPlots, and the
packages used by the manuscript. It builds the supplied manuscript from its
recorded figure and table inputs. No solver license is needed for this route.
The supplied graph PDFs preserve the original layouts exactly. Differences in
TeX installations can affect typography and PDF metadata.

## Recompute the evaluation figures and tables from recorded data

```bash
cd code
python3.11 -m venv .venv
.venv/bin/python -m pip install -r reproduction/requirements.txt
.venv/bin/python -m pip install -e '.[evaluation]'
.venv/bin/python reproduction/reproduce.py export --output ../regenerated
```

This executes the numerical exporters against the included results. It requires
LaTeX for graph drawings, but does not rerun optimization. The numerical results
and exact publication inputs are supplied separately so that later formatting
changes are distinguishable from changes in the experiments. The original
machine paths retained in result records are provenance, not external data
dependencies. Portable copies of path fields are resolved by the export command.

## Rerun the optimization experiments

See `code/reproduction/README.md` and `code/check/OGDF_EXPERIMENT_SETTINGS.md`.
Julia 1.12.6 and the recorded Gurobi environments are required for the application
experiments. Gurobi software and licenses are not distributed in this ZIP.
OGDF planarization uses ABACUS and CLP. The Julia sources and dependency files
are included under `code/third_party/`. New packages include the pinned
TopologyOptimizer, BendersNetworkDesign, and OGDF submodule contents and revision
records. See `code/reproduction/DEPENDENCIES.md` for the retained PIG settings.

Use new output names and keep the supplied results unchanged. The main study
uses 100 input sets in each domain, five methods, 600-second generation-time
planarizations, and 3600-second final planarizations. Source-skewness runs use
86400 seconds. The sampling alternatives use unlimited SCOPF solves, while
the consecutive reference and normalization use the recorded 10800-second
TAL=0 baseline. Main SCOTS uses TAL=1 and a separate SCOPF warmstart.

Recorded data reproduce the reported numerical plots. Independent optimization
reruns need not return identical time-limited bounds or tied heuristic solutions
because timing and parallel execution affect their trajectories. The ZIP does
not claim that a fresh full optimization campaign was rerun during packaging.

Verbose solver traces, caches, redundant plot renderings, and unsuccessful
attempts are omitted. Final summaries, input files, checkpoints, graph data,
and settings needed by the plot exporters are included. Existing reused SND
results retain their original termination states, bounds, and runtimes.
