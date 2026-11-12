# Planar Instance Generation

Source code and reproduction scripts for *Composition of Near-Planar
Infrastructure Network Instances with Certified Skewness Bounds* by Tim Donkiewicz.
The code extracts and composes PGLib and SNDlib networks using five methods:
random (R), high degree (HD), Kamada–Kawai (KK), heuristic outer face (OF-H),
and exact outer face (OF-B).

## Repository and recorded results

The `main` branch contains source code, experiment settings, tests,
reproduction instructions, and the 1,000 final generated instances from the
main experiment. Original benchmark inputs, recorded solver results, solver
binaries, and manuscript files are excluded from Git.

A fresh checkout can rerun the experiments after installing the dependencies
and obtaining the benchmark inputs. To reproduce the reported numerical plots
without rerunning optimization, use the separate `planar-evaluation-20261110.zip`
archive. Its contents and checksum are described in
[the recorded-data instructions](reproduction/RECORDED_DATA.md). The archive
must be obtained separately and is not downloaded by cloning this repository.
Time limits and parallel execution can cause independent optimization runs to
return different bounds, heuristic solutions, and runtimes.

## Installation

Use x86-64 Linux, Python 3.11, Julia 1.12.6, CMake, Git, and a C++17 compiler.
Install Julia with Juliaup and select version 1.12.6. A working Gurobi license is
required for the power-flow and communication-flow feasibility checks and the
application experiments. OGDF planarization uses ABACUS and CLP, without Gurobi.
TeX Live with TikZ, PGFPlots, and `standalone` is needed for the graph drawings
and manuscript, but not for the short example's PDF/PNG plots.

```bash
git clone --recurse-submodules https://github.com/tidonk/PlanarInstanceGeneration.git
cd PlanarInstanceGeneration
juliaup add 1.12.6
juliaup default 1.12.6
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r reproduction/requirements.txt
python -m pip install -e '.[evaluation,plots]'
python check/scripts/build_ogdf.py --jobs 16
python check/scripts/prepare_topology_optimizer.py --instantiate
```

OGDF, TopologyOptimizer.jl, and BendersNetworkDesign.jl are Git submodules
pinned to the revisions in [reproduction/dependencies.json](reproduction/dependencies.json).
For an existing checkout, run `git submodule update --init --recursive`.
GitHub's automatically generated source ZIP does not contain the submodule
contents. Use a recursive clone or the complete evaluation archive.

The native OGDF interface and its build-time thread limit are maintained here.
The build script exports the pinned upstream source into `build/ogdf` and leaves
the submodule unchanged. TopologyOptimizer is the renamed InnOpTEM package.
Its setup script prepares `build/TopologyOptimizer` with the recorded PIG
defaults from [reproduction/settings/topology-default.toml](reproduction/settings/topology-default.toml).
The application runs then apply [check/eval/scots_benders.toml](check/eval/scots_benders.toml)
and their run-specific limits. See [the dependency notes](reproduction/DEPENDENCIES.md)
for the comparison with the public solver settings.

Licenses are listed in [THIRD_PARTY.md](THIRD_PARTY.md). Installation and
compilation are separate from the experiment runtimes below. Set `PIG_JULIA`
if Julia is not on `PATH`. Use the same `JULIA_DEPOT_PATH` for installation
and the later Julia runs:

```bash
export JULIA_DEPOT_PATH="$PWD/build/julia-depot"
```

For the full SND application experiment, also install its Julia dependencies:

```bash
export JULIA_DEPOT_PATH="$PWD/build/julia-depot"
julia +1.12.6 --project=third_party/BendersNetworkDesign -e 'using Pkg; Pkg.instantiate(); using BendersNetworkDesign'
```

## Benchmark data

The [published instances](instances/README.md) contain the exact 500 PGLib
and 500 SNDlib composites used in the main evaluation, with a catalogue,
source attribution, and SHA-256 checksums. Verify them from the repository root:

```bash
sha256sum -c instances/SHA256SUMS
```

To generate new instances, download the original inputs from
[PGLib-OPF](https://github.com/power-grid-lib/pglib-opf) and
[SNDlib](https://sndlib.zib.de/) under their respective terms:

```bash
python check/scripts/fetch_inputs.py --dataset pglib
python check/scripts/fetch_inputs.py --dataset sndlib
```

PGLib is pinned to `dc6be4b2f85ca0e776952ec22cbd4c22396ea5a3` (v23.07).
[The input inventory](reproduction/input-files.csv) records all 66 PGLib and
26 SNDlib filenames and SHA-256 hashes. SNDlib has no corresponding Git revision.
Its download service currently redirects to a host with a certificate error.
If automatic download fails, obtain the XML files from the data owner and place
them under `check/data/sndlib/`. Verify manually obtained files with:

```bash
python check/scripts/fetch_inputs.py --verify-only
```

The script rejects different contents rather than silently substituting a
newer dataset. The short example needs only PGLib.

## Short example: approximately 10–30 minutes

After installation and data download:

```bash
python check/scripts/run_quick_example.py --name quick_example
```

This generates **five input sets and 25 final composites**, applying all five
methods to each input set. It computes fresh OPF dispatches, runs generation
with a 10-second exact-planarization limit, and evaluates skewness with a
20-second limit per graph. Four final-evaluation processes run concurrently,
with one solver thread each. It does not use saved generated graphs or solver
results. Use a fresh name for another run.

The test on the innoptem server took **13.0 minutes**, including Julia startup
and source screening. OF-H was provably better than R, HD, and KK on two of
five inputs; OF-B was on three. The fixed seeds are illustrative, not a random
sample for estimating the methods' overall performance. Different hardware,
solver timing, or versions can change results. The script reports what it
actually obtains and does not require a particular method to win.

Open `check/results/quick_example/figures/`. It contains PDF and PNG versions
of the generation-runtime comparison and the added-skewness comparisons
corresponding to Figures 3, 4a, and 5a, plus `summary.csv`. Bounds are shown
explicitly. `timing.json` records total elapsed time. This short run excludes
SNDlib, the application solves, and the supplementary studies.

## Full reproduction

The [experiment guide](reproduction/README.md) gives the workflow and maps every
evaluation figure and table to its generating code. The additional
[coupling-flow workflow](reproduction/FLOW_MEASUREMENTS.md) includes the
measurement scripts and both plot exporters. The main experiment has
100 input sets per dataset, 1,000 final composites, and separate application
solves. Prepare the SND source pool, then prepare and submit the HTCondor DAG:

```bash
python check/scripts/preprocess_sndlib.py
python check/eval/make_reuse_dag.py paper_run --prepare-sources
condor_submit_dag check/eval/paper_run.dag
```

Review the shared filesystem, environment, and resource settings before using
another cluster. The [cluster guide](check/eval/README.md) describes dependencies
and job priorities. The [algorithm settings](check/OGDF_EXPERIMENT_SETTINGS.md)
list the OGDF defaults and modifications. Each final composite is evaluated
with reusable constituent solutions and bounds.

After the required runs finish, copy `reproduction/paper.json`, set the result
paths, and generate only the manuscript's evaluation inputs:

```bash
python check/scripts/export_paper.py --config reproduction/paper.json \
    --output check/results/paper_export
```

The output directory must be empty. Copy its `resources/evaluation/` contents
into the manuscript checkout and compile the manuscript there. The exporter
requires complete comparisons and fails on missing or inconsistent results.
`--parts generation` exports only generation and skewness results, without
waiting for the application and supplementary studies. It does not edit prose.

## Validation and citation

```bash
python -m unittest discover -s tests
python -m pip install -e '.[docs]'
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

See [CITATION.cff](CITATION.cff) for citation metadata and
[CONTRIBUTORS.md](CONTRIBUTORS.md) for project and library credits. The arXiv identifier will
be added when assigned. Original software is licensed under
**GPL-3.0-or-later**; original documentation is under **CC BY 4.0**. Third-party
sources and benchmark data retain their own terms. See [LICENSE](LICENSE),
[the documentation license](docs/LICENSE-CC-BY-4.0.txt), and
[THIRD_PARTY.md](THIRD_PARTY.md).
