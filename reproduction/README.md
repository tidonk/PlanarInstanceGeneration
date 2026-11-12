# Reproducing the evaluation

Plot inputs and their publication history are listed in [PLOT_DATA.md](PLOT_DATA.md).
To export the recorded paper results from this checkout, follow
[RECORDED_DATA.md](RECORDED_DATA.md). The results archive is separate from Git.

Run commands from the repository root after following the installation and
input-download instructions in the top-level README. All results stay under
`check/results/`; no input or generated instance belongs in Git.

## Main experiment

`check/eval/ogdf_600_3600.env` defines 100 input sets per dataset and five
methods per set. Generation uses seeds 1 through 100. Each input set uses the
same extracted constituents for all methods; a failed method causes the whole
set to be retried. Final solves share valid constituent results and use equal
budgets across methods.

```bash
python check/scripts/preprocess_sndlib.py
python check/eval/make_reuse_dag.py paper_run --prepare-sources
condor_submit_dag check/eval/paper_run.dag
```

The source pool path defaults to `check/results/pglib_source_pool_innoptem`.
Use a clean checkout for an independent full reproduction. The source merge
refuses an existing pool. For repeated campaigns in one checkout, specify a
fresh pool in the generation settings and in the source-merge submit arguments.
Preparing a DAG does not submit it. Inspect `check/eval/*.sub` for memory,
filesystem-domain, and wall-time requirements on another cluster.

Completed generation releases five final-planarization jobs, with higher
priority than generation, and five application jobs. Dataset statistics and
application summaries are produced automatically. Run
`python check/scripts/summarize_ogdf_progress.py paper_run` for an intermediate
statistics. Incomplete comparisons are kept separate from the full test set.

A full experiment takes days on a cluster. Solver limits alone do not give a
reliable elapsed-time forecast: generation can retry, constituents also require
solves, and application callbacks and initialization add time. The short example
in the main README was measured separately at 13.0 minutes.

## Additional evaluations

These are separate from the main DAG. The main DAG alone does not reproduce
the appendix and supplementary tables.

| Evaluation | Entry point and settings |
|---|---|
| PGLib original-source skewness | `evaluate_pglib_source_skewness.py SOURCE --time-limit 86400 --threads 16` |
| PGLib skewness without lines between different voltage levels | Same command with `--voltage-separated`, 86,400 seconds |
| SND source skewness | `evaluate_snd_source_skewness.py --time-limit 86400` on the screened source library |
| Source application reference | `run_scopf_benders.py` with the main SCOTS settings, and `run_snd_benders.py --ignore-manifest-julia`, one run per screened source |
| PGLib coupling-flow measurements | Separate SCOPF runs with switching disabled (`--topological-action-limit 0`), then `prepare_scopf_flow_profile.py` and its generated PGLib application jobs |
| SND coupling-flow measurements | `prepare_snd_flow_measurements.py`, `run_snd_flow_measurements.py`, and `collect_snd_flow_measurements.py`; see [FLOW_MEASUREMENTS.md](FLOW_MEASUREMENTS.md) |
| Sampling alternatives | `check/eval/make_sampling_sensitivity_dag.py`: consecutive, equidistant, adaptive skip-one, and preference for non-leaves; 600/3,600-second planarizations and unlimited SCOPF solves for the alternatives |
| Load normalization | `prepare_scopf_normalization_sensitivity.py`, then the generated 10,800-second TAL=0 SCOPF jobs |
| SND constituent objectives | `prepare_snd_piece_comparison.py`, then the generated SND application jobs with the main settings |

Use `python check/scripts/SCRIPT.py --help` for the result-directory arguments.
All additional source runs must use the same screened source library as the main
experiment. Sampling and normalization comparisons check that the compared
inputs agree. The flow table uses a separate SCOPF baseline without switching, specified by
`applications.flow_baseline`; the SCOTS results cannot be substituted.
The TAL=0 sensitivity runs are separate from the main TAL=1 SCOTS
application experiment. The consecutive sampling reference uses the 10,800-second
TAL=0 baseline. Alternative sampling SCOPF runs have no solver time limit. The application entry points accept `--dry-run` to
write their settings and command without starting a solve.

The required source libraries are named in `quick-sources.json` for PGLib and
in the result of `preprocess_sndlib.py` for SNDlib. The complete original input
inventory, including sources removed by screening, is `input-files.csv`.

## Generating the paper inputs

Copy `paper.json` and replace its paths with those of the completed runs.
Paths are relative to the code repository unless absolute. This file specifies
inputs; it is not evidence that those runs have completed. The example input
number is explicit so the graph and its legend always refer to the same case.

```bash
python check/scripts/export_paper.py --config reproduction/paper.json \
    --output check/results/paper_export
```

The destination must be empty. The exporter writes `resources/evaluation/`,
ready to copy into the manuscript repository. It never changes manuscript prose
or overwrites an existing manuscript. An export failure can leave partial output;
use a fresh destination after resolving the missing or inconsistent input.

| Paper output | Export code in `check/scripts/` |
|---|---|
| Source-skewness tables | `export_paper_source_skewness.py` |
| Generation and final-evaluation runtimes | `export_paper_runtime_plot.py` |
| Normalized added-skewness distributions | `export_paper_skewness_cdf.py` |
| Added-skewness distribution on jointly exact inputs | `export_paper_skewness_distribution.py` |
| Representative composites and last join | `export_paper_last_join.py` |
| Structural bounds and global characteristics | `export_paper_structure_tikz.py` |
| Application outcomes and runtimes | `export_paper_application_outcomes.py`, `export_paper_exact_tables.py` |
| Summary and complete skewness tables | `export_paper.py`, `export_paper_skewness_tables.py` |
| Generation/application counts | `export_paper.py` |
| Coupling-flow boxplots | `export_paper_scopf_flow_boxplot.py`, `export_paper_snd_flow_boxplot.py` |
| Earlier coupling-flow table | `export_paper_scopf_flows.py` |
| Merge-vertex degree distribution | `analyze_merge_node_degrees.py`, `export_paper_merge_vertex_degrees.py` |
| Sampling tables | `export_paper_sampling.py` |
| Load-normalization table | `compare_scopf_normalization.py` |
| Source/extracted/composite characteristics | `export_paper_extraction.py` |
| SND constituent-objective table | `summarize_snd_piece_comparison.py` |

The coupling-flow workflow includes both the measurement and plotting scripts.
See [FLOW_MEASUREMENTS.md](FLOW_MEASUREMENTS.md) for preparation, optimization,
collection, and export commands. It accepts completed generation directories
and result summaries, so no evaluation ZIP is required. Saved observations are
optional local files and are excluded from Git.

Use `--parts flows` to export both panels from the `flow_measurements` paths in
`paper.json`. This group is separate from the historical five export groups
below because it requires the additional SND flow solves.

`--parts generation applications sources example supplement` selects groups;
the default includes all five. Diagnostic CSV/JSON summaries stay in the code
repository. The graph drawings require LaTeX. Other plots are editable TikZ
and PGFPlots sources and inherit the manuscript's fonts and macros.
Conceptual illustrations and pseudocode are manuscript sources, not experiment
outputs; they remain in the manuscript repository.

For the quick example, `run_quick_example.py` calls the production generation
and evaluation entry points, then `export_quick_example.py`. Its Figure 5a
shows bounds for all five inputs, unlike the full paper's jointly exact subset
plot. The filled/open endpoints denote guaranteed/possible fractions, not
statistical confidence intervals. The example does not replace the full study.

## Source and data provenance

The 66 PGLib files can be downloaded and checked against the pinned revision.
SNDlib's 26 filenames and hashes identify the original XML files even if its
service is unavailable. The original data and all generated instances are
excluded from the repository and source archive.

The two Julia libraries and OGDF are pinned as Git submodules. Initialize them
with `git submodule update --init --recursive`. The TopologyOptimizer setup
uses the recorded PIG defaults as described in [DEPENDENCIES.md](DEPENDENCIES.md).
OGDF is built with portable x86-64 settings. `build/`, environments, caches, solver
binaries, and results are excluded. See `THIRD_PARTY.md` for attribution.

Future DAGs produce evaluation data with `--no-compile`; they do not build
review PDFs or graph galleries. Application jobs start after their generation
job, independently of this data processing.


## Completed evaluation and data package, 10 November 2026

The production checkout is `planar-instance-generation-code`. Its local source
changes are part of the experiment and are not identified by the base Git commit
alone. This repository contains the portable
source distribution and the consolidated plot exporters, imported from commit
`1f3abe9c8e06c2c3f042395dab95570dfa7f6028` of the development repository.
The data package records the production and portable sources separately with
file checksums.

The final data paths are in `paper.json`. All 48 original and voltage-separated
source results are present. The sampling alternatives have 300 completed
final-skewness records and 300 completed SCOPF records. Sampling statistics use
69 identical input sequences across all four policies. The main applications
have 500 recorded outcomes per domain, including unsuccessful solves. Reused SND
records preserve their original runtimes, bounds, and termination states.

Create the complete local ZIP from the portable checkout:

```bash
python check/scripts/package_evaluation.py \
  --production /path/to/planar-instance-generation-code \
  --paper /path/to/planar-instance-generation \
  --output /tmp/planar-evaluation-20261110 \
  --zip /tmp/planar-evaluation-20261110.zip
```

The output directory must be new. This command collects original inputs,
generated instances, checkpoints, graph data, settings, final result summaries,
the precise native source and executables, and the manuscript build inputs.
It excludes caches, redundant renderings, verbose solver traces, and unsuccessful
attempt directories. Original SND records referenced by reused results are
included. It never starts or submits optimization jobs.

Read `PACKAGE_README.md` for the extracted ZIP's commands. Use `reproduce.py
verify` to verify the copied files, `reproduce.py export --output NEW_DIRECTORY`
to recompute numerical plots from recorded data, and `reproduce.py build-paper`
to rebuild the exact recorded paper figures. A Gurobi license is needed only
when rerunning application optimization. Independently rerunning time-limited
or parallel optimization need not reproduce the same search trajectory.

The current manuscript's figure numbering and layout are authoritative. Earlier
numbering in the plot inventory is historical. The source table now contains
all completed source results. The composite-example legend uses normal text size.
