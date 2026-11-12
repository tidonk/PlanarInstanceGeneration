# Paper experiment workflow

The current configuration is [ogdf_600_3600.env](ogdf_600_3600.env). It prepares
100 accepted sets of extracted networks for each of PGLib and SNDlib, with
all five merge methods applied to each set. The exact algorithms, modifications,
versions, and time accounting are documented in
[OGDF_EXPERIMENT_SETTINGS.md](../OGDF_EXPERIMENT_SETTINGS.md).

## Preparation

The repository, Python environment, native builds, benchmark inputs, and result
directories must be accessible from every compute node. Install the dependencies
and prepare OGDF and InnOpTEM as described in the top-level README. The SND
source pool must already exist. The PGLib pool is screened with the current
InnOpTEM OPF implementation before generation.

From the repository root, prepare a new workflow without submitting it:

```bash
.venv/bin/python check/eval/make_reuse_dag.py NEW_EXPERIMENT --prepare-sources
```

The command creates `check/eval/NEW_EXPERIMENT.dag`, a list of source jobs, and log directories.
With `--prepare-sources`, it creates one job per PGLib input file. Each job
applies the size and Euler filters before attempting OPF. A merge job waits for
all source jobs, validates their common settings and saved solutions, and
publishes the source pool before releasing PGLib generation.
It rejects an existing workflow name. It uses `ogdf_600_3600.env` unless an
explicit `--settings` file is supplied. On a later rerun, choose a fresh source
pool path in the settings and source-screening job as well. Source screening
refuses to overwrite an existing result directory.

After reviewing the generated workflow, submission is a separate command:

```bash
condor_submit_dag -batch-name pig-NEW_EXPERIMENT-main-workflow check/eval/NEW_EXPERIMENT.dag
```

## Execution

Each generation job draws one set of constituents and applies all five methods
to those same constituents. A failed method causes a retry for the complete
set. Jobs write separate result directories named
`check/results/<experiment>_<dataset>_job_i<NNN>`.

Each successful generation job releases five independent final-planarization
jobs and an array of five application jobs. Application jobs depend only on
that generated set and have priority 0. Final planarizations have scheduler
priority 100, source screening and its merge step have priority 50, and
generation has priority 0. The DAG also
prioritizes submission of final-planarization nodes. These priorities favor
finishing comparisons before starting more generation jobs, subject to resource
availability. Those jobs share locked constituent checkpoints and receive the same
available generation bounds. A data processing job consolidates each dataset and computes the paper statistics
after its final solves finish, reusing the completed planarization checkpoints.
It does not compile a review PDF or graph galleries. A separate application summary waits for every
application array and the dataset statistics. Application outcomes do not change
generation acceptance.

| Step | Configuration |
|---|---|
| PGLib acceptance | InnOpTEM base-case DC-OPF, 300 seconds, one thread |
| SND acceptance | Fixed-capacity multicommodity flow, one thread |
| Heuristic planarization | Default OGDF Fast10, maximum 16 threads |
| Exact generation planarization | OGDF/ABACUS/CLP, 600 seconds per graph |
| Final and shared constituent planarization | Same exact solver, 3,600 seconds per graph |
| PGLib application | SCOTS, one permitted switching action, 10,800 seconds |
| SCOPF warmstart for SCOTS | Separate 10,800-second allowance |
| SND application | Benders method, 10,800 seconds |

Generation and final-planarization jobs request 16 logical CPUs, corresponding
to eight physical cores on the intended nodes. The exact search remains
serial. Each application job requests one CPU. Memory requests and scheduler
wall limits are specified in the submit files. Native solver limits do not
include external Fast10 or Python preprocessing, which remain part of the
recorded total runtime.

## Entry points

| File | Role |
|---|---|
| `make_reuse_dag.py` | Prepare the current complete workflow |
| `pglib_sources.sub` | Apply filters and screen one PGLib source with InnOpTEM |
| `pglib_source_pool.sub`, `merge_pglib_source_pools.py` | Validate and combine all source results before generation |
| `pglib_one.sub`, `job.sh` | Generate one set for either dataset |
| `pglib_cohort_skewness.sub`, `snd_cohort_skewness.sub` | Evaluate one final graph with shared constituent solves |
| `report.sh`, `postprocess.py` | Consolidate outputs and export report data |
| `prepare_application.sh`, `make_application_jobs.py` | Derive application inputs from accepted composites |
| `application_job.sh` | Run one PGLib or SND application solve |
| `summarize_applications.py` | Summarize completed application outputs |

Use the Python scripts' `--help` for individual runs. OGDF retains its constructor defaults.

## Checking results before the full workflow finishes

From the repository root, refresh an early report without starting any solves:

```bash
.venv/bin/python check/scripts/summarize_ogdf_progress.py paper_run
```

Open `check/results/paper_run_progress/index.html`. The same directory contains
`results.csv` and `summary.json`. Replace the experiment name for another run.
Add `--watch 300` to refresh every five minutes while the process remains running.
The command writes only its own report files and leaves experiment outputs intact.

The report counts generated inputs and completed final solves for both datasets.
Method summaries use only inputs with all five final results. The OF-H versus
OF-B comparison also includes completed pairs while other methods are still
running, with that separate denominator stated. It distinguishes exact final
skewness, proven zero added skewness, proven strict improvements, and smaller
upper bounds without a proof of improvement. It includes selection times,
final solve times, input IDs for decisive comparisons, and application job states.
Bounds use the same composition calculations as the final reports.
Inconsistent settings or bounds are reported and excluded from comparisons.
Partial results can favor easier inputs and do not establish the full-study claims.

The dataset statistics and application summaries remain automatic in the DAG;
review PDFs are not generated.
Application jobs become eligible immediately after their generation job finishes.
Check scheduler progress separately with `condor_q`; a prepared DAG has no running
jobs until it has been submitted.

## Results and scope

Saved planarization results include input and backend settings. A matching
completed checkpoint can be reused. Incompatible settings require a fresh
result directory. Retained edges and bound supports are validated when read.
Generation and final-evaluation logs preserve the native solver output.

The main workflow does not include full-source skewness, source application
runs, or the two supplementary sensitivity studies. Those require separate
input lists based on the newly screened sources. The current paper settings
export report data without compiling aggregate PDFs. Use the report scripts'
compilation option when a rendered review report is needed.

For another cluster, check shared-storage access, `FileSystemDomain`, Julia
paths, Gurobi licensing, memory, and CPU allocation before submission.

Julia workers install dependencies directly with `Pkg.instantiate()` in
`/local/tmp/$USER/.julia-pig-installed`. The installer also downloads lazy artifacts. Package caches are never copied from
home. Only the pinned `Project.toml` and `Manifest.toml` are staged in a local
environment for installation, and their hashes must remain unchanged. The
source repository and result files remain shared, as in `dispatch_testset.sh`.
Julia 1.12.6 itself is downloaded from the official distribution, checked against
its SHA-256 checksum, and extracted under `/local/tmp/$USER/julia-runtime`.

The supplementary DAG begins with one 300-second TAL=0 solve using the original
pinned InnOpTEM source. It performs no extra package precompilation. Startup
without output or CPU activity is limited to 900 seconds. SCOPF/SCOTS additionally limit total
process time to finite solver budgets plus 1800 seconds. A SCOPF `--time-limit 0`
disables the solver and total process time limits. Sampling alternatives use this
setting and disable the submission file's runtime hold; the reused TAL=0 baseline
retains its 10800-second budget. Both applications preserve
complete results if Julia stalls during shutdown and bound process cleanup.
Already running processes retain their original environment.

Queue names use `pig-<experiment>-<dataset>-<task>[-<variant>]`. Workflow controllers
use `pig-<experiment>-<purpose>-workflow`; pass `-batch-name` when submitting a DAG.
`batch_names.inc` maps campaign variants to readable labels without renaming result directories.
