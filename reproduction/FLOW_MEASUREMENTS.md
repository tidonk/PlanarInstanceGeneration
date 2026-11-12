# Coupling-flow measurements

The measurement scripts compute the observations used by the two flow-boxplot
exporters. The exporters read completed observations and draw the plots.
Neither workflow requires a public ZIP. Run these commands from the repository
root after installing the evaluation dependencies in the main README.

## PGLib

Run the separate SCOPF evaluation described in [README.md](README.md), with
switching disabled and without voltage normalization. Its summary CSV records
mean absolute base-case flow on coupling and inherited edges. Export the ratio
of those means for each feasible composite:

```bash
python check/scripts/export_paper_scopf_flow_boxplot.py \
  --summary check/results/ogdf_20261104_pglib_scopf_tal0/summary.csv \
  --paper-dir check/results/flow_plots
```

The paper uses 310 feasible composites among 500 completed runs. A new run can
have a different number of feasible composites. Incomplete branch-flow
observations are rejected.

## SND

Prepare inputs from the generated composites and the matching source-pool
solutions. Substitute the directories of your completed generation run:

```bash
python check/scripts/prepare_snd_flow_measurements.py \
  --run check/results/paper_run_snd \
  --source-pool check/results/sndlib_source_pool \
  --output check/results/snd_flow_measurements
```

The preparation checks the inherited capacities, source mappings, demands, and
stored feasible routing. It records input and source-solution hashes. The
prepared directory contains `inputs/`, `all.jobs`, and a two-case `pilot.jobs`
list covering the smallest and largest models. `--archive` also accepts a
private copy of the original evaluation archive in place of `--run` and
`--source-pool`.

The solver fixes inherited capacities and enables the existing zero-cost
module on every coupling edge. It first minimizes routing cost and then
minimizes total flow while retaining optimal routing cost within the recorded
numerical tolerance. There are no failures or capacity decisions in this
measurement. Each edge's flow sums both directions over all commodities.

For one prepared input:

```bash
python check/scripts/run_snd_flow_measurements.py \
  check/results/snd_flow_measurements/inputs/INPUT.json.gz \
  check/results/snd_flow_measurements/results/INPUT \
  --validate-baseline
```

Replace `INPUT` with a name from `pilot.jobs` or `all.jobs`. Baseline validation
sets coupling capacities to zero and checks the saved generation objective.
The paper applies this check to the pilot cases. Each output directory must be
new. The defaults retain one solver thread, seed zero, feasibility and
optimality tolerances of `1e-8`, a 6 GiB solver memory limit, an 8 GiB process
address-space limit, and 3,600 seconds per optimization step. The time limit
can be set with `--time-limit`; a result is accepted only when both steps finish
optimally. The process limit can be adjusted with `--memory-limit-gib`.

Run the remaining entries sequentially or through your scheduler. For example,
the following loop processes every input in a fresh measurement directory:

```bash
while read -r instance estimated_seconds; do
  python check/scripts/run_snd_flow_measurements.py \
    "check/results/snd_flow_measurements/inputs/${instance}.json.gz" \
    "check/results/snd_flow_measurements/results/${instance}"
done < check/results/snd_flow_measurements/all.jobs
```

Do not run that full loop over directories already created by the pilot.
Use `remaining.jobs` after completing the pilot instead. Once all 500 solves
finish, collect their results and generate the second panel:

```bash
python check/scripts/collect_snd_flow_measurements.py \
  check/results/snd_flow_measurements \
  --output check/results/snd_flow_measurements/collected
python check/scripts/export_paper_snd_flow_boxplot.py \
  --results check/results/snd_flow_measurements/collected/results.json \
  --paper-dir check/results/flow_plots
```

Collection requires 100 distinct input indices per method, optimal status in
both steps, matching prepared-input hashes, and consistent finite flow ratios.
It writes the individual results, a CSV summary, and aggregate statistics.
The solver model follows the experiment's `run-long.py`, SHA-256
`6aa04fef0110252c1bcd4e9893c7d86241c4a9ecfd0acb02e385e369f61c58dc`.
The public script exposes paths and resource limits as arguments and applies
the process memory limit only when invoked, without changing an importing
process's limits.

## Plot outputs and saved observations

Both exporters write `resources/evaluation/fig-eval-*-flows.tikz` and five
matching `.dat` files beneath the requested paper directory. Existing flow
files are protected from overwriting. The manuscript supplies the panel layout
and `pigMethod0` through `pigMethod4` colors. Boxplot quartiles and whiskers are
computed from raw ratios before applying the display transformation.

The optional `flows` group in `export_paper.py` calls both exporters using
`flow_measurements` in [paper.json](paper.json):

```bash
python check/scripts/export_paper.py --config reproduction/paper.json \
  --parts flows --output check/results/new_flow_export
```

Saved per-instance measurements can also be supplied with `--data-dir`.
The local `reproduction/flow-plot-data/` directory is ignored by Git and is not
part of the public source checkout. This route checks the recorded counts,
ratios, and medians. `--verify-paper-results` additionally requires the original
source checksum when reading a summary or a private archive. Fresh results use
their own statistics and source checksum. Independent runs need not select the
same routing among multiple optima.
