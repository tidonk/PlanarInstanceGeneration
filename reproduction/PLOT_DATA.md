# Paper figure and table data

Paths are relative to the code repository. `P` = paper repository; `S` = `check/scripts`;
`G` = `check/results/ogdf_20261104_{pglib,snd}`.
`Data path` lists source inputs, never the generated manuscript figures.
Times are plot-data export times in Europe/Berlin, not experiment runtimes.
**Whenever a figure or table is updated, update its generation time and Up to date cell here.**
✓ means the plot reflects the current completed experiment data (or is a static illustration).
Leave the cell empty for older, mixed, or unverified results. Do not change the time merely on rebuilding LaTeX.

| Number | Description | Script path | Data path | Data generation time | Up to date |
|---|---|---|---|---|---|
| Fig. 1 | Schematic networks | `P/resources/fig-input-networks.tex` | `P/resources/country-composition/` | Static | ✓ |
| Fig. 2 | Matching example | `P/resources/fig-noncrossing-matching.tex` | Embedded in script | Static | ✓ |
| Fig. 3 | Generation runtimes | `S/export_paper_runtime_plot.py` | `G/tables/`, `G/evaluation/paired_strategy_outcomes.csv` | 2026-11-07 07:43 | ✓ |
| Fig. 4 | Added skewness bounds | `S/export_paper_skewness_cdf.py` | `G/evaluation/paired_strategy_outcomes.csv` (all inputs) | 2026-11-07 07:43 | ✓ |
| Fig. 5 | Exact added skewness | `S/export_paper_skewness_distribution.py` | `G/evaluation/paired_strategy_outcomes.csv` (jointly exact inputs; `export_paper.py` first generates the table read by the plotter) | 2026-11-07 07:43 | ✓ |
| Fig. 6 | Example i070 | `S/export_paper.py`, `S/export_paper_last_join.py` | PGLib `G/tables/accepted_instances.csv`, `G/plots/composite/221_i070{,_2,_3,_4,_5}/`; bounds from paired CSV | 2026-11-06 08:44 | ✓ |
| Fig. 7–8 | Structural changes | `S/export_paper_structure_tikz.py` | `G/evaluation/structural{,_exact}.csv`, paired CSVs | 2026-11-07 07:43 | ✓ |
| Fig. 9(a) | Application outcomes: SCOTS | `S/export_paper_application_outcomes.py` | `A_SCOTS` | 2026-11-07 07:43 | ✓ |
| Fig. 9(b) | Application outcomes: SND | `S/export_paper_application_outcomes.py` | `A_SND`, `AS_SND` | 2026-11-07 07:43 | ✓ |
| Fig. 10(a) | Application runtimes: SCOTS | `S/export_paper_exact_tables.py` | `A_SCOTS` | 2026-11-07 07:43 | ✓ |
| Fig. 10(b) | Application runtimes: SND | `S/export_paper_exact_tables.py` | `A_SND`, `AS_SND` | 2026-11-07 07:43 | ✓ |
| Fig. 11(a) | Evaluation funnel: SCOTS | `S/export_paper.py` | PGLib `G/tables/attempts.csv`, `A_SCOTS` | 2026-11-07 07:43 | ✓ |
| Fig. 11(b) | Evaluation funnel: SND | `S/export_paper.py` | `A_SND`, SND `G/tables/attempts.csv` | 2026-11-07 07:43 | ✓ |
| Fig. 12 | Selected vertex degrees | `S/analyze_merge_node_degrees.py`, `S/export_paper_merge_vertex_degrees.py` | `G/tables/accepted_instances.csv`, `G/plots/merged/`, root metadata | 2026-11-07 07:43 | ✓ |
| Fig. S1 | Extraction | `S/export_paper_extraction.py` | `G/evaluation/structural{,_exact}.csv`, paired CSVs | 2026-11-07 07:43 | ✓ |
| Fig. S2–S6 | Last join of i070 | `S/export_paper_last_join.py` | PGLib `G/plots/merged/221_i070{,_2,_3,_4,_5}/stage_003/`, accepted-instance table | 2026-11-06 08:47 | ✓ |
| Fig. S7–S8 | Algorithm diagrams | `P/resources/fig-{pipeline-overview,example}.tex` | `P/resources/fig-toy-defs.tex` | Static | ✓ |
| Fig. (unnumbered supplement) | Source networks | `S/generate_source_graphs.py` | `<pglib>/pglib_opf_<case>.m`, `<topohub>/data/sndlib/<case>.json`; case lists in script | Static | ✓ |
| Tab. 1 | Literature comparison | `P/resources/tab-modifications.tex` | Cited studies in `P/literature.bib`; manually maintained | Static | ✓ |
| Tab. 2 | Source skewness bounds | `S/refresh_paper_source_skewness.py` (partial update), `S/export_paper_source_skewness.py` (previous entries) | `R/ogdf_sources_20261106_{pglib,voltage}/*.json`, `R/ogdf_sources_20261106_snd/skewness.csv` | 2026-11-09 12:54 | ✓ |
| Tab. 3 | Skewness summary | `S/export_paper.py` | `G/evaluation/paired_strategy_outcomes.csv` | 2026-11-05 | ✓ |
| Tab. 4 | SCOPF coupling flows | `S/export_paper_scopf_flows.py` | `R/ogdf_20261104_pglib_scopf_tal0/summary.csv`, feasible-run `logs/*/adapter_result.json` | 2026-11-07 13:12 | ✓ |
| Tab. S1–S2 | Complete added skewness | `S/export_paper.py`, `S/export_paper_skewness_tables.py` | `G/evaluation/paired_strategy_outcomes.csv` | 2026-11-05 | ✓ |
| Tab. S3 | Sampling: skewness | `S/export_paper_sampling.py` | `R/ogdf_20261104_pglib{,_exact}/`, `R/ogdf_sampling_20261106_<policy>_pglib_job_i*/`, `R/ogdf_sampling_20261106_<policy>_pglib_exact/` | 2026-11-09 12:54 | ✓ |
| Tab. S4 | Sampling: SCOPF outcomes | `S/export_paper_sampling.py` | `R/ogdf_20261104_pglib_scopf_tal0/summary.csv`, `R/ogdf_sampling_20261106_<policy>_pglib_scopf/summary.csv` | 2026-11-09 12:54 | ✓ |
| Tab. S5 | Voltage normalization | `S/compare_scopf_normalization.py` | `R/ogdf_20261104_pglib_scopf_tal0/summary.csv`, `R/ogdf_20261104_normalized_pglib_scopf/{summary,normalization_sensitivity}.csv` | 2026-11-07 13:12 | ✓ |
| Tab. S6 | SND constituent objectives | `S/summarize_snd_piece_comparison.py` | `R/ogdf_20261104_{snd_benders,piece_comparison_snd_pieces_benders}/{jobs.csv,logs/*/summary.json}` | 2026-11-07 13:12 | ✓ |

`R` = `check/results`; `B` = `R/merge_count_600_3600_20260911_log2`;
Historical table paths follow the original exporters; numerical reproduction has not been rechecked.
`Q` = `R/sampling_600_3600_20260927_log2`; policies: `equidistant`, `adaptive_skip_one`, `prefer_non_leaf`.

Use this checkout after providing the recorded data as described in
[RECORDED_DATA.md](RECORDED_DATA.md):
`python check/scripts/export_paper.py --config reproduction/paper.json --output /tmp/paper-plots --parts generation example`
(empty output directory). Settings are also saved as `check/paper-plot-data.json` in the experiment checkout.
Preserve both `G` directories. Dataset versions/hashes are in `README.md` and `reproduction/input-files.csv`.

Figs. 9(a), 10(a), and 11(a) use all 500 completed SCOTS runs (TAL=1).
`A_SCOTS` = `check/results/ogdf_20261104_pglib_scopf/summary.csv`;
`A_SND` = `check/results/ogdf_20261104_snd_benders/summary.csv`;
`AS_SND` = `check/results/snd_source_benders_10800/summary.csv`.
Fig. 10 includes globally optimal runs only and SCOPF warmstarting.
SCOTS source rows use all 16 completed matching reruns in
`check/results/ogdf_20261106_pglib_source_scots/summary.csv`.
Both applications now include the source comparison in the outcomes and runtime plots.
The application plots use 500 recorded composite outcomes per domain. SND runs ending
at the outer time limit remain in the no-solution category. Input hashes are checked
against every result, including reused SND results.
The SND percentage in the text now agrees with the refreshed plot: 214/500 = 42.8%.

Outstanding: completion of Tab. 2 and Tabs. S3–S4. Tabs. 4, S5, and S6 now use the completed current results. Check marks describe generated data only. The numerical statements and the normalization qualification are synchronized below.

2026-11-06: Table 2 is submitted as DAG 849580 (`check/eval/ogdf_sources_20261106.dag`):
48 OGDF solves, 24 h each, nice-user priority -100; voltage-separated results precede the original PGLib solves.
SND constituent solves: 376 normal jobs submitted as cluster 849375.
DAG 849376 passed its local Julia TAL=0 check and releases:
500 normal TAL=0 solves (Table 4 and the normalization baseline), 500 nice-user normalized solves,
300 nice-user sampling generations + 300 final skewness solves + 300 SCOPF solves,
The consecutive-sampling SCOPF baseline reuses the OF-B subset of the TAL=0 runs.
The sampling alternatives have no solver time limit. The baseline retains the main TAL=0 run budget of 10800 s; no 300-second filtering is applied.
Sensitivity jobs remain nice-user priority 0; source skewness uses nice-user scheduling at priority -100. The cache check is normal priority 1000,
with 2 CPUs and 14 GiB; TAL=0/normalization use 10800 s and no redundant SCOPF warmstart.
Julia packages are installed directly with `Pkg.instantiate()` in `/local/tmp/$USER/.julia-pig-installed`, without copying home caches. Julia 1.12.6 is downloaded and verified on local disk. The former cache gate now checks one 300-second TAL=0 solve with the original runtime; no extra precompilation is performed.
Submitted/prepared work is not a refreshed figure or table: export times and check marks stay unchanged.
Tab. 2 concerns fixed source graphs; verify input hashes and historical bounds before reuse, rather than automatically rerunning.

New experiment input paths are recorded in `check/paper-plot-data.json` and `reproduction/paper.json`;
Tab. S4 reuses `ogdf_20261104_pglib_scopf_tal0` (OF-B only), never the TAL=1 application.
Cluster 849382 is redundant and performs no solves; its existing DAG entries finish successfully to preserve the running workflow.

2026-11-06 recovery: node path failures triggered DAGMan's 100-hold limit.
SND resumes as cluster 849726 (487 jobs; 13 completed results retained); TAL=0 SCOPF resumes as 849727 (500 jobs).
These replacement batches run independently of the failed DAG nodes; summarize their existing result directories after completion.
No plot data changed during recovery.

Node recovery: all 64 nodes pass shared-file access and log-file creation checks. Temporary node and restart restrictions are removed from current jobs and submit files. No held jobs or new failures remain at this check; diagnostics are in `R/job_health_20261106/`.

2026-11-07: 300 sampling skewness jobs stopped because the worker required all five
methods although the saved sampling settings specify OF-B only. The worker now
checks completeness against the saved method list. Four regression tests pass.
Job 849768.0 completes with valid exact bounds for both subinstances and the composite;
the remaining 299 existing jobs are released once after that check.
Diagnostics: `R/job_health_20261107/`. Plot exports and input checks:
`R/paper_refresh_20261107/`. Generation and structural figure inputs reproduce
identically. The changed figures are SCOTS outcomes and runtimes, and SND outcomes,
runtimes, and evaluation funnel. Source-skewness and sampling results remain incomplete.

2026-11-07 12:03: all 182 currently waiting jobs use normal accounting after the
requested priority change (10 never-started applications and 172 skewness retries).
The nice-user accounting assignment is cleared; running jobs keep their assignment.
There are no held jobs. Sampling has 122/300 skewness results and 290/300 SCOPF
results; 11 source-skewness jobs and 6 sampling skewness jobs are running.
Queue records: `R/job_health_20261107/priority-120210/`. Plot export dates are unchanged.

2026-11-07 13:12: refreshed Tables 4, S5, and S6, and the completed cells of Table 2.
Exported tables and diagnostics are in `R/paper_tables_20261107/`.
Table 2 checks PGLib source SHA-256 values, graph sizes, and retained planar subgraphs.
All 16 SNDlib rows match the current source-pool graph signatures and valid stored planar subgraphs;
the numerical SNDlib table is unchanged. Eleven original PGLib cells and four voltage-separated
cells retain the previous table and are not marked current. The completed original
`case200_activ` interval changes from [4, 7] to [1, 6]; no historical lower bound is combined
with the new result without matching input evidence. All 48 source solves have a requested 24-hour limit;
11 are currently running and four original PGLib jobs await their
voltage-separated predecessors. Across all variants, 33/48 source results have returned.

Table S6 uses all 500 recorded composite outcomes and all 376 constituent outcomes.
Constituents: 309 optimal, 56 solver time limits, 11 external wall-clock limits.
Composites: 130 optimal, 119 solver time limits, 251 external wall-clock limits.
The table includes only the 130 jointly optimal comparisons, from 30 inputs:
83 lower, one equal, 46 higher, and median relative difference -8.6%.
The 506 current XML inputs used in the comparison or its constituent eligibility checks
match the SHA-256 values in their result summaries. External time limits remain recorded
outcomes and are not counted as optimal solves.

Table 4 and its caption use all 310 feasible TAL=0 results (500 outcomes total), with full coupling and
inherited-edge flow observations. Table S5 pairs all 500 baseline and normalized outcomes:
310 versus 288 feasible, with five gained and 27 lost. HD stays at 70 feasible runs.

Following approval of numerical updates, the corresponding manuscript values are synchronized:

- Table 2's caption now gives limits up to 24 hours for the mixed PGLib results and
  24 hours for SNDlib. The table remains incomplete for the current PGLib runs.
- The main text and supplement now give 130 SND comparisons; the main median is -8.6%.
  The supplement's 30 inputs remain correct.
- The normalization statement now excludes HD from the methods with fewer feasible runs.
  The current 343 extracted subinstances have scale factors from 0.000726 to 885.339.
- The coupling-flow median ratios in the text are R 134.5%, HD 280.1%, KK 70.2%,
  OF-H 99.4%, and OF-B 104.9%.
- The main SND feasibility percentage is 42.8%, matching the existing figure.
These corrections preserve the conclusions about interaction between subinstances,
comparable mean coupling and inherited-edge flows under outer face selection, and
no overall feasibility benefit from normalization. Other manuscript wording is unchanged.

Sampling remains incomplete: at this check there are 133/300 skewness results
(96 equidistant, 37 adaptive skip-one, zero prefer-non-leaf) and 290/300 SCOPF results.
No sampling tables are updated. The live queue has 18 running computation jobs,
170 waiting jobs, and no held jobs (excluding two workflow managers).
The supplementary workflow still records its removed original SCOPF node as failed;
the independent replacement run has all 500 results. That workflow flag is distinct
from a missing result in the regenerated tables. Existing plot inputs are already
marked current from the 07:43 export; this refresh changes generated tables only.

Validation: the SND objective and normalization unit tests pass (two tests total).
Source-table checks reject changed source hashes and invalid retained subgraphs.
Both documents are rebuilt with separate output directories under
`/tmp/pig-tables-20261107/`; the updated table pages are rendered and inspected.
There are no undefined citations or references or LaTeX errors. The main document
retains its existing two overfull boxes at the InnOpTEM note and the existing
4.77-point oversized float; the supplement has no overfull boxes or oversized floats.
No Git index changes, commits, pushes, or queue changes are made in this refresh.

2026-11-08 12:35: Table 2 is refreshed from 44/48 returned source results. All input hashes, graph sizes, and retained planar subgraphs pass the exporter checks. Four original PGLib cells (case89_pegase, case1354_pegase, case1888_rte, case1951_rte) retain the previous values. The SNDlib table is unchanged. Sampling has 247/300 final-skewness results (100 equidistant, 96 adaptive skip-one, 51 prefer-non-leaf), and 290/300 SCOPF results (100, 100, and 90 individual result summaries). Tables S3-S4 remain unchanged until the comparison is complete. Main experiment plots and the completed application tables already reflect the available results. There are no held jobs. Manuscript prose and the Git index are unchanged. Validation details are in `R/paper_tables_20261108/`.
The 34-page main document builds without LaTeX errors or undefined references/citations. Page 20 is rendered and inspected. The two existing overfull boxes and the existing oversized float remain unchanged. The rebuilt PDF is copied to the paper checkout.

2026-11-09 12:54: All 48 source results, 300 sampling final-skewness results, and 300 sampling SCOPF results are available. The PIG queue is empty. Table 2 now includes the final four original PGLib results. The source exporter verifies hashes, graph sizes, and retained planar subgraphs. Tables S3-S4 reproduce the current manuscript inputs exactly, with 69 matched input sequences. All 400 sampling application input hashes (including the consecutive baseline) match their result records. The three alternative sampling policies each have 100 completed outcomes: feasible/infeasible counts are 60/40, 57/43, and 61/39. The consecutive baseline has 62/38. Diagnostics are in `R/paper_tables_20261109/`. No manuscript prose or Git index entries are changed.
Both documents build without LaTeX errors or undefined citations/references. The source table (now Table 3, main page 19) and sampling tables (supplement page 5) are rendered and inspected. The main document retains two existing 13.5395pt overfull boxes and the existing 4.76674pt oversized float; the supplement has no such warnings. Both rebuilt PDFs are copied to the paper checkout. Table numbers in the earlier inventory precede the latest manuscript reordering.
