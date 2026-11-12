# Submission checks, 5 November 2026

The short example ran on the innoptem server with seeds 1, 3, 4, 13, and 15,
10-second exact solves during generation, 20-second final solves, four evaluation
processes, and one solver thread each. Installation and input download preceded
the timed run. Total elapsed time was 782.9 seconds, including source screening
and Julia initialization.

| Method | Completed inputs | Exact skewness | Zero added skewness certified | Provably better than all R/HD/KK |
|---|---:|---:|---:|---:|
| R | 5 | 1 | 1 | 0 |
| HD | 5 | 0 | 0 | 0 |
| KK | 5 | 0 | 0 | 0 |
| OF-H | 5 | 3 | 3 | 2 |
| OF-B | 5 | 4 | 3 | 3 |

These are illustrative short-limit results, not estimates for the full study.
No strict OF-B improvement over OF-H was proved on these five inputs.

The 66 PGLib inputs were downloaded from the recorded revision and all hashes
matched. All 92 locally available PGLib/SNDlib inputs matched the inventory.
Automatic SNDlib download failed at the owner's redirected HTTPS service.

The 69 included InnOpTEM files and 22 included BendersNetworkDesign files match
their recorded Git revisions byte for byte. The additional revision records
identify the selected files. Unrelated experiment settings are omitted. The bundled InnOpTEM source also
loaded successfully under Julia 1.12.6 from a fresh prepared directory. The
bundled BendersNetworkDesign source loaded under the same Julia version.

Generation, application, and supplementary exporters were exercised on existing
complete results. Graph examples were exported from a newly generated input.
The resulting PDFs were compiled and representative plots inspected. Source-table
checks cover input identity and reject mismatching files. A short native SND
source run covered all 16 retained sources and returned consistent bounds.
The documentation build completed with warnings treated as errors.

The updated test suite passed 307 tests. Obsolete study-specific tests were
removed together with their scripts; publication and input-identity checks were
added. These checks validate the packaging and export paths. They do not
establish completion of the ongoing full OGDF experiment.
