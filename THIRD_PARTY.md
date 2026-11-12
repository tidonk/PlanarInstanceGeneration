# Third-party sources and benchmark data

Original software in this repository is licensed under GPL-3.0-or-later.
Original documentation is licensed under CC BY 4.0. The source dependencies
retain their own licenses and are pinned as Git submodules:

| Component | Revision | License | Source |
|---|---|---|---|
| OGDF | `5b6795655399b9d8e2921afec9d97bab9107d5ee` | GPL v2/v3 with upstream exceptions | https://github.com/ogdf/ogdf |
| TopologyOptimizer.jl | `cadf8b63fe0a6b44c99646d9988e3df63b24c5f7` | Apache-2.0, with separate terms for incorporated PowerModels.jl code | https://github.com/tidonk/TopologyOptimizer.jl |
| BendersNetworkDesign.jl | `dbd9b4253a2a65eab1b1defa1e5029d4caf191d6` | CC-BY-4.0 | https://github.com/tidonk/BendersNetworkDesign.jl |

After initializing the submodules, consult their original licenses and
[TopologyOptimizer's notices](third_party/TopologyOptimizer/THIRD_PARTY_NOTICES.md),
including [the PowerModels.jl license](third_party/TopologyOptimizer/LICENSES/PowerModels.txt).
The [dependency notes](reproduction/DEPENDENCIES.md) explain the relation to
the recorded experiment revisions and settings. OGDF includes COIN libraries
with their upstream notices. Gurobi must be installed and licensed separately.

## Benchmark inputs

PGLib-OPF: https://github.com/power-grid-lib/pglib-opf, release `v23.07`, commit
`dc6be4b2f85ca0e776952ec22cbd4c22396ea5a3`. The data have their own CC BY license.

SNDlib: https://sndlib.zib.de/. These XML files were distributed as a downloadable
collection. The exact 26 filenames and SHA-256 hashes, together with the 66
PGLib filenames and hashes, are recorded in `reproduction/input-files.csv`.
Obtain the data from their owners and observe the respective terms. The download
script verifies every file and refuses mismatches.

The 1,000 final generated instances are distributed under `instances/`.
Their [data notices](instances/DATA_NOTICES.md) describe the source terms,
attribution, and modifications. The catalogue maps each composite to its
original sources. The software license does not replace these data terms.

## Separate evaluation ZIP

The optional ZIP created by `check/scripts/package_evaluation.py` includes the
initialized submodule sources, original benchmark files, generated instances,
and final numerical records. Data and dependency sources retain their original
terms. Original benchmark inputs and recorded solver results are excluded from
this Git repository. The ZIP includes the native executables used in the experiments,
but no Gurobi software, license files, or Julia package caches.
