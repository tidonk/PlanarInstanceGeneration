# Pinned source dependencies

Initialize the dependencies with `git submodule update --init --recursive`.
The Git entries and [dependencies.json](dependencies.json) identify the exact
commits. Updating a submodule is a change to the experiment source and should
be validated before recording a new pin. Do not use `git submodule update --remote`
when reproducing this revision.

| Dependency | Commit | Role |
|---|---|---|
| TopologyOptimizer.jl | `cadf8b63fe0a6b44c99646d9988e3df63b24c5f7` | Power-flow feasibility and SCOTS |
| BendersNetworkDesign.jl | `dbd9b4253a2a65eab1b1defa1e5029d4caf191d6` | SND application solves |
| OGDF | `5b6795655399b9d8e2921afec9d97bab9107d5ee` | Heuristic and exact planarization |

## TopologyOptimizer and evaluation settings

The recorded experiment uses InnOpTEM revision
`f502511dd59ad30608f1e3a12a34324d44aab7ee`. The pinned TopologyOptimizer source
matches that solver implementation after the package-name change, the
addition of attribution comments, and an updated precompilation input path.
Its Julia dependency versions are unchanged.
The dependency-file project hash changes with the renamed project.

TopologyOptimizer's public defaults enable the integrated paper's Branch Fixing
configuration. The PIG application records use a different configuration.
All 500 saved main PIG settings inherit branch fixing and angle-bound settings
from the recorded runtime defaults:

| Setting | PIG evaluation | TopologyOptimizer default |
|---|---|---|
| `BENDERS.FIX_FORCED_ON_BRANCHES` | `false` | `true` |
| `BIG_M_MULT.SET_MAX_ANGLE_CLOSE` | `false` | `true` |
| `BIG_M_MULT.MAX_ANG_OFF` | `30`, inactive | `0.5235987755982988`, radians |

The PIG overrides already enable partial decomposition and the recorded
subproblem and cut selection settings. The full original default file is kept
in [settings/topology-default.toml](settings/topology-default.toml).
Complete overrides for the main SCOTS and SCOPF flow evaluations are available
in [settings/scots.toml](settings/scots.toml) and
[settings/scopf.toml](settings/scopf.toml), with
[usage instructions](settings/README.md).
`prepare_topology_optimizer.py` exports the pinned source into
`build/TopologyOptimizer` and replaces only that build copy's default settings.
It includes the input required by the solver's precompilation workload.
The submodule remains unchanged. The application commands then apply the same
per-run overrides as the recorded evaluation. `prepare_innoptem.py` remains an
alias for the renamed setup command.

The original experiment code, settings, and results remain in the separate
November 2026 evaluation archive. The public package-name change does not imply
that the complete optimization campaign has been rerun with the renamed package.

## BendersNetworkDesign

The pinned public revision has the same solver source, project dependencies,
and default settings as the recorded revision
`45135498c12912c0563791fabf88b25ab8617e62`. Its Git history and citation metadata
are separate from the original development repository.

## OGDF

The submodule contains the unmodified pinned upstream revision. The build
script exports it into `build/ogdf/source`. It creates a separate
`MaxCPlanarMaster.cpp` with the recorded thread cap for native initialization.
The static Euler bounds, reuse of constituent solutions, and completion after
time limits are implemented in `native/ogdf/exact.cpp` in this repository.
`native/ogdf/heuristic.cpp` supplies the heuristic interface. No OGDF fork or
uncommitted submodule modification is required.

## Complete archives

`package_evaluation.py` includes the initialized submodules' source files and
adds revision records to the resulting archive. This makes the complete archive
independent of Git metadata. GitHub's automatic source ZIP has a different
scope and does not include submodule contents.
