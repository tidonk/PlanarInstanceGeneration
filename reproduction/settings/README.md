# PGLib application settings

These files reproduce the PGLib application settings from *Composition of
Near-Planar Infrastructure Network Instances with Certified Skewness Bounds*.
Use the TopologyOptimizer.jl revision pinned in
[dependencies.json](../dependencies.json), with its Julia 1.12.6 and
Gurobi 13.0.2 dependencies. Follow the repository installation instructions
and run `python check/scripts/prepare_topology_optimizer.py --instantiate`
to prepare the runtime.

| File | Application | Switching limit | Initialization |
| --- | --- | --- | --- |
| [scots.toml](scots.toml) | Main SCOTS evaluation | 1 | SCOPF, up to 10,800 seconds |
| [scopf.toml](scopf.toml) | SCOPF flow evaluation | 0 | Disabled |

Both configurations use one thread and a 10,800-second main solve limit. SCOTS
initialization has a separate time budget. Set `TECHNICAL.SEED` to the input
index, 1 through 100, using the same seed for all five generation methods.
The files use seed 1 as an example.

The files override the solver defaults through `params_file`. They explicitly
disable branch fixing and the additional angle-bound setting enabled by the
public defaults. `BIG_M_MULT.MAX_ANG_OFF = 30` preserves the recorded value
and is inactive because `SET_MAX_ANGLE_CLOSE = false`. They also reproduce
the same effective settings when used with the prepared runtime and its
recorded defaults in [topology-default.toml](topology-default.toml).

From the PlanarInstanceGeneration repository root, start Julia with:

```sh
julia +1.12.6 --startup-file=no --project=build/TopologyOptimizer --threads=1
```

Use a MATPOWER input prepared by the application runner. The recorded runs
store that input as `innoptem_input.m`, including the original preparation
of coupling branch ratings.

```julia
using TopologyOptimizer

result = TopologyOptimizer.main(
    "/path/to/innoptem_input.m";
    application = "DC_SCOTS_TIME_SLACK_BENDERS",
    params_file = "reproduction/settings/scots.toml",
    additional_params = Dict(:TECHNICAL => Dict(:SEED => 31)),
    slack_types = [:ratea],
    busbars = false,
    timesteps = 1,
    plot = false,
)
```

For the flow evaluation, use `scopf.toml` with the same input and seed.
These files cover the two application evaluations above. Source screening,
generation feasibility checks, and sensitivity experiments have separate
settings. The existing experiment commands continue to create their settings
from the recorded defaults and run-specific parameters.

The configurations were checked against all 500 recorded SCOTS settings and
all 500 recorded SCOPF settings in `planar-evaluation-20261110.zip`, whose
SHA-256 is
`88fc1741f03b6482936da699a3aee1f768c2fd969408da8149da62354781e753`.
The corresponding result directories are `ogdf_20261104_pglib_scopf` and
`ogdf_20261104_pglib_scopf_tal0`. Their effective parameters, including inherited
defaults from experiment revision `f502511dd59ad30608f1e3a12a34324d44aab7ee`,
match the supplied files after substituting each run's seed. This check compares
settings and does not rerun optimization.
