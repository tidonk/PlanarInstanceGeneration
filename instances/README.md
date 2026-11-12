# Published instances

These are the exact 1,000 final composites used in the main evaluation of
*Composition of Near-Planar Infrastructure Network Instances with Certified
Skewness Bounds* by Tim Donkiewicz. The instance files are unchanged copies
from the completed experiment.

| Dataset | Directory | Format | Instances |
| --- | --- | --- | --- |
| PGLib | [pglib/](pglib/) | MATPOWER v2 `.m` | 500 |
| SNDlib | [snd/](snd/) | SNDlib 1.0 `.xml` | 500 |

Each dataset has 100 input sets and five methods per set: random (R), high
degree (HD), Kamada–Kawai (KK), heuristic outer face (OF-H), and exact outer
face (OF-B). All five methods use the same extracted constituents for a given
input set. The instance files total 203,197,203 bytes (about 194 MiB).

[catalogue.csv](catalogue.csv) identifies the dataset, file, input index,
method, generation seed, ordered source networks and extracted constituents,
node and branch/link counts, and SHA-256 hash. Source names can repeat within
an input set. The generation seed is the recorded integer seed for the
accepted generation attempt. Application experiments instead use the input
index, 1 through 100, as their solver seed. Keep generation seeds as integers
or text when importing the CSV to avoid rounding.

Filenames preserve the experiment names: `<nodes>_i<input>` for R, with
suffixes `_2`, `_3`, `_4`, and `_5` for HD, KK, OF-H, and OF-B. The catalogue
is the authoritative mapping. PGLib filenames preserve their original numeric
prefix even though the MATLAB function name begins with `pglib_generated_`.
Use the repository's `read_pglib` parser or PowerModels' MATPOWER parser.

## Verification and source attribution

From the repository root:

```bash
sha256sum -c instances/SHA256SUMS
```

The checksums cover all 1,000 instance files and the PGLib coupling-branch
indices needed by the application runner. [sources.csv](sources.csv) records
the original source filenames, hashes, and URLs. See
[DATA_NOTICES.md](DATA_NOTICES.md) for attribution and data terms.

## Using the application runners

The existing runners work in `check/results/`. After verification, copy the
instances into fresh directories there and restore the PGLib coupling-branch
metadata used to prepare application inputs:

```bash
python - <<'PY'
import json
from pathlib import Path
import shutil

for dataset, extension in (("pglib", "m"), ("snd", "xml")):
    target = Path("check/results") / f"paper_{dataset}"
    target.mkdir(parents=True, exist_ok=False)
    (target / "instances").mkdir()
    for source in sorted((Path("instances") / dataset).glob(f"*.{extension}")):
        shutil.copyfile(source, target / "instances" / source.name)

coupling = json.loads(Path("instances/pglib/coupling-branches.json").read_text())
for stem, rows in coupling.items():
    target = Path("check/results/paper_pglib/plots/composite") / stem
    target.mkdir(parents=True)
    metadata = {"artifact_stem": stem, "coupling_branch_rows_zero_based": rows}
    (target / "metadata.json").write_text(json.dumps(metadata) + "\n")
PY
```

For example, after installing the dependencies as described in the top-level
README, prepare the first PGLib SCOTS and SND application runs:

```bash
python check/scripts/run_scopf_benders.py \
  check/results/paper_pglib/instances/237_i001.m \
  --run paper_pglib_scots --seed 1 --topological-action-limit 1 \
  --scopf-warmstart --benders-settings check/eval/scots_benders.toml --dry-run
python check/scripts/run_snd_benders.py \
  check/results/paper_snd/instances/29_i001.xml \
  --run paper_snd_application --seed 1 --ignore-manifest-julia --dry-run
```

Remove `--dry-run` to solve the instances. A Gurobi license is required for
optimization. The PGLib runner uses the recorded coupling-branch indices to
prepare finite line ratings in its separate `innoptem_input.m` file. The
published `.m` files retain the original generated ratings. The complete
PGLib application settings are documented in
[reproduction/settings/](../reproduction/settings/README.md).

These files provide the main benchmark inputs. Recorded solver outcomes,
generation checkpoints, constituent solutions, and supplementary experiments
remain separate. See the [reproduction guide](../reproduction/README.md) for
the full workflow and [recorded-data instructions](../reproduction/RECORDED_DATA.md)
for recreating plots from saved observations. Independent optimization runs
can return different bounds, heuristic solutions, and runtimes.
