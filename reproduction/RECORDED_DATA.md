# Reproducing the recorded plots

The source repository excludes the paper results. The separate archive
`planar-evaluation-20261110.zip` contains the original inputs, generated graphs,
recorded solver outcomes, and the manuscript build inputs from 10 November 2026.
Its SHA-256 is:

```text
88fc1741f03b6482936da699a3aee1f768c2fd969408da8149da62354781e753
```

The archive is 407,715,543 bytes. No public archive download URL is assigned in
this source snapshot. Obtain the archive separately before following these
steps. Cloning the repository alone does not provide the recorded results.

Check the downloaded archive and extract it to a separate directory:

```bash
sha256sum /path/to/planar-evaluation-20261110.zip
unzip /path/to/planar-evaluation-20261110.zip -d /path/to/evaluation
python3.11 /path/to/evaluation/planar-evaluation-20261110/code/reproduction/reproduce.py verify
```

After installing the Python dependencies from the top-level README, use a fresh
checkout without existing `check/data` or `check/results` paths. Link the
recorded data into the ignored directories, then run this checkout's exporters:

```bash
ln -s /path/to/evaluation/planar-evaluation-20261110/code/check/data check/data
ln -s /path/to/evaluation/planar-evaluation-20261110/code/check/results check/results
python check/scripts/fetch_inputs.py --verify-only
python check/scripts/export_paper.py --config reproduction/paper.json \
  --output /path/to/new-paper-export
```

The output directory must be empty. The exporter writes numerical plot inputs
and tables under `resources/evaluation/`. Graph drawings require LaTeX. This
route uses recorded outcomes and does not require a Gurobi license or a new
optimization run. The benchmark files retain their original terms.

For the exact recorded manuscript figures, build the manuscript supplied in the
archive using the commands in [PACKAGE_README.md](PACKAGE_README.md). The source
repository does not contain that manuscript. Typography and PDF metadata can
vary between TeX installations.

For independent experiment runs, use [README.md](README.md), download the
benchmark inputs, and choose fresh result names. Independent runs need not
repeat time-limited search trajectories or measured runtimes.
