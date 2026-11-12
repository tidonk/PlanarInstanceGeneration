# Data sources and attribution

The files under `pglib/` and `snd/` are derived benchmark instances generated
by Tim Donkiewicz for *Composition of Near-Planar Infrastructure Network
Instances with Certified Skewness Bounds*. They are modified compositions of
the sources identified in [catalogue.csv](catalogue.csv), rather than original
PGLib or SNDlib networks. The original filenames, SHA-256 hashes, and source
URLs are listed in [sources.csv](sources.csv).

The generation procedure extracts connected subnetworks, selects coupling
vertices, and connects the subnetworks. It also transforms the application
data, including power injections or communication demands, as described by
the generation code and paper. Source authors do not endorse these derived
instances. The repository's software license does not replace the terms of
the underlying data.

## PGLib-OPF

Source: [IEEE PES Power Grid Library — Optimal Power Flow](https://github.com/power-grid-lib/pglib-opf),
release v23.07, revision `dc6be4b2f85ca0e776952ec22cbd4c22396ea5a3`.

The PGLib-OPF data are distributed under the
[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).
The derived PGLib instances are distributed under the same license.
[LICENSES/PGLib-OPF.txt](LICENSES/PGLib-OPF.txt) reproduces the upstream license
notice, and [LICENSES/CC-BY-4.0.txt](LICENSES/CC-BY-4.0.txt) contains the license
text. Copyright holders, original references, and case-specific notices are
preserved in [LICENSES/PGLib-OPF-sources.txt](LICENSES/PGLib-OPF-sources.txt).
That file contains the original attribution headers from every PGLib case
used in the published composites. Retain these notices and the source mapping when
redistributing the dataset.

## SNDlib

Source: [Survivable Network Design Library](https://sndlib.zib.de/).

Reference: S. Orlowski, R. Wessäly, M. Pióro, and A. Tomaszewski,
“SNDlib 1.0—Survivable Network Design Library,” *Networks* 55(3),
276–286 (2010), [doi:10.1002/net.20371](https://doi.org/10.1002/net.20371).

[LICENSES/SNDlib-sources.txt](LICENSES/SNDlib-sources.txt) preserves the XML
metadata and comments present in each original source, including origin and
contributor credits where supplied. SNDlib provides downloadable benchmark
data but the recorded source files do not supply a common license text.
This repository therefore does not assign GPL or CC BY to the underlying
SNDlib data. The original providers retain their rights and applicable terms.
