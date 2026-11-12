Command-line workflows
======================

The scripts are run from the repository root. Use ``--help`` for the complete
set of solver, plotting, resource, and reproducibility controls.

PGLib / DC power flow
---------------------

.. code-block:: console

   python check/scripts/preprocess_pglib.py --time-limit 300
   python check/scripts/generate_pglib.py \
     --result-name pglib_generation \
     --count 100 \
     --num-subinstances 2 5 \
     --min-source-nodes 73 \
     --target-size 100 1000 \
     --size-range 0.15 0.85 \
     --outer-face-sampling consecutive --opf-time-limit 300
   python check/scripts/evaluate_pglib.py pglib_generation

The gate uses strict base-case DC-OPF from InnOpTEM with branch switching
disabled. The separate SCOPF/SCOTS application runner is
``check/scripts/run_scopf_benders.py``. The prepared paper settings select
SCOTS with one permitted switching action and native SCOPF warmstarting.

SNDlib / multicommodity flow
----------------------------

.. code-block:: console

   python check/scripts/preprocess_sndlib.py
   python check/scripts/generate_sndlib.py \
     --result-name snd_generation \
     --count 100 \
     --num-subinstances 2 5 \
     --size-range 0.15 0.85 \
     --outer-face-sampling consecutive
   python check/scripts/evaluate_sndlib.py snd_generation

Without an explicit ``--target-size``, SNDlib samples targets between the
smallest loaded source instance and 100 nodes, inclusive.

``--outer-face-sampling`` controls both outer face methods.
``consecutive`` is the default. ``equidistant`` spreads endpoints around the
eligible cyclic boundary; ``consecutive`` selects a random cyclic arc;
``adaptive_skip_one`` selects a random arc with one-node gaps distributed as
evenly as possible, falling back toward consecutive selection when the face
does not contain enough unused boundary nodes. It always returns the requested
number of distinct endpoints.

``--outer-face-prefer-non-leaf`` is orthogonal to the spacing policy. It first
selects from outer-face vertices whose degree in the unplanarized input graph
is not one. Degree-one vertices remain eligible only when the requested
coupling count exceeds the number of non-leaf boundary candidates.

Preprocessing constructs fixed capacities and checks them with continuous
multicommodity flow. The final survivable-network-design application runner is
``check/scripts/run_snd_benders.py``.

Artifacts
---------

Every named generation run uses a consistent collision-safe stem across:

``instances/``
   Accepted final composites only. For the configured PGLib run, every file
   is validated to contain 100--1000 buses.

``subinstances/``
   Extracted randomized-BFS pieces. These may be smaller than the final target
   range; their exact allocated sizes are recorded in metadata and tables.

``plots/``
   TikZ sources, standalone PDFs, source-color legends, and one PDF per
   pairwise merge stage.

``logs/``
   OGDF, application solver, and LaTeX logs.

``tables/`` and ``evaluation/``
   Attempts, runtimes, structural properties, skewness certificates,
   preservation audits, and paired-strategy comparisons.
