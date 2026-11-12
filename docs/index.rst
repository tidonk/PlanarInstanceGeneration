Planar Instance Generation
==========================

This package constructs reproducible near-planar power and communication
network instances from PGLib and SNDlib sources. It provides source screening,
flow-preserving extraction, sequential composition, merge-node strategies,
DC-only feasibility gates, publication-ready TikZ, and evaluation artifacts.

Python runs the construction and evaluation workflow. C++ executables use OGDF
for planarization. Julia runs the InnOpTEM power models and the final
application evaluations.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   getting-started
   pipeline
   cli
   development

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index

Scope
-----

All generated instances, plots, tables, and logs are written below
``check/results``. Raw inputs below ``check/data`` are treated as immutable.
Power-system optimization is DC-only; the package does not run AC models.
