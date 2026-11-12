Development
===========

Tests
-----

The package uses the standard-library test runner and has no test-framework
dependency:

.. code-block:: console

   python -m unittest discover -s tests -v

Tests must be deterministic, avoid external writes, and use temporary
directories for artifacts. Optional solver integrations are isolated so the
base suite can run without Gurobi.

Documentation
-------------

Build the HTML documentation with warnings treated as errors:

.. code-block:: console

   python -m pip install -e '.[docs]'
   python -m sphinx -W --keep-going -b html docs docs/_build/html

Open ``docs/_build/html/index.html`` after a successful build. Generated HTML
is ignored by Git and published as a GitLab job artifact.

Package layout
--------------

``planar_instance_generation.io``
   Strict benchmark readers, writers, and artifact naming.

``planar_instance_generation.core``
   Extraction, composition, selection, planarization, and serialization.

``planar_instance_generation.gates``
   InnOpTEM DC-OPF and Gurobi multicommodity-flow feasibility checks.

``planar_instance_generation.evaluation``
   Source pools, experiment orchestration, metrics, and application runners.

``planar_instance_generation.plt``
   Kamada--Kawai TikZ plots, boxplots, and A4 reports.
