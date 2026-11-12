Getting started
===============

Requirements
------------

Python 3.11 or newer is required. Install the package from the repository root:

.. code-block:: console

   git submodule update --init --recursive
   python -m pip install -e .

Install the application dependencies and build the solvers:

.. code-block:: console

   python -m pip install -e '.[evaluation]'
   python check/scripts/build_ogdf.py --jobs 16
   python check/scripts/prepare_topology_optimizer.py --instantiate

The OGDF build requires Git, CMake, and a C++17 compiler. It uses ABACUS and
CLP for exact planarization. The preparation script exports the pinned
InnOpTEM source and installs its Julia 1.12.6 dependencies. Gurobi and a
working license are required by the application models, including the
feasibility checks, but not by planarization.

Use ``python -m pip install -e '.[docs]'`` to build this documentation.
The Python ``opf`` and ``planarization`` extras contain no dependencies:
their native and Julia builds are prepared separately by the commands above.

First API call
--------------

The number of coupling edges is derived independently at every merge stage:

.. code-block:: python

   import networkx as nx

   from planar_instance_generation.core import coupling_pair_count

   left = nx.path_graph(20)
   right = nx.cycle_graph(12)
   count = coupling_pair_count(left, right)

``count`` is ``min(n_min, max(3, ceil(log2(n_min))))``, where ``n_min`` is the
smaller node count. Every strategy must realize exactly this many pairs.
Candidate-set and outer-face sizes are feasibility conditions; a shortage
rejects and redraws the complete paired strategy attempt.

Repository data
---------------

Place downloaded PGLib ``.m`` files under ``check/data/pglib`` and raw SNDlib
``.xml`` files under ``check/data/sndlib``. The structured runners refuse to
overwrite existing result runs; choose a new ``--result-name`` for every run.
