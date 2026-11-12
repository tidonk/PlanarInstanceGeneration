Generation pipeline
===================

The two domains share one experiment structure:

1. Read and validate raw benchmark instances.
2. Exclude simple source topologies that violate the Euler bound by default.
3. Solve the domain feasibility gate once and cache each retained source.
4. Draw sources with replacement and extract connected induced subinstances by
   randomized breadth-first search.
5. Compensate flow removed at the extraction boundary.
6. Join the pieces left-associatively, recomputing merge nodes at every stage.
7. Accept only feasible final instances that are attribute-distinct from earlier
   accepted outputs of the same method. Cross-method equality is allowed.
8. Write instances, statistics, plots, solver logs, and settings below
   ``check/results/<result-name>``.

Coupling count
--------------

For the join of the current composite :math:`C_{r-1}` and the next extracted
graph :math:`G_r`, let

.. math::

   n_{\min,r} = \min\left\{|V(C_{r-1})|, |V(G_r)|\right\}.

The implementation requests

.. math::

   q_r = \min\left\{
       n_{\min,r},
       \max\left\{3, \left\lceil\log_2 n_{\min,r}\right\rceil\right\}
   \right\}.

Every strategy must select exactly :math:`q_r` endpoint-disjoint pairs.
Candidate-set sizes, including the eligible distinct nodes on the smaller
outer face, are feasibility conditions only. They never reduce :math:`q_r`.
If any strategy cannot supply the request, the complete paired attempt is
rejected and redrawn for every strategy. Stage statistics record both
``requested_pair_count`` and ``pair_count``, and candidate metadata records
the per-stage request tuple.

Selection strategies
--------------------

``random``
   Uniformly sample :math:`q_r` distinct nodes without replacement from each
   graph, then pair the two sample orders. Every node is eligible. At stages
   after the first, the left graph is the complete growing composite.

``high_degree``
   Rank nodes by degree and pair equal ranks. Recorded seeds resolve degree
   ties without favoring graph insertion order.

``kamada_kawai``
   Reembed both graphs separately, place them side by side at equal scale, and
   greedily choose short cross-graph pairs without endpoint reuse. Its two
   candidate sets are the convex-hull vertices of the separate embeddings.

``heuristic_outer_face``
   Use OGDF's ``PlanarSubgraphFast`` with its default ten runs, then select
   vertices on the resulting outer faces.

``exact_outer_face``
   Use OGDF's native exact maximum planar subgraph algorithm, with ABACUS and
   CLP. Nonplanar biconnected blocks are solved separately. Planar blocks,
   including bridges, are retained. Native initialization uses
   ``MaximalPlanarSubgraphSimple`` and 25 runs of ``PlanarSubgraphFast``.
   Native separation, branching, and heuristic controls retain their defaults.

Both methods use the additions below. OF-H denotes heuristic outer face
selection and OF-B denotes exact outer face selection.

Static Euler inequalities
-------------------------

A simple planar graph with :math:`n\geq3` vertices has at most
:math:`3n-6` edges. If it contains no triangle, the bound is :math:`2n-4`.
Consequently, a subgraph with :math:`m` edges requires at least
:math:`m-3n+6` deletions, or :math:`m-2n+4` when it is triangle-free.

For each nonplanar block, start with all its vertices and repeatedly remove a
vertex of minimum current degree, stopping at three vertices. Ties follow the
stored vertex order. At every step, evaluate the ordinary bound and, if no
triangles remain, the triangle-free bound. Keep the subset giving the largest
positive bound for each inequality family. The first subset is retained on a
tie. This produces at most two distinct inequalities per block.

The code maintains the triangle count as vertices are removed. It examines
one nested sequence of vertex subsets, so it can miss useful inequalities on
other subsets. It uses the generalized Euler inequality family described by
`Chimani and Wiedera (ESA 2018, Section 3)
<https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESA.2018.19>`_.
The subset search is our preprocessing procedure, not their LP separation.

OF-B inserts these inequalities before branch-and-cut. OF-H uses their lower
bounds for certification. There are no additional custom separation rounds
during search.

Reuse across sequential joins
-----------------------------

At an outer face join, retain the two selected planar subgraphs and all new
coupling edges. Their union is planar by construction. At the next
planarization, compare this retained solution with a fresh default Fast10
solution, keeping the one with more edges and choosing the fresh result on
ties. OF-B supplies that candidate to the native exact solver. Its own
initialization may improve the candidate further. There are no additional
custom greedy passes or searches through earlier solutions.

A valid deletion lower bound :math:`L_Q` on an unchanged edge subgraph
:math:`Q` gives the constraint

.. math::

   \sum_{e\in E(Q)}x_e\geq L_Q,

where :math:`x_e=1` means that edge :math:`e` is deleted. Both methods retain
these bounds. OF-B inserts constraints whose supports lie within one solve
block. Bounds spanning several blocks remain available outside the solver.
Only bounds on disjoint edge sets are added together. When a valid lower bound
equals the deletion count of a verified planar solution, optimality is proved
without an exact solve.

Final evaluation uses the same constituent information for all five methods.
It solves extracted constituents once and shares their results across the
five final graphs. Generation bounds on unchanged supports are also available
to every method. The retained constituent solutions are joined by all coupling
edges if the union is planar, and otherwise by a connecting forest. This
candidate is compared with fresh Fast10 using the same rule as above.

A connecting forest retains :math:`p-1` coupling edges for :math:`p` connected
constituents in a connected composite. This gives the constructive upper bound
:math:`\sum_i d_i+|M|-(p-1)`, where :math:`d_i` is the deletion count of
constituent solution :math:`i`. The reporting code also retains this bound.

Resources, bounds, and saved results
----------------------------------------

The paper settings cap both standalone and native initialization heuristics at
16 threads. Fast10 uses at most ten workers. Branch-and-cut and static Euler
preprocessing are serial. Native exact budgets are 600 seconds during generation
and 3,600 seconds during final evaluation, shared across blocks. Both native
processor time and elapsed time are limited. Preprocessing and external Fast10
are outside the native timer and included in total runtime.

The supplied complete planar solution covers blocks left unsolved at timeout.
Primal bounds are deletion counts in verified planar solutions. For a valid
retained-edge dual bound :math:`D` on a block with :math:`m` edges, the deletion
lower bound is :math:`\max\{0,\lceil m-D-10^{-4}\rceil\}`. Inconsistent
bounds are flagged and discarded. Solver status alone does not prove optimality.

Results record graph order, seeds, retained edges, bound supports, and backend
versions. Saved solves with incompatible settings are rejected. Parallel
heuristics can choose different tied solutions despite equal seeds.
``greedy_start_upper_bound`` is the retained field name for the fresh OGDF
heuristic's deletion count. ``piece_start_upper_bound`` records the reused
solution's deletion count.

Voltage-separated certificates
------------------------------

``check/scripts/evaluate_pglib_source_skewness.py --voltage-separated`` solves
the graph obtained by omitting lines between different voltage levels.
For the original graph, ``--voltage-warm-start`` optionally reuses a matching
saved bound and planar solution from that subgraph. Source hashes, the edge
rule, and backend settings are checked before reuse. This option is not part
of the main generation workflow. It does not alter the generated power data
or transfer a full source bound to a smaller extracted piece.

Attempt and extraction diagnostics
----------------------------------

Attempt CSVs retain ``status`` and free-text ``reason`` and additionally record
``individual_status``, ``retry_trigger_strategy`` and ``retry_trigger_status``.
A method that passed before another method failed has individual status
``accepted`` but overall status ``paired_retry``; a method skipped after a
failure has individual status ``not_run``. Shared extraction or size failures
are recorded as such without attributing them to a merge method.

Piece statistics record the compensated-node fraction and total absolute nodal
compensation. PGLib additionally records compensation-generator capacity and
the original positive demand on retained buses, before compensation. Their
ratio is undefined (``None``) when that original demand is zero. SND compensation
uses the source routing unit and does not include later synthetic demands.

Euler density screen
--------------------

The default source filter excludes a simple topology when
:math:`|V|\geq3` and :math:`|E|>3|V|-6`. This is a sufficient certificate of
nonplanarity, not a planarity test. Both preprocessors expose
``--no-exclude-euler-bound-violations`` for explicit comparison runs and record
the true edge count, bound, and exclusion status in CSV and JSON.
