# Planarization and experiment settings

The implementation uses OGDF revision
`5b6795655399b9d8e2921afec9d97bab9107d5ee` (foxglove-202510).
OF-H and OF-B use the same face selection and coupling rules, with heuristic
and exact planarization, respectively. Their graphs can differ after a join.

| Component | Configuration | Modification to OGDF |
|---|---|---|
| Heuristic | `PlanarSubgraphFast<double>`, 10 runs | Constructor default; explicit seed and thread cap |
| Exact solver | Maximum planar subgraph, ABACUS and bundled CLP | Access to the internal master for external solutions, constraints, and bounds |
| Exact initialization | Simple maximal planar subgraph and Fast with 25 runs | Native procedure retained |
| Pricing | Disabled | Public module default |
| Decomposition | Separate maximal biconnected subgraphs | Public module decomposition retained |
| Search | Serial branch and cut, 0% requested gap | Native defaults |
| Resources | 8 physical cores, 16 logical CPUs per planarization job | Heuristics capped at 16 threads; Fast10 uses at most 10 |
| Generation | 600 seconds per exact graph solve | Budget shared across the biconnected subgraphs |
| Final evaluation | 3,600 seconds per graph | Same solver for all five construction methods |

The following additions are used. They are not claims of novelty.

**Static Euler inequalities.** For each nonplanar maximal biconnected subgraph,
repeatedly remove a vertex of minimum degree. At every remaining vertex set
`S` with at least three vertices, compute `|E(S)| - 3|S| + 6`. If the induced
graph has no triangle, also compute `|E(S)| - 2|S| + 4`. Triangle counts are
maintained throughout the removal sequence. Keep the largest positive value
for each expression, using the earliest visited set on a tie. For retained-edge
variables `y`, add the corresponding inequalities
`sum(y[e], e in E(S)) <= 3|S| - 6` or `<= 2|S| - 4` before solving.
There are at most two distinct rows per biconnected subgraph. OF-H uses the
same calculation to obtain lower bounds on the number of edges that must be
deleted; it has no linear relaxation to strengthen. The search is deterministic
and examines one nested sequence of vertex sets. It adds no separation rounds
during branch and cut. OGDF's own separation remains unchanged.

These Euler inequalities are described by Chimani and Wiedera,
[ESA 2018, Section 3](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESA.2018.19).
The static minimum-degree search here is a different procedure, so their
experiments do not establish its performance benefit. The implementation is in
[`core/ogdf.py`](../planar_instance_generation/core/ogdf.py).

**Reuse between joins.** A constituent or earlier intermediate graph with a
known lower bound `L` on its skewness supplies
`sum(y[e], e in E(Q)) <= |E(Q)| - L`. Bounds on disjoint edge sets can be added;
overlapping bounds are retained separately. Individual bounds are passed to
the relevant biconnected solve. A global bound may also be retained outside
the solver. A lower bound matching a verified feasible solution proves
optimality without further search.

For outer face joins, the selected planar subgraphs together with all new
coupling edges give a planar solution. At the next stage, compare this solution
with a fresh default Fast10 result, preferring Fast10 on a tie. Pass the better
solution to the exact method as well. The native exact initialization still
runs, and the supplied solution is accepted when it improves the incumbent.
For final evaluation of R, HD, and KK, use a forest of coupling edges to join
the constituent planar subgraphs, then compare with Fast10. There are no
additional greedy extension passes or searches through older solutions.

**Limits and result handling.** The native interface enforces CPU and wall
limits shared across biconnected solves. Native initialization is included;
Python preprocessing and the external Fast10 call are recorded separately.
Thus 600 and 3,600 seconds are not inclusive limits on the whole Python call.
A complete planar solution is retained when some subgraphs remain unsolved.
Returned solutions are checked for planarity, connectivity, and edge coverage.
A retained-edge dual bound `D` on `m` edges gives the integer lower bound
`ceil(m - D - 1e-4)` on the number of deletions. A contradictory bound is flagged
and discarded, preserving other valid bounds. Raw solver output is retained.

The native source patch caps Fast's threads inside exact initialization. Other
integration changes are in [`native/ogdf/exact.cpp`](../native/ogdf/exact.cpp).
Seeds, input order, source and executable hashes, and solver settings are saved.
Parallel heuristic scheduling can change tied solutions despite a fixed seed.

## Application models

Generation uses InnOpTEM base-case DC optimal power flow for PGLib, with no
switching, contingencies, or feasibility slack, and a fixed-capacity
multicommodity-flow feasibility test for SNDlib. The PGLib limit is 300 seconds;
both tests use one thread. OptiMods is not used.

Final PGLib applications solve security-constrained optimal transmission
switching (SCOTS), allowing one switching action. The limit is 10,800 seconds,
plus a separate 10,800-second security-constrained optimal power flow (SCOPF)
warmstart. Final SND applications use BendersNetworkDesign with a 10,800-second
limit. Application solves use one thread. Their feasibility results do not
affect generation acceptance.

InnOpTEM is pinned to `f502511dd59ad30608f1e3a12a34324d44aab7ee`, with Julia
1.12.6, PowerModels 0.21.6, and Gurobi 13.0.2. The Python SND check uses Gurobi
13.0.3; the pinned Julia SND application uses Gurobi 12.0.3. See
[THIRD_PARTY.md](../THIRD_PARTY.md) for the source snapshots and licenses.

## Source submodules

The public source checkout pins OGDF, TopologyOptimizer.jl, and
BendersNetworkDesign.jl as submodules. The original experiment revisions above
remain the historical record. See [the dependency notes](../reproduction/DEPENDENCIES.md)
for the corresponding public revisions, retained PIG defaults, and the OGDF
build modification.
