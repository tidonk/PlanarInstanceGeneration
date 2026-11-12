// OGDF exact planarization with static bounds and a complete supplied solution.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <ogdf/basic/Graph.h>
#include <ogdf/basic/Logger.h>
#include <ogdf/basic/extended_graph_alg.h>
#include <ogdf/basic/simple_graph_alg.h>
#include <ogdf/cluster/ClusterGraph.h>
#include <ogdf/cluster/internal/MaxCPlanarMaster.h>
#include <ogdf/cluster/internal/MaxPlanarEdgesConstraint.h>
#include <ogdf/external/abacus.h>
#include <sstream>
#include <string>
#include <vector>
using namespace ogdf;
using Clock = std::chrono::steady_clock;
using MaxBase = cluster_planarity::MaxCPlanarMaster;

struct BoundRow {
  int lower;
  std::vector<int> edges;
};
class TimedMaster : public MaxBase {
  long wallLimit_;
  const Graph &graph_;
  const std::vector<int> &warm_;
  const std::vector<BoundRow> &rows_;
  int lower_;

public:
  bool warmAccepted = false;
  double initialPrimal = 0, initialDual = 0;
  TimedMaster(const ClusterGraph &cg, const std::string &cpuLimit,
              long wallLimit, const Graph &graph, const std::vector<int> &warm,
              const std::vector<BoundRow> &rows, int lower)
      : MaxBase(cg, nullptr, 1, 1, 0.4, 5, 10, 10, 10, 0.8, 0.8, false, 0.4,
                cpuLimit.c_str(), false, false, 15, 0.3, 0.3),
        wallLimit_(wallLimit), graph_(graph), warm_(warm), rows_(rows),
        lower_(lower) {}

protected:
  void initializeParameters() override {
    MaxBase::initializeParameters();
    maxCowTime(wallLimit_);
    requiredGuarantee(0.0);
  }
  void initializePools(ArrayBuffer<abacus::Constraint *> &constraints,
                       ArrayBuffer<abacus::Variable *> &variables,
                       int varPoolSize, int cutPoolSize,
                       bool dynamicCutPool = false) override {
    ArrayBuffer<abacus::Constraint *> extended(constraints.size() +
                                               rows_.size());
    for (auto *c : constraints)
      extended.push(c);
    std::vector<edge> edges;
    for (edge e : graph_.edges)
      edges.push_back(e);
    for (const auto &row : rows_) {
      List<NodePair> pairs;
      for (int i : row.edges)
        pairs.pushBack(NodePair(edges.at(i)->source(), edges.at(i)->target()));
      extended.push(new cluster_planarity::MaxPlanarEdgesConstraint(
          this, (int)row.edges.size() - row.lower, pairs));
    }
    MaxBase::initializePools(extended, variables, varPoolSize, cutPoolSize,
                             dynamicCutPool);
  }
  void initializeOptimization() override {
    MaxBase::initializeOptimization();
    {
      const int retained = graph_.numberOfEdges() - warm_.size();
      if (betterPrimal(retained)) {
        std::vector<bool> drop(graph_.maxEdgeIndex() + 1, false);
        for (int i : warm_)
          drop.at(i) = true;
        List<NodePair> original, connection;
        List<edge> deleted;
        for (edge e : graph_.edges) {
          if (drop[e->index()])
            deleted.pushBack(e);
          else
            original.pushBack(NodePair(e->source(), e->target()));
        }
        updateBestSubGraph(original, connection, deleted);
        primalBound(retained);
        warmAccepted = true;
      }
    }
    if (lower_ > 0 && graph_.numberOfEdges() - lower_ < dualBound())
      dualBound(graph_.numberOfEdges() - lower_);
    initialPrimal = primalBound();
    initialDual = dualBound();
  }
};
struct Result {
  int n = 0, m = 0, lb = 0, ub = 0, nlp = 0, nsub = 0, kcuts = 0;
  double elapsed = 0, primal = 0, dual = 0;
  bool feasible = false, native = false, warmAccepted = false;
  int boundRows = 0;
  double initialPrimal = 0, initialDual = 0;
  std::string status = "Unprocessed", error, boundWarning;
  std::vector<int> deleted, originalEdges;
};
std::string esc(const std::string &s) {
  std::ostringstream o;
  o << '"';
  for (char c : s) {
    switch (c) {
    case '"':
      o << "\\\"";
      break;
    case '\\':
      o << "\\\\";
      break;
    case '\n':
      o << "\\n";
      break;
    case '\r':
      o << "\\r";
      break;
    default:
      o << c;
    }
  }
  o << '"';
  return o.str();
}
std::string duration(long s) {
  std::ostringstream o;
  o << s / 3600 << ":" << (s / 60) % 60 << ":" << s % 60;
  return o.str();
}
bool feasibleAfter(const Graph &g, const std::vector<int> &deleted) {
  Graph h;
  std::vector<node> nodes(g.maxNodeIndex() + 1);
  for (node v : g.nodes)
    nodes[v->index()] = h.newNode();
  std::vector<bool> drop(g.maxEdgeIndex() + 1, false);
  for (int i : deleted) {
    if (i < 0 || i > g.maxEdgeIndex() || drop[i])
      return false;
    drop[i] = true;
  }
  for (edge e : g.edges)
    if (!drop[e->index()])
      h.newEdge(nodes[e->source()->index()], nodes[e->target()->index()]);
  return isConnected(h) && isPlanar(h);
}
int main(int argc, char **argv) {
  if (argc != 7) {
    std::cerr << "usage: driver INPUT OUTPUT_JSON TIME_LIMIT SEED CERTIFICATES "
                 "THREADS\n";
    return 2;
  }
  std::ifstream in(argv[1]);
  int n, m;
  if (!(in >> n >> m) || n < 0 || m < 0)
    return 2;
  Graph g;
  std::vector<node> nodes(n);
  for (int i = 0; i < n; ++i)
    nodes[i] = g.newNode();
  for (int i = 0; i < m; ++i) {
    int u, v;
    if (!(in >> u >> v) || u < 0 || v < 0 || u >= n || v >= n || u == v)
      return 2;
    g.newEdge(nodes[u], nodes[v]);
  }
  if (!isConnected(g)) {
    std::cerr << "input must be connected\n";
    return 2;
  }
  if (!isSimple(g)) {
    std::cerr << "input must be simple\n";
    return 2;
  }
  if (std::stoi(argv[6]) < 1)
    return 2;
  setenv("PIG_OGDF_THREADS", argv[6], 1);
  std::ifstream cert(argv[5]);
  int globalLower, nwarm, nrows;
  if (!(cert >> globalLower >> nwarm >> nrows) || globalLower < 0 ||
      nwarm < 0 || nrows < 0)
    return 2;
  std::vector<int> warm(nwarm);
  for (int &i : warm)
    if (!(cert >> i))
      return 2;
  if (!feasibleAfter(g, warm)) {
    std::cerr << "supplied subgraph must be planar and connected\n";
    return 2;
  }
  std::vector<BoundRow> rows;
  for (int j = 0; j < nrows; ++j) {
    BoundRow r;
    int k;
    if (!(cert >> r.lower >> k) || r.lower <= 0 || k < r.lower)
      return 2;
    r.edges.resize(k);
    for (int &i : r.edges)
      if (!(cert >> i) || i < 0 || i >= m)
        return 2;
    std::vector<int> unique = r.edges;
    std::sort(unique.begin(), unique.end());
    if (std::adjacent_find(unique.begin(), unique.end()) != unique.end())
      return 2;
    rows.push_back(r);
  }
  double budget = std::stod(argv[3]);
  if (!std::isfinite(budget) || budget < 0)
    return 2;
  int seed = std::stoi(argv[4]);
  if (seed < 0)
    return 2;
  setSeed(seed);
  std::srand(seed);
  const auto begin = Clock::now();
  auto elapsed = [&]() {
    return std::chrono::duration<double>(Clock::now() - begin).count();
  };
  Logger::globalLogLevel(Logger::Level::Alarm);
  EdgeArray<int> component(g);
  int count = biconnectedComponents(g, component);
  std::vector<std::vector<edge>> blocks(count);
  for (edge e : g.edges)
    blocks[component[e]].push_back(e);
  // Public wrapper uses pushFront for each block except when only one exists.
  if (count > 1)
    for (auto &b : blocks)
      std::reverse(b.begin(), b.end());
  int planarBlocks = 0, lower = 0;
  std::vector<int> deleted;
  std::vector<Result> results;
  for (size_t bi = 0; bi < blocks.size(); ++bi) {
    auto &edges = blocks[bi];
    Graph b;
    std::vector<node> bnodes(n, nullptr);
    std::vector<int> original;
    if (count == 1)
      for (node v : g.nodes)
        bnodes[v->index()] = b.newNode();
    for (edge e : edges) {
      int u = e->source()->index(), v = e->target()->index();
      if (!bnodes[u])
        bnodes[u] = b.newNode();
      if (!bnodes[v])
        bnodes[v] = b.newNode();
      b.newEdge(bnodes[u], bnodes[v]);
      original.push_back(e->index());
    }
    if (isPlanar(b)) {
      ++planarBlocks;
      continue;
    }
    Result r;
    r.originalEdges = original;
    r.n = b.numberOfNodes();
    r.m = b.numberOfEdges();
    r.lb = std::max(1, r.m - 3 * r.n + 6);
    std::vector<int> localWarm;
    std::vector<BoundRow> localRows;
    std::vector<int> local(m, -1);
    for (size_t i = 0; i < original.size(); ++i)
      local[original[i]] = i;
    for (int i : warm)
      if (local[i] >= 0)
        localWarm.push_back(local[i]);
    std::vector<int> localDeleted = localWarm;
    r.ub = localDeleted.size();
    r.feasible = true;
    for (const auto &row : rows) {
      BoundRow lr;
      lr.lower = row.lower;
      for (int i : row.edges)
        if (local[i] >= 0)
          lr.edges.push_back(local[i]);
      if (lr.edges.empty())
        continue;
      if (lr.edges.size() != row.edges.size()) {
        std::cerr << "certificate spans separate solve blocks\n";
        return 3;
      }
      r.lb = std::max(r.lb, lr.lower);
      localRows.push_back(lr);
    }
    if (count == 1)
      r.lb = std::max(r.lb, globalLower);
    if (globalLower == (int)warm.size())
      r.lb = r.ub;
    r.boundRows = localRows.size();
    if (r.lb > r.ub) {
      std::cerr << "certificate contradicts feasible solution\n";
      return 3;
    }
    double remaining = r.lb == r.ub ? 0 : budget - elapsed();
    if (remaining >= 1.0) {
      const auto start = Clock::now();
      long seconds = std::max(1L, (long)std::ceil(remaining));
      ClusterGraph cg(b);
      TimedMaster master(cg, duration(seconds), seconds, b, localWarm,
                         localRows, r.lb);
      abacus::Master::STATUS stat = abacus::Master::Error;
      bool completed = false;
      try {
        stat = master.optimize();
        completed = true;
      } catch (const std::exception &e) {
        r.error = e.what();
      } catch (...) {
        r.error = "unknown exception";
      }
      r.elapsed = std::chrono::duration<double>(Clock::now() - start).count();
      r.status = abacus::Master::STATUS_[stat];
      if (completed && stat == abacus::Master::Error)
        r.error = "native solver returned Error";
      if (completed) {
        r.warmAccepted = master.warmAccepted;
        r.initialPrimal = master.initialPrimal;
        r.initialDual = master.initialDual;
        r.primal = master.primalBound();
        r.dual = master.dualBound();
        r.nlp = master.nLp();
        r.nsub = master.nSub();
        r.kcuts = master.addedKConstraints();
        List<edge> nativeDeleted;
        master.getDeletedEdges(nativeDeleted);
        std::vector<int> candidate;
        for (edge e : nativeDeleted)
          candidate.push_back(e->index());
        if (feasibleAfter(b, candidate)) {
          r.native = true;
          if (candidate.size() <= localDeleted.size()) {
            localDeleted = candidate;
            r.ub = candidate.size();
          }
          if (std::abs((r.m - (int)candidate.size()) -
                       std::floor(r.primal + 1e-4)) > 1e-5)
            r.error = "native solution and primal bound disagree";
        } else {
          r.error = "native deleted edges do not yield a planar graph";
        }
        if (stat != abacus::Master::Error && std::isfinite(r.dual) &&
            r.dual >= -1e-4 && r.dual <= r.m + 1e-4) {
          int nativeLB = std::max(0, (int)std::ceil(r.m - r.dual - 1e-4));
          if (nativeLB <= r.ub)
            r.lb = std::max(r.lb, nativeLB);
          else
            r.boundWarning = "native dual bound contradicts feasible solution";
        }
      }
    } else
      r.status = r.lb == r.ub ? "CertifiedOptimal" : "GlobalTimeLimit";
    lower += r.lb;
    for (int index : localDeleted) {
      r.deleted.push_back(original[index]);
      deleted.push_back(original[index]);
    }
    results.push_back(r);
  }
  lower = std::max(lower, globalLower);
  bool rowsValid = true;
  std::vector<bool> drop(m, false);
  for (int i : deleted)
    drop[i] = true;
  for (const auto &row : rows) {
    int total = 0;
    for (int i : row.edges)
      total += drop[i];
    if (total < row.lower)
      rowsValid = false;
  }
  bool feasible =
      rowsValid && lower <= (int)deleted.size() && feasibleAfter(g, deleted);
  bool error = !feasible;
  for (auto &r : results)
    if (!r.error.empty())
      error = true;
  std::sort(deleted.begin(), deleted.end());
  double total = elapsed();
  std::ofstream out(argv[2]);
  out << std::setprecision(16);
  out << "{\n\"implementation\":\"OGDF foxglove-202510 / ABACUS / "
         "CLP\",\n\"commit\":\"5b6795655399b9d8e2921afec9d97bab9107d5ee\",\n";
  out << "\"n\":" << n << ",\"m\":" << m << ",\"seed\":" << seed
      << ",\"time_limit_seconds\":" << budget << ",\n";
  out << "\"runtime_seconds\":" << total
      << ",\"deletion_lower_bound\":" << lower
      << ",\"deletion_upper_bound\":" << deleted.size() << ",\n";
  out << "\"status\":"
      << esc(error                          ? "ERROR"
             : lower == (int)deleted.size() ? "OPTIMAL"
                                            : "TIME_LIMIT")
      << ",\"validated_planar\":" << (feasible ? "true" : "false") << ",\n";
  out << "\"biconnected_components\":" << count
      << ",\"planar_components\":" << planarBlocks
      << ",\"deleted_edge_indices\":[";
  for (size_t i = 0; i < deleted.size(); ++i) {
    if (i)
      out << ',';
    out << deleted[i];
  }
  out << "],\n\"blocks\":[";
  for (size_t i = 0; i < results.size(); ++i) {
    auto &r = results[i];
    if (i)
      out << ',';
    out << "{\"n\":" << r.n << ",\"m\":" << r.m
        << ",\"status\":" << esc(r.status)
        << ",\"runtime_seconds\":" << r.elapsed
        << ",\"deletion_lower_bound\":" << r.lb
        << ",\"deletion_upper_bound\":" << r.ub
        << ",\"retained_primal_bound_raw\":" << r.primal
        << ",\"retained_dual_bound\":" << r.dual
        << ",\"native_feasible_solution\":" << (r.native ? "true" : "false")
        << ",\"bound_rows\":" << r.boundRows
        << ",\"warm_start_accepted\":" << (r.warmAccepted ? "true" : "false")
        << ",\"initial_primal\":" << r.initialPrimal
        << ",\"initial_dual\":" << r.initialDual
        << ",\"linear_programs\":" << r.nlp << ",\"subproblems\":" << r.nsub
        << ",\"kuratowski_cuts\":" << r.kcuts << ",\"error\":" << esc(r.error)
        << ",\"bound_warning\":" << esc(r.boundWarning)
        << ",\"original_edge_indices\":[";
    for (size_t j = 0; j < r.originalEdges.size(); ++j) {
      if (j)
        out << ',';
      out << r.originalEdges[j];
    }
    out << "]}";
  }
  out << "]\n}\n";
  return feasible && !error ? 0 : 1;
}
