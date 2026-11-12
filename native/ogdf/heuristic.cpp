// Standalone OGDF PlanarSubgraphFast with constructor-default algorithm
// settings.
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <ogdf/basic/Graph.h>
#include <ogdf/basic/Logger.h>
#include <ogdf/basic/extended_graph_alg.h>
#include <ogdf/basic/simple_graph_alg.h>
#include <ogdf/planarity/PlanarSubgraphFast.h>
#include <vector>
using namespace ogdf;
int main(int argc, char **argv) {
  if (argc != 5)
    return 2;
  std::ifstream input(argv[1]);
  int n, m;
  if (!(input >> n >> m) || n < 1 || m < 0)
    return 2;
  Graph g;
  std::vector<node> nodes(n);
  for (node &v : nodes)
    v = g.newNode();
  for (int i = 0; i < m; ++i) {
    int u, v;
    if (!(input >> u >> v) || u < 0 || v < 0 || u >= n || v >= n)
      return 2;
    g.newEdge(nodes[u], nodes[v]);
  }
  if (!isSimple(g) || !isConnected(g))
    return 2;
  int seed = std::stoi(argv[3]);
  setSeed(seed);
  std::srand(seed);
  Logger::globalLogLevel(Logger::Level::Alarm);
  PlanarSubgraphFast<double> solver;
  const auto defaultThreads = solver.maxThreads();
  // Only the resource limit is configured. The run count stays at its default.
  const unsigned int threads = std::stoul(argv[4]);
  solver.maxThreads(threads);
  auto start = std::chrono::steady_clock::now();
  List<edge> deleted;
  if (!isPlanar(g) && !Module::isSolution(solver.call(g, deleted)))
    return 3;
  double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  std::ofstream out(argv[2]);
  out << std::setprecision(16) << "{\"seed\":" << seed
      << ",\"runs\":" << solver.runs()
      << ",\"thread_limit\":" << solver.maxThreads()
      << ",\"constructor_thread_limit\":" << defaultThreads
      << ",\"seconds\":" << seconds << ",\"deleted_edge_indices\":[";
  bool first = true;
  for (edge e : deleted) {
    if (!first)
      out << ',';
    first = false;
    out << e->index();
  }
  out << "]}\n";
}
