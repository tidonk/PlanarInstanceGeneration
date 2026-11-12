"""Lift a planar certificate of a spanning subgraph to its original graph."""

import networkx as nx

from planar_instance_generation.core.planarization import CertifiedPlanarPiece


def lift_subgraph_certificate(graph, subgraph, planar_subgraph, lower_bound):
    """Keep the subgraph lower bound and connect its planar incumbent by bridges.

    The bound is valid by subgraph monotonicity. Adding a forest between the
    incumbent's components preserves planarity and restores exactly the original
    connected components, as required by the shared-piece warm-start interface.
    This does not transfer source bounds to extracted induced subgraphs.
    """
    if set(graph) != set(subgraph) or set(graph) != set(planar_subgraph):
        raise ValueError("certificates must span the original node set")
    if any(not graph.has_edge(u, v) for u, v in subgraph.edges):
        raise ValueError("certificate subgraph has foreign edges")
    if any(not subgraph.has_edge(u, v) for u, v in planar_subgraph.edges):
        raise ValueError("planar certificate has foreign edges")
    if not nx.check_planarity(planar_subgraph)[0]:
        raise ValueError("certificate is not planar")
    if (int(lower_bound) != lower_bound or lower_bound < 0 or
            lower_bound > subgraph.number_of_edges() - planar_subgraph.number_of_edges()):
        raise ValueError("inconsistent subgraph skewness bound")
    retained = planar_subgraph.copy()
    components = nx.utils.UnionFind(graph)
    for u, v in retained.edges:
        components.union(u, v)
    for u, v, attributes in graph.edges(data=True):
        if components[u] != components[v]:
            retained.add_edge(u, v, **attributes)
            components.union(u, v)
    return CertifiedPlanarPiece(graph.copy(), retained, int(lower_bound))
