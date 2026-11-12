"""Attribute-preserving isomorphism for released SNDlib instances."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

import networkx as nx

from planar_instance_generation.core.snd import SndInstance
from planar_instance_generation.core.snd_serialization import serialize_snd_instance
from planar_instance_generation.io.sndlib import (
    SndlibCapacityModule,
    SndlibNetwork,
)


_NodeData = Mapping[str, object]
_MultiEdgeData = Mapping[object, Mapping[str, object]]


def sndlib_isomorphic(left: SndlibNetwork, right: SndlibNetwork) -> bool:
    """Return whether two released networks differ only by IDs and row order.

    Node coordinates; link modules, costs, and endpoints; demand directions,
    values, path limits, and admissible paths are all preserved exactly. A
    link's XML ``source``/``target`` order is only a reference orientation for
    the undirected SND model, so flipping it does not change identity.
    """

    if not isinstance(left, SndlibNetwork) or not isinstance(right, SndlibNetwork):
        raise TypeError("left and right must be SndlibNetwork objects")
    left.validate()
    right.validate()
    if _global_signature(left) != _global_signature(right):
        return False

    left_graph = _isomorphism_graph(left)
    right_graph = _isomorphism_graph(right)
    if _attribute_histograms(left_graph) != _attribute_histograms(right_graph):
        return False
    matcher = nx.algorithms.isomorphism.MultiDiGraphMatcher(
        left_graph,
        right_graph,
        node_match=_node_match,
        edge_match=_multiedge_match,
    )
    return matcher.is_isomorphic()


def snd_isomorphic(left: SndInstance, right: SndInstance) -> bool:
    """Compare core instances by the gate-ready XML models they release.

    Installed reference capacities, reference flows, source provenance, and
    exactly inert internal commodities are not XML instance data. They are
    intentionally excluded from the isomorphism comparison.
    """

    if not isinstance(left, SndInstance) or not isinstance(right, SndInstance):
        raise TypeError("left and right must be SndInstance objects")
    return sndlib_isomorphic(
        serialize_snd_instance(left).network,
        serialize_snd_instance(right).network,
    )


def _global_signature(network: SndlibNetwork) -> tuple[object, ...]:
    return (
        network.version,
        network.coordinates_type,
        network.metadata,
        len(network.nodes),
        len(network.links),
        len(network.demands),
    )


def _isomorphism_graph(network: SndlibNetwork) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    nodes = {node.id: ("node", index) for index, node in enumerate(network.nodes)}
    links = {link.id: ("link", index) for index, link in enumerate(network.links)}

    for index, node in enumerate(network.nodes):
        graph.add_node(("node", index), signature=("node", node.x, node.y))
    for index, link in enumerate(network.links):
        link_node = ("link", index)
        graph.add_node(
            link_node,
            signature=(
                "link",
                _module_signature(link.preinstalled_module),
                tuple(
                    sorted(
                        _module_signature(module)
                        for module in link.additional_modules
                    )
                ),
                link.routing_cost,
                link.setup_cost,
            ),
        )
        graph.add_edge(link_node, nodes[link.source], signature=("endpoint",))
        graph.add_edge(link_node, nodes[link.target], signature=("endpoint",))

    for demand_index, demand in enumerate(network.demands):
        demand_node = ("demand", demand_index)
        graph.add_node(
            demand_node,
            signature=("demand", demand.value, demand.max_path_length),
        )
        graph.add_edge(
            demand_node,
            nodes[demand.source],
            signature=("demand_source",),
        )
        graph.add_edge(
            demand_node,
            nodes[demand.target],
            signature=("demand_target",),
        )
        for path_index, path in enumerate(demand.admissible_paths):
            path_node = ("path", demand_index, path_index)
            graph.add_node(path_node, signature=("path", len(path.link_ids)))
            graph.add_edge(demand_node, path_node, signature=("admissible_path",))
            for position, link_id in enumerate(path.link_ids):
                graph.add_edge(
                    path_node,
                    links[link_id],
                    signature=("path_link", position),
                )
    return graph


def _module_signature(
    module: SndlibCapacityModule | None,
) -> tuple[float, float] | None:
    if module is None:
        return None
    return module.capacity, module.cost


def _attribute_histograms(
    graph: nx.MultiDiGraph,
) -> tuple[Counter[object], Counter[object]]:
    return (
        Counter(attributes["signature"] for _, attributes in graph.nodes(data=True)),
        Counter(
            attributes["signature"]
            for _, _, _, attributes in graph.edges(data=True, keys=True)
        ),
    )


def _node_match(left: _NodeData, right: _NodeData) -> bool:
    return left.get("signature") == right.get("signature")


def _multiedge_match(left: _MultiEdgeData, right: _MultiEdgeData) -> bool:
    return Counter(
        attributes.get("signature") for attributes in left.values()
    ) == Counter(attributes.get("signature") for attributes in right.values())
