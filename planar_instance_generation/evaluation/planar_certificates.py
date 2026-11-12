"""Persist generation certificates and transfer them fairly between merge rules."""

from dataclasses import replace

import networkx as nx

from planar_instance_generation.core.planarization import CertifiedPlanarPiece


def generation_certificates(candidate):
    """Save every input planarization in final-composite node coordinates."""
    records = []
    stages = candidate.stages
    for index, stage in enumerate(stages):
        graph = (
            stage.merge.case.topology(active_only=True)
            if hasattr(stage.merge, "case")
            else stage.merge.instance.topology()
        )
        for side in ("left", "right"):
            result = getattr(stage.selection, side + "_planarization", None)
            if result is None:
                continue
            source = getattr(stage.merge, side)
            mapping = dict(
                source.bus_ids if hasattr(source, "bus_ids") else source.node_ids
            )
            original = graph.subgraph(mapping.values()).copy()
            planar = nx.relabel_nodes(result.planar_subgraph, mapping, copy=True)
            bounds = [
                (tuple((mapping[u], mapping[v]) for u, v in support), lower)
                for support, lower in result.bound_rows
            ]
            for later in stages[index + 1 :]:
                source = later.merge.left
                relabel = (
                    source.bus_ids if hasattr(source, "bus_ids") else source.node_ids
                )
                original = nx.relabel_nodes(original, relabel, copy=True)
                planar = nx.relabel_nodes(planar, relabel, copy=True)
                bounds = [
                    (tuple((relabel[u], relabel[v]) for u, v in support), lower)
                    for support, lower in bounds
                ]
            records.append(
                {
                    "stage": stage.stage_index,
                    "side": side,
                    "nodes": list(original),
                    "edges": list(original.edges()),
                    "planar_edges": list(planar.edges()),
                    "bound_rows": bounds,
                    "backend": result.solver_metadata.get("policy"),
                    "lower_bound": int(result.omitted_lower_bound),
                    "upper_bound": int(result.omitted_upper_bound),
                }
            )
    return records


def attach_generation_certificates(tasks, dependencies, metadata_sources):
    """Give every rule the same pool, translated through source-local labels.

    metadata_sources contains (metadata, source task keys in source order).
    Transfer a bound only when all edges in its support remain present.
    Extra target edges cannot invalidate it. Restrict a planar subgraph and
    reconnect its components by a forest.
    """
    from .planarization_batch import source_mappings

    pool = []
    for metadata, keys in metadata_sources:
        canonical = {}
        for position, (_stem, mapping) in enumerate(source_mappings(metadata)):
            canonical.update(
                (target, (tuple(keys), position, source))
                for source, target in mapping.items()
            )
        for record in metadata.get("planarization_certificates", ()):
            original = nx.Graph()
            original.add_nodes_from(canonical[node] for node in record["nodes"])
            original.add_edges_from(
                (canonical[u], canonical[v]) for u, v in record["edges"]
            )
            planar = nx.Graph()
            planar.add_nodes_from(original)
            planar.add_edges_from(
                (canonical[u], canonical[v]) for u, v in record["planar_edges"]
            )
            if not nx.check_planarity(planar)[0] or not all(
                original.has_edge(u, v) for u, v in planar.edges()
            ):
                raise ValueError("invalid persisted planar certificate")
            lower = record["lower_bound"]
            if (
                type(lower) is not int
                or not 0
                <= lower
                <= original.number_of_edges() - planar.number_of_edges()
            ):
                raise ValueError("invalid persisted lower bound")
            bounds = [
                (tuple((canonical[u], canonical[v]) for u, v in support), value)
                for support, value in record.get("bound_rows", ())
            ]
            pool.append((original, planar, lower, bounds))

    prepared = []
    for task in tasks:
        mappings = dependencies.get(task.cache_key)
        labels = (
            {
                (tuple(key for key, _ in mappings), position, source): target
                for position, (_key, mapping) in enumerate(mappings)
                for source, target in mapping.items()
            }
            if mappings
            else None
        )
        certificates = list(task.prior_certificates)
        seen = set()
        for original, planar, lower, bounds in pool:
            if labels is None:
                # Identical piece tasks can occur more than once in a final.
                # Restrict each occurrence separately; never identify vertices.
                positions = {
                    node[:2] for node in original if node[0][node[1]] == task.cache_key
                }
                translations = [
                    {node: node[2] for node in original if node[:2] == position}
                    for position in sorted(positions)
                ]
            else:
                translations = [labels]
            for translation in translations:
                certificate = _transfer(
                    task.graph, original, planar, lower, translation, bounds
                )
                if certificate is None:
                    continue
                signature = (
                    frozenset(certificate.graph),
                    frozenset(map(frozenset, certificate.graph.edges)),
                    frozenset(map(frozenset, certificate.planar_subgraph.edges)),
                    certificate.lower_bound,
                    frozenset(
                        (frozenset(map(frozenset, support)), value)
                        for support, value in certificate.bound_rows
                    ),
                )
                if signature not in seen:
                    seen.add(signature)
                    certificates.append(certificate)
        prepared.append(replace(task, prior_certificates=tuple(certificates)))
    return tuple(prepared)


def _transfer(graph, original, planar, lower, labels, bounds=()):
    nodes = set(original).intersection(labels)
    if not nodes:
        return None
    target = graph.subgraph(labels[n] for n in nodes).copy()
    retained = nx.Graph()
    retained.add_nodes_from(target)
    retained.add_edges_from(
        (labels[u], labels[v])
        for u, v in planar.edges()
        if u in nodes and v in nodes and target.has_edge(labels[u], labels[v])
    )
    lost = sum(
        not (u in nodes and v in nodes and target.has_edge(labels[u], labels[v]))
        for u, v in original.edges()
    )
    bound = lower if lost == 0 else 0
    components = nx.utils.UnionFind(target)
    for u, v in retained.edges():
        components.union(u, v)
    for u, v in target.edges():
        if components[u] != components[v]:
            retained.add_edge(u, v)
            components.union(u, v)
    rows = tuple(
        (tuple((labels[u], labels[v]) for u, v in support), value)
        for support, value in bounds
        if all(
            u in nodes and v in nodes and target.has_edge(labels[u], labels[v])
            for u, v in support
        )
    )
    return CertifiedPlanarPiece(target, retained, bound, rows)
