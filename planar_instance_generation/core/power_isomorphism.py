"""Attribute-preserving isomorphism for MATPOWER/PGLib cases."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping

import networkx as nx
import numpy as np

from planar_instance_generation.io.pglib import PglibCase, PglibFormatError


_NodeData = Mapping[str, object]
_MultiEdgeData = Mapping[object, Mapping[str, object]]


def pglib_isomorphic(left: PglibCase, right: PglibCase) -> bool:
    """Return whether two cases differ only by bus, area, and row ordering.

    Case names and numeric identifiers are deliberately ignored. All numeric
    model attributes are compared exactly: bus data, generator rows paired
    with their active and optional reactive cost rows, every branch (including
    inactive and parallel branches), and optional area data. Area membership
    and reference buses are represented explicitly, allowing area identifiers
    to be relabelled consistently with bus identifiers. Branch ``from`` and
    ``to`` roles remain directed because transformer phase shifts make their
    orientation meaningful.

    Numeric model attributes are compared exactly; numerically close cases are
    therefore not considered isomorphic.
    """

    if not isinstance(left, PglibCase) or not isinstance(right, PglibCase):
        raise TypeError("left and right must be PglibCase instances")
    left.validate()
    right.validate()
    _validate_area_data(left)
    _validate_area_data(right)

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


def _global_signature(case: PglibCase) -> tuple[object, ...]:
    return (
        case.version,
        float(case.base_mva),
        case.bus.shape,
        case.gen.shape,
        case.branch.shape,
        case.gencost.shape,
        case.areas is not None,
        None if case.areas is None else case.areas.shape,
    )


def _validate_area_data(case: PglibCase) -> None:
    _integer_values(
        case.bus[:, 6], f"{case.name}: mpc.bus(:, 7) area identifiers"
    )
    if case.areas is None:
        return

    area_ids = _integer_values(
        case.areas[:, 0], f"{case.name}: mpc.areas(:, 1) area identifiers"
    )
    if len(set(area_ids)) != len(area_ids):
        raise PglibFormatError(
            f"{case.name}: mpc.areas contains duplicate area identifiers"
        )

    reference_buses = _integer_values(
        case.areas[:, 1], f"{case.name}: mpc.areas(:, 2) reference buses"
    )
    known_buses = {int(value) for value in case.bus[:, 0]}
    unknown = set(reference_buses) - known_buses
    if unknown:
        preview = ", ".join(map(str, sorted(unknown)[:8]))
        raise PglibFormatError(
            f"{case.name}: mpc.areas references unknown bus(es): {preview}"
        )

    # Bus areas need not have an explicit legacy mpc.areas row. Conversely,
    # unused rows are retained as model data and therefore affect isomorphism.


def _integer_values(values: np.ndarray, context: str) -> tuple[int, ...]:
    rounded = np.rint(values)
    if not np.array_equal(values, rounded):
        raise PglibFormatError(f"{context} must be integers")
    return tuple(int(value) for value in rounded)


def _isomorphism_graph(case: PglibCase) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    generator_records = _generators_by_bus(case)

    for row in case.bus:
        bus_id = int(row[0])
        area_id = int(row[6])
        bus_attributes = tuple(row[1:6]) + tuple(row[7:])
        graph.add_node(
            ("bus", bus_id),
            signature=("bus", bus_attributes, generator_records.get(bus_id, ())),
        )
        graph.add_node(("area", area_id), signature=("area", False, ()))
        graph.add_edge(
            ("bus", bus_id),
            ("area", area_id),
            signature=("area_membership",),
        )

    if case.areas is not None:
        for row in case.areas:
            area_id = int(row[0])
            reference_bus = int(row[1])
            area_node = ("area", area_id)
            graph.add_node(area_node, signature=("area", True, tuple(row[2:])))
            graph.add_edge(
                area_node,
                ("bus", reference_bus),
                signature=("area_reference",),
            )

    for row in case.branch:
        graph.add_edge(
            ("bus", int(row[0])),
            ("bus", int(row[1])),
            signature=("branch", tuple(row[2:])),
        )
    return graph


def _generators_by_bus(case: PglibCase) -> dict[int, tuple[tuple[object, ...], ...]]:
    generator_count = case.gen.shape[0]
    has_reactive_cost = case.gencost.shape[0] == 2 * generator_count
    records: defaultdict[int, list[tuple[object, ...]]] = defaultdict(list)
    for index, row in enumerate(case.gen):
        active_cost = tuple(case.gencost[index])
        reactive_cost = (
            tuple(case.gencost[index + generator_count])
            if has_reactive_cost
            else None
        )
        records[int(row[0])].append(
            (tuple(row[1:]), active_cost, reactive_cost)
        )
    return {bus: tuple(sorted(values)) for bus, values in records.items()}


def _attribute_histograms(
    graph: nx.MultiDiGraph,
) -> tuple[Counter[object], Counter[object]]:
    nodes = Counter(attributes["signature"] for _, attributes in graph.nodes(data=True))
    edges = Counter(
        attributes["signature"]
        for _, _, _, attributes in graph.edges(data=True, keys=True)
    )
    return nodes, edges


def _node_match(left: _NodeData, right: _NodeData) -> bool:
    return left.get("signature") == right.get("signature")


def _multiedge_match(left: _MultiEdgeData, right: _MultiEdgeData) -> bool:
    return Counter(
        attributes.get("signature") for attributes in left.values()
    ) == Counter(attributes.get("signature") for attributes in right.values())
