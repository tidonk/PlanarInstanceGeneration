"""Strict, namespace-aware I/O for SNDlib 1.0 XML instances."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import re
from typing import Final
import xml.etree.ElementTree as ET

import networkx as nx


SNDLIB_NAMESPACE: Final = "http://sndlib.zib.de/network"
_NUMBER: Final = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)$"
)


class SndlibFormatError(ValueError):
    """Raised when XML is not in the supported SNDlib 1.0 format."""


@dataclass(frozen=True, slots=True)
class SndlibMetadata:
    granularity: str
    time: str
    unit: str
    origin: str

    def __post_init__(self) -> None:
        for field in ("granularity", "time", "unit", "origin"):
            _validate_text(field, getattr(self, field))


@dataclass(frozen=True, slots=True)
class SndlibNode:
    id: str
    x: float
    y: float

    def __post_init__(self) -> None:
        _validate_id("node", self.id)
        _validate_finite("node x-coordinate", self.x)
        _validate_finite("node y-coordinate", self.y)


@dataclass(frozen=True, slots=True)
class SndlibCapacityModule:
    capacity: float
    cost: float

    def __post_init__(self) -> None:
        _validate_nonnegative("module capacity", self.capacity)
        _validate_nonnegative("module cost", self.cost)


@dataclass(frozen=True, slots=True)
class SndlibLink:
    """One XML link, retaining its ordered source and target exactly."""

    id: str
    source: str
    target: str
    preinstalled_module: SndlibCapacityModule | None = None
    additional_modules: tuple[SndlibCapacityModule, ...] = ()
    routing_cost: float | None = None
    setup_cost: float | None = None

    def __post_init__(self) -> None:
        _validate_id("link", self.id)
        _validate_id("link source", self.source)
        _validate_id("link target", self.target)
        if self.source == self.target:
            raise SndlibFormatError(f"link {self.id!r} is a self-loop")
        object.__setattr__(self, "additional_modules", tuple(self.additional_modules))
        if self.preinstalled_module is None and not self.additional_modules:
            raise SndlibFormatError(f"link {self.id!r} has no capacity module")
        if self.routing_cost is not None:
            _validate_nonnegative(f"link {self.id!r} routing cost", self.routing_cost)
        if self.setup_cost is not None:
            _validate_nonnegative(f"link {self.id!r} setup cost", self.setup_cost)


@dataclass(frozen=True, slots=True)
class SndlibAdmissiblePath:
    id: str
    link_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_id("admissible path", self.id)
        object.__setattr__(self, "link_ids", tuple(self.link_ids))
        if not self.link_ids:
            raise SndlibFormatError(f"admissible path {self.id!r} is empty")
        for link_id in self.link_ids:
            _validate_id("admissible-path link", link_id)


@dataclass(frozen=True, slots=True)
class SndlibDemand:
    id: str
    source: str
    target: str
    value: float
    max_path_length: int | None = None
    admissible_paths: tuple[SndlibAdmissiblePath, ...] = ()

    def __post_init__(self) -> None:
        _validate_id("demand", self.id)
        _validate_id("demand source", self.source)
        _validate_id("demand target", self.target)
        if self.source == self.target:
            raise SndlibFormatError(f"demand {self.id!r} has identical terminals")
        _validate_nonnegative(f"demand {self.id!r} value", self.value)
        if self.max_path_length is not None and self.max_path_length <= 0:
            raise SndlibFormatError("maximum path length must be positive")
        object.__setattr__(self, "admissible_paths", tuple(self.admissible_paths))
        path_ids = [path.id for path in self.admissible_paths]
        _reject_duplicates("admissible path", path_ids)


@dataclass(frozen=True, slots=True)
class SndlibNetwork:
    """An SNDlib network with exact link orientation and capacity choices.

    SNDlib 1.0 XML does not declare whether links are semantically directed.
    :meth:`topology` therefore gives the undirected physical projection, while
    :meth:`directed_link_graph` retains each XML source-to-target record.
    """

    version: str
    coordinates_type: str
    nodes: tuple[SndlibNode, ...]
    links: tuple[SndlibLink, ...]
    demands: tuple[SndlibDemand, ...]
    metadata: SndlibMetadata | None = None

    def __post_init__(self) -> None:
        if self.version != "1.0":
            raise SndlibFormatError(
                f"only SNDlib version '1.0' is supported, got {self.version!r}"
            )
        _validate_text("coordinates type", self.coordinates_type)
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "links", tuple(self.links))
        object.__setattr__(self, "demands", tuple(self.demands))
        self.validate()

    @property
    def routing_unit(self) -> str | None:
        """Return the capacity/demand unit recorded by SNDlib metadata."""

        return None if self.metadata is None else self.metadata.unit

    def validate(self) -> None:
        """Fail fast on broken identifiers, references, and routed paths."""

        node_ids = [node.id for node in self.nodes]
        link_ids = [link.id for link in self.links]
        demand_ids = [demand.id for demand in self.demands]
        _reject_duplicates("node", node_ids)
        _reject_duplicates("link", link_ids)
        _reject_duplicates("demand", demand_ids)
        known_nodes = set(node_ids)
        known_links = {link.id: link for link in self.links}

        for link in self.links:
            unknown = {link.source, link.target} - known_nodes
            if unknown:
                raise SndlibFormatError(
                    f"link {link.id!r} references unknown node(s): {_preview(unknown)}"
                )
        for demand in self.demands:
            unknown = {demand.source, demand.target} - known_nodes
            if unknown:
                raise SndlibFormatError(
                    f"demand {demand.id!r} references unknown node(s): {_preview(unknown)}"
                )
            for path in demand.admissible_paths:
                missing = set(path.link_ids) - known_links.keys()
                if missing:
                    raise SndlibFormatError(
                        f"path {path.id!r} of demand {demand.id!r} references "
                        f"unknown link(s): {_preview(missing)}"
                    )
                _validate_walk(demand, path, known_links)

    def topology(self) -> nx.Graph:
        """Return a simple undirected projection with XML-link provenance."""

        graph = nx.Graph(coordinates_type=self.coordinates_type)
        for index, node in enumerate(self.nodes):
            graph.add_node(node.id, node_index=index, x=node.x, y=node.y)
        for index, link in enumerate(self.links):
            orientation = (link.source, link.target)
            if graph.has_edge(link.source, link.target):
                edge = graph[link.source][link.target]
                edge["multiplicity"] += 1
                edge["link_indices"] = (*edge["link_indices"], index)
                edge["link_ids"] = (*edge["link_ids"], link.id)
                edge["orientations"] = (*edge["orientations"], orientation)
            else:
                graph.add_edge(
                    link.source,
                    link.target,
                    multiplicity=1,
                    link_indices=(index,),
                    link_ids=(link.id,),
                    orientations=(orientation,),
                )
        return graph

    def directed_link_graph(self) -> nx.MultiDiGraph:
        """Return the literal ordered XML links without inferring semantics."""

        graph = nx.MultiDiGraph(coordinates_type=self.coordinates_type)
        for index, node in enumerate(self.nodes):
            graph.add_node(node.id, node_index=index, x=node.x, y=node.y)
        for index, link in enumerate(self.links):
            graph.add_edge(
                link.source,
                link.target,
                key=link.id,
                link_index=index,
                link=link,
            )
        return graph


def read_sndlib(path: str | Path) -> SndlibNetwork:
    """Read an SNDlib 1.0 XML file without resolving external entities."""

    source = Path(path)
    try:
        data = source.read_bytes()
    except OSError as error:
        raise SndlibFormatError(f"cannot read {source}: {error}") from error
    upper = data.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise SndlibFormatError("DTD and entity declarations are not supported")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as error:
        raise SndlibFormatError(f"malformed XML in {source}: {error}") from error

    _expect_tag(root, "network")
    _expect_attributes(root, {"version"})
    version = _required_attribute(root, "version")
    _expect_children(root, {"meta", "networkStructure", "demands"})
    metadata_element = _optional_child(root, "meta")
    structure = _required_child(root, "networkStructure")
    demands_element = _required_child(root, "demands")

    _expect_attributes(structure, set())
    _expect_children(structure, {"nodes", "links"})
    nodes_element = _required_child(structure, "nodes")
    links_element = _required_child(structure, "links")

    _expect_attributes(nodes_element, {"coordinatesType"})
    _expect_children(nodes_element, {"node"})
    _expect_attributes(links_element, set())
    _expect_children(links_element, {"link"})
    _expect_attributes(demands_element, set())
    _expect_children(demands_element, {"demand"})

    network = SndlibNetwork(
        version=version,
        coordinates_type=_required_attribute(nodes_element, "coordinatesType"),
        nodes=tuple(_parse_node(element) for element in _children(nodes_element, "node")),
        links=tuple(_parse_link(element) for element in _children(links_element, "link")),
        demands=tuple(
            _parse_demand(element) for element in _children(demands_element, "demand")
        ),
        metadata=None if metadata_element is None else _parse_metadata(metadata_element),
    )
    return network


def write_sndlib(network: SndlibNetwork, path: str | Path) -> Path:
    """Write deterministic, canonical SNDlib 1.0 XML and return its path."""

    network.validate()
    ET.register_namespace("", SNDLIB_NAMESPACE)
    root = ET.Element(_qualified("network"), {"version": network.version})
    if network.metadata is not None:
        meta = ET.SubElement(root, _qualified("meta"))
        _text_child(meta, "granularity", network.metadata.granularity)
        _text_child(meta, "time", network.metadata.time)
        _text_child(meta, "unit", network.metadata.unit)
        _text_child(meta, "origin", network.metadata.origin)

    structure = ET.SubElement(root, _qualified("networkStructure"))
    nodes = ET.SubElement(
        structure,
        _qualified("nodes"),
        {"coordinatesType": network.coordinates_type},
    )
    for node in network.nodes:
        element = ET.SubElement(nodes, _qualified("node"), {"id": node.id})
        coordinates = ET.SubElement(element, _qualified("coordinates"))
        _text_child(coordinates, "x", _format_number(node.x))
        _text_child(coordinates, "y", _format_number(node.y))

    links = ET.SubElement(structure, _qualified("links"))
    for link in network.links:
        element = ET.SubElement(links, _qualified("link"), {"id": link.id})
        _text_child(element, "source", link.source)
        _text_child(element, "target", link.target)
        if link.routing_cost is not None:
            _text_child(element, "routingCost", _format_number(link.routing_cost))
        if link.setup_cost is not None:
            _text_child(element, "setupCost", _format_number(link.setup_cost))
        if link.preinstalled_module is not None:
            module = ET.SubElement(element, _qualified("preInstalledModule"))
            _write_module(module, link.preinstalled_module)
        if link.additional_modules:
            additional = ET.SubElement(element, _qualified("additionalModules"))
            for value in link.additional_modules:
                module = ET.SubElement(additional, _qualified("addModule"))
                _write_module(module, value)

    demands = ET.SubElement(root, _qualified("demands"))
    for demand in network.demands:
        element = ET.SubElement(demands, _qualified("demand"), {"id": demand.id})
        _text_child(element, "source", demand.source)
        _text_child(element, "target", demand.target)
        _text_child(element, "demandValue", _format_number(demand.value))
        if demand.max_path_length is not None:
            _text_child(element, "maxPathLength", str(demand.max_path_length))
        if demand.admissible_paths:
            paths = ET.SubElement(element, _qualified("admissiblePaths"))
            for path_value in demand.admissible_paths:
                path_element = ET.SubElement(
                    paths,
                    _qualified("admissiblePath"),
                    {"id": path_value.id},
                )
                for link_id in path_value.link_ids:
                    _text_child(path_element, "linkId", link_id)

    ET.indent(root, space=" ")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = ET.tostring(
        root,
        encoding="ISO-8859-1",
        xml_declaration=True,
        short_empty_elements=False,
    ) + b"\n"
    try:
        destination.write_bytes(payload)
    except OSError as error:
        raise SndlibFormatError(f"cannot write {destination}: {error}") from error
    return destination


def _parse_metadata(element: ET.Element) -> SndlibMetadata:
    _expect_attributes(element, set())
    _expect_children(element, {"granularity", "time", "unit", "origin"})
    return SndlibMetadata(
        granularity=_required_text(element, "granularity"),
        time=_required_text(element, "time"),
        unit=_required_text(element, "unit"),
        origin=_required_text(element, "origin"),
    )


def _parse_node(element: ET.Element) -> SndlibNode:
    _expect_attributes(element, {"id"})
    _expect_children(element, {"coordinates"})
    coordinates = _required_child(element, "coordinates")
    _expect_attributes(coordinates, set())
    _expect_children(coordinates, {"x", "y"})
    return SndlibNode(
        id=_required_attribute(element, "id"),
        x=_required_float(coordinates, "x"),
        y=_required_float(coordinates, "y"),
    )


def _parse_link(element: ET.Element) -> SndlibLink:
    _expect_attributes(element, {"id"})
    _expect_children(
        element,
        {
            "source",
            "target",
            "routingCost",
            "setupCost",
            "preInstalledModule",
            "additionalModules",
        },
    )
    preinstalled = _optional_child(element, "preInstalledModule")
    additional = _optional_child(element, "additionalModules")
    routing_cost = _optional_float(element, "routingCost")
    setup_cost = _optional_float(element, "setupCost")
    modules: tuple[SndlibCapacityModule, ...] = ()
    if additional is not None:
        _expect_attributes(additional, set())
        _expect_children(additional, {"addModule"})
        module_elements = _children(additional, "addModule")
        if not module_elements:
            raise SndlibFormatError("additionalModules must contain an addModule")
        modules = tuple(_parse_module(value) for value in module_elements)
    return SndlibLink(
        id=_required_attribute(element, "id"),
        source=_required_text(element, "source"),
        target=_required_text(element, "target"),
        preinstalled_module=None if preinstalled is None else _parse_module(preinstalled),
        additional_modules=modules,
        routing_cost=routing_cost,
        setup_cost=setup_cost,
    )


def _parse_module(element: ET.Element) -> SndlibCapacityModule:
    _expect_attributes(element, set())
    _expect_children(element, {"capacity", "cost"})
    return SndlibCapacityModule(
        capacity=_required_float(element, "capacity"),
        cost=_required_float(element, "cost"),
    )


def _parse_demand(element: ET.Element) -> SndlibDemand:
    _expect_attributes(element, {"id"})
    _expect_children(
        element,
        {"source", "target", "demandValue", "maxPathLength", "admissiblePaths"},
    )
    paths_element = _optional_child(element, "admissiblePaths")
    paths: tuple[SndlibAdmissiblePath, ...] = ()
    if paths_element is not None:
        _expect_attributes(paths_element, set())
        _expect_children(paths_element, {"admissiblePath"})
        path_elements = _children(paths_element, "admissiblePath")
        if not path_elements:
            raise SndlibFormatError("admissiblePaths must contain an admissiblePath")
        paths = tuple(_parse_admissible_path(value) for value in path_elements)
    return SndlibDemand(
        id=_required_attribute(element, "id"),
        source=_required_text(element, "source"),
        target=_required_text(element, "target"),
        value=_required_float(element, "demandValue"),
        max_path_length=_optional_positive_integer(element, "maxPathLength"),
        admissible_paths=paths,
    )


def _parse_admissible_path(element: ET.Element) -> SndlibAdmissiblePath:
    _expect_attributes(element, {"id"})
    _expect_children(element, {"linkId"})
    return SndlibAdmissiblePath(
        id=_required_attribute(element, "id"),
        link_ids=tuple(_leaf_text(value) for value in _children(element, "linkId")),
    )


def _validate_walk(
    demand: SndlibDemand,
    path: SndlibAdmissiblePath,
    links: dict[str, SndlibLink],
) -> None:
    current = demand.source
    for link_id in path.link_ids:
        link = links[link_id]
        if current == link.source:
            current = link.target
        elif current == link.target:
            current = link.source
        else:
            raise SndlibFormatError(
                f"path {path.id!r} of demand {demand.id!r} is not contiguous at "
                f"link {link_id!r}"
            )
    if current != demand.target:
        raise SndlibFormatError(
            f"path {path.id!r} of demand {demand.id!r} ends at {current!r}, "
            f"not {demand.target!r}"
        )


def _qualified(local_name: str) -> str:
    return f"{{{SNDLIB_NAMESPACE}}}{local_name}"


def _expect_tag(element: ET.Element, local_name: str) -> None:
    expected = _qualified(local_name)
    if element.tag != expected:
        raise SndlibFormatError(
            f"expected SNDlib element {local_name!r} in namespace "
            f"{SNDLIB_NAMESPACE!r}, got {element.tag!r}"
        )


def _expect_attributes(element: ET.Element, allowed: set[str]) -> None:
    unknown = set(element.attrib) - allowed
    if unknown:
        raise SndlibFormatError(
            f"unsupported attribute(s) on {_local_name(element)}: {_preview(unknown)}"
        )


def _expect_children(element: ET.Element, allowed: set[str]) -> None:
    allowed_tags = {_qualified(name) for name in allowed}
    unknown = [child.tag for child in element if child.tag not in allowed_tags]
    if unknown:
        raise SndlibFormatError(
            f"unsupported child element(s) of {_local_name(element)}: {_preview(unknown)}"
        )


def _children(element: ET.Element, local_name: str) -> list[ET.Element]:
    return [child for child in element if child.tag == _qualified(local_name)]


def _required_child(element: ET.Element, local_name: str) -> ET.Element:
    matches = _children(element, local_name)
    if len(matches) != 1:
        raise SndlibFormatError(
            f"{_local_name(element)} must contain exactly one {local_name}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _optional_child(element: ET.Element, local_name: str) -> ET.Element | None:
    matches = _children(element, local_name)
    if len(matches) > 1:
        raise SndlibFormatError(
            f"{_local_name(element)} contains duplicate {local_name} elements"
        )
    return None if not matches else matches[0]


def _required_attribute(element: ET.Element, name: str) -> str:
    value = element.get(name)
    if value is None:
        raise SndlibFormatError(f"{_local_name(element)} is missing attribute {name!r}")
    _validate_text(f"{_local_name(element)} {name}", value)
    return value


def _required_text(element: ET.Element, local_name: str) -> str:
    return _leaf_text(_required_child(element, local_name))


def _leaf_text(element: ET.Element) -> str:
    _expect_attributes(element, set())
    if len(element):
        raise SndlibFormatError(f"{_local_name(element)} must contain only text")
    value = "" if element.text is None else element.text.strip()
    _validate_text(_local_name(element), value)
    return value


def _required_float(element: ET.Element, local_name: str) -> float:
    return _parse_number(_required_text(element, local_name), local_name)


def _optional_float(element: ET.Element, local_name: str) -> float | None:
    child = _optional_child(element, local_name)
    return None if child is None else _parse_number(_leaf_text(child), local_name)


def _optional_positive_integer(element: ET.Element, local_name: str) -> int | None:
    child = _optional_child(element, local_name)
    if child is None:
        return None
    text = _leaf_text(child)
    if not text.isdecimal() or int(text) <= 0:
        raise SndlibFormatError(f"{local_name} must be a positive integer, got {text!r}")
    return int(text)


def _parse_number(text: str, context: str) -> float:
    if _NUMBER.fullmatch(text) is None:
        raise SndlibFormatError(f"invalid numeric value {text!r} in {context}")
    value = float(text)
    _validate_finite(context, value)
    return value


def _write_module(parent: ET.Element, module: SndlibCapacityModule) -> None:
    _text_child(parent, "capacity", _format_number(module.capacity))
    _text_child(parent, "cost", _format_number(module.cost))


def _text_child(parent: ET.Element, local_name: str, value: str) -> None:
    child = ET.SubElement(parent, _qualified(local_name))
    child.text = value


def _format_number(value: float) -> str:
    return repr(float(value))


def _validate_text(context: str, value: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SndlibFormatError(f"{context} must be nonempty text without surrounding space")


def _validate_id(context: str, value: str) -> None:
    _validate_text(f"{context} identifier", value)


def _validate_finite(context: str, value: float) -> None:
    if not isfinite(value):
        raise SndlibFormatError(f"{context} must be finite")


def _validate_nonnegative(context: str, value: float) -> None:
    _validate_finite(context, value)
    if value < 0:
        raise SndlibFormatError(f"{context} must be nonnegative")


def _reject_duplicates(context: str, values: list[str]) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise SndlibFormatError(f"duplicate {context} identifier: {value!r}")
        seen.add(value)


def _preview(values: Iterable[str]) -> str:
    return ", ".join(repr(value) for value in sorted(values)[:8])


def _local_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]
