"""
Topology Analyzer for AEC Systems.

Graph-based analysis of MEP system connectivity, flow paths, and spatial
relationships. Builds topological graphs from classified geometry to understand:

1. System Connectivity - Which elements are connected
2. Flow Paths - Trace from source to terminal
3. Branches and Mains - Identify hierarchy in piping/ductwork
4. Spatial Relationships - Containment, adjacency

This is Phase D: Geometry Intelligence in the Semantic Intelligence Pipeline.

Usage:
    >>> analyzer = TopologyAnalyzer()
    >>> graph = await analyzer.build_system_graph(
    ...     lines=classified_lines,
    ...     symbols=smart_symbols,
    ... )
    >>> path = graph.find_path(valve_id, fixture_id)
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from aec_agent.mcp.tools.geometry_classifier import (
    ClassifiedLine,
    GeometrySystem,
    GeometryType,
)

logger = structlog.get_logger(__name__)


class NodeType(str, Enum):
    """Type of node in the topology graph."""
    # Equipment nodes
    EQUIPMENT = "equipment"
    VALVE = "valve"
    FITTING = "fitting"
    TERMINAL = "terminal"  # Diffuser, outlet, fixture
    SOURCE = "source"      # AHU, panel, main

    # Geometry nodes
    JUNCTION = "junction"      # Where lines meet
    BEND = "bend"              # Change in direction
    REDUCTION = "reduction"    # Size change
    TEE = "tee"                # Three-way junction
    CROSS = "cross"            # Four-way junction

    # Spatial nodes
    ROOM = "room"
    ZONE = "zone"
    FLOOR = "floor"

    UNKNOWN = "unknown"


class EdgeType(str, Enum):
    """Type of edge (connection) in the topology graph."""
    # Physical connections
    PIPE_RUN = "pipe_run"
    DUCT_RUN = "duct_run"
    CONDUIT_RUN = "conduit_run"
    WIRE_RUN = "wire_run"

    # Logical connections
    FEEDS = "feeds"             # Power/flow direction
    RETURNS_TO = "returns_to"   # Return path
    BRANCHES_FROM = "branches_from"
    CONNECTS_TO = "connects_to"

    # Spatial relationships
    CONTAINS = "contains"
    ADJACENT_TO = "adjacent_to"
    SERVES = "serves"
    ON_LEVEL = "on_level"

    UNKNOWN = "unknown"


@dataclass
class TopologyNode:
    """A node in the topology graph."""
    id: str
    node_type: NodeType
    position: tuple[float, float]
    system: GeometrySystem = GeometrySystem.UNKNOWN
    data: dict[str, Any] = field(default_factory=dict)
    # For symbols
    symbol_type: str | None = None
    block_name: str | None = None
    # For geometry
    geometry_type: GeometryType | None = None
    # Computed
    degree: int = 0  # Number of connections


@dataclass
class TopologyEdge:
    """An edge (connection) in the topology graph."""
    id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    system: GeometrySystem = GeometrySystem.UNKNOWN
    length: float = 0.0
    size: str | None = None  # Pipe/duct size
    flow_direction: str | None = None  # "forward", "reverse", None
    data: dict[str, Any] = field(default_factory=dict)
    # Original geometry
    geometry_indices: list[int] = field(default_factory=list)


@dataclass
class SystemPath:
    """A path through the system from source to terminal."""
    path_id: str
    nodes: list[str]  # List of node IDs
    edges: list[str]  # List of edge IDs
    system: GeometrySystem
    total_length: float
    source_node: str
    terminal_node: str
    branch_level: int = 0  # 0 = main, 1 = branch, 2 = sub-branch


@dataclass
class SystemBranch:
    """A branch in the system topology."""
    branch_id: str
    parent_id: str | None  # None for main
    child_ids: list[str]
    level: int
    nodes: list[str]
    edges: list[str]
    system: GeometrySystem


@dataclass
class TopologyGraph:
    """
    Graph representation of system topology.

    Provides methods for traversing, analyzing, and querying
    the connectivity of MEP systems.
    """
    nodes: dict[str, TopologyNode] = field(default_factory=dict)
    edges: dict[str, TopologyEdge] = field(default_factory=dict)
    adjacency: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    systems: dict[GeometrySystem, list[str]] = field(default_factory=lambda: defaultdict(list))

    def add_node(self, node: TopologyNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        if node.system != GeometrySystem.UNKNOWN:
            self.systems[node.system].append(node.id)

    def add_edge(self, edge: TopologyEdge) -> None:
        """Add an edge to the graph."""
        self.edges[edge.id] = edge
        self.adjacency[edge.source_id].append(edge.target_id)
        self.adjacency[edge.target_id].append(edge.source_id)

        # Update node degrees
        if edge.source_id in self.nodes:
            self.nodes[edge.source_id].degree += 1
        if edge.target_id in self.nodes:
            self.nodes[edge.target_id].degree += 1

    def get_neighbors(self, node_id: str) -> list[str]:
        """Get all nodes connected to the given node."""
        return self.adjacency.get(node_id, [])

    def get_edges_for_node(self, node_id: str) -> list[TopologyEdge]:
        """Get all edges connected to a node."""
        return [
            edge for edge in self.edges.values()
            if edge.source_id == node_id or edge.target_id == node_id
        ]

    def get_nodes_by_type(self, node_type: NodeType) -> list[TopologyNode]:
        """Get all nodes of a specific type."""
        return [node for node in self.nodes.values() if node.node_type == node_type]

    def get_nodes_by_system(self, system: GeometrySystem) -> list[TopologyNode]:
        """Get all nodes in a specific system."""
        node_ids = self.systems.get(system, [])
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]

    def find_path(
        self,
        start_id: str,
        end_id: str,
        system: GeometrySystem | None = None,
    ) -> list[str] | None:
        """
        Find a path between two nodes using BFS.

        Args:
            start_id: Starting node ID
            end_id: Ending node ID
            system: Optional system filter

        Returns:
            List of node IDs forming the path, or None if no path exists
        """
        if start_id not in self.nodes or end_id not in self.nodes:
            return None

        visited = {start_id}
        queue = [(start_id, [start_id])]

        while queue:
            current, path = queue.pop(0)

            if current == end_id:
                return path

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    # Filter by system if specified
                    if system and self.nodes[neighbor].system != system:
                        continue
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def find_all_paths(
        self,
        start_id: str,
        end_id: str,
        max_length: int = 50,
    ) -> list[list[str]]:
        """Find all paths between two nodes up to max_length."""
        if start_id not in self.nodes or end_id not in self.nodes:
            return []

        paths = []

        def dfs(current: str, path: list[str], visited: set[str]) -> None:
            if len(path) > max_length:
                return

            if current == end_id:
                paths.append(path.copy())
                return

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)

        dfs(start_id, [start_id], {start_id})
        return paths

    def find_connected_components(self) -> list[list[str]]:
        """Find all connected components in the graph."""
        visited: set[str] = set()
        components: list[list[str]] = []

        for node_id in self.nodes:
            if node_id not in visited:
                component: list[str] = []
                stack = [node_id]

                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)
                        stack.extend(
                            n for n in self.get_neighbors(current)
                            if n not in visited
                        )

                components.append(component)

        return components

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the graph."""
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "node_types": {
                t.value: len([n for n in self.nodes.values() if n.node_type == t])
                for t in NodeType
            },
            "systems": {
                s.value: len(nodes) for s, nodes in self.systems.items()
            },
            "connected_components": len(self.find_connected_components()),
            "average_degree": (
                sum(n.degree for n in self.nodes.values()) / len(self.nodes)
                if self.nodes else 0
            ),
        }


class TopologyAnalyzer:
    """
    Analyzer for building and querying system topology.

    Builds a graph from classified geometry and symbols, then provides
    methods for analyzing connectivity, finding paths, and identifying
    system structure.

    Args:
        proximity_tolerance: Distance tolerance for connecting elements
        min_edge_length: Minimum edge length to consider

    Example:
        >>> analyzer = TopologyAnalyzer()
        >>> graph = await analyzer.build_system_graph(
        ...     lines=classified_lines,
        ...     symbols=smart_symbols,
        ... )
        >>> sources = graph.get_nodes_by_type(NodeType.SOURCE)
        >>> terminals = graph.get_nodes_by_type(NodeType.TERMINAL)
        >>> for source in sources:
        ...     for terminal in terminals:
        ...         path = graph.find_path(source.id, terminal.id)
        ...         if path:
        ...             print(f"Path: {' -> '.join(path)}")
    """

    def __init__(
        self,
        proximity_tolerance: float = 10.0,
        min_edge_length: float = 5.0,
    ):
        self.proximity_tolerance = proximity_tolerance
        self.min_edge_length = min_edge_length
        self._node_counter = 0
        self._edge_counter = 0

    def _generate_node_id(self) -> str:
        """Generate a unique node ID."""
        self._node_counter += 1
        return f"node_{self._node_counter}"

    def _generate_edge_id(self) -> str:
        """Generate a unique edge ID."""
        self._edge_counter += 1
        return f"edge_{self._edge_counter}"

    async def build_system_graph(
        self,
        lines: list[ClassifiedLine] | None = None,
        symbols: list[Any] | None = None,  # List[SmartSymbol]
        rooms: list[Any] | None = None,  # List[Room] or similar
    ) -> TopologyGraph:
        """
        Build a topology graph from classified elements.

        Args:
            lines: Classified lines (pipes, ducts, conduits)
            symbols: Smart symbols with classification
            rooms: Room boundaries for containment

        Returns:
            TopologyGraph with nodes and edges
        """
        lines = lines or []
        symbols = symbols or []
        rooms = rooms or []

        graph = TopologyGraph()

        logger.info(
            "Building topology graph",
            lines=len(lines),
            symbols=len(symbols),
            rooms=len(rooms),
        )

        # Step 1: Create nodes from symbols
        symbol_nodes = self._create_symbol_nodes(symbols)
        for node in symbol_nodes:
            graph.add_node(node)

        # Step 2: Create junction nodes at line intersections
        junction_nodes = self._create_junction_nodes(lines)
        for node in junction_nodes:
            graph.add_node(node)

        # Step 3: Create edges from lines
        edges = self._create_edges_from_lines(lines, graph)
        for edge in edges:
            graph.add_edge(edge)

        # Step 4: Connect symbols to nearby junctions
        connection_edges = self._connect_symbols_to_graph(symbol_nodes, graph)
        for edge in connection_edges:
            graph.add_edge(edge)

        # Step 5: Add room containment relationships
        if rooms:
            containment_edges = self._create_containment_edges(rooms, graph)
            for edge in containment_edges:
                graph.add_edge(edge)

        logger.info(
            "Topology graph built",
            nodes=len(graph.nodes),
            edges=len(graph.edges),
            components=len(graph.find_connected_components()),
        )

        return graph

    def _create_symbol_nodes(
        self,
        symbols: list[Any],
    ) -> list[TopologyNode]:
        """Create nodes from smart symbols."""
        nodes = []

        for i, symbol in enumerate(symbols):
            # Determine node type from symbol
            node_type = self._symbol_to_node_type(symbol)
            system = self._symbol_to_system(symbol)

            position = (0.0, 0.0)
            if hasattr(symbol, 'position'):
                if isinstance(symbol.position, tuple):
                    position = symbol.position
                elif hasattr(symbol.position, 'x') and hasattr(symbol.position, 'y'):
                    position = (symbol.position.x, symbol.position.y)

            node = TopologyNode(
                id=f"symbol_{i}",
                node_type=node_type,
                position=position,
                system=system,
                symbol_type=getattr(symbol, 'type', None),
                block_name=getattr(symbol, 'block_name', None),
                data={
                    "subtype": getattr(symbol, 'subtype', None),
                    "size": getattr(symbol, 'size', None),
                },
            )
            nodes.append(node)

        return nodes

    def _symbol_to_node_type(self, symbol: Any) -> NodeType:
        """Determine node type from symbol classification."""
        sym_type = getattr(symbol, 'type', '').lower()
        sym_category = getattr(symbol, 'category', '').lower()

        # Valves
        if 'valve' in sym_type:
            return NodeType.VALVE

        # Fittings
        if any(f in sym_type for f in ['fitting', 'elbow', 'tee', 'coupling']):
            return NodeType.FITTING

        # Terminals
        if any(t in sym_type for t in ['diffuser', 'grille', 'register', 'outlet', 'fixture']):
            return NodeType.TERMINAL

        # Sources
        if any(s in sym_type for s in ['ahu', 'panel', 'pump', 'boiler', 'chiller']):
            return NodeType.SOURCE

        # Equipment
        if sym_category in ['mechanical', 'electrical', 'plumbing']:
            return NodeType.EQUIPMENT

        return NodeType.UNKNOWN

    def _symbol_to_system(self, symbol: Any) -> GeometrySystem:
        """Determine system from symbol classification."""
        sym_category = getattr(symbol, 'category', '').lower()
        sym_system = getattr(symbol, 'system', '').lower()

        # Check explicit system
        if sym_system:
            for system in GeometrySystem:
                if system.value in sym_system:
                    return system

        # Infer from category
        if sym_category == 'plumbing':
            return GeometrySystem.DOMESTIC_COLD  # Default plumbing
        if sym_category == 'mechanical':
            return GeometrySystem.SUPPLY_AIR  # Default HVAC
        if sym_category == 'electrical':
            return GeometrySystem.POWER
        if sym_category == 'fire':
            return GeometrySystem.FIRE_ALARM

        return GeometrySystem.UNKNOWN

    def _create_junction_nodes(
        self,
        lines: list[ClassifiedLine],
    ) -> list[TopologyNode]:
        """Create junction nodes at line intersections."""
        nodes = []
        endpoints: dict[tuple[int, int], list[int]] = defaultdict(list)

        # Bucket endpoints by rounded position
        for i, line in enumerate(lines):
            start_key = (round(line.start[0]), round(line.start[1]))
            end_key = (round(line.end[0]), round(line.end[1]))
            endpoints[start_key].append(i)
            endpoints[end_key].append(i)

        # Create junction nodes where multiple lines meet
        for pos_key, line_indices in endpoints.items():
            if len(line_indices) >= 2:
                # Determine junction type by number of connections
                if len(line_indices) == 3:
                    junction_type = NodeType.TEE
                elif len(line_indices) >= 4:
                    junction_type = NodeType.CROSS
                else:
                    junction_type = NodeType.JUNCTION

                # Determine system from connected lines
                systems = [
                    lines[i].classification.system
                    for i in line_indices
                    if lines[i].classification.system != GeometrySystem.UNKNOWN
                ]
                system = systems[0] if systems else GeometrySystem.UNKNOWN

                node = TopologyNode(
                    id=self._generate_node_id(),
                    node_type=junction_type,
                    position=(float(pos_key[0]), float(pos_key[1])),
                    system=system,
                    data={"connected_lines": line_indices},
                )
                nodes.append(node)

        return nodes

    def _create_edges_from_lines(
        self,
        lines: list[ClassifiedLine],
        graph: TopologyGraph,
    ) -> list[TopologyEdge]:
        """Create edges from classified lines."""
        edges = []

        for i, line in enumerate(lines):
            # Skip non-system lines
            if line.classification.geometry_type in [
                GeometryType.DIMENSION_LINE,
                GeometryType.LEADER,
                GeometryType.UNKNOWN,
            ]:
                continue

            # Find nodes at endpoints
            start_node = self._find_node_at_position(line.start, graph)
            end_node = self._find_node_at_position(line.end, graph)

            if not start_node:
                # Create a new node
                start_node = TopologyNode(
                    id=self._generate_node_id(),
                    node_type=NodeType.JUNCTION,
                    position=line.start,
                    system=line.classification.system,
                )
                graph.add_node(start_node)

            if not end_node:
                end_node = TopologyNode(
                    id=self._generate_node_id(),
                    node_type=NodeType.JUNCTION,
                    position=line.end,
                    system=line.classification.system,
                )
                graph.add_node(end_node)

            # Determine edge type from geometry type
            edge_type = self._geometry_to_edge_type(line.classification.geometry_type)

            length = self._distance(line.start, line.end)

            edge = TopologyEdge(
                id=self._generate_edge_id(),
                source_id=start_node.id,
                target_id=end_node.id,
                edge_type=edge_type,
                system=line.classification.system,
                length=length,
                size=line.classification.size,
                geometry_indices=[i],
            )
            edges.append(edge)

        return edges

    def _geometry_to_edge_type(self, geometry_type: GeometryType) -> EdgeType:
        """Convert geometry type to edge type."""
        mapping = {
            GeometryType.PIPE: EdgeType.PIPE_RUN,
            GeometryType.DUCT: EdgeType.DUCT_RUN,
            GeometryType.DUCT_RECTANGULAR: EdgeType.DUCT_RUN,
            GeometryType.DUCT_ROUND: EdgeType.DUCT_RUN,
            GeometryType.CONDUIT: EdgeType.CONDUIT_RUN,
            GeometryType.WIRE_RUN: EdgeType.WIRE_RUN,
        }
        return mapping.get(geometry_type, EdgeType.CONNECTS_TO)

    def _find_node_at_position(
        self,
        position: tuple[float, float],
        graph: TopologyGraph,
    ) -> TopologyNode | None:
        """Find a node near the given position."""
        for node in graph.nodes.values():
            if self._distance(position, node.position) < self.proximity_tolerance:
                return node
        return None

    def _connect_symbols_to_graph(
        self,
        symbol_nodes: list[TopologyNode],
        graph: TopologyGraph,
    ) -> list[TopologyEdge]:
        """Connect symbol nodes to nearby junction nodes."""
        edges = []

        for symbol_node in symbol_nodes:
            # Find nearest junction node
            nearest_junction = None
            nearest_dist = float('inf')

            for node in graph.nodes.values():
                if node.node_type in [NodeType.JUNCTION, NodeType.TEE, NodeType.CROSS]:
                    dist = self._distance(symbol_node.position, node.position)
                    if dist < nearest_dist and dist < self.proximity_tolerance * 3:
                        nearest_dist = dist
                        nearest_junction = node

            if nearest_junction:
                edge = TopologyEdge(
                    id=self._generate_edge_id(),
                    source_id=symbol_node.id,
                    target_id=nearest_junction.id,
                    edge_type=EdgeType.CONNECTS_TO,
                    system=symbol_node.system,
                    length=nearest_dist,
                )
                edges.append(edge)

        return edges

    def _create_containment_edges(
        self,
        rooms: list[Any],
        graph: TopologyGraph,
    ) -> list[TopologyEdge]:
        """Create containment edges for equipment in rooms."""
        edges = []

        for i, room in enumerate(rooms):
            room_node = TopologyNode(
                id=f"room_{i}",
                node_type=NodeType.ROOM,
                position=self._get_room_center(room),
                data={"room_name": getattr(room, 'name', None)},
            )
            graph.add_node(room_node)

            # Find equipment/terminals inside this room
            for node in graph.nodes.values():
                if node.node_type in [NodeType.EQUIPMENT, NodeType.TERMINAL]:
                    if self._point_in_room(node.position, room):
                        edge = TopologyEdge(
                            id=self._generate_edge_id(),
                            source_id=room_node.id,
                            target_id=node.id,
                            edge_type=EdgeType.CONTAINS,
                        )
                        edges.append(edge)

        return edges

    def _get_room_center(self, room: Any) -> tuple[float, float]:
        """Get the center point of a room."""
        if hasattr(room, 'center'):
            return room.center
        if hasattr(room, 'boundary') and room.boundary:
            points = room.boundary
            x = sum(p[0] for p in points) / len(points)
            y = sum(p[1] for p in points) / len(points)
            return (x, y)
        return (0.0, 0.0)

    def _point_in_room(
        self,
        point: tuple[float, float],
        room: Any,
    ) -> bool:
        """Check if a point is inside a room boundary (simplified)."""
        if not hasattr(room, 'boundary') or not room.boundary:
            return False

        # Simple bounding box check
        boundary = room.boundary
        min_x = min(p[0] for p in boundary)
        max_x = max(p[0] for p in boundary)
        min_y = min(p[1] for p in boundary)
        max_y = max(p[1] for p in boundary)

        return min_x <= point[0] <= max_x and min_y <= point[1] <= max_y

    def _distance(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
    ) -> float:
        """Calculate distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    async def find_system_paths(
        self,
        graph: TopologyGraph,
        system: GeometrySystem | None = None,
    ) -> list[SystemPath]:
        """
        Find all paths from sources to terminals in a system.

        Args:
            graph: The topology graph
            system: Optional system filter

        Returns:
            List of SystemPath objects
        """
        paths = []
        sources = graph.get_nodes_by_type(NodeType.SOURCE)
        terminals = graph.get_nodes_by_type(NodeType.TERMINAL)

        # Filter by system if specified
        if system:
            sources = [s for s in sources if s.system == system]
            terminals = [t for t in terminals if t.system == system]

        path_counter = 0
        for source in sources:
            for terminal in terminals:
                node_path = graph.find_path(source.id, terminal.id)
                if node_path:
                    # Calculate path details
                    edge_ids = self._get_edge_ids_for_path(node_path, graph)
                    total_length = sum(
                        graph.edges[eid].length for eid in edge_ids
                        if eid in graph.edges
                    )

                    path_counter += 1
                    paths.append(
                        SystemPath(
                            path_id=f"path_{path_counter}",
                            nodes=node_path,
                            edges=edge_ids,
                            system=source.system,
                            total_length=total_length,
                            source_node=source.id,
                            terminal_node=terminal.id,
                        )
                    )

        return paths

    def _get_edge_ids_for_path(
        self,
        node_path: list[str],
        graph: TopologyGraph,
    ) -> list[str]:
        """Get edge IDs for a node path."""
        edge_ids = []
        for i in range(len(node_path) - 1):
            source = node_path[i]
            target = node_path[i + 1]
            for edge in graph.edges.values():
                if (edge.source_id == source and edge.target_id == target) or \
                   (edge.source_id == target and edge.target_id == source):
                    edge_ids.append(edge.id)
                    break
        return edge_ids

    async def identify_branches(
        self,
        graph: TopologyGraph,
        system: GeometrySystem | None = None,
    ) -> list[SystemBranch]:
        """
        Identify branch structure in a system.

        Traces from sources and identifies main runs vs branches.

        Args:
            graph: The topology graph
            system: Optional system filter

        Returns:
            List of SystemBranch objects
        """
        branches = []
        visited_edges: set[str] = set()

        sources = graph.get_nodes_by_type(NodeType.SOURCE)
        if system:
            sources = [s for s in sources if s.system == system]

        branch_counter = 0

        for source in sources:
            # BFS from source to identify main and branches
            queue = [(source.id, None, 0)]  # (node_id, parent_branch_id, level)

            while queue:
                current_id, parent_branch_id, level = queue.pop(0)
                current = graph.nodes.get(current_id)
                if not current:
                    continue

                edges = graph.get_edges_for_node(current_id)
                unvisited_edges = [e for e in edges if e.id not in visited_edges]

                if current.degree >= 3 and level > 0:
                    # This is a branch point - create branches
                    for edge in unvisited_edges:
                        visited_edges.add(edge.id)
                        branch_counter += 1

                        next_node = (
                            edge.target_id
                            if edge.source_id == current_id
                            else edge.source_id
                        )

                        branch = SystemBranch(
                            branch_id=f"branch_{branch_counter}",
                            parent_id=parent_branch_id,
                            child_ids=[],
                            level=level,
                            nodes=[current_id, next_node],
                            edges=[edge.id],
                            system=current.system,
                        )
                        branches.append(branch)

                        queue.append((next_node, branch.branch_id, level + 1))
                else:
                    # Continue on main/current branch
                    for edge in unvisited_edges:
                        visited_edges.add(edge.id)
                        next_node = (
                            edge.target_id
                            if edge.source_id == current_id
                            else edge.source_id
                        )
                        queue.append((next_node, parent_branch_id, level))

        return branches

    async def analyze_connectivity(
        self,
        graph: TopologyGraph,
    ) -> dict[str, Any]:
        """
        Analyze system connectivity and return statistics.

        Args:
            graph: The topology graph

        Returns:
            Dictionary with connectivity analysis
        """
        components = graph.find_connected_components()
        stats = graph.get_statistics()

        # Find isolated nodes (degree 0)
        isolated = [
            node.id for node in graph.nodes.values() if node.degree == 0
        ]

        # Find dead ends (degree 1, not terminals or sources)
        dead_ends = [
            node.id for node in graph.nodes.values()
            if node.degree == 1
            and node.node_type not in [NodeType.TERMINAL, NodeType.SOURCE]
        ]

        return {
            **stats,
            "connected_components": len(components),
            "largest_component_size": max(len(c) for c in components) if components else 0,
            "isolated_nodes": isolated,
            "dead_ends": dead_ends,
            "is_fully_connected": len(components) == 1,
        }


# Convenience functions
async def build_system_graph(
    lines: list[ClassifiedLine] | None = None,
    symbols: list[Any] | None = None,
    rooms: list[Any] | None = None,
    **analyzer_kwargs: Any,
) -> TopologyGraph:
    """
    Build a topology graph from classified elements.

    Convenience function that creates a TopologyAnalyzer and builds the graph.

    Args:
        lines: Classified lines (pipes, ducts, conduits)
        symbols: Smart symbols with classification
        rooms: Room boundaries for containment
        **analyzer_kwargs: Additional arguments for TopologyAnalyzer

    Returns:
        TopologyGraph with nodes and edges

    Example:
        >>> graph = await build_system_graph(
        ...     lines=classified_result.classified_lines,
        ...     symbols=vectorization_result.smart_symbols,
        ... )
        >>> print(f"Graph has {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    """
    analyzer = TopologyAnalyzer(**analyzer_kwargs)
    return await analyzer.build_system_graph(lines, symbols, rooms)


async def find_connected_equipment(
    graph: TopologyGraph,
    equipment_id: str,
    max_depth: int = 10,
) -> list[str]:
    """
    Find all equipment connected to a given piece of equipment.

    Args:
        graph: The topology graph
        equipment_id: ID of the starting equipment
        max_depth: Maximum search depth

    Returns:
        List of connected equipment node IDs
    """
    if equipment_id not in graph.nodes:
        return []

    connected: list[str] = []
    visited: set[str] = {equipment_id}
    queue = [(equipment_id, 0)]

    equipment_types = {NodeType.EQUIPMENT, NodeType.VALVE, NodeType.TERMINAL, NodeType.SOURCE}

    while queue:
        current_id, depth = queue.pop(0)

        if depth > max_depth:
            continue

        for neighbor_id in graph.get_neighbors(current_id):
            if neighbor_id not in visited:
                visited.add(neighbor_id)
                neighbor = graph.nodes.get(neighbor_id)
                if neighbor and neighbor.node_type in equipment_types:
                    connected.append(neighbor_id)
                queue.append((neighbor_id, depth + 1))

    return connected
