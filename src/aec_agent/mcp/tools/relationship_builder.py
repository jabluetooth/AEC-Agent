"""
Relationship Builder for AEC Elements.

Infers relationships between detected elements using spatial analysis,
connectivity detection, and topological reasoning. This is Phase E of
the Semantic Intelligence Pipeline.

Relationship Types:
1. CONTAINMENT - Room contains equipment (point-in-polygon)
2. CONNECTIVITY - Pipe connects to valve (endpoint proximity)
3. BRANCHES_FROM - Duct branches from main (topology analysis)
4. FEEDS - Panel feeds circuit (flow direction)
5. SERVES - Diffuser serves room (spatial proximity + containment)
6. ADJACENT_TO - Room adjacent to room (shared wall)
7. ON_LEVEL - Element on floor level

Usage:
    >>> builder = RelationshipBuilder()
    >>> relationships = await builder.build_relationships(
    ...     rooms=detected_rooms,
    ...     symbols=smart_symbols,
    ...     geometry=classified_geometry,
    ...     topology=system_topology,
    ... )
    >>> for rel in relationships:
    ...     print(f"{rel.source_id} --{rel.relation_type.value}--> {rel.target_id}")
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import structlog

from aec_agent.mcp.tools.geometry_classifier import (
    ClassifiedLine,
    GeometrySystem,
    GeometryType,
)
from aec_agent.mcp.tools.topology_analyzer import (
    TopologyGraph,
    NodeType,
    EdgeType,
)

logger = structlog.get_logger(__name__)


class RelationType(str, Enum):
    """Type of relationship between elements."""
    # Containment relationships
    CONTAINS = "contains"           # Room contains equipment
    CONTAINED_BY = "contained_by"   # Equipment is in room

    # Connectivity relationships
    CONNECTED_TO = "connected_to"   # Physical connection (pipe to valve)
    BRANCHES_FROM = "branches_from" # Branch from main run
    MERGES_INTO = "merges_into"     # Multiple paths merge

    # Flow relationships
    FEEDS = "feeds"                 # Upstream element feeds downstream
    FED_BY = "fed_by"               # Downstream element fed by upstream
    SUPPLIES = "supplies"           # Supply air/water to
    RETURNS_TO = "returns_to"       # Return path to source

    # Spatial relationships
    SERVES = "serves"               # Equipment serves room/area
    SERVED_BY = "served_by"         # Room served by equipment
    ADJACENT_TO = "adjacent_to"     # Shares boundary
    NEAR = "near"                   # Within proximity threshold
    ABOVE = "above"                 # Vertically above
    BELOW = "below"                 # Vertically below

    # Structural relationships
    ON_LEVEL = "on_level"           # On specific floor level
    HOSTS = "hosts"                 # Wall hosts door/window
    HOSTED_BY = "hosted_by"         # Door hosted by wall

    # Annotation relationships
    LABELS = "labels"               # Text labels element
    LABELED_BY = "labeled_by"       # Element labeled by text
    REFERENCES = "references"       # References another element
    REFERENCED_BY = "referenced_by" # Referenced by another


@dataclass
class InferredRelationship:
    """A relationship inferred between two elements."""
    source_id: str  # ID of source element
    target_id: str  # ID of target element
    relation_type: RelationType
    id: UUID = field(default_factory=uuid4)
    confidence: float = 1.0  # 0.0 - 1.0
    distance: float | None = None  # Distance in drawing units
    source_type: str = ""  # Type of source element
    target_type: str = ""  # Type of target element
    system: GeometrySystem = GeometrySystem.UNKNOWN
    bidirectional: bool = False  # True if relationship goes both ways
    reasoning: str = ""  # Explanation for relationship
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoomBoundary:
    """A room boundary for containment detection."""
    id: str
    name: str | None = None
    boundary: list[tuple[float, float]] = field(default_factory=list)
    center: tuple[float, float] | None = None
    area: float = 0.0
    level: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationshipBuilderResult:
    """Result from relationship building."""
    relationships: list[InferredRelationship] = field(default_factory=list)
    containment_relationships: list[InferredRelationship] = field(default_factory=list)
    connectivity_relationships: list[InferredRelationship] = field(default_factory=list)
    spatial_relationships: list[InferredRelationship] = field(default_factory=list)
    flow_relationships: list[InferredRelationship] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)


class RelationshipBuilder:
    """
    Builder for inferring relationships between AEC elements.

    Uses multiple strategies to detect relationships:
    - Point-in-polygon for containment
    - Endpoint proximity for connectivity
    - Graph traversal for flow paths
    - Spatial analysis for adjacency

    Args:
        proximity_tolerance: Distance tolerance for connectivity detection
        containment_margin: Margin for point-in-polygon tests
        adjacency_threshold: Max distance for adjacency relationships
        min_confidence: Minimum confidence to report relationship

    Example:
        >>> builder = RelationshipBuilder()
        >>> result = await builder.build_relationships(
        ...     rooms=[room1, room2],
        ...     symbols=smart_symbols,
        ...     geometry=classified_lines,
        ... )
        >>> print(f"Found {len(result.relationships)} relationships")
    """

    def __init__(
        self,
        proximity_tolerance: float = 15.0,
        containment_margin: float = 5.0,
        adjacency_threshold: float = 50.0,
        min_confidence: float = 0.5,
    ):
        self.proximity_tolerance = proximity_tolerance
        self.containment_margin = containment_margin
        self.adjacency_threshold = adjacency_threshold
        self.min_confidence = min_confidence

    async def build_relationships(
        self,
        rooms: list[RoomBoundary] | None = None,
        symbols: list[Any] | None = None,  # List[SmartSymbol]
        geometry: list[ClassifiedLine] | None = None,
        topology: TopologyGraph | None = None,
        texts: list[Any] | None = None,  # List[ParsedAnnotation]
    ) -> RelationshipBuilderResult:
        """
        Build relationships between all detected elements.

        Args:
            rooms: Room boundaries for containment detection
            symbols: Smart symbols with classification
            geometry: Classified geometry (pipes, ducts, etc.)
            topology: System topology graph
            texts: Parsed text annotations

        Returns:
            RelationshipBuilderResult with all inferred relationships
        """
        rooms = rooms or []
        symbols = symbols or []
        geometry = geometry or []
        texts = texts or []

        result = RelationshipBuilderResult()

        logger.info(
            "Building relationships",
            rooms=len(rooms),
            symbols=len(symbols),
            geometry=len(geometry),
            has_topology=topology is not None,
            texts=len(texts),
        )

        # Step 1: Containment relationships (room contains equipment)
        containment = await self._build_containment_relationships(rooms, symbols)
        result.containment_relationships = containment
        result.relationships.extend(containment)

        # Step 2: Connectivity relationships (endpoint proximity)
        connectivity = await self._build_connectivity_relationships(symbols, geometry)
        result.connectivity_relationships = connectivity
        result.relationships.extend(connectivity)

        # Step 3: Spatial relationships (adjacency, proximity)
        spatial = await self._build_spatial_relationships(rooms, symbols)
        result.spatial_relationships = spatial
        result.relationships.extend(spatial)

        # Step 4: Flow relationships (from topology graph)
        if topology:
            flow = await self._build_flow_relationships(topology, symbols)
            result.flow_relationships = flow
            result.relationships.extend(flow)

        # Step 5: Text labeling relationships
        if texts and symbols:
            labeling = await self._build_labeling_relationships(texts, symbols)
            result.relationships.extend(labeling)

        # Compute statistics
        result.statistics = self._compute_statistics(result)

        logger.info(
            "Relationship building complete",
            total=len(result.relationships),
            containment=len(result.containment_relationships),
            connectivity=len(result.connectivity_relationships),
            spatial=len(result.spatial_relationships),
            flow=len(result.flow_relationships),
        )

        return result

    async def _build_containment_relationships(
        self,
        rooms: list[RoomBoundary],
        symbols: list[Any],
    ) -> list[InferredRelationship]:
        """Build containment relationships using point-in-polygon tests."""
        relationships = []

        for room in rooms:
            if not room.boundary or len(room.boundary) < 3:
                continue

            for symbol in symbols:
                position = self._get_symbol_position(symbol)
                if position is None:
                    continue

                if self._point_in_polygon(position, room.boundary):
                    # Room contains this symbol
                    rel = InferredRelationship(
                        source_id=room.id,
                        target_id=self._get_symbol_id(symbol),
                        relation_type=RelationType.CONTAINS,
                        confidence=0.95,
                        source_type="room",
                        target_type=getattr(symbol, 'type', 'symbol'),
                        reasoning=f"Symbol at {position} is inside room boundary",
                        metadata={
                            "room_name": room.name,
                            "symbol_position": position,
                        },
                    )
                    relationships.append(rel)

                    # Also add SERVES relationship for terminals
                    if hasattr(symbol, 'type') and symbol.type in [
                        'diffuser', 'grille', 'register', 'outlet', 'light', 'detector'
                    ]:
                        serves_rel = InferredRelationship(
                            source_id=self._get_symbol_id(symbol),
                            target_id=room.id,
                            relation_type=RelationType.SERVES,
                            confidence=0.90,
                            source_type=getattr(symbol, 'type', 'terminal'),
                            target_type="room",
                            reasoning=f"Terminal device inside room typically serves that room",
                        )
                        relationships.append(serves_rel)

        return relationships

    async def _build_connectivity_relationships(
        self,
        symbols: list[Any],
        geometry: list[ClassifiedLine],
    ) -> list[InferredRelationship]:
        """Build connectivity relationships based on endpoint proximity."""
        relationships = []

        # Get symbol positions
        symbol_positions: list[tuple[str, tuple[float, float], str, Any]] = []
        for symbol in symbols:
            pos = self._get_symbol_position(symbol)
            if pos:
                symbol_positions.append((
                    self._get_symbol_id(symbol),
                    pos,
                    getattr(symbol, 'type', 'symbol'),
                    symbol,
                ))

        # Check geometry endpoints against symbol positions
        for i, line in enumerate(geometry):
            if line.classification.geometry_type == GeometryType.UNKNOWN:
                continue

            line_id = f"line_{i}"
            line_type = line.classification.geometry_type.value

            # Check both endpoints
            for endpoint in [line.start, line.end]:
                for sym_id, sym_pos, sym_type, symbol in symbol_positions:
                    dist = self._distance(endpoint, sym_pos)

                    if dist < self.proximity_tolerance:
                        # Line connects to symbol
                        rel = InferredRelationship(
                            source_id=line_id,
                            target_id=sym_id,
                            relation_type=RelationType.CONNECTED_TO,
                            confidence=max(0.5, 1.0 - (dist / self.proximity_tolerance)),
                            distance=dist,
                            source_type=line_type,
                            target_type=sym_type,
                            system=line.classification.system,
                            bidirectional=True,
                            reasoning=f"Line endpoint within {dist:.1f} units of symbol",
                        )
                        relationships.append(rel)

        # Check symbol-to-symbol connections through geometry
        # Symbols connected by the same line segment are CONNECTED_TO each other
        connected_pairs: set[tuple[str, str]] = set()
        for line in geometry:
            start_symbols = []
            end_symbols = []

            for sym_id, sym_pos, sym_type, _ in symbol_positions:
                if self._distance(line.start, sym_pos) < self.proximity_tolerance:
                    start_symbols.append(sym_id)
                if self._distance(line.end, sym_pos) < self.proximity_tolerance:
                    end_symbols.append(sym_id)

            # Symbols at opposite ends are connected
            for start_sym in start_symbols:
                for end_sym in end_symbols:
                    if start_sym != end_sym:
                        pair = tuple(sorted([start_sym, end_sym]))
                        if pair not in connected_pairs:
                            connected_pairs.add(pair)
                            rel = InferredRelationship(
                                source_id=start_sym,
                                target_id=end_sym,
                                relation_type=RelationType.CONNECTED_TO,
                                confidence=0.85,
                                distance=self._line_length(line),
                                source_type="symbol",
                                target_type="symbol",
                                system=line.classification.system,
                                bidirectional=True,
                                reasoning="Symbols connected by classified line segment",
                            )
                            relationships.append(rel)

        return relationships

    async def _build_spatial_relationships(
        self,
        rooms: list[RoomBoundary],
        symbols: list[Any],
    ) -> list[InferredRelationship]:
        """Build spatial relationships (adjacency, proximity)."""
        relationships = []

        # Room adjacency (rooms sharing wall segments)
        for i, room1 in enumerate(rooms):
            for room2 in rooms[i + 1:]:
                if self._rooms_adjacent(room1, room2):
                    rel = InferredRelationship(
                        source_id=room1.id,
                        target_id=room2.id,
                        relation_type=RelationType.ADJACENT_TO,
                        confidence=0.90,
                        source_type="room",
                        target_type="room",
                        bidirectional=True,
                        reasoning="Rooms share wall segment",
                        metadata={
                            "room1_name": room1.name,
                            "room2_name": room2.name,
                        },
                    )
                    relationships.append(rel)

        # Symbol proximity (symbols near each other)
        symbol_positions = [
            (self._get_symbol_id(s), self._get_symbol_position(s), s)
            for s in symbols
        ]

        for i, (id1, pos1, sym1) in enumerate(symbol_positions):
            if pos1 is None:
                continue
            for id2, pos2, sym2 in symbol_positions[i + 1:]:
                if pos2 is None:
                    continue

                dist = self._distance(pos1, pos2)
                if dist < self.adjacency_threshold and dist > 0:
                    rel = InferredRelationship(
                        source_id=id1,
                        target_id=id2,
                        relation_type=RelationType.NEAR,
                        confidence=max(0.5, 1.0 - (dist / self.adjacency_threshold)),
                        distance=dist,
                        source_type=getattr(sym1, 'type', 'symbol'),
                        target_type=getattr(sym2, 'type', 'symbol'),
                        bidirectional=True,
                        reasoning=f"Symbols within {dist:.1f} units",
                    )
                    relationships.append(rel)

        return relationships

    async def _build_flow_relationships(
        self,
        topology: TopologyGraph,
        symbols: list[Any],
    ) -> list[InferredRelationship]:
        """Build flow relationships from topology graph."""
        relationships = []

        # Find source nodes (AHU, panel, pump, etc.)
        sources = topology.get_nodes_by_type(NodeType.SOURCE)

        # Find terminal nodes (diffuser, outlet, fixture)
        terminals = topology.get_nodes_by_type(NodeType.TERMINAL)

        # Build FEEDS relationships from sources through the graph
        for source in sources:
            # Find all reachable terminals
            for terminal in terminals:
                path = topology.find_path(source.id, terminal.id)
                if path and len(path) > 1:
                    rel = InferredRelationship(
                        source_id=source.id,
                        target_id=terminal.id,
                        relation_type=RelationType.FEEDS,
                        confidence=0.85,
                        source_type=source.symbol_type or "source",
                        target_type=terminal.symbol_type or "terminal",
                        system=source.system,
                        reasoning=f"Path exists from source to terminal ({len(path)} nodes)",
                        metadata={
                            "path": path,
                            "path_length": len(path),
                        },
                    )
                    relationships.append(rel)

        # Build BRANCHES_FROM relationships for tees
        tees = topology.get_nodes_by_type(NodeType.TEE)
        for tee in tees:
            edges = topology.get_edges_for_node(tee.id)
            if len(edges) >= 3:
                # Find the main (longest/largest) edge
                main_edge = max(edges, key=lambda e: e.length)
                branch_edges = [e for e in edges if e.id != main_edge.id]

                for branch in branch_edges:
                    # Determine direction (which node is the branch)
                    branch_node_id = (
                        branch.target_id if branch.source_id == tee.id
                        else branch.source_id
                    )
                    main_node_id = (
                        main_edge.target_id if main_edge.source_id == tee.id
                        else main_edge.source_id
                    )

                    rel = InferredRelationship(
                        source_id=branch_node_id,
                        target_id=main_node_id,
                        relation_type=RelationType.BRANCHES_FROM,
                        confidence=0.75,
                        source_type="branch",
                        target_type="main",
                        system=tee.system,
                        reasoning="Branch detected at tee junction",
                    )
                    relationships.append(rel)

        return relationships

    async def _build_labeling_relationships(
        self,
        texts: list[Any],
        symbols: list[Any],
    ) -> list[InferredRelationship]:
        """Build relationships between text labels and symbols."""
        relationships = []

        for text in texts:
            text_pos = self._get_text_position(text)
            if text_pos is None:
                continue

            # Find nearest symbol
            nearest_symbol = None
            nearest_dist = float('inf')

            for symbol in symbols:
                sym_pos = self._get_symbol_position(symbol)
                if sym_pos is None:
                    continue

                dist = self._distance(text_pos, sym_pos)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_symbol = symbol

            if nearest_symbol and nearest_dist < self.adjacency_threshold:
                text_id = getattr(text, 'id', f"text_{id(text)}")
                rel = InferredRelationship(
                    source_id=text_id,
                    target_id=self._get_symbol_id(nearest_symbol),
                    relation_type=RelationType.LABELS,
                    confidence=max(0.5, 1.0 - (nearest_dist / self.adjacency_threshold)),
                    distance=nearest_dist,
                    source_type="text",
                    target_type=getattr(nearest_symbol, 'type', 'symbol'),
                    reasoning=f"Text is nearest to symbol at {nearest_dist:.1f} units",
                    metadata={
                        "text_content": getattr(text, 'text', str(text)),
                    },
                )
                relationships.append(rel)

        return relationships

    def _point_in_polygon(
        self,
        point: tuple[float, float],
        polygon: list[tuple[float, float]],
    ) -> bool:
        """
        Check if a point is inside a polygon using ray casting algorithm.

        Args:
            point: (x, y) coordinates
            polygon: List of (x, y) vertices

        Returns:
            True if point is inside polygon
        """
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def _rooms_adjacent(
        self,
        room1: RoomBoundary,
        room2: RoomBoundary,
    ) -> bool:
        """Check if two rooms share a wall segment."""
        if not room1.boundary or not room2.boundary:
            return False

        # Check if any edge of room1 overlaps with any edge of room2
        for i in range(len(room1.boundary)):
            p1 = room1.boundary[i]
            p2 = room1.boundary[(i + 1) % len(room1.boundary)]

            for j in range(len(room2.boundary)):
                p3 = room2.boundary[j]
                p4 = room2.boundary[(j + 1) % len(room2.boundary)]

                # Check if edges are collinear and overlapping
                if self._edges_overlap(p1, p2, p3, p4):
                    return True

        return False

    def _edges_overlap(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
        p3: tuple[float, float],
        p4: tuple[float, float],
        tolerance: float = 5.0,
    ) -> bool:
        """Check if two edges are collinear and overlapping."""
        # Check if edges are parallel and close
        dx1, dy1 = p2[0] - p1[0], p2[1] - p1[1]
        dx2, dy2 = p4[0] - p3[0], p4[1] - p3[1]

        len1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
        len2 = math.sqrt(dx2 * dx2 + dy2 * dy2)

        if len1 < 1 or len2 < 1:
            return False

        # Normalize
        dx1, dy1 = dx1 / len1, dy1 / len1
        dx2, dy2 = dx2 / len2, dy2 / len2

        # Check if parallel (dot product of normals close to 0)
        cross = abs(dx1 * dy2 - dy1 * dx2)
        if cross > 0.1:  # Not parallel
            return False

        # Check if edges are close together (perpendicular distance)
        # Distance from p3 to line (p1, p2)
        dist = abs((p3[1] - p1[1]) * dx1 - (p3[0] - p1[0]) * dy1)
        if dist > tolerance:
            return False

        # Check if projections overlap
        proj1_start = p1[0] * dx1 + p1[1] * dy1
        proj1_end = p2[0] * dx1 + p2[1] * dy1
        proj2_start = p3[0] * dx1 + p3[1] * dy1
        proj2_end = p4[0] * dx1 + p4[1] * dy1

        if proj1_start > proj1_end:
            proj1_start, proj1_end = proj1_end, proj1_start
        if proj2_start > proj2_end:
            proj2_start, proj2_end = proj2_end, proj2_start

        overlap = min(proj1_end, proj2_end) - max(proj1_start, proj2_start)
        min_length = min(proj1_end - proj1_start, proj2_end - proj2_start)

        return overlap > min_length * 0.1  # At least 10% overlap

    def _get_symbol_position(self, symbol: Any) -> tuple[float, float] | None:
        """Get position from a symbol object."""
        if hasattr(symbol, 'position'):
            pos = symbol.position
            if isinstance(pos, tuple) and len(pos) >= 2:
                return (float(pos[0]), float(pos[1]))
            if hasattr(pos, 'x') and hasattr(pos, 'y'):
                return (float(pos.x), float(pos.y))
        if hasattr(symbol, 'center'):
            center = symbol.center
            if isinstance(center, tuple) and len(center) >= 2:
                return (float(center[0]), float(center[1]))
        if hasattr(symbol, 'x') and hasattr(symbol, 'y'):
            return (float(symbol.x), float(symbol.y))
        return None

    def _get_symbol_id(self, symbol: Any) -> str:
        """Get ID from a symbol object."""
        if hasattr(symbol, 'id'):
            return str(symbol.id)
        if hasattr(symbol, 'block_name'):
            return f"{symbol.block_name}_{id(symbol)}"
        return f"symbol_{id(symbol)}"

    def _get_text_position(self, text: Any) -> tuple[float, float] | None:
        """Get position from a text object."""
        if hasattr(text, 'position'):
            pos = text.position
            if isinstance(pos, tuple) and len(pos) >= 2:
                return (float(pos[0]), float(pos[1]))
            if hasattr(pos, 'x') and hasattr(pos, 'y'):
                return (float(pos.x), float(pos.y))
        if hasattr(text, 'center'):
            return text.center
        return None

    def _distance(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
    ) -> float:
        """Calculate distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _line_length(self, line: ClassifiedLine) -> float:
        """Calculate length of a classified line."""
        return self._distance(line.start, line.end)

    def _compute_statistics(
        self,
        result: RelationshipBuilderResult,
    ) -> dict[str, Any]:
        """Compute statistics about inferred relationships."""
        type_counts: dict[str, int] = {}
        system_counts: dict[str, int] = {}
        confidence_sum = 0.0

        for rel in result.relationships:
            rel_type = rel.relation_type.value
            type_counts[rel_type] = type_counts.get(rel_type, 0) + 1
            confidence_sum += rel.confidence

            if rel.system != GeometrySystem.UNKNOWN:
                system = rel.system.value
                system_counts[system] = system_counts.get(system, 0) + 1

        total = len(result.relationships)

        return {
            "total_relationships": total,
            "containment_count": len(result.containment_relationships),
            "connectivity_count": len(result.connectivity_relationships),
            "spatial_count": len(result.spatial_relationships),
            "flow_count": len(result.flow_relationships),
            "type_counts": type_counts,
            "system_counts": system_counts,
            "average_confidence": confidence_sum / total if total > 0 else 0,
            "bidirectional_count": sum(1 for r in result.relationships if r.bidirectional),
        }


# Convenience function
async def build_relationships(
    rooms: list[RoomBoundary] | None = None,
    symbols: list[Any] | None = None,
    geometry: list[ClassifiedLine] | None = None,
    topology: TopologyGraph | None = None,
    texts: list[Any] | None = None,
    **builder_kwargs: Any,
) -> RelationshipBuilderResult:
    """
    Build relationships between detected elements.

    Convenience function that creates a RelationshipBuilder and runs it.

    Args:
        rooms: Room boundaries for containment detection
        symbols: Smart symbols with classification
        geometry: Classified geometry (pipes, ducts, etc.)
        topology: System topology graph
        texts: Parsed text annotations
        **builder_kwargs: Additional arguments for RelationshipBuilder

    Returns:
        RelationshipBuilderResult with all inferred relationships

    Example:
        >>> result = await build_relationships(
        ...     rooms=detected_rooms,
        ...     symbols=smart_symbols,
        ...     geometry=classified_lines,
        ... )
        >>> for rel in result.containment_relationships:
        ...     print(f"Room {rel.source_id} contains {rel.target_id}")
    """
    builder = RelationshipBuilder(**builder_kwargs)
    return await builder.build_relationships(
        rooms=rooms,
        symbols=symbols,
        geometry=geometry,
        topology=topology,
        texts=texts,
    )


async def find_elements_in_room(
    room: RoomBoundary,
    symbols: list[Any],
) -> list[Any]:
    """
    Find all symbols contained within a room.

    Args:
        room: Room boundary
        symbols: List of symbols to check

    Returns:
        List of symbols inside the room
    """
    builder = RelationshipBuilder()
    contained = []

    for symbol in symbols:
        pos = builder._get_symbol_position(symbol)
        if pos and builder._point_in_polygon(pos, room.boundary):
            contained.append(symbol)

    return contained


async def find_connected_chain(
    start_symbol: Any,
    symbols: list[Any],
    geometry: list[ClassifiedLine],
    max_depth: int = 10,
) -> list[str]:
    """
    Find chain of connected symbols starting from a given symbol.

    Args:
        start_symbol: Starting symbol
        symbols: All symbols
        geometry: Classified geometry connecting symbols
        max_depth: Maximum chain length

    Returns:
        List of symbol IDs in the chain
    """
    builder = RelationshipBuilder()
    connectivity = await builder._build_connectivity_relationships(symbols, geometry)

    # Build adjacency map
    adjacency: dict[str, set[str]] = {}
    for rel in connectivity:
        if rel.relation_type == RelationType.CONNECTED_TO:
            if rel.source_id not in adjacency:
                adjacency[rel.source_id] = set()
            if rel.target_id not in adjacency:
                adjacency[rel.target_id] = set()
            adjacency[rel.source_id].add(rel.target_id)
            adjacency[rel.target_id].add(rel.source_id)

    # BFS to find chain
    start_id = builder._get_symbol_id(start_symbol)
    chain = [start_id]
    visited = {start_id}
    queue = [(start_id, 0)]

    while queue and len(chain) < max_depth:
        current, depth = queue.pop(0)
        for neighbor in adjacency.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                chain.append(neighbor)
                queue.append((neighbor, depth + 1))

    return chain
