"""
Polyline Auto-Join Module.

Automatically joins connected line segments into polylines to reduce
entity count and improve drawing efficiency.

Algorithm:
1. Build endpoint graph from all lines
2. Find connected chains using graph traversal
3. Convert chains to polylines (LWPOLYLINE)
4. Preserve vertex order (CW/CCW for closed polylines)
5. Handle branches and intersections

Features:
- Endpoint snapping within tolerance
- Closed polyline detection
- Chain optimization (removes intermediate points on collinear segments)
- Preserves layer and linetype information
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import math

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PolylineConfig:
    """Configuration for polyline building."""
    endpoint_tolerance: float = 0.01  # Distance to consider endpoints connected
    collinear_tolerance: float = 0.1  # Angle tolerance for collinear points (radians)
    min_chain_length: int = 2  # Minimum segments to form a polyline
    remove_collinear: bool = True  # Remove intermediate points on straight lines
    detect_closed: bool = True  # Detect and mark closed polylines


@dataclass
class PolylineResult:
    """Result of polyline building."""
    polylines: List[Any]  # List of EntityToCreate with type=LWPOLYLINE
    original_line_count: int
    polyline_count: int
    closed_count: int
    reduction_ratio: float  # Entity count reduction
    removed_collinear: int  # Removed collinear points

    def to_dict(self) -> dict:
        return {
            "original_line_count": self.original_line_count,
            "polyline_count": self.polyline_count,
            "closed_count": self.closed_count,
            "reduction_ratio": f"{self.reduction_ratio:.1%}",
            "removed_collinear": self.removed_collinear,
        }


class EndpointGraph:
    """
    Graph structure for line endpoints.

    Maps endpoints to connected line segments for efficient traversal.
    """

    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance
        self._nodes: Dict[Tuple[float, float], List[int]] = defaultdict(list)
        self._lines: List[Tuple[Tuple[float, float], Tuple[float, float], int]] = []
        self._grid_size = tolerance * 10  # Grid cell size for spatial hashing

    def _snap_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Snap point to grid for consistent hashing."""
        return (
            round(point[0] / self._grid_size) * self._grid_size,
            round(point[1] / self._grid_size) * self._grid_size,
        )

    def _find_nearby_node(self, point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Find existing node within tolerance."""
        snapped = self._snap_point(point)

        # Check nearby grid cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_point = (
                    snapped[0] + dx * self._grid_size,
                    snapped[1] + dy * self._grid_size,
                )
                if check_point in self._nodes:
                    # Check actual distance
                    dist = math.sqrt(
                        (point[0] - check_point[0]) ** 2 +
                        (point[1] - check_point[1]) ** 2
                    )
                    if dist <= self.tolerance:
                        return check_point

        return None

    def add_line(self, start: Tuple[float, float], end: Tuple[float, float], line_idx: int) -> None:
        """Add a line segment to the graph."""
        # Snap endpoints to existing nodes or create new ones
        node_start = self._find_nearby_node(start)
        if node_start is None:
            node_start = self._snap_point(start)

        node_end = self._find_nearby_node(end)
        if node_end is None:
            node_end = self._snap_point(end)

        # Store line
        self._lines.append((node_start, node_end, line_idx))

        # Add to adjacency lists
        self._nodes[node_start].append(len(self._lines) - 1)
        self._nodes[node_end].append(len(self._lines) - 1)

    def get_degree(self, node: Tuple[float, float]) -> int:
        """Get the degree (number of connections) of a node."""
        return len(self._nodes.get(node, []))

    def get_connected_lines(self, node: Tuple[float, float]) -> List[int]:
        """Get line indices connected to a node."""
        return self._nodes.get(node, [])

    def get_line(self, line_idx: int) -> Tuple[Tuple[float, float], Tuple[float, float], int]:
        """Get line by index (returns start, end, original_idx)."""
        return self._lines[line_idx]

    def get_other_endpoint(self, line_idx: int, from_node: Tuple[float, float]) -> Tuple[float, float]:
        """Get the endpoint of a line that is not from_node."""
        start, end, _ = self._lines[line_idx]
        if self._points_equal(start, from_node):
            return end
        return start

    def _points_equal(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        """Check if two points are equal within tolerance."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) <= self.tolerance

    def find_chains(self) -> List[List[int]]:
        """
        Find all chains (paths) in the graph.

        A chain is a sequence of connected lines where interior nodes
        have degree 2 (straight path, no branches).

        Returns:
            List of chains, each chain is a list of line indices
        """
        visited_lines: Set[int] = set()
        chains: List[List[int]] = []

        # Find starting points: nodes with degree != 2 (endpoints, junctions)
        # or any unvisited node for closed loops
        start_nodes = [
            node for node, lines in self._nodes.items()
            if len(lines) != 2 and lines
        ]

        # Process chains from each starting node
        for start_node in start_nodes:
            for line_idx in self._nodes[start_node]:
                if line_idx in visited_lines:
                    continue

                chain = self._trace_chain(start_node, line_idx, visited_lines)
                if chain:
                    chains.append(chain)

        # Find closed loops (all nodes have degree 2)
        for node, line_indices in self._nodes.items():
            if len(line_indices) == 2:
                for line_idx in line_indices:
                    if line_idx not in visited_lines:
                        chain = self._trace_chain(node, line_idx, visited_lines)
                        if chain:
                            chains.append(chain)

        return chains

    def _trace_chain(
        self,
        start_node: Tuple[float, float],
        first_line: int,
        visited: Set[int],
    ) -> List[int]:
        """Trace a chain starting from a node and line."""
        chain = [first_line]
        visited.add(first_line)

        current_node = self.get_other_endpoint(first_line, start_node)

        while True:
            # Get connected lines at current node
            connected = self.get_connected_lines(current_node)

            # Find next unvisited line
            next_line = None
            for line_idx in connected:
                if line_idx not in visited:
                    next_line = line_idx
                    break

            if next_line is None:
                # No more lines to follow
                break

            # Check if we should continue (degree 2 = straight path)
            if len(connected) != 2:
                # This is a junction or endpoint
                break

            chain.append(next_line)
            visited.add(next_line)
            current_node = self.get_other_endpoint(next_line, current_node)

        return chain


def _is_collinear(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    tolerance: float = 0.1,
) -> bool:
    """Check if three points are collinear."""
    # Vector from p1 to p2
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    # Vector from p2 to p3
    v2 = (p3[0] - p2[0], p3[1] - p2[1])

    len1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    len2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

    if len1 < 1e-10 or len2 < 1e-10:
        return True

    # Compute angle between vectors
    cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2)
    cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp for numerical stability

    angle = math.acos(abs(cos_angle))

    return angle < tolerance


def _simplify_collinear(
    points: List[Tuple[float, float]],
    tolerance: float = 0.1,
) -> Tuple[List[Tuple[float, float]], int]:
    """
    Remove collinear intermediate points.

    Returns:
        Tuple of (simplified_points, removed_count)
    """
    if len(points) <= 2:
        return points, 0

    simplified = [points[0]]
    removed = 0

    for i in range(1, len(points) - 1):
        if not _is_collinear(simplified[-1], points[i], points[i + 1], tolerance):
            simplified.append(points[i])
        else:
            removed += 1

    simplified.append(points[-1])

    return simplified, removed


def _check_closed(
    points: List[Tuple[float, float]],
    tolerance: float = 0.01,
) -> bool:
    """Check if a polyline is closed (first point == last point)."""
    if len(points) < 3:
        return False

    dist = math.sqrt(
        (points[0][0] - points[-1][0]) ** 2 +
        (points[0][1] - points[-1][1]) ** 2
    )
    return dist <= tolerance


def _extract_points_from_chain(
    graph: EndpointGraph,
    chain: List[int],
    entities: List[Any],
) -> List[Tuple[float, float]]:
    """Extract ordered points from a chain of line indices."""
    if not chain:
        return []

    points = []

    # Get first line
    start, end, orig_idx = graph.get_line(chain[0])
    points.append(start)
    current_end = end

    for line_idx in chain:
        start, end, _ = graph.get_line(line_idx)

        # Determine correct order
        if len(points) > 0:
            last_point = points[-1]
            dist_to_start = math.sqrt(
                (last_point[0] - start[0]) ** 2 +
                (last_point[1] - start[1]) ** 2
            )
            dist_to_end = math.sqrt(
                (last_point[0] - end[0]) ** 2 +
                (last_point[1] - end[1]) ** 2
            )

            if dist_to_start < dist_to_end:
                # start connects to previous
                points.append(end)
            else:
                # end connects to previous
                points.append(start)
        else:
            points.append(end)

    return points


def build_polylines(
    entities: List[Any],
    config: Optional[PolylineConfig] = None,
) -> PolylineResult:
    """
    Convert connected line segments to polylines.

    Args:
        entities: List of EntityToCreate objects
        config: Optional configuration

    Returns:
        PolylineResult with new polyline entities
    """
    if config is None:
        config = PolylineConfig()

    # Separate lines from other entities
    from .adaptive_extraction import EntityType, EntityToCreate, ExtractionSource

    lines = []
    other_entities = []

    for entity in entities:
        entity_type = entity.entity_type
        if hasattr(entity_type, 'value'):
            entity_type = entity_type.value

        if entity_type.lower() == "line":
            lines.append(entity)
        else:
            other_entities.append(entity)

    if not lines:
        return PolylineResult(
            polylines=other_entities,
            original_line_count=0,
            polyline_count=0,
            closed_count=0,
            reduction_ratio=0.0,
            removed_collinear=0,
        )

    # Build endpoint graph
    graph = EndpointGraph(tolerance=config.endpoint_tolerance)

    for idx, line in enumerate(lines):
        props = line.properties
        start = props.get("start", (0, 0))
        end = props.get("end", (0, 0))

        if isinstance(start, (list, tuple)) and isinstance(end, (list, tuple)):
            graph.add_line(tuple(start[:2]), tuple(end[:2]), idx)

    # Find chains
    chains = graph.find_chains()

    # Convert chains to polylines
    polylines = []
    total_removed_collinear = 0
    closed_count = 0

    for chain in chains:
        if len(chain) < config.min_chain_length:
            # Keep as individual lines
            for line_idx in chain:
                _, _, orig_idx = graph.get_line(line_idx)
                polylines.append(lines[orig_idx])
            continue

        # Extract points
        points = _extract_points_from_chain(graph, chain, entities)

        if len(points) < 2:
            continue

        # Remove collinear points if enabled
        if config.remove_collinear:
            points, removed = _simplify_collinear(points, config.collinear_tolerance)
            total_removed_collinear += removed

        # Check if closed
        is_closed = False
        if config.detect_closed:
            is_closed = _check_closed(points, config.endpoint_tolerance)
            if is_closed:
                closed_count += 1
                # Remove duplicate last point if closed
                if len(points) > 1:
                    points = points[:-1]

        # Get layer from first line in chain
        _, _, first_orig_idx = graph.get_line(chain[0])
        first_line = lines[first_orig_idx]
        layer = first_line.layer
        linetype = getattr(first_line, 'linetype', 'Continuous')
        lineweight = getattr(first_line, 'lineweight', 0.0)

        # Create polyline entity
        polyline = EntityToCreate(
            entity_type=EntityType.LWPOLYLINE,
            layer=layer,
            properties={
                "points": points,
                "vertices": points,
                "closed": is_closed,
            },
            source=ExtractionSource.HYBRID,
            linetype=linetype,
            lineweight=lineweight,
        )
        polylines.append(polyline)

    # Add other (non-line) entities
    polylines.extend(other_entities)

    # Calculate reduction
    original_count = len(lines)
    new_polyline_count = len([p for p in polylines if hasattr(p, 'entity_type') and
                              (p.entity_type == EntityType.LWPOLYLINE or
                               (hasattr(p.entity_type, 'value') and p.entity_type.value == 'lwpolyline'))])

    reduction = 1.0 - (new_polyline_count / original_count) if original_count > 0 else 0.0

    logger.info(
        "polyline_builder_complete",
        original_lines=original_count,
        polylines=new_polyline_count,
        closed=closed_count,
        reduction=f"{reduction:.1%}",
        removed_collinear=total_removed_collinear,
    )

    return PolylineResult(
        polylines=polylines,
        original_line_count=original_count,
        polyline_count=new_polyline_count,
        closed_count=closed_count,
        reduction_ratio=reduction,
        removed_collinear=total_removed_collinear,
    )


def auto_join_lines(
    entities: List[Any],
    endpoint_tolerance: float = 0.01,
    remove_collinear: bool = True,
) -> List[Any]:
    """
    Convenience function to auto-join lines into polylines.

    Args:
        entities: List of EntityToCreate objects
        endpoint_tolerance: Distance to consider endpoints connected
        remove_collinear: Remove intermediate collinear points

    Returns:
        List of entities with lines converted to polylines where possible
    """
    config = PolylineConfig(
        endpoint_tolerance=endpoint_tolerance,
        remove_collinear=remove_collinear,
    )

    result = build_polylines(entities, config)
    return result.polylines
