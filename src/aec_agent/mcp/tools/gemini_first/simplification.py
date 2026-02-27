"""
Line and Polygon Simplification Module.

Implements algorithms for reducing vertex count while preserving shape:
- Ramer-Douglas-Peucker (RDP): For geometric/architectural shapes
- Visvalingam-Whyatt: For organic/cartographic shapes

Based on research from PDF_TO_VECTOR_NEW.md.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union
import heapq
import math

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class SimplificationMethod(str, Enum):
    """Available simplification algorithms."""
    RDP = "rdp"  # Ramer-Douglas-Peucker
    VISVALINGAM = "visvalingam"  # Visvalingam-Whyatt


@dataclass
class SimplificationConfig:
    """Configuration for line simplification."""
    method: SimplificationMethod = SimplificationMethod.RDP

    # RDP parameters
    rdp_epsilon: float = 2.0  # Maximum distance tolerance (pixels)

    # Visvalingam parameters
    visvalingam_threshold: float = 10.0  # Minimum triangle area
    visvalingam_target_count: Optional[int] = None  # Target number of points

    # General
    preserve_endpoints: bool = True
    min_points: int = 2  # Minimum points to keep


@dataclass
class SimplificationResult:
    """Result of simplification operation."""
    points: List[Tuple[float, float]]
    original_count: int
    final_count: int
    reduction_ratio: float  # 0-1, how much was removed
    method_used: str


Point = Tuple[float, float]
PointList = List[Point]


def _perpendicular_distance(point: Point, line_start: Point, line_end: Point) -> float:
    """
    Calculate perpendicular distance from point to line segment.

    Uses the formula: |cross(AB, AP)| / |AB|
    """
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end

    # Line vector
    dx = x2 - x1
    dy = y2 - y1

    # If line is a point, return distance to that point
    line_length_sq = dx * dx + dy * dy
    if line_length_sq == 0:
        return math.sqrt((x - x1) ** 2 + (y - y1) ** 2)

    # Calculate perpendicular distance using cross product
    cross = abs(dy * x - dx * y + x2 * y1 - y2 * x1)
    return cross / math.sqrt(line_length_sq)


def _triangle_area(p1: Point, p2: Point, p3: Point) -> float:
    """
    Calculate area of triangle formed by three points.

    Uses the shoelace formula (half the absolute value of the cross product).
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)


def rdp_simplify(
    points: PointList,
    epsilon: float,
    preserve_endpoints: bool = True,
) -> PointList:
    """
    Ramer-Douglas-Peucker line simplification.

    Recursively removes points that deviate less than epsilon from
    the straight line between endpoints.

    Best for: Geometric shapes, architectural lines, straight segments

    Args:
        points: List of (x, y) coordinate tuples
        epsilon: Maximum distance tolerance (same units as coordinates)
        preserve_endpoints: Always keep first and last points

    Returns:
        Simplified list of points

    Time complexity: O(n²) worst case, O(n log n) average
    Space complexity: O(n)

    Example:
        >>> points = [(0, 0), (1, 0.1), (2, -0.1), (3, 0.2), (4, 0)]
        >>> simplified = rdp_simplify(points, epsilon=0.5)
        >>> print(simplified)  # [(0, 0), (4, 0)]
    """
    if len(points) <= 2:
        return points

    # Find point with maximum distance from line
    start = points[0]
    end = points[-1]

    max_dist = 0.0
    max_idx = 0

    for i in range(1, len(points) - 1):
        dist = _perpendicular_distance(points[i], start, end)
        if dist > max_dist:
            max_dist = dist
            max_idx = i

    # If max distance exceeds epsilon, recursively simplify
    if max_dist > epsilon:
        # Split and recurse
        left = rdp_simplify(points[:max_idx + 1], epsilon, preserve_endpoints)
        right = rdp_simplify(points[max_idx:], epsilon, preserve_endpoints)

        # Combine (avoid duplicating the split point)
        return left[:-1] + right
    else:
        # All points within tolerance, keep only endpoints
        if preserve_endpoints:
            return [start, end]
        else:
            return [start, end]


def rdp_simplify_iterative(
    points: PointList,
    epsilon: float,
) -> PointList:
    """
    Iterative (non-recursive) RDP implementation.

    Uses explicit stack to avoid recursion depth limits on large datasets.
    """
    if len(points) <= 2:
        return points

    # Mark which points to keep
    keep = [False] * len(points)
    keep[0] = True
    keep[-1] = True

    # Stack of ranges to process
    stack = [(0, len(points) - 1)]

    while stack:
        start_idx, end_idx = stack.pop()

        if end_idx - start_idx <= 1:
            continue

        # Find point with maximum distance
        max_dist = 0.0
        max_idx = start_idx

        start_pt = points[start_idx]
        end_pt = points[end_idx]

        for i in range(start_idx + 1, end_idx):
            dist = _perpendicular_distance(points[i], start_pt, end_pt)
            if dist > max_dist:
                max_dist = dist
                max_idx = i

        if max_dist > epsilon:
            keep[max_idx] = True
            stack.append((start_idx, max_idx))
            stack.append((max_idx, end_idx))

    # Build result
    return [points[i] for i in range(len(points)) if keep[i]]


class VisvalingamHeapItem:
    """Heap item for Visvalingam-Whyatt algorithm."""

    def __init__(self, index: int, area: float):
        self.index = index
        self.area = area
        self.removed = False

    def __lt__(self, other):
        return self.area < other.area


def visvalingam_simplify(
    points: PointList,
    threshold: Optional[float] = None,
    target_count: Optional[int] = None,
    preserve_endpoints: bool = True,
) -> PointList:
    """
    Visvalingam-Whyatt line simplification.

    Iteratively removes points that contribute the least "effective area"
    (the area of the triangle formed with neighbors).

    Best for: Organic shapes, cartographic data, smooth curves

    Args:
        points: List of (x, y) coordinate tuples
        threshold: Minimum triangle area to keep (mutually exclusive with target_count)
        target_count: Target number of points to keep
        preserve_endpoints: Always keep first and last points

    Returns:
        Simplified list of points

    Time complexity: O(n log n)
    Space complexity: O(n)

    Example:
        >>> points = [(0, 0), (1, 2), (2, 1), (3, 3), (4, 0)]
        >>> simplified = visvalingam_simplify(points, target_count=3)
    """
    n = len(points)
    if n <= 2:
        return points

    # Must specify either threshold or target_count
    if threshold is None and target_count is None:
        threshold = 0.0  # Keep all points

    if target_count is not None:
        target_count = max(2, min(target_count, n))

    # Build doubly-linked list structure
    prev_idx = [i - 1 for i in range(n)]
    next_idx = [i + 1 for i in range(n)]
    prev_idx[0] = -1
    next_idx[-1] = -1

    # Calculate initial areas and build heap
    heap_items = [None] * n
    heap = []

    for i in range(1, n - 1):
        area = _triangle_area(points[prev_idx[i]], points[i], points[next_idx[i]])
        item = VisvalingamHeapItem(i, area)
        heap_items[i] = item
        heapq.heappush(heap, item)

    # Track how many points remain
    remaining = n
    current_min_area = 0.0

    # Remove points until we hit threshold or target count
    while heap:
        if target_count is not None and remaining <= target_count:
            break

        if threshold is not None and remaining <= 2:
            break

        # Get point with minimum area
        item = heapq.heappop(heap)

        if item.removed:
            continue

        # Check threshold
        if threshold is not None and item.area >= threshold:
            break

        # Remove this point
        idx = item.index
        item.removed = True
        remaining -= 1
        current_min_area = item.area

        # Update links
        p = prev_idx[idx]
        n_idx = next_idx[idx]

        if p >= 0:
            next_idx[p] = n_idx
        if n_idx >= 0 and n_idx < len(prev_idx):
            prev_idx[n_idx] = p

        # Recalculate areas for affected neighbors
        for neighbor in [p, n_idx]:
            if neighbor <= 0 or neighbor >= n - 1:
                continue
            if heap_items[neighbor] is None or heap_items[neighbor].removed:
                continue

            # Mark old item as removed
            heap_items[neighbor].removed = True

            # Calculate new area
            pp = prev_idx[neighbor]
            nn = next_idx[neighbor]
            if pp < 0 or nn < 0 or nn >= n:
                continue

            new_area = _triangle_area(points[pp], points[neighbor], points[nn])
            # Ensure area doesn't decrease below current minimum (to maintain topological consistency)
            new_area = max(new_area, current_min_area)

            # Create new heap item
            new_item = VisvalingamHeapItem(neighbor, new_area)
            heap_items[neighbor] = new_item
            heapq.heappush(heap, new_item)

    # Build result by following the linked list
    result = []
    idx = 0
    while idx >= 0 and idx < n:
        if idx == 0 or idx == n - 1 or (heap_items[idx] is not None and not heap_items[idx].removed):
            result.append(points[idx])
        idx = next_idx[idx] if idx < n - 1 else -1

    # Ensure we have the full path
    if not result:
        return [points[0], points[-1]]

    return result


def simplify_line(
    points: PointList,
    config: Optional[SimplificationConfig] = None,
) -> SimplificationResult:
    """
    Simplify a polyline using configured algorithm.

    Args:
        points: List of (x, y) coordinate tuples
        config: Simplification configuration

    Returns:
        SimplificationResult with simplified points and statistics

    Example:
        >>> config = SimplificationConfig(method=SimplificationMethod.RDP, rdp_epsilon=2.0)
        >>> result = simplify_line(points, config)
        >>> print(f"Reduced from {result.original_count} to {result.final_count} points")
    """
    config = config or SimplificationConfig()
    original_count = len(points)

    if original_count <= config.min_points:
        method_str = config.method.value if hasattr(config.method, 'value') else str(config.method)
        return SimplificationResult(
            points=list(points),
            original_count=original_count,
            final_count=original_count,
            reduction_ratio=0.0,
            method_used=method_str,
        )

    if config.method == SimplificationMethod.RDP:
        simplified = rdp_simplify_iterative(
            points,
            config.rdp_epsilon,
        )
    elif config.method == SimplificationMethod.VISVALINGAM:
        simplified = visvalingam_simplify(
            points,
            threshold=config.visvalingam_threshold,
            target_count=config.visvalingam_target_count,
            preserve_endpoints=config.preserve_endpoints,
        )
    else:
        simplified = list(points)

    # Ensure minimum points
    if len(simplified) < config.min_points:
        simplified = list(points)[:config.min_points]

    final_count = len(simplified)
    reduction = 1.0 - (final_count / original_count) if original_count > 0 else 0.0
    method_str = config.method.value if hasattr(config.method, 'value') else str(config.method)

    logger.debug(
        "line_simplified",
        method=method_str,
        original_count=original_count,
        final_count=final_count,
        reduction_ratio=reduction,
    )

    return SimplificationResult(
        points=simplified,
        original_count=original_count,
        final_count=final_count,
        reduction_ratio=reduction,
        method_used=method_str,
    )


def simplify_polygon(
    points: PointList,
    config: Optional[SimplificationConfig] = None,
) -> SimplificationResult:
    """
    Simplify a closed polygon.

    Handles the closure by temporarily duplicating the first point
    and ensuring the result remains closed.

    Args:
        points: List of polygon vertices (will be treated as closed)
        config: Simplification configuration

    Returns:
        SimplificationResult with simplified polygon vertices
    """
    config = config or SimplificationConfig()

    if len(points) < 4:  # Triangle or less
        method_str = config.method.value if hasattr(config.method, 'value') else str(config.method)
        return SimplificationResult(
            points=list(points),
            original_count=len(points),
            final_count=len(points),
            reduction_ratio=0.0,
            method_used=method_str,
        )

    # Check if already closed
    is_closed = (
        len(points) > 1 and
        abs(points[0][0] - points[-1][0]) < 0.001 and
        abs(points[0][1] - points[-1][1]) < 0.001
    )

    # For closed polygons, duplicate first point at end
    working_points = list(points)
    if not is_closed:
        working_points.append(points[0])

    # Simplify as if it were open
    result = simplify_line(working_points, config)
    simplified = result.points

    # Ensure closure
    if len(simplified) > 1:
        if not (
            abs(simplified[0][0] - simplified[-1][0]) < 0.001 and
            abs(simplified[0][1] - simplified[-1][1]) < 0.001
        ):
            simplified.append(simplified[0])

    return SimplificationResult(
        points=simplified,
        original_count=len(points),
        final_count=len(simplified),
        reduction_ratio=result.reduction_ratio,
        method_used=result.method_used,
    )


def simplify_entities(
    entities: List[dict],
    config: Optional[SimplificationConfig] = None,
) -> Tuple[List[dict], dict]:
    """
    Simplify geometry in a list of entity dictionaries.

    Processes entities with 'points', 'vertices', or 'start'/'end' properties.

    Args:
        entities: List of entity dictionaries
        config: Simplification configuration

    Returns:
        Tuple of (simplified entities, statistics dict)
    """
    config = config or SimplificationConfig()
    stats = {
        "total_entities": len(entities),
        "entities_simplified": 0,
        "total_points_before": 0,
        "total_points_after": 0,
    }

    simplified_entities = []

    for entity in entities:
        props = entity.get("properties", entity)
        simplified_entity = dict(entity)
        simplified_props = dict(props)

        # Check for polyline points
        if "points" in props or "vertices" in props:
            key = "points" if "points" in props else "vertices"
            points = props[key]

            if isinstance(points, list) and len(points) > 2:
                # Convert to tuples if needed
                point_tuples = [
                    (p[0], p[1]) if isinstance(p, (list, tuple)) else (p.get("x", 0), p.get("y", 0))
                    for p in points
                ]

                stats["total_points_before"] += len(point_tuples)

                # Determine if polygon or line
                is_closed = entity.get("entity_type") in ("polygon", "closed_polyline")
                if is_closed:
                    result = simplify_polygon(point_tuples, config)
                else:
                    result = simplify_line(point_tuples, config)

                # Convert back to original format
                if isinstance(points[0], dict):
                    simplified_points = [{"x": p[0], "y": p[1]} for p in result.points]
                else:
                    simplified_points = [list(p) for p in result.points]

                simplified_props[key] = simplified_points
                stats["total_points_after"] += len(simplified_points)
                stats["entities_simplified"] += 1

        # Update entity
        if "properties" in simplified_entity:
            simplified_entity["properties"] = simplified_props
        else:
            simplified_entity = simplified_props

        simplified_entities.append(simplified_entity)

    # Calculate overall reduction
    if stats["total_points_before"] > 0:
        stats["reduction_ratio"] = 1.0 - (stats["total_points_after"] / stats["total_points_before"])
    else:
        stats["reduction_ratio"] = 0.0

    logger.info(
        "entities_simplified",
        total_entities=stats["total_entities"],
        entities_simplified=stats["entities_simplified"],
        points_before=stats["total_points_before"],
        points_after=stats["total_points_after"],
        reduction_ratio=stats["reduction_ratio"],
    )

    return simplified_entities, stats


# Convenience functions
def quick_simplify_rdp(points: PointList, epsilon: float = 2.0) -> PointList:
    """Quick RDP simplification."""
    result = simplify_line(
        points,
        SimplificationConfig(method=SimplificationMethod.RDP, rdp_epsilon=epsilon)
    )
    return result.points


def quick_simplify_visvalingam(points: PointList, target_ratio: float = 0.5) -> PointList:
    """Quick Visvalingam simplification to target ratio of original points."""
    target_count = max(2, int(len(points) * target_ratio))
    result = simplify_line(
        points,
        SimplificationConfig(
            method=SimplificationMethod.VISVALINGAM,
            visvalingam_target_count=target_count
        )
    )
    return result.points
