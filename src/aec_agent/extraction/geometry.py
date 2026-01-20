"""
Geometry conversion utilities for PostGIS.

Converts AutoCAD and Revit geometry to WKT (Well-Known Text) format
for storage in PostgreSQL with PostGIS.
"""

from typing import Dict, Any, Optional, List, Tuple
import math

from aec_agent.db.models import BoundsInfo, CentroidInfo


def autocad_geometry_to_wkt(geometry_info: Dict[str, Any]) -> Optional[str]:
    """
    Convert AutoCAD geometry to WKT format.

    Supports: LINE, CIRCLE, ARC, POLYLINE, POINT, LWPOLYLINE

    Args:
        geometry_info: Geometry dict from AutoCAD sidecar
            - type: Geometry type
            - points: List of [x, y, z] coordinates
            - center: [x, y, z] for circles/arcs
            - radius: For circles/arcs
            - start_angle, end_angle: For arcs

    Returns:
        WKT string or None if unsupported
    """
    geom_type = geometry_info.get("type", "").upper()

    if geom_type == "LINE":
        points = geometry_info.get("points", [])
        if len(points) >= 2:
            start = points[0]
            end = points[1]
            return f"LINESTRING Z({start[0]} {start[1]} {_z(start)}, {end[0]} {end[1]} {_z(end)})"

    elif geom_type == "POINT":
        points = geometry_info.get("points", [])
        if points:
            p = points[0]
            return f"POINT Z({p[0]} {p[1]} {_z(p)})"
        center = geometry_info.get("center")
        if center:
            return f"POINT Z({center[0]} {center[1]} {_z(center)})"

    elif geom_type == "CIRCLE":
        center = geometry_info.get("center", [0, 0, 0])
        radius = geometry_info.get("radius", 1)
        # Approximate circle as polygon with 32 points
        points = _circle_to_polygon(center, radius, 32)
        coords = ", ".join(f"{p[0]} {p[1]} {_z(center)}" for p in points)
        return f"POLYGON Z(({coords}))"

    elif geom_type == "ARC":
        center = geometry_info.get("center", [0, 0, 0])
        radius = geometry_info.get("radius", 1)
        start_angle = geometry_info.get("start_angle", 0)
        end_angle = geometry_info.get("end_angle", 360)
        # Approximate arc as linestring
        points = _arc_to_linestring(center, radius, start_angle, end_angle, 16)
        coords = ", ".join(f"{p[0]} {p[1]} {_z(center)}" for p in points)
        return f"LINESTRING Z({coords})"

    elif geom_type in ("POLYLINE", "LWPOLYLINE", "POLYLINE2D", "POLYLINE3D"):
        points = geometry_info.get("points", [])
        if len(points) >= 2:
            coords = ", ".join(f"{p[0]} {p[1]} {_z(p)}" for p in points)
            # Check if closed
            is_closed = geometry_info.get("closed", False)
            if is_closed and len(points) >= 3:
                # Close the polygon
                coords += f", {points[0][0]} {points[0][1]} {_z(points[0])}"
                return f"POLYGON Z(({coords}))"
            return f"LINESTRING Z({coords})"

    elif geom_type == "SPLINE":
        # Splines are complex - use control points as approximation
        points = geometry_info.get("points", [])
        if len(points) >= 2:
            coords = ", ".join(f"{p[0]} {p[1]} {_z(p)}" for p in points)
            return f"LINESTRING Z({coords})"

    elif geom_type == "HATCH":
        # Hatches have loops - use first loop boundary
        loops = geometry_info.get("loops", [])
        if loops and len(loops[0]) >= 3:
            points = loops[0]
            coords = ", ".join(f"{p[0]} {p[1]} {_z(p)}" for p in points)
            # Close polygon
            coords += f", {points[0][0]} {points[0][1]} {_z(points[0])}"
            return f"POLYGON Z(({coords}))"

    elif geom_type == "TEXT" or geom_type == "MTEXT":
        # Text uses insertion point
        position = geometry_info.get("position") or geometry_info.get("center")
        if position:
            return f"POINT Z({position[0]} {position[1]} {_z(position)})"

    elif geom_type == "INSERT":
        # Block reference - use insertion point
        position = geometry_info.get("position") or geometry_info.get("center")
        if position:
            return f"POINT Z({position[0]} {position[1]} {_z(position)})"

    return None


def revit_location_to_wkt(location_info: Dict[str, Any]) -> Optional[str]:
    """
    Convert Revit element location to WKT format.

    Supports: POINT (columns, furniture), CURVE (walls, beams), AREA (rooms)

    Args:
        location_info: Location dict from Revit sidecar
            - type: POINT, CURVE, or AREA
            - point: {x, y, z} for points
            - start, end: {x, y, z} for curves
            - boundary: List of {x, y, z} for areas

    Returns:
        WKT string or None if unsupported
    """
    loc_type = location_info.get("type", "").upper()

    if loc_type == "POINT":
        point = location_info.get("point", {})
        x = point.get("x", 0)
        y = point.get("y", 0)
        z = point.get("z", 0)
        return f"POINT Z({x} {y} {z})"

    elif loc_type == "CURVE":
        start = location_info.get("start", {})
        end = location_info.get("end", {})
        sx, sy, sz = start.get("x", 0), start.get("y", 0), start.get("z", 0)
        ex, ey, ez = end.get("x", 0), end.get("y", 0), end.get("z", 0)
        return f"LINESTRING Z({sx} {sy} {sz}, {ex} {ey} {ez})"

    elif loc_type == "AREA":
        boundary = location_info.get("boundary", [])
        if len(boundary) >= 3:
            coords = ", ".join(
                f"{p.get('x', 0)} {p.get('y', 0)} {p.get('z', 0)}"
                for p in boundary
            )
            # Close polygon
            first = boundary[0]
            coords += f", {first.get('x', 0)} {first.get('y', 0)} {first.get('z', 0)}"
            return f"POLYGON Z(({coords}))"

    return None


def compute_centroid(geometry_info: Dict[str, Any], source: str = "autocad") -> Optional[CentroidInfo]:
    """
    Compute centroid from geometry info.

    Args:
        geometry_info: Geometry dict
        source: 'autocad' or 'revit'

    Returns:
        CentroidInfo or None
    """
    if source == "autocad":
        return _compute_autocad_centroid(geometry_info)
    else:
        return _compute_revit_centroid(geometry_info)


def _compute_autocad_centroid(geometry_info: Dict[str, Any]) -> Optional[CentroidInfo]:
    """Compute centroid for AutoCAD geometry."""
    geom_type = geometry_info.get("type", "").upper()

    # For circles/arcs, center is the centroid
    if geom_type in ("CIRCLE", "ARC"):
        center = geometry_info.get("center")
        if center:
            return CentroidInfo(x=center[0], y=center[1], z=_z(center))

    # For points, the point itself is the centroid
    if geom_type == "POINT":
        points = geometry_info.get("points", [])
        if points:
            p = points[0]
            return CentroidInfo(x=p[0], y=p[1], z=_z(p))
        center = geometry_info.get("center")
        if center:
            return CentroidInfo(x=center[0], y=center[1], z=_z(center))

    # For lines/polylines, compute average of all points
    points = geometry_info.get("points", [])
    if points:
        x = sum(p[0] for p in points) / len(points)
        y = sum(p[1] for p in points) / len(points)
        z = sum(_z(p) for p in points) / len(points)
        return CentroidInfo(x=x, y=y, z=z)

    # For text/blocks, use position
    position = geometry_info.get("position") or geometry_info.get("center")
    if position:
        return CentroidInfo(x=position[0], y=position[1], z=_z(position))

    return None


def _compute_revit_centroid(location_info: Dict[str, Any]) -> Optional[CentroidInfo]:
    """Compute centroid for Revit location."""
    loc_type = location_info.get("type", "").upper()

    if loc_type == "POINT":
        point = location_info.get("point", {})
        return CentroidInfo(
            x=point.get("x", 0),
            y=point.get("y", 0),
            z=point.get("z", 0)
        )

    elif loc_type == "CURVE":
        start = location_info.get("start", {})
        end = location_info.get("end", {})
        return CentroidInfo(
            x=(start.get("x", 0) + end.get("x", 0)) / 2,
            y=(start.get("y", 0) + end.get("y", 0)) / 2,
            z=(start.get("z", 0) + end.get("z", 0)) / 2
        )

    elif loc_type == "AREA":
        boundary = location_info.get("boundary", [])
        if boundary:
            x = sum(p.get("x", 0) for p in boundary) / len(boundary)
            y = sum(p.get("y", 0) for p in boundary) / len(boundary)
            z = sum(p.get("z", 0) for p in boundary) / len(boundary)
            return CentroidInfo(x=x, y=y, z=z)

    return None


def compute_bounds(geometry_info: Dict[str, Any], source: str = "autocad") -> Optional[BoundsInfo]:
    """
    Compute bounding box from geometry info.

    Args:
        geometry_info: Geometry dict
        source: 'autocad' or 'revit'

    Returns:
        BoundsInfo or None
    """
    if source == "autocad":
        return _compute_autocad_bounds(geometry_info)
    else:
        return _compute_revit_bounds(geometry_info)


def _compute_autocad_bounds(geometry_info: Dict[str, Any]) -> Optional[BoundsInfo]:
    """Compute bounds for AutoCAD geometry."""
    # Check for pre-computed bounds
    if "bounds" in geometry_info:
        b = geometry_info["bounds"]
        return BoundsInfo(
            min_x=b.get("min_x", b.get("minX", 0)),
            min_y=b.get("min_y", b.get("minY", 0)),
            min_z=b.get("min_z", b.get("minZ", 0)),
            max_x=b.get("max_x", b.get("maxX", 0)),
            max_y=b.get("max_y", b.get("maxY", 0)),
            max_z=b.get("max_z", b.get("maxZ", 0)),
        )

    # Compute from geometry
    points = _extract_all_points(geometry_info)
    if not points:
        return None

    return BoundsInfo(
        min_x=min(p[0] for p in points),
        min_y=min(p[1] for p in points),
        min_z=min(_z(p) for p in points),
        max_x=max(p[0] for p in points),
        max_y=max(p[1] for p in points),
        max_z=max(_z(p) for p in points),
    )


def _compute_revit_bounds(location_info: Dict[str, Any]) -> Optional[BoundsInfo]:
    """Compute bounds for Revit location."""
    # Check for pre-computed bounds
    if "bounds" in location_info:
        b = location_info["bounds"]
        return BoundsInfo(
            min_x=b.get("min_x", b.get("minX", 0)),
            min_y=b.get("min_y", b.get("minY", 0)),
            min_z=b.get("min_z", b.get("minZ", 0)),
            max_x=b.get("max_x", b.get("maxX", 0)),
            max_y=b.get("max_y", b.get("maxY", 0)),
            max_z=b.get("max_z", b.get("maxZ", 0)),
        )

    loc_type = location_info.get("type", "").upper()
    points = []

    if loc_type == "POINT":
        point = location_info.get("point", {})
        points = [(point.get("x", 0), point.get("y", 0), point.get("z", 0))]

    elif loc_type == "CURVE":
        start = location_info.get("start", {})
        end = location_info.get("end", {})
        points = [
            (start.get("x", 0), start.get("y", 0), start.get("z", 0)),
            (end.get("x", 0), end.get("y", 0), end.get("z", 0)),
        ]

    elif loc_type == "AREA":
        boundary = location_info.get("boundary", [])
        points = [(p.get("x", 0), p.get("y", 0), p.get("z", 0)) for p in boundary]

    if not points:
        return None

    return BoundsInfo(
        min_x=min(p[0] for p in points),
        min_y=min(p[1] for p in points),
        min_z=min(p[2] for p in points),
        max_x=max(p[0] for p in points),
        max_y=max(p[1] for p in points),
        max_z=max(p[2] for p in points),
    )


# =============================================================================
# Helper Functions
# =============================================================================

def _z(point: List) -> float:
    """Get Z coordinate, defaulting to 0."""
    return point[2] if len(point) > 2 else 0.0


def _extract_all_points(geometry_info: Dict[str, Any]) -> List[Tuple[float, float, float]]:
    """Extract all points from geometry for bounds calculation."""
    points = []
    geom_type = geometry_info.get("type", "").upper()

    if geom_type == "CIRCLE":
        center = geometry_info.get("center", [0, 0, 0])
        radius = geometry_info.get("radius", 0)
        # Bounding box of circle
        points = [
            (center[0] - radius, center[1] - radius, _z(center)),
            (center[0] + radius, center[1] + radius, _z(center)),
        ]

    elif geom_type == "ARC":
        center = geometry_info.get("center", [0, 0, 0])
        radius = geometry_info.get("radius", 0)
        start_angle = geometry_info.get("start_angle", 0)
        end_angle = geometry_info.get("end_angle", 360)
        # Arc points
        arc_points = _arc_to_linestring(center, radius, start_angle, end_angle, 8)
        points = [(p[0], p[1], _z(center)) for p in arc_points]

    else:
        raw_points = geometry_info.get("points", [])
        points = [(p[0], p[1], _z(p)) for p in raw_points]

        center = geometry_info.get("center")
        if center and not points:
            points = [(center[0], center[1], _z(center))]

        position = geometry_info.get("position")
        if position and not points:
            points = [(position[0], position[1], _z(position))]

    return points


def _circle_to_polygon(
    center: List[float],
    radius: float,
    num_points: int = 32
) -> List[Tuple[float, float]]:
    """Convert circle to polygon approximation."""
    points = []
    for i in range(num_points + 1):  # +1 to close the polygon
        angle = 2 * math.pi * i / num_points
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        points.append((x, y))
    return points


def _arc_to_linestring(
    center: List[float],
    radius: float,
    start_angle: float,
    end_angle: float,
    num_points: int = 16
) -> List[Tuple[float, float]]:
    """Convert arc to linestring approximation."""
    # Convert degrees to radians
    start_rad = math.radians(start_angle)
    end_rad = math.radians(end_angle)

    # Handle wrap-around
    if end_rad < start_rad:
        end_rad += 2 * math.pi

    points = []
    for i in range(num_points + 1):
        t = i / num_points
        angle = start_rad + t * (end_rad - start_rad)
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        points.append((x, y))
    return points
