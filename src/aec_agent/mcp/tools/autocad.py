"""
MCP tools for AutoCAD automation.

Uses command-based API format for AutoCAD sidecar:
- POST to root endpoint with {"command": "...", "params": {...}}
- Authorization: Bearer token
"""


import structlog

from aec_agent.mcp.concurrency import with_tool_lock
from aec_agent.mcp.server import get_lock, mcp
from aec_agent.mcp.sidecar_client import SidecarError, call_autocad_command

from .base import ErrorCode, error_result

logger = structlog.get_logger(__name__)


# =============================================================================
# Layer Operations
# =============================================================================

@mcp.tool()
async def autocad_list_layers() -> dict:
    """
    List all layers in the current AutoCAD drawing.

    Returns:
        List of layers with name, color, and visibility state
    """
    try:
        result = await call_autocad_command("audit_layers")
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
    except Exception as e:
        logger.error("Unexpected error in autocad_list_layers", error=str(e), exc_info=True)
        return error_result(ErrorCode.INTERNAL_ERROR, f"Unexpected error: {str(e)}")


@mcp.tool()
@with_tool_lock(get_lock())
async def autocad_create_layer(name: str, color: int = 7) -> dict:
    """
    Create a new layer in AutoCAD.

    Args:
        name: Layer name (e.g., "Walls", "Dimensions")
        color: AutoCAD color index 1-255 (default 7=white)
               Common colors: 1=red, 2=yellow, 3=green, 4=cyan, 5=blue, 6=magenta

    Returns:
        Created layer details

    Example:
        autocad_create_layer("Electrical", 1) - Creates red "Electrical" layer
    """
    if not name or not name.strip():
        return error_result(ErrorCode.INVALID_PARAMS, "Layer name is required")

    if not isinstance(color, int) or not (1 <= color <= 255):
        return error_result(
            ErrorCode.INVALID_PARAMS,
            "Color must be an integer between 1 and 255"
        )

    try:
        result = await call_autocad_command(
            "create_layer",
            {"name": name.strip(), "color": color}
        )
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
    except Exception as e:
        logger.error("Unexpected error in autocad_create_layer", error=str(e), exc_info=True)
        return error_result(ErrorCode.INTERNAL_ERROR, f"Unexpected error: {str(e)}")


@mcp.tool()
@with_tool_lock(get_lock())
async def autocad_set_layer_state(
    name: str,
    is_on: bool | None = None,
    is_frozen: bool | None = None
) -> dict:
    """
    Change layer visibility state in AutoCAD.

    Args:
        name: Layer name
        is_on: Set layer on (True) or off (False)
        is_frozen: Set layer frozen (True) or thawed (False)

    Returns:
        Updated layer state

    Example:
        autocad_set_layer_state("Construction", is_frozen=True) - Freeze layer
    """
    if not name or not name.strip():
        return error_result(ErrorCode.INVALID_PARAMS, "Layer name is required")

    if is_on is None and is_frozen is None:
        return error_result(
            ErrorCode.INVALID_PARAMS,
            "At least one of is_on or is_frozen must be specified"
        )

    params = {"name": name.strip()}
    if is_on is not None:
        params["is_off"] = not bool(is_on)  # AutoCAD uses is_off, not is_on
    if is_frozen is not None:
        params["is_frozen"] = bool(is_frozen)

    try:
        result = await call_autocad_command("modify_layer", params)
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)


# =============================================================================
# Drawing Operations
# =============================================================================

@mcp.tool()
@with_tool_lock(get_lock())
async def autocad_draw_line(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    layer: str | None = None
) -> dict:
    """
    Draw a line in AutoCAD.

    Args:
        start_x: Start X coordinate
        start_y: Start Y coordinate
        end_x: End X coordinate
        end_y: End Y coordinate
        layer: Layer name to draw on (optional, uses current if not specified)

    Returns:
        Created line entity details

    Example:
        autocad_draw_line(0, 0, 100, 100, "Construction") - Draw diagonal line
    """
    # Validate coordinates
    for coord_name, coord_val in [
        ("start_x", start_x), ("start_y", start_y),
        ("end_x", end_x), ("end_y", end_y)
    ]:
        if not isinstance(coord_val, (int, float)):
            return error_result(
                ErrorCode.INVALID_PARAMS,
                f"{coord_name} must be a number"
            )

    # AutoCAD expects arrays: {"start": [x, y], "end": [x, y]}
    params = {
        "start": [float(start_x), float(start_y)],
        "end": [float(end_x), float(end_y)]
    }

    if layer:
        params["layer"] = layer.strip()

    try:
        result = await call_autocad_command("draw_line", params)
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)


@mcp.tool()
@with_tool_lock(get_lock())
async def autocad_draw_circle(
    center_x: float,
    center_y: float,
    radius: float,
    layer: str | None = None
) -> dict:
    """
    Draw a circle in AutoCAD.

    Args:
        center_x: Center X coordinate
        center_y: Center Y coordinate
        radius: Circle radius
        layer: Layer name to draw on (optional)

    Returns:
        Created circle entity details

    Example:
        autocad_draw_circle(50, 50, 25) - Draw circle with radius 25 at (50,50)
    """
    for coord_name, coord_val in [
        ("center_x", center_x), ("center_y", center_y), ("radius", radius)
    ]:
        if not isinstance(coord_val, (int, float)):
            return error_result(
                ErrorCode.INVALID_PARAMS,
                f"{coord_name} must be a number"
            )

    if radius <= 0:
        return error_result(ErrorCode.INVALID_PARAMS, "Radius must be positive")

    # AutoCAD expects: {"center": [x, y], "radius": r}
    params = {
        "center": [float(center_x), float(center_y)],
        "radius": float(radius)
    }

    if layer:
        params["layer"] = layer.strip()

    try:
        result = await call_autocad_command("draw_circle", params)
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)


@mcp.tool()
@with_tool_lock(get_lock())
async def autocad_draw_rectangle(
    corner1_x: float,
    corner1_y: float,
    corner2_x: float,
    corner2_y: float,
    layer: str | None = None
) -> dict:
    """
    Draw a rectangle in AutoCAD defined by two corner points.

    Args:
        corner1_x: First corner X coordinate
        corner1_y: First corner Y coordinate
        corner2_x: Opposite corner X coordinate
        corner2_y: Opposite corner Y coordinate
        layer: Layer name to draw on (optional)

    Returns:
        Created rectangle (polyline) entity details

    Example:
        autocad_draw_rectangle(0, 0, 100, 50) - Draw 100x50 rectangle
    """
    for coord_name, coord_val in [
        ("corner1_x", corner1_x), ("corner1_y", corner1_y),
        ("corner2_x", corner2_x), ("corner2_y", corner2_y)
    ]:
        if not isinstance(coord_val, (int, float)):
            return error_result(
                ErrorCode.INVALID_PARAMS,
                f"{coord_name} must be a number"
            )

    # AutoCAD expects: {"corner1": [x, y], "corner2": [x, y]}
    params = {
        "corner1": [float(corner1_x), float(corner1_y)],
        "corner2": [float(corner2_x), float(corner2_y)]
    }

    if layer:
        params["layer"] = layer.strip()

    try:
        result = await call_autocad_command("draw_rectangle", params)
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)


# =============================================================================
# Query Operations
# =============================================================================

@mcp.tool()
async def autocad_get_drawing_info() -> dict:
    """
    Get current AutoCAD drawing information.

    Returns:
        Drawing name, path, and statistics
    """
    try:
        result = await call_autocad_command("get_drawing_info")
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)


@mcp.tool()
async def autocad_get_entities(
    layer: str | None = None,
    entity_type: str | None = None,
    limit: int = 100
) -> dict:
    """
    Get entities in the drawing, optionally filtered by layer or type.

    Args:
        layer: Layer name to filter by (optional)
        entity_type: Entity type to filter by (e.g., "Line", "Circle", "Polyline")
        limit: Maximum number of entities to return (default 100)

    Returns:
        List of entities with their properties
    """
    params = {"limit": limit, "include_geometry": True}
    if layer:
        params["layer"] = layer.strip()
    if entity_type:
        params["entity_type"] = entity_type

    try:
        result = await call_autocad_command("get_entities", params)
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)


# =============================================================================
# Entity Management
# =============================================================================

@mcp.tool()
@with_tool_lock(get_lock())
async def autocad_delete_entity(handle: str) -> dict:
    """
    Delete an entity from the AutoCAD drawing by its handle.

    Use autocad_get_entities to find entity handles first.

    Args:
        handle: Entity handle (hex string, e.g. "1A3")

    Returns:
        Deletion confirmation with entity type

    Example:
        autocad_delete_entity("1A3") - Delete entity with handle 1A3
    """
    if not handle or not handle.strip():
        return error_result(ErrorCode.INVALID_PARAMS, "handle is required")

    try:
        result = await call_autocad_command("delete_entity", {"handle": handle.strip()})
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
    except Exception as e:
        logger.error("Unexpected error in autocad_delete_entity", error=str(e), exc_info=True)
        return error_result(ErrorCode.INTERNAL_ERROR, f"Unexpected error: {str(e)}")


# =============================================================================
# Advanced Drawing Operations (Arc, Ellipse, Spline)
# =============================================================================

@mcp.tool()
@with_tool_lock(get_lock())
async def autocad_draw_arc(
    center_x: float,
    center_y: float,
    radius: float,
    start_angle: float,
    end_angle: float,
    layer: str | None = None
) -> dict:
    """
    Draw a circular arc in AutoCAD.

    Args:
        center_x: Center X coordinate
        center_y: Center Y coordinate
        radius: Arc radius
        start_angle: Start angle in degrees (0 = +X axis, counter-clockwise)
        end_angle: End angle in degrees
        layer: Layer name to draw on (optional)

    Returns:
        Created arc entity details

    Example:
        autocad_draw_arc(50, 50, 25, 0, 90) - Quarter circle arc
    """
    if radius <= 0:
        return error_result(ErrorCode.INVALID_PARAMS, "Radius must be positive")

    params = {
        "center": [float(center_x), float(center_y), 0.0],
        "radius": float(radius),
        "start_angle": float(start_angle),
        "end_angle": float(end_angle),
    }
    if layer:
        params["layer"] = layer.strip()

    try:
        result = await call_autocad_command("draw_arc", params)
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
    except Exception as e:
        logger.error("Unexpected error in autocad_draw_arc", error=str(e), exc_info=True)
        return error_result(ErrorCode.INTERNAL_ERROR, f"Unexpected error: {str(e)}")


@mcp.tool()
@with_tool_lock(get_lock())
async def autocad_draw_ellipse(
    center_x: float,
    center_y: float,
    major_end_x: float,
    major_end_y: float,
    axis_ratio: float,
    start_angle: float = 0.0,
    end_angle: float = 360.0,
    layer: str | None = None
) -> dict:
    """
    Draw an ellipse (or elliptical arc) in AutoCAD.

    The major axis is defined by the vector from center to major_end.
    The minor axis length is major_length * axis_ratio.

    For a full ellipse, use start_angle=0, end_angle=360.
    For an elliptical arc, specify the angular range.

    Args:
        center_x: Center X coordinate
        center_y: Center Y coordinate
        major_end_x: Major axis endpoint X (relative to center)
        major_end_y: Major axis endpoint Y (relative to center)
        axis_ratio: Minor-to-major axis ratio (0 < ratio <= 1)
        start_angle: Start angle in degrees (default 0, full ellipse)
        end_angle: End angle in degrees (default 360, full ellipse)
        layer: Layer name to draw on (optional)

    Returns:
        Created ellipse entity details

    Example:
        autocad_draw_ellipse(50, 50, 30, 0, 0.5) - Ellipse with 2:1 ratio
    """
    if axis_ratio <= 0 or axis_ratio > 1:
        return error_result(ErrorCode.INVALID_PARAMS, "axis_ratio must be > 0 and <= 1")

    params = {
        "center": [float(center_x), float(center_y), 0.0],
        "major_axis_endpoint": [float(major_end_x), float(major_end_y), 0.0],
        "axis_ratio": float(axis_ratio),
        "start_angle": float(start_angle),
        "end_angle": float(end_angle),
    }
    if layer:
        params["layer"] = layer.strip()

    try:
        result = await call_autocad_command("draw_ellipse", params)
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
    except Exception as e:
        logger.error("Unexpected error in autocad_draw_ellipse", error=str(e), exc_info=True)
        return error_result(ErrorCode.INTERNAL_ERROR, f"Unexpected error: {str(e)}")


@mcp.tool()
@with_tool_lock(get_lock())
async def autocad_draw_spline(
    points: list[list[float]],
    closed: bool = False,
    layer: str | None = None
) -> dict:
    """
    Draw a spline (smooth curve) through fit points in AutoCAD.

    Creates a NURBS spline that passes through the given fit points.
    Useful for complex curves that cannot be represented as arcs or ellipses.

    Args:
        points: List of fit points as [[x, y], ...] or [[x, y, z], ...]
                At least 2 points required.
        closed: Whether the spline is closed (default False)
        layer: Layer name to draw on (optional)

    Returns:
        Created spline entity details

    Example:
        autocad_draw_spline([[0,0], [10,20], [30,10], [50,25]]) - Smooth curve
    """
    if not points or len(points) < 2:
        return error_result(ErrorCode.INVALID_PARAMS, "At least 2 fit points required")

    # Ensure 3D points
    pts_3d = []
    for pt in points:
        if len(pt) < 2:
            return error_result(ErrorCode.INVALID_PARAMS, "Each point must have at least x and y")
        pts_3d.append([float(pt[0]), float(pt[1]), float(pt[2]) if len(pt) > 2 else 0.0])

    params = {
        "fit_points": pts_3d,
        "closed": closed,
    }
    if layer:
        params["layer"] = layer.strip()

    try:
        result = await call_autocad_command("draw_spline", params)
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
    except Exception as e:
        logger.error("Unexpected error in autocad_draw_spline", error=str(e), exc_info=True)
        return error_result(ErrorCode.INTERNAL_ERROR, f"Unexpected error: {str(e)}")
