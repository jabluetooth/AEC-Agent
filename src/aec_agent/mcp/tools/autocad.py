"""
MCP tools for AutoCAD automation.
"""

from typing import Optional, List

from aec_agent.mcp.server import mcp, get_lock, get_cache
from aec_agent.mcp.concurrency import with_tool_lock
from aec_agent.mcp.sidecar_client import call_sidecar, SidecarError
from .base import success_result, error_result, ErrorCode

import structlog

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
        result = await call_sidecar(
            endpoint="/layers",
            method="GET",
            sidecar_type="autocad"
        )
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)


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
        result = await call_sidecar(
            endpoint="/layers/create",
            method="POST",
            payload={"name": name.strip(), "color": color},
            sidecar_type="autocad"
        )
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)


@mcp.tool()
@with_tool_lock(get_lock())
async def autocad_set_layer_state(
    name: str,
    is_on: Optional[bool] = None,
    is_frozen: Optional[bool] = None
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

    payload = {"name": name.strip()}
    if is_on is not None:
        payload["is_on"] = bool(is_on)
    if is_frozen is not None:
        payload["is_frozen"] = bool(is_frozen)

    try:
        result = await call_sidecar(
            endpoint="/layers/state",
            method="POST",
            payload=payload,
            sidecar_type="autocad"
        )
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
    layer: Optional[str] = None
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

    payload = {
        "start": {"x": float(start_x), "y": float(start_y)},
        "end": {"x": float(end_x), "y": float(end_y)}
    }

    if layer:
        payload["layer"] = layer.strip()

    try:
        result = await call_sidecar(
            endpoint="/draw/line",
            method="POST",
            payload=payload,
            sidecar_type="autocad"
        )
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)


@mcp.tool()
@with_tool_lock(get_lock())
async def autocad_draw_circle(
    center_x: float,
    center_y: float,
    radius: float,
    layer: Optional[str] = None
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

    payload = {
        "center": {"x": float(center_x), "y": float(center_y)},
        "radius": float(radius)
    }

    if layer:
        payload["layer"] = layer.strip()

    try:
        result = await call_sidecar(
            endpoint="/draw/circle",
            method="POST",
            payload=payload,
            sidecar_type="autocad"
        )
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
    layer: Optional[str] = None
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

    payload = {
        "corner1": {"x": float(corner1_x), "y": float(corner1_y)},
        "corner2": {"x": float(corner2_x), "y": float(corner2_y)}
    }

    if layer:
        payload["layer"] = layer.strip()

    try:
        result = await call_sidecar(
            endpoint="/draw/rectangle",
            method="POST",
            payload=payload,
            sidecar_type="autocad"
        )
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
        result = await call_sidecar(
            endpoint="/drawing/info",
            method="GET",
            sidecar_type="autocad"
        )
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)


@mcp.tool()
async def autocad_count_entities(layer: Optional[str] = None) -> dict:
    """
    Count entities in the drawing, optionally filtered by layer.

    Args:
        layer: Layer name to filter by (optional, counts all if not specified)

    Returns:
        Entity counts by type
    """
    endpoint = "/entities/count"
    if layer:
        endpoint += f"?layer={layer}"

    try:
        result = await call_sidecar(
            endpoint=endpoint,
            method="GET",
            sidecar_type="autocad"
        )
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
