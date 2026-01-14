"""
MCP tools for Revit automation.
"""

from typing import Optional

from aec_agent.mcp.server import mcp, get_lock, get_cache
from aec_agent.mcp.concurrency import with_tool_lock
from aec_agent.mcp.sidecar_client import call_sidecar, SidecarError
from aec_agent.config.settings import get_settings
from .base import success_result, error_result, ErrorCode

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Level Operations
# =============================================================================

@mcp.tool()
async def revit_list_levels() -> dict:
    """
    List all levels in the current Revit document.

    Returns:
        List of levels with id, name, and elevation in meters
    """
    try:
        result = await call_sidecar(
            endpoint="/mcp/levels",
            method="GET",
            sidecar_type="revit"
        )
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)


@mcp.tool()
@with_tool_lock(get_lock())
async def revit_create_level(name: str, elevation: float) -> dict:
    """
    Create a new level in Revit.

    Args:
        name: Name for the new level (e.g., "Level 3", "Roof")
        elevation: Elevation in meters above ground level

    Returns:
        Created level details including id

    Example:
        revit_create_level("Level 2", 3.5) - Creates level at 3.5m height
    """
    if not name or not name.strip():
        return error_result(ErrorCode.INVALID_PARAMS, "Level name is required")

    if not isinstance(elevation, (int, float)):
        return error_result(ErrorCode.INVALID_PARAMS, "Elevation must be a number")

    try:
        result = await call_sidecar(
            endpoint="/mcp/levels/create",
            method="POST",
            payload={"name": name.strip(), "elevation": float(elevation)},
            sidecar_type="revit"
        )
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)


# =============================================================================
# Wall Operations
# =============================================================================

@mcp.tool()
async def revit_list_walls() -> dict:
    """
    List all walls in the current Revit document.

    Returns:
        List of walls with id, type, level, and length
    """
    try:
        result = await call_sidecar(
            endpoint="/mcp/walls",
            method="GET",
            sidecar_type="revit"
        )
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)


@mcp.tool()
@with_tool_lock(get_lock())
async def revit_create_wall(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    level_name: str,
    height: float = 3.0
) -> dict:
    """
    Create a wall in Revit between two points.

    Args:
        start_x: Start X coordinate in meters
        start_y: Start Y coordinate in meters
        end_x: End X coordinate in meters
        end_y: End Y coordinate in meters
        level_name: Name of the level to place the wall on
        height: Wall height in meters (default 3.0)

    Returns:
        Created wall details including id

    Example:
        revit_create_wall(0, 0, 10, 0, "Level 1", 3.0) - Creates 10m wall on Level 1
    """
    # Validate coordinates
    for coord_name, coord_val in [
        ("start_x", start_x), ("start_y", start_y),
        ("end_x", end_x), ("end_y", end_y), ("height", height)
    ]:
        if not isinstance(coord_val, (int, float)):
            return error_result(
                ErrorCode.INVALID_PARAMS,
                f"{coord_name} must be a number"
            )

    if not level_name or not level_name.strip():
        return error_result(ErrorCode.INVALID_PARAMS, "level_name is required")

    # Check if start and end are different
    if start_x == end_x and start_y == end_y:
        return error_result(
            ErrorCode.INVALID_PARAMS,
            "Start and end points cannot be the same"
        )

    # Look up level ID from cache
    cache = get_cache()
    level = await cache.get_level_by_name(level_name.strip())

    if not level:
        # Try fetching from sidecar
        levels_result = await call_sidecar(
            endpoint="/mcp/levels",
            method="GET",
            sidecar_type="revit"
        )
        if levels_result.get("success"):
            for lvl in levels_result.get("data", {}).get("levels", []):
                if lvl["name"].lower() == level_name.strip().lower():
                    level = lvl
                    break

    if not level:
        return error_result(
            ErrorCode.ELEMENT_NOT_FOUND,
            f"Level '{level_name}' not found"
        )

    try:
        result = await call_sidecar(
            endpoint="/mcp/walls/create",
            method="POST",
            payload={
                "start": {"x": float(start_x), "y": float(start_y)},
                "end": {"x": float(end_x), "y": float(end_y)},
                "level_id": level["id"],
                "height": float(height)
            },
            sidecar_type="revit"
        )
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)


# =============================================================================
# Room Operations
# =============================================================================

@mcp.tool()
async def revit_list_rooms(use_cache: bool = True) -> dict:
    """
    List all rooms in the current Revit document.

    Args:
        use_cache: If True, uses cached data for faster response.
                  If False, queries Revit directly (slower but current).

    Returns:
        List of rooms with id, name, number, level, and area
    """
    endpoint = "/mcp/rooms/cached" if use_cache else "/mcp/rooms"

    try:
        result = await call_sidecar(
            endpoint=endpoint,
            method="GET",
            sidecar_type="revit"
        )
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)


# =============================================================================
# Document Operations
# =============================================================================

@mcp.tool()
async def revit_get_document_status() -> dict:
    """
    Get current Revit document information.

    Returns:
        Document title, path, modification state, and worksharing info
    """
    try:
        result = await call_sidecar(
            endpoint="/mcp/status",
            method="GET",
            sidecar_type="revit"
        )
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)


@mcp.tool()
@with_tool_lock(get_lock())
async def revit_delete_element(element_id: int) -> dict:
    """
    Delete an element from Revit by its ID.

    Args:
        element_id: The Revit element ID to delete

    Returns:
        Success confirmation or error

    Warning:
        This operation cannot be undone via the API. Use with caution.
    """
    if not isinstance(element_id, int):
        return error_result(ErrorCode.INVALID_PARAMS, "element_id must be an integer")

    try:
        result = await call_sidecar(
            endpoint="/mcp/elements/delete",
            method="POST",
            payload={"element_id": element_id},
            sidecar_type="revit"
        )
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
