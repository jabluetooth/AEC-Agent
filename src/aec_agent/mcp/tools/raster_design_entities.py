"""
MCP tools for AutoCAD Raster Design integration — REM primitives, selection,
and follower VTools.

Split from the former monolithic raster_design.py. Contains:
- raster_create_primitive: create a REM primitive from raster data (isline/isarc/iscircle/issmart)
- raster_select_entities: select raster entities within a rectangular region (isebrcon/isebrsmart)
- raster_follower: semi-automatic follower for tracing raster lines/contours (vfpline/vfcontour/vf3dpoly)
"""


import structlog

from aec_agent.mcp.concurrency import with_tool_lock
from aec_agent.mcp.server import get_lock, mcp
from aec_agent.mcp.sidecar_client import SidecarError, call_autocad_command

from .base import ErrorCode, error_result, success_result
from .pdf_converter import convert_pdf_to_bitonal_tiff

logger = structlog.get_logger(__name__)


# =============================================================================
# REM Primitives — isline, isarc, iscircle, issmart
# =============================================================================

@mcp.tool()
@with_tool_lock(get_lock())
async def raster_create_primitive(
    primitive_type: str = "smart",
    point_x: float | None = None,
    point_y: float | None = None,
) -> dict:
    """
    Create a REM (Raster Entity Manipulation) primitive from raster data.

    Detects a raster entity (line, arc, or circle) and creates an overlay
    primitive that represents it as a vector object. Use "smart" to
    auto-detect the best primitive type.

    Provide coordinates to target a specific raster entity. If omitted,
    AutoCAD prompts for interactive picking.

    Args:
        primitive_type: Primitive type. Options:
            - "smart": Auto-detect best type (issmart, default)
            - "line": Line primitive (isline)
            - "arc": Arc primitive (isarc)
            - "circle": Circle primitive (iscircle)
        point_x: X coordinate of raster entity to convert (optional)
        point_y: Y coordinate of raster entity to convert (optional)

    Returns:
        Queued operation status

    Example:
        raster_create_primitive("smart", point_x=100.0, point_y=200.0)
    """
    valid_types = {"smart", "line", "arc", "circle"}
    if primitive_type not in valid_types:
        return error_result(
            ErrorCode.INVALID_PARAMS,
            f"Invalid primitive_type: {primitive_type}. Valid: {', '.join(sorted(valid_types))}"
        )

    params = {"primitive_type": primitive_type}
    if point_x is not None and point_y is not None:
        params["point"] = [float(point_x), float(point_y)]

    try:
        result = await call_autocad_command("raster_create_primitive", params)
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
    except Exception as e:
        logger.error("Unexpected error in raster_create_primitive", error=str(e), exc_info=True)
        return error_result(ErrorCode.INTERNAL_ERROR, f"Unexpected error: {str(e)}")


# =============================================================================
# Select Raster Entities — isebrcon, isebrsmart
# =============================================================================

@mcp.tool()
@with_tool_lock(get_lock())
async def raster_select_entities(
    method: str = "smart",
    corner1_x: float | None = None,
    corner1_y: float | None = None,
    corner2_x: float | None = None,
    corner2_y: float | None = None,
) -> dict:
    """
    Select raster entities within a rectangular region.

    Uses Raster Design's bitonal region selection to identify and select
    complete raster entities. Selected entities become REM objects that
    can be converted to primitives with raster_create_primitive.

    Args:
        method: Selection method. Options:
            - "smart": Smart entity detection (isebrsmart, default)
            - "crossing": Crossing rectangle selection (isebrcon)
        corner1_x: First corner X coordinate (optional)
        corner1_y: First corner Y coordinate (optional)
        corner2_x: Opposite corner X coordinate (optional)
        corner2_y: Opposite corner Y coordinate (optional)

    Returns:
        Queued operation status

    Example:
        raster_select_entities("smart", 0, 0, 1000, 1000)
    """
    valid_methods = {"smart", "crossing"}
    if method not in valid_methods:
        return error_result(
            ErrorCode.INVALID_PARAMS,
            f"Invalid method: {method}. Valid: {', '.join(sorted(valid_methods))}"
        )

    params = {"method": method}
    if (corner1_x is not None and corner1_y is not None and
            corner2_x is not None and corner2_y is not None):
        params["corner1"] = [float(corner1_x), float(corner1_y)]
        params["corner2"] = [float(corner2_x), float(corner2_y)]

    try:
        result = await call_autocad_command("raster_select_entities", params)
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
    except Exception as e:
        logger.error("Unexpected error in raster_select_entities", error=str(e), exc_info=True)
        return error_result(ErrorCode.INTERNAL_ERROR, f"Unexpected error: {str(e)}")


# =============================================================================
# Follower VTools — vfpline, vfcontour, vf3dpoly
# =============================================================================

@mcp.tool()
@with_tool_lock(get_lock())
async def raster_follower(
    follower_type: str = "polyline",
    start_point_x: float | None = None,
    start_point_y: float | None = None,
    target_layer: str | None = None,
) -> dict:
    """
    Semi-automatic follower for tracing raster lines and contours.

    The follower traces along raster data from a starting point,
    automatically following the raster path and creating vector entities.
    Useful for complex geometry like contour lines and winding paths.

    Provide a starting point to begin tracing. If omitted, AutoCAD
    prompts for interactive picking.

    Args:
        follower_type: Follower type. Options:
            - "polyline": Follow raster polyline path (vfpline, default)
            - "contour": Follow raster contour line (vfcontour)
            - "3dpoly": Create 3D polyline from raster (vf3dpoly)
        start_point_x: X coordinate to start following from (optional)
        start_point_y: Y coordinate to start following from (optional)
        target_layer: Layer to place followed entities on (optional)

    Returns:
        Queued operation status

    Example:
        raster_follower("polyline", start_point_x=50.0, start_point_y=100.0)
    """
    valid_types = {"polyline", "contour", "3dpoly"}
    if follower_type not in valid_types:
        return error_result(
            ErrorCode.INVALID_PARAMS,
            f"Invalid follower_type: {follower_type}. Valid: {', '.join(sorted(valid_types))}"
        )

    params = {"follower_type": follower_type}
    if start_point_x is not None and start_point_y is not None:
        params["start_point"] = [float(start_point_x), float(start_point_y)]
    if target_layer:
        params["target_layer"] = target_layer.strip()

    try:
        result = await call_autocad_command("raster_follower", params)
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
    except Exception as e:
        logger.error("Unexpected error in raster_follower", error=str(e), exc_info=True)
        return error_result(ErrorCode.INTERNAL_ERROR, f"Unexpected error: {str(e)}")
