"""
Smart drawing tools with automatic coordinate resolution.

These tools accept element references (natural language or IDs)
and automatically resolve them to coordinates before executing.
"""

from uuid import UUID

import structlog

from aec_agent.mcp.concurrency import with_tool_lock
from aec_agent.mcp.server import get_lock, mcp
from aec_agent.mcp.sidecar_client import call_autocad_command
from aec_agent.mcp.tools.base import (
    ErrorCode,
    error_result,
    safe_tool,
    success_result,
)
from aec_agent.mcp.tools.metadata import MetadataErrorCode, _get_active_project_id, _get_services

logger = structlog.get_logger(__name__)


async def _resolve_element_to_point(
    reference: str,
    project_id: UUID,
) -> dict | None:
    """
    Resolve element reference to centroid coordinates.

    Args:
        reference: Element ID or description
        project_id: Project to search

    Returns:
        Dict with x, y, z coordinates or None
    """
    pool, embeddings = await _get_services()
    if not pool:
        return None

    from aec_agent.semantic.search import resolve_element

    if embeddings:
        element = await resolve_element(reference, project_id, pool, embeddings)
    else:
        from aec_agent.db.repository import ElementRepository
        repo = ElementRepository(pool)
        element = await repo.get_element_by_source_id(project_id, reference)

    if element and element.centroid:
        return element.centroid.model_dump()

    return None


@mcp.tool()
@safe_tool
@with_tool_lock(get_lock())
async def draw_line_between(
    from_element: str,
    to_element: str,
    layer: str | None = None,
) -> dict:
    """
    Draw a line between two elements, resolving coordinates automatically.

    Instead of specifying exact coordinates, you can reference elements
    by description or ID. The tool resolves the centroid of each element
    and draws a line between them.

    Args:
        from_element: Source element (ID or description like "the east wall")
        to_element: Target element (ID or description like "column C3")
        layer: Layer name for the new line (optional)

    Returns:
        Created line details with resolved coordinates

    Example:
        draw_line_between("the reception desk", "the main entrance")
        draw_line_between("column A1", "column B1", layer="STRUCTURE")
    """
    pool, _ = await _get_services()

    if not pool:
        return error_result(
            MetadataErrorCode.DATABASE_NOT_CONFIGURED,
            "Database not configured - use autocad_draw_line with coordinates instead"
        )

    project_id = await _get_active_project_id()
    if not project_id:
        return error_result(
            MetadataErrorCode.PROJECT_NOT_FOUND,
            "No active project - run sync_metadata first"
        )

    # Resolve source element
    from_point = await _resolve_element_to_point(from_element, project_id)
    if not from_point:
        return error_result(
            MetadataErrorCode.ELEMENT_NOT_FOUND,
            f"Could not resolve 'from' element: {from_element}"
        )

    # Resolve target element
    to_point = await _resolve_element_to_point(to_element, project_id)
    if not to_point:
        return error_result(
            MetadataErrorCode.ELEMENT_NOT_FOUND,
            f"Could not resolve 'to' element: {to_element}"
        )

    # Draw the line using AutoCAD tool
    params = {
        "start_x": from_point["x"],
        "start_y": from_point["y"],
        "end_x": to_point["x"],
        "end_y": to_point["y"],
    }
    if layer:
        params["layer"] = layer

    response = await call_autocad_command("draw_line", params)

    if response.get("success"):
        result_data = response.get("data", {})
        result_data["from_element"] = from_element
        result_data["to_element"] = to_element
        result_data["from_point"] = from_point
        result_data["to_point"] = to_point

        return success_result(
            data=result_data,
            message=f"Drew line from '{from_element}' to '{to_element}'"
        )
    else:
        error = response.get("error", {})
        return error_result(
            error.get("code", ErrorCode.SIDECAR_ERROR),
            error.get("message", "Failed to draw line"),
            error.get("details")
        )


@mcp.tool()
@safe_tool
@with_tool_lock(get_lock())
async def draw_circle_at(
    element: str,
    radius: float,
    layer: str | None = None,
) -> dict:
    """
    Draw a circle centered on an element.

    Args:
        element: Center element (ID or description)
        radius: Circle radius
        layer: Layer name for the circle (optional)

    Returns:
        Created circle details

    Example:
        draw_circle_at("column C3", radius=0.5, layer="ANNOTATION")
    """
    pool, _ = await _get_services()

    if not pool:
        return error_result(
            MetadataErrorCode.DATABASE_NOT_CONFIGURED,
            "Database not configured"
        )

    project_id = await _get_active_project_id()
    if not project_id:
        return error_result(
            MetadataErrorCode.PROJECT_NOT_FOUND,
            "No active project"
        )

    # Resolve element
    center = await _resolve_element_to_point(element, project_id)
    if not center:
        return error_result(
            MetadataErrorCode.ELEMENT_NOT_FOUND,
            f"Could not resolve element: {element}"
        )

    # Draw the circle
    params = {
        "center_x": center["x"],
        "center_y": center["y"],
        "radius": radius,
    }
    if layer:
        params["layer"] = layer

    response = await call_autocad_command("draw_circle", params)

    if response.get("success"):
        result_data = response.get("data", {})
        result_data["reference_element"] = element
        result_data["center"] = center

        return success_result(
            data=result_data,
            message=f"Drew circle at '{element}'"
        )
    else:
        error = response.get("error", {})
        return error_result(
            error.get("code", ErrorCode.SIDECAR_ERROR),
            error.get("message", "Failed to draw circle"),
            error.get("details")
        )


@mcp.tool()
@safe_tool
@with_tool_lock(get_lock())
async def draw_rectangle_around(
    element: str,
    padding: float = 0.5,
    layer: str | None = None,
) -> dict:
    """
    Draw a rectangle around an element's bounding box.

    Args:
        element: Element to surround (ID or description)
        padding: Padding around the bounding box in meters (default 0.5)
        layer: Layer name for the rectangle (optional)

    Returns:
        Created rectangle details

    Example:
        draw_rectangle_around("the reception desk", padding=1.0)
    """
    pool, _ = await _get_services()

    if not pool:
        return error_result(
            MetadataErrorCode.DATABASE_NOT_CONFIGURED,
            "Database not configured"
        )

    project_id = await _get_active_project_id()
    if not project_id:
        return error_result(
            MetadataErrorCode.PROJECT_NOT_FOUND,
            "No active project"
        )

    # Resolve element and get bounds
    from aec_agent.semantic.search import resolve_element

    pool, embeddings = await _get_services()

    if embeddings:
        resolved = await resolve_element(element, project_id, pool, embeddings)
    else:
        from aec_agent.db.repository import ElementRepository
        repo = ElementRepository(pool)
        resolved = await repo.get_element_by_source_id(project_id, element)

    if not resolved:
        return error_result(
            MetadataErrorCode.ELEMENT_NOT_FOUND,
            f"Could not resolve element: {element}"
        )

    if not resolved.bounds:
        # Use centroid with default size
        if resolved.centroid:
            bounds = {
                "min_x": resolved.centroid.x - 1,
                "min_y": resolved.centroid.y - 1,
                "max_x": resolved.centroid.x + 1,
                "max_y": resolved.centroid.y + 1,
            }
        else:
            return error_result(
                MetadataErrorCode.ELEMENT_NOT_FOUND,
                f"Element has no bounds or centroid: {element}"
            )
    else:
        bounds = resolved.bounds.model_dump()

    # Add padding
    params = {
        "corner1_x": bounds["min_x"] - padding,
        "corner1_y": bounds["min_y"] - padding,
        "corner2_x": bounds["max_x"] + padding,
        "corner2_y": bounds["max_y"] + padding,
    }
    if layer:
        params["layer"] = layer

    response = await call_autocad_command("draw_rectangle", params)

    if response.get("success"):
        result_data = response.get("data", {})
        result_data["reference_element"] = element
        result_data["padding"] = padding

        return success_result(
            data=result_data,
            message=f"Drew rectangle around '{element}'"
        )
    else:
        error = response.get("error", {})
        return error_result(
            error.get("code", ErrorCode.SIDECAR_ERROR),
            error.get("message", "Failed to draw rectangle"),
            error.get("details")
        )


@mcp.tool()
@safe_tool
async def get_distance_between(
    element1: str,
    element2: str,
) -> dict:
    """
    Calculate the distance between two elements.

    Args:
        element1: First element (ID or description)
        element2: Second element (ID or description)

    Returns:
        Distance in meters and element details
    """
    pool, _ = await _get_services()

    if not pool:
        return error_result(
            MetadataErrorCode.DATABASE_NOT_CONFIGURED,
            "Database not configured"
        )

    project_id = await _get_active_project_id()
    if not project_id:
        return error_result(
            MetadataErrorCode.PROJECT_NOT_FOUND,
            "No active project"
        )

    # Resolve both elements
    point1 = await _resolve_element_to_point(element1, project_id)
    if not point1:
        return error_result(
            MetadataErrorCode.ELEMENT_NOT_FOUND,
            f"Could not resolve element: {element1}"
        )

    point2 = await _resolve_element_to_point(element2, project_id)
    if not point2:
        return error_result(
            MetadataErrorCode.ELEMENT_NOT_FOUND,
            f"Could not resolve element: {element2}"
        )

    # Calculate 3D distance
    import math
    distance = math.sqrt(
        (point2["x"] - point1["x"]) ** 2 +
        (point2["y"] - point1["y"]) ** 2 +
        (point2.get("z", 0) - point1.get("z", 0)) ** 2
    )

    # Calculate 2D distance (plan view)
    distance_2d = math.sqrt(
        (point2["x"] - point1["x"]) ** 2 +
        (point2["y"] - point1["y"]) ** 2
    )

    return success_result(
        data={
            "distance_3d": round(distance, 3),
            "distance_2d": round(distance_2d, 3),
            "element1": {
                "reference": element1,
                "point": point1,
            },
            "element2": {
                "reference": element2,
                "point": point2,
            },
        },
        message=f"Distance: {distance_2d:.2f}m (plan), {distance:.2f}m (3D)"
    )
