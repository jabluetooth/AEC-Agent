"""
MCP tools for AutoCAD Raster Design integration — status, query, and storage.

Split from the former monolithic raster_design.py. Contains:
- raster_get_status: list raster images in the current drawing
- raster_get_entity_count: count entities, optionally filtered by layer
- raster_store_vectorized: extract entities from the drawing and store in PostgreSQL
"""


import structlog

from aec_agent.mcp.concurrency import with_tool_lock
from aec_agent.mcp.server import get_lock, mcp
from aec_agent.mcp.sidecar_client import SidecarError, call_autocad_command

from .base import ErrorCode, error_result, success_result
from .pdf_converter import convert_pdf_to_bitonal_tiff

logger = structlog.get_logger(__name__)


# =============================================================================
# Status & Query
# =============================================================================

@mcp.tool()
async def raster_get_status() -> dict:
    """
    Get information about all raster images in the current AutoCAD drawing.

    Returns image names, dimensions, file paths, resolution, and load status.
    Also checks if AutoCAD Raster Design is available.

    Returns:
        List of raster images with their properties, and Raster Design availability
    """
    try:
        result = await call_autocad_command("raster_get_status")
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
    except Exception as e:
        logger.error("Unexpected error in raster_get_status", error=str(e), exc_info=True)
        return error_result(ErrorCode.INTERNAL_ERROR, f"Unexpected error: {str(e)}")


@mcp.tool()
async def raster_get_entity_count(
    layer: str | None = None
) -> dict:
    """
    Count entities in the drawing, optionally filtered by layer.

    Useful for checking results after PDF import or vectorization operations.
    Returns total count and breakdown by entity type and layer.

    Args:
        layer: Filter by layer name (optional, counts all if omitted)

    Returns:
        Entity counts (total, by type, by layer)

    Example:
        raster_get_entity_count("Vectorized")
    """
    params = {}
    if layer:
        params["layer"] = layer.strip()

    try:
        result = await call_autocad_command("raster_get_entity_count", params)
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
    except Exception as e:
        logger.error("Unexpected error in raster_get_entity_count", error=str(e), exc_info=True)
        return error_result(ErrorCode.INTERNAL_ERROR, f"Unexpected error: {str(e)}")


# =============================================================================
# PostgreSQL Storage
# =============================================================================

@mcp.tool()
@with_tool_lock(get_lock())
async def raster_store_vectorized(
    file_path: str | None = None,
    force: bool = True,
) -> dict:
    """
    Extract all entities from the current AutoCAD drawing and store in PostgreSQL.

    Performs full extraction including:
    - Entity metadata (type, layer, color, linetype)
    - Geometry (PostGIS spatial data)
    - Embeddings (pgvector for semantic search)
    - Spatial relationships between entities

    Use after vectorization to persist results for knowledge base referencing.

    Args:
        file_path: Path to DWG file (optional, uses active document if omitted)
        force: Force re-extraction even if unchanged (default True)

    Returns:
        Extraction summary with project_id, entity counts, and breakdown

    Example:
        raster_store_vectorized()
    """
    from aec_agent.mcp.server import get_database_pool, get_sync_manager

    pool = get_database_pool()
    sm = get_sync_manager()

    if not pool:
        return error_result(
            ErrorCode.MISSING_CONFIG,
            "PostgreSQL database not configured. Set DATABASE_URL."
        )

    if not sm:
        return error_result(
            ErrorCode.INTERNAL_ERROR,
            "Sync manager not available."
        )

    try:
        # Get file path from active drawing if not provided
        if not file_path:
            drawing_info = await call_autocad_command("get_drawing_info")
            if drawing_info.get("success") and drawing_info.get("data"):
                file_path = drawing_info["data"].get("file_path", "")
            if not file_path:
                return error_result(
                    ErrorCode.INVALID_PARAMS,
                    "No file_path provided and no active drawing found"
                )

        # Run full sync (extraction + embeddings + relationships)
        result = await sm.trigger_full_sync("autocad", file_path, force=force)

        if not result.success:
            return error_result(
                ErrorCode.INTERNAL_ERROR,
                "Extraction failed",
                "; ".join(result.errors)
            )

        # Get entity breakdown from the drawing
        count_result = await call_autocad_command("raster_get_entity_count")
        type_breakdown = {}
        layer_breakdown = {}
        if count_result.get("success") and count_result.get("data"):
            type_breakdown = count_result["data"].get("by_type", {})
            layer_breakdown = count_result["data"].get("by_layer", {})

        return success_result(
            data={
                "project_id": str(result.project_id),
                "elements_extracted": result.elements_extracted,
                "elements_updated": result.elements_updated,
                "embeddings_generated": result.embeddings_generated,
                "relationships_computed": result.relationships_computed,
                "duration_ms": result.duration_ms,
                "by_entity_type": type_breakdown,
                "by_layer": layer_breakdown,
            },
            message=f"Stored {result.elements_extracted} entities in PostgreSQL"
        )

    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
    except Exception as e:
        logger.error("Failed to store vectorized entities", error=str(e), exc_info=True)
        return error_result(ErrorCode.INTERNAL_ERROR, f"Storage failed: {str(e)}")
