"""
Metadata and semantic search MCP tools.

Provides tools for:
- Semantic search for elements
- Spatial proximity queries
- Element resolution from natural language
"""

from typing import Optional, List
from uuid import UUID

import structlog

from aec_agent.mcp.server import mcp
from aec_agent.mcp.tools.base import (
    success_result,
    error_result,
    safe_tool,
    ErrorCode,
)

logger = structlog.get_logger(__name__)


# Error codes for metadata operations
class MetadataErrorCode:
    DATABASE_NOT_CONFIGURED = 6001
    ELEMENT_NOT_FOUND = 6002
    AMBIGUOUS_REFERENCE = 6003
    EMBEDDING_ERROR = 6004
    PROJECT_NOT_FOUND = 6005


async def _get_services():
    """Get database and semantic services."""
    from aec_agent.db.connection import get_database_pool
    from aec_agent.semantic.embeddings import get_embedding_service

    pool = await get_database_pool()
    embeddings = get_embedding_service()

    return pool, embeddings


async def _get_active_project_id() -> Optional[UUID]:
    """Get the currently active project ID from cache or recent sync."""
    from aec_agent.mcp.server import get_cache

    cache = get_cache()
    if cache:
        project_id = await cache.get_metadata("active_project_id")
        if project_id:
            try:
                return UUID(project_id)
            except ValueError:
                pass
    return None


@mcp.tool()
@safe_tool
async def find_elements(
    query: str,
    source: Optional[str] = None,
    category: Optional[str] = None,
    layer: Optional[str] = None,
    limit: int = 10,
) -> dict:
    """
    Search for elements using natural language.

    Uses semantic search (vector similarity) combined with filters
    to find relevant CAD elements.

    Args:
        query: Natural language search (e.g., "fire-rated walls", "columns on level 1")
        source: Filter by source ("autocad" or "revit")
        category: Filter by category (Revit) or entity type (AutoCAD)
        layer: Filter by layer name (AutoCAD only)
        limit: Maximum results to return (default 10)

    Returns:
        List of matching elements with id, description, and location

    Example queries:
        - "exterior walls"
        - "doors near the elevator"
        - "columns on level 2"
        - "fire rated walls with 1HR rating"
    """
    pool, embeddings = await _get_services()

    if not pool:
        return error_result(
            MetadataErrorCode.DATABASE_NOT_CONFIGURED,
            "Database not configured",
            "Set DATABASE_URL to enable semantic search"
        )

    if not embeddings:
        return error_result(
            MetadataErrorCode.EMBEDDING_ERROR,
            "Embedding service not available",
            "Install sentence-transformers: pip install sentence-transformers"
        )

    project_id = await _get_active_project_id()
    if not project_id:
        return error_result(
            MetadataErrorCode.PROJECT_NOT_FOUND,
            "No active project",
            "Extract a drawing first using the extraction tools"
        )

    from aec_agent.semantic.search import SemanticSearch

    search = SemanticSearch(pool, embeddings)

    # Map category to entity_type for AutoCAD
    entity_type = None
    if source == "autocad" and category:
        entity_type = category.upper()
        category = None

    elements = await search.search(
        query=query,
        project_id=project_id,
        entity_type=entity_type,
        category=category,
        layer=layer,
        limit=limit,
    )

    results = [e.to_search_result() for e in elements]

    return success_result(
        data={"elements": results, "count": len(results)},
        message=f"Found {len(results)} elements"
    )


@mcp.tool()
@safe_tool
async def get_nearby_elements(
    element_id: str,
    distance: float = 1.0,
    category: Optional[str] = None,
    limit: int = 20,
) -> dict:
    """
    Find elements near a specific element.

    Uses spatial proximity queries (PostGIS) to find nearby elements.

    Args:
        element_id: Source element ID (Handle, ElementId, or UUID)
        distance: Search radius in meters (default 1.0)
        category: Filter by category/type
        limit: Maximum results (default 20)

    Returns:
        List of nearby elements with distances
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

    from aec_agent.db.repository import ElementRepository

    repo = ElementRepository(pool)

    # Resolve element ID
    element = await repo.get_element_by_source_id(project_id, element_id)
    if not element:
        # Try as UUID
        try:
            element = await repo.get_element(UUID(element_id))
        except ValueError:
            pass

    if not element:
        return error_result(
            MetadataErrorCode.ELEMENT_NOT_FOUND,
            f"Element not found: {element_id}"
        )

    # Get nearby elements
    nearby = await repo.get_nearby_elements(element.id, distance, limit)

    # Filter by category if specified
    if category:
        category_upper = category.upper()
        nearby = [
            e for e in nearby
            if (e.category and category_upper in e.category.upper()) or
               (e.entity_type and category_upper in e.entity_type.upper())
        ]

    results = [e.to_search_result() for e in nearby]

    return success_result(
        data={
            "elements": results,
            "count": len(results),
            "reference_element": element.source_id,
        },
        message=f"Found {len(results)} elements within {distance}m"
    )


@mcp.tool()
@safe_tool
async def get_related_elements(
    element_id: str,
    relation_type: Optional[str] = None,
    limit: int = 20,
) -> dict:
    """
    Get elements related to a specific element.

    Finds elements connected through spatial or logical relationships.

    Args:
        element_id: Element ID to find relations for
        relation_type: Filter by relation type:
            - "intersects": Geometries touch/overlap
            - "near": Within distance threshold
            - "hosts": Element hosts another (e.g., wall hosts door)
            - "connected_to": Structural connection
            - "on_layer": Same AutoCAD layer
        limit: Maximum results (default 20)

    Returns:
        List of related elements with relationship details
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

    from aec_agent.db.repository import ElementRepository

    repo = ElementRepository(pool)

    # Resolve element ID
    element = await repo.get_element_by_source_id(project_id, element_id)
    if not element:
        try:
            element = await repo.get_element(UUID(element_id))
        except ValueError:
            pass

    if not element:
        return error_result(
            MetadataErrorCode.ELEMENT_NOT_FOUND,
            f"Element not found: {element_id}"
        )

    # Get related elements
    relation_types = [relation_type] if relation_type else None
    related = await repo.get_related_elements(element.id, relation_types, limit)

    results = [e.to_search_result() for e in related]

    return success_result(
        data={
            "elements": results,
            "count": len(results),
            "reference_element": element.source_id,
            "relation_filter": relation_type,
        },
        message=f"Found {len(results)} related elements"
    )


@mcp.tool()
@safe_tool
async def resolve_coordinates(
    element_reference: str,
) -> dict:
    """
    Get coordinates for an element reference.

    Resolves a natural language description or ID to specific coordinates.
    This is the core function for smart drawing tools.

    Args:
        element_reference: Element ID or natural language description
            Examples: "column C3", "the east wall", "123456" (ElementId)

    Returns:
        Centroid coordinates {x, y, z} and bounding box
    """
    pool, embeddings = await _get_services()

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

    from aec_agent.semantic.search import resolve_element

    if embeddings:
        element = await resolve_element(
            element_reference, project_id, pool, embeddings
        )
    else:
        # Fall back to ID-only resolution
        from aec_agent.db.repository import ElementRepository
        repo = ElementRepository(pool)
        element = await repo.get_element_by_source_id(project_id, element_reference)

    if not element:
        return error_result(
            MetadataErrorCode.ELEMENT_NOT_FOUND,
            f"Could not resolve: {element_reference}",
            "Try using a more specific description or the element ID"
        )

    result = {
        "element_id": str(element.id),
        "source_id": element.source_id,
        "source": element.source,
        "entity_type": element.entity_type,
        "description": element.description,
    }

    if element.centroid:
        result["centroid"] = element.centroid.model_dump()

    if element.bounds:
        result["bounds"] = element.bounds.model_dump()

    return success_result(data=result)


@mcp.tool()
@safe_tool
async def sync_metadata(
    source: str = "autocad",
    file_path: Optional[str] = None,
) -> dict:
    """
    Trigger metadata extraction and synchronization.

    Extracts all entities from the current drawing/model into the
    PostgreSQL database for semantic search.

    Args:
        source: Source application ("autocad" or "revit")
        file_path: Optional file path override

    Returns:
        Sync statistics including elements extracted and relationships computed
    """
    pool, embeddings = await _get_services()

    if not pool:
        return error_result(
            MetadataErrorCode.DATABASE_NOT_CONFIGURED,
            "Database not configured",
            "Set DATABASE_URL environment variable"
        )

    # Get file path from sidecar if not provided
    if not file_path:
        from aec_agent.mcp.sidecar_client import call_autocad_command, call_sidecar

        try:
            if source == "autocad":
                response = await call_autocad_command("get_drawing_info", {})
                if response.get("success"):
                    file_path = response.get("data", {}).get("file_path")
            else:
                response = await call_sidecar(
                    endpoint="/mcp/status",
                    method="GET",
                    sidecar_type="revit"
                )
                if response.get("success"):
                    file_path = response.get("data", {}).get("path")
        except Exception as e:
            logger.warning("Could not get file path from sidecar", error=str(e))

    if not file_path:
        file_path = f"unknown_{source}_document"

    from aec_agent.extraction.sync_manager import SyncManager

    sync_manager = SyncManager(pool, embeddings)
    result = await sync_manager.trigger_full_sync(source, file_path)

    # Store active project ID
    cache = await _get_cache()
    if cache:
        await cache.set_metadata("active_project_id", str(result.project_id))

    if result.success:
        return success_result(
            data={
                "project_id": str(result.project_id),
                "elements_extracted": result.elements_extracted,
                "relationships_computed": result.relationships_computed,
                "embeddings_generated": result.embeddings_generated,
                "duration_ms": result.duration_ms,
            },
            message=f"Synced {result.elements_extracted} elements"
        )
    else:
        return error_result(
            ErrorCode.INTERNAL_ERROR,
            "Sync failed",
            "; ".join(result.errors)
        )


async def _get_cache():
    """Get cache manager."""
    from aec_agent.mcp.server import get_cache
    return get_cache()
