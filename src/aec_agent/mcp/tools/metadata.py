"""
Metadata and semantic search MCP tools.

Provides tools for:
- Semantic search for elements
- Spatial proximity queries
- Element resolution from natural language
- Cache-first data access
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
from aec_agent.mcp.tools.cache_helpers import (
    get_element_with_context,
    is_cache_fresh,
    get_cache_stats,
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


async def _get_file_context_from_sidecar(source: Optional[str] = None) -> dict:
    """Fall back to direct sidecar query when PostgreSQL is not available."""
    from aec_agent.mcp.sidecar_client import call_autocad_command, call_sidecar, SidecarError

    # Determine which sidecar to query
    if source is None:
        # Try to read from cache which source is active
        from aec_agent.mcp.server import get_cache
        cache = get_cache()
        if cache:
            source = await cache.get_metadata("active_source")
        if not source:
            source = "autocad"  # default

    if source == "autocad":
        try:
            drawing_info = await call_autocad_command("get_drawing_info", {})
            if drawing_info.get("success"):
                data = drawing_info.get("data", {})
                return success_result(
                    data={
                        "project": {
                            "name": data.get("file_name", "Unknown"),
                            "source": "autocad",
                            "file_path": data.get("file_path", ""),
                        },
                        "summary": {
                            "total_elements": data.get("entity_count", 0),
                            "layer_count": data.get("layer_count", 0),
                        },
                        "drawing_info": data,
                        "note": "Direct sidecar query (PostgreSQL not available for cached metadata).",
                    },
                    message=(
                        f"Drawing: {data.get('file_name', 'Unknown')} — "
                        f"{data.get('entity_count', 0)} entities, "
                        f"{data.get('layer_count', 0)} layers"
                    ),
                )
            else:
                return error_result(
                    ErrorCode.SIDECAR_ERROR,
                    drawing_info.get("error", {}).get("message", "Sidecar query failed"),
                )
        except SidecarError as e:
            return error_result(e.code, e.message, e.details)
        except Exception as e:
            return error_result(
                ErrorCode.SIDECAR_ERROR,
                f"Could not reach AutoCAD sidecar: {e}. Is AutoCAD running with the AEC Agent plugin?",
            )
    elif source == "revit":
        try:
            status = await call_sidecar(
                endpoint="/mcp/status", method="GET", sidecar_type="revit"
            )
            if status.get("success"):
                data = status.get("data", {})
                return success_result(
                    data={
                        "project": {
                            "name": data.get("title", "Unknown"),
                            "source": "revit",
                            "file_path": data.get("path", ""),
                        },
                        "summary": data,
                        "note": "Direct sidecar query (PostgreSQL not available for cached metadata).",
                    },
                    message=f"Revit model: {data.get('title', 'Unknown')}",
                )
            else:
                return error_result(
                    ErrorCode.SIDECAR_ERROR,
                    status.get("error", {}).get("message", "Sidecar query failed"),
                )
        except Exception as e:
            return error_result(
                ErrorCode.SIDECAR_ERROR,
                f"Could not reach Revit sidecar: {e}. Is Revit running with the AEC Agent extension?",
            )

    return error_result(ErrorCode.INVALID_PARAMS, f"Unknown source: {source}")


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
    Uses cache-first pattern for element lookup.

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

    # Cache-first element lookup
    element_data = await get_element_with_context(
        source_id=element_id,
        project_id=project_id,
        fallback_to_sidecar=False,  # Spatial queries require PostgreSQL
    )

    if not element_data:
        return error_result(
            MetadataErrorCode.ELEMENT_NOT_FOUND,
            f"Element not found: {element_id}"
        )

    # Get the actual element object for spatial query
    from aec_agent.db.repository import ElementRepository
    repo = ElementRepository(pool)

    # Use the resolved element ID from cache lookup
    element_uuid = element_data.get("id") or element_data.get("element_id")
    if not element_uuid:
        return error_result(
            MetadataErrorCode.ELEMENT_NOT_FOUND,
            f"Element not found: {element_id}"
        )

    try:
        element = await repo.get_element(UUID(str(element_uuid)))
    except (ValueError, TypeError):
        element = None

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
    Uses cache-first pattern for element lookup.

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

    # Cache-first element lookup
    element_data = await get_element_with_context(
        source_id=element_id,
        project_id=project_id,
        include_related=True,  # Hint that we need relations
        fallback_to_sidecar=False,
    )

    if not element_data:
        return error_result(
            MetadataErrorCode.ELEMENT_NOT_FOUND,
            f"Element not found: {element_id}"
        )

    # Get the actual element object for relationship query
    from aec_agent.db.repository import ElementRepository
    repo = ElementRepository(pool)

    element_uuid = element_data.get("id") or element_data.get("element_id")
    if not element_uuid:
        return error_result(
            MetadataErrorCode.ELEMENT_NOT_FOUND,
            f"Element not found: {element_id}"
        )

    try:
        element = await repo.get_element(UUID(str(element_uuid)))
    except (ValueError, TypeError):
        element = None

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


@mcp.tool()
@safe_tool
async def get_file_context(
    source: Optional[str] = None,
) -> dict:
    """
    Get a compact summary of the current drawing/model.

    Returns element counts by type, layers, categories, and key stats.
    Uses PostgreSQL cache if available, otherwise queries the sidecar directly.

    Args:
        source: Filter by source ("autocad" or "revit"), or None for all

    Returns:
        Compact file context: element counts, layers, categories, families
    """
    pool, _ = await _get_services()

    if not pool:
        # No PostgreSQL — fall back to direct sidecar query
        return await _get_file_context_from_sidecar(source)

    project_id = await _get_active_project_id()
    if not project_id:
        # No active project in cache — fall back to direct sidecar query
        return await _get_file_context_from_sidecar(source)

    async with pool.acquire() as conn:
        # Project info
        project = await conn.fetchrow(
            "SELECT name, source, file_path, extracted_at FROM projects WHERE id = $1",
            project_id,
        )
        if not project:
            return error_result(MetadataErrorCode.PROJECT_NOT_FOUND, "Project not found")

        # Element count by type
        type_counts = await conn.fetch(
            """
            SELECT entity_type, COUNT(*) as count
            FROM elements
            WHERE project_id = $1 AND deleted_at IS NULL
            GROUP BY entity_type
            ORDER BY count DESC
            LIMIT 30
            """,
            project_id,
        )

        # Layer summary
        layers = await conn.fetch(
            """
            SELECT layer, COUNT(*) as count
            FROM elements
            WHERE project_id = $1 AND deleted_at IS NULL AND layer IS NOT NULL
            GROUP BY layer
            ORDER BY count DESC
            LIMIT 30
            """,
            project_id,
        )

        # Category summary (Revit)
        categories = await conn.fetch(
            """
            SELECT category, COUNT(*) as count
            FROM elements
            WHERE project_id = $1 AND deleted_at IS NULL AND category IS NOT NULL
            GROUP BY category
            ORDER BY count DESC
            LIMIT 20
            """,
            project_id,
        )

        # Family summary (Revit)
        families = await conn.fetch(
            """
            SELECT family, COUNT(*) as count
            FROM elements
            WHERE project_id = $1 AND deleted_at IS NULL AND family IS NOT NULL
            GROUP BY family
            ORDER BY count DESC
            LIMIT 20
            """,
            project_id,
        )

        # Total counts
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM elements WHERE project_id = $1 AND deleted_at IS NULL",
            project_id,
        )

        with_embeddings = await conn.fetchval(
            "SELECT COUNT(*) FROM elements WHERE project_id = $1 AND deleted_at IS NULL AND embedding IS NOT NULL",
            project_id,
        )

        relationship_count = await conn.fetchval(
            "SELECT COUNT(*) FROM element_relationships WHERE project_id = $1",
            project_id,
        )

    context = {
        "project": {
            "id": str(project_id),
            "name": project["name"],
            "source": project["source"],
            "file_path": project["file_path"],
            "extracted_at": project["extracted_at"].isoformat() if project["extracted_at"] else None,
        },
        "summary": {
            "total_elements": total or 0,
            "elements_with_embeddings": with_embeddings or 0,
            "relationship_count": relationship_count or 0,
        },
        "element_types": {row["entity_type"]: row["count"] for row in type_counts},
        "layers": {row["layer"]: row["count"] for row in layers},
        "categories": {row["category"]: row["count"] for row in categories},
        "families": {row["family"]: row["count"] for row in families},
    }

    return success_result(
        data=context,
        message=f"File context: {total} elements across {len(type_counts)} types"
    )


async def _get_cache():
    """Get cache manager."""
    from aec_agent.mcp.server import get_cache
    return get_cache()


@mcp.tool()
@safe_tool
async def get_cache_status() -> dict:
    """
    Get cache status and statistics.

    Returns information about the PostgreSQL cache including:
    - Whether cache is available and configured
    - Number of cached elements for active project
    - Cache freshness status
    - Last sync timestamp

    Returns:
        Cache status and statistics
    """
    pool, embeddings = await _get_services()

    status = {
        "database_configured": pool is not None,
        "embeddings_available": embeddings is not None,
    }

    project_id = await _get_active_project_id()
    if project_id:
        status["active_project_id"] = str(project_id)

        # Get detailed cache stats
        stats = await get_cache_stats(project_id)
        status.update(stats)

        # Check cache freshness
        is_fresh = await is_cache_fresh(project_id, max_age_seconds=300)
        status["cache_fresh"] = is_fresh
    else:
        status["active_project_id"] = None
        status["cache_fresh"] = False

        # Check if we at least have an active file tracked in SQLite
        from aec_agent.mcp.server import get_cache
        cache = get_cache()
        if cache:
            active_file = await cache.get_metadata("active_file_path")
            active_source = await cache.get_metadata("active_source")
            if active_file:
                status["active_file_path"] = active_file
                status["active_source"] = active_source
                status["message"] = (
                    f"Drawing open: {active_file}. "
                    "PostgreSQL not synced — use autocad_get_drawing_info or "
                    "autocad_get_entities for direct access."
                )
            else:
                status["message"] = (
                    "No active project. Open a drawing in AutoCAD/Revit first, "
                    "or use autocad_get_drawing_info to check the sidecar."
                )
        else:
            status["message"] = "No active project. Open a drawing/model first."

    return success_result(data=status)
