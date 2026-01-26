"""
Cache-first helpers for MCP tools.

Provides functions to check PostgreSQL cache before calling sidecars,
reducing sidecar calls and improving response times.
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from uuid import UUID

import structlog

from aec_agent.config.settings import get_settings

logger = structlog.get_logger(__name__)


async def get_element_with_context(
    source_id: str,
    source: str = "autocad",
    project_id: Optional[UUID] = None,
    include_related: bool = False,
    fallback_to_sidecar: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Get element data, checking PostgreSQL cache first.

    Falls back to sidecar only if cache miss and fallback enabled.

    Args:
        source_id: Element Handle (AutoCAD) or ElementId (Revit)
        source: 'autocad' or 'revit'
        project_id: Project UUID (uses active project if not specified)
        include_related: Include related elements
        fallback_to_sidecar: Call sidecar if not in cache

    Returns:
        Element data dict or None
    """
    settings = get_settings()

    # Try PostgreSQL cache first if configured
    if settings.database_url:
        try:
            from aec_agent.db.connection import get_database_pool
            from aec_agent.db.repository import ElementRepository

            pool = await get_database_pool()
            if pool:
                if not project_id:
                    project_id = await _get_active_project_id()

                if project_id:
                    repo = ElementRepository(pool)
                    element = await repo.get_element_by_source_id(
                        project_id, source_id, source
                    )

                    if element:
                        result = element.to_search_result()
                        result["_cache_hit"] = True
                        result["_source"] = "postgresql"

                        if include_related:
                            related = await repo.get_related_elements(
                                element.id, limit=10
                            )
                            result["related_elements"] = [
                                r.to_search_result() for r in related
                            ]

                        logger.debug(
                            "Cache hit",
                            source_id=source_id,
                            source=source,
                            project_id=str(project_id)
                        )
                        return result
        except Exception as e:
            logger.warning("Cache lookup failed", error=str(e))

    # Fall back to sidecar if allowed
    if fallback_to_sidecar:
        logger.debug(
            "Cache miss, calling sidecar",
            source_id=source_id,
            source=source
        )
        return await _get_from_sidecar(source_id, source)

    return None


async def _get_active_project_id() -> Optional[UUID]:
    """Get the currently active project ID from cache."""
    from aec_agent.mcp.server import get_cache

    cache = get_cache()
    if cache:
        project_id_str = await cache.get_metadata("active_project_id")
        if project_id_str:
            try:
                return UUID(project_id_str)
            except ValueError:
                pass
    return None


async def _get_from_sidecar(source_id: str, source: str) -> Optional[Dict[str, Any]]:
    """Get element data from sidecar."""
    from aec_agent.mcp.sidecar_client import call_sidecar, call_autocad_command

    try:
        if source == "autocad":
            response = await call_autocad_command(
                "get_entity",
                {"handle": source_id, "include_geometry": True}
            )
        else:
            response = await call_sidecar(
                endpoint=f"/mcp/elements/{source_id}",
                method="GET",
                sidecar_type="revit"
            )

        if response.get("success"):
            data = response.get("data", {})
            data["_cache_hit"] = False
            data["_source"] = "sidecar"
            return data
    except Exception as e:
        logger.warning("Sidecar call failed", source_id=source_id, error=str(e))

    return None


async def batch_get_elements(
    source_ids: List[str],
    source: str = "autocad",
    project_id: Optional[UUID] = None,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Batch get elements, using cache for available ones.

    Returns dict mapping source_id to element data (or None if not found).
    """
    results: Dict[str, Optional[Dict[str, Any]]] = {}
    cache_misses: List[str] = []

    settings = get_settings()

    if settings.database_url:
        try:
            from aec_agent.db.connection import get_database_pool
            from aec_agent.db.repository import ElementRepository

            pool = await get_database_pool()
            if pool:
                if not project_id:
                    project_id = await _get_active_project_id()

                if project_id:
                    repo = ElementRepository(pool)
                    for source_id in source_ids:
                        element = await repo.get_element_by_source_id(
                            project_id, source_id, source
                        )
                        if element:
                            result = element.to_search_result()
                            result["_cache_hit"] = True
                            results[source_id] = result
                        else:
                            cache_misses.append(source_id)
                else:
                    cache_misses = list(source_ids)
        except Exception as e:
            logger.warning("Batch cache lookup failed", error=str(e))
            cache_misses = list(source_ids)
    else:
        cache_misses = list(source_ids)

    # Batch fetch from sidecar for misses
    for source_id in cache_misses:
        data = await _get_from_sidecar(source_id, source)
        results[source_id] = data

    logger.debug(
        "Batch get completed",
        total=len(source_ids),
        cache_hits=len(source_ids) - len(cache_misses),
        cache_misses=len(cache_misses)
    )

    return results


async def is_cache_fresh(
    project_id: UUID,
    max_age_seconds: int = 300,
) -> bool:
    """
    Check if cache is fresh enough to use without sidecar calls.

    Args:
        project_id: Project UUID
        max_age_seconds: Maximum age in seconds (default 5 minutes)

    Returns:
        True if cache is fresh, False otherwise
    """
    settings = get_settings()

    if not settings.database_url:
        return False

    try:
        from aec_agent.db.connection import get_database_pool
        from aec_agent.db.repository import ElementRepository

        pool = await get_database_pool()
        if not pool:
            return False

        repo = ElementRepository(pool)
        project = await repo.get_project(project_id)

        if project and project.extracted_at:
            age = (datetime.now(timezone.utc) - project.extracted_at).total_seconds()
            return age < max_age_seconds
    except Exception as e:
        logger.warning("Cache freshness check failed", error=str(e))

    return False


async def get_cache_stats(project_id: Optional[UUID] = None) -> Dict[str, Any]:
    """
    Get cache statistics for debugging and monitoring.

    Returns:
        Dict with cache statistics
    """
    stats = {
        "database_configured": False,
        "project_id": None,
        "element_count": 0,
        "last_extracted": None,
        "cache_fresh": False,
    }

    settings = get_settings()

    if not settings.database_url:
        return stats

    stats["database_configured"] = True

    try:
        from aec_agent.db.connection import get_database_pool
        from aec_agent.db.repository import ElementRepository

        pool = await get_database_pool()
        if not pool:
            return stats

        if not project_id:
            project_id = await _get_active_project_id()

        if project_id:
            stats["project_id"] = str(project_id)

            repo = ElementRepository(pool)
            project = await repo.get_project(project_id)

            if project:
                stats["last_extracted"] = (
                    project.extracted_at.isoformat() if project.extracted_at else None
                )
                stats["cache_fresh"] = await is_cache_fresh(project_id)

                # Get element count
                async with pool.acquire() as conn:
                    count = await conn.fetchval(
                        "SELECT COUNT(*) FROM elements WHERE project_id = $1 AND deleted_at IS NULL",
                        project_id
                    )
                    stats["element_count"] = count or 0
    except Exception as e:
        logger.warning("Failed to get cache stats", error=str(e))

    return stats
