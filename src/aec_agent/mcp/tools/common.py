"""
Common MCP tools shared across AutoCAD and Revit.
"""

from typing import Optional

from aec_agent.mcp.server import mcp, get_lock, get_cache, trigger_background_sync
from aec_agent.mcp.sidecar_client import check_sidecar_health
from aec_agent.config.settings import get_settings
from .base import success_result, error_result


@mcp.tool()
async def ping() -> dict:
    """
    Test if the MCP server is running.

    Returns:
        Success response with server info
    """
    settings = get_settings()
    return success_result(
        data={
            "service": "aec-agent-mcp",
            "version": "0.1.0",
            "environment": settings.environment.value,
        },
        message="AEC Agent MCP server is running"
    )


@mcp.tool()
async def get_server_status() -> dict:
    """
    Get current server status including lock stats and cache info.

    Returns:
        Server status with concurrency and cache metrics
    """
    settings = get_settings()
    lock = get_lock()

    return success_result(data={
        "environment": settings.environment.value,
        "llm_provider": settings.llm_provider.value,
        "mcp_server_port": settings.mcp_server_port,
        "sidecar_port": settings.mcp_listener_port,
        "concurrency": lock.stats,
    })


@mcp.tool()
async def check_sidecar(sidecar_type: str = "revit") -> dict:
    """
    Check if a sidecar (AutoCAD or Revit) is healthy and responsive.

    Args:
        sidecar_type: "autocad" or "revit"

    Returns:
        Health status of the sidecar
    """
    if sidecar_type not in ("autocad", "revit"):
        return error_result(4002, f"Invalid sidecar_type: {sidecar_type}. Use 'autocad' or 'revit'")

    health = await check_sidecar_health(sidecar_type)
    return success_result(data=health)


@mcp.tool()
async def sync_cache(categories: list[str] = None) -> dict:
    """
    Trigger a cache sync from the sidecar.

    Args:
        categories: List of categories to sync (e.g., ["rooms", "levels"]).
                   Default syncs all.

    Returns:
        Sync results with counts
    """
    from aec_agent.mcp.sidecar_client import call_sidecar, SidecarError

    if categories is None:
        categories = ["rooms", "levels", "walls"]

    try:
        result = await call_sidecar(
            endpoint="/mcp/cache/sync",
            method="POST",
            payload={"categories": categories},
            sidecar_type="revit"
        )
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)


@mcp.tool()
async def notify_file_opened(
    source: str,
    file_path: str,
    document_title: Optional[str] = None,
    force_sync: bool = False,
) -> dict:
    """
    Notify the server that a file was opened.

    Called by sidecar event hooks when AutoCAD/Revit opens a document.
    Triggers background sync to PostgreSQL for caching.

    Args:
        source: 'autocad' or 'revit'
        file_path: Full path to the opened file
        document_title: Optional document title for display
        force_sync: Force re-extraction even if file unchanged

    Returns:
        Sync status (queued, debounced, or skipped)
    """
    if source not in ("autocad", "revit"):
        return error_result(4002, f"Invalid source: {source}. Use 'autocad' or 'revit'")

    if not file_path:
        return error_result(4002, "file_path is required")

    # Trigger background sync
    result = await trigger_background_sync(source, file_path, force=force_sync)

    return success_result(
        data=result,
        message=f"File open notification received: {document_title or file_path}"
    )
