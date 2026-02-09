"""
AEC Agent MCP Server entry point.

Run with: python -m aec_agent.server

Serves both:
- MCP SSE transport (for LLM tool calls)
- REST endpoints (for sidecar event notifications)
"""

import json

import structlog
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from aec_agent.config.settings import get_settings
from aec_agent.mcp.server import mcp, trigger_background_sync

# Import tools to register them with the MCP server
from aec_agent.mcp.tools import (  # noqa: F401
    autocad,
    common,
    mep_tools,
    metadata,
    raster_design,
    revit,
)
from aec_agent.utils.logging import setup_logging

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# REST endpoints for sidecar event hooks
# ---------------------------------------------------------------------------

async def handle_notify_file_opened(request: Request) -> JSONResponse:
    """
    REST endpoint called by AutoCAD/Revit sidecars when a document is opened.

    The sidecars POST here to trigger PostgreSQL metadata sync.
    """
    try:
        body = await request.json()
    except (json.JSONDecodeError, Exception):
        return JSONResponse(
            {"success": False, "error": "Invalid JSON body"},
            status_code=400,
        )

    source = body.get("source", "")
    file_path = body.get("file_path", "")
    force_sync = body.get("force_sync", False)

    if source not in ("autocad", "revit"):
        return JSONResponse(
            {"success": False, "error": f"Invalid source: {source}"},
            status_code=400,
        )

    if not file_path:
        return JSONResponse(
            {"success": False, "error": "file_path is required"},
            status_code=400,
        )

    result = await trigger_background_sync(source, file_path, force=force_sync)

    logger.info(
        "File open notification received",
        source=source,
        file_path=file_path,
        status=result.get("status"),
    )

    return JSONResponse({"success": True, "data": result})


async def handle_health(request: Request) -> JSONResponse:
    """Simple health check endpoint."""
    return JSONResponse({"status": "ok", "service": "aec-agent-mcp"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run the MCP server with REST endpoints."""
    import uvicorn

    settings = get_settings()
    setup_logging(settings.log_level, settings.log_format)

    logger.info(
        "Starting AEC Agent MCP Server",
        environment=settings.environment.value,
        llm_provider=settings.llm_provider.value,
        port=settings.mcp_server_port,
    )

    # Build a combined ASGI app: REST routes + MCP SSE
    rest_routes = [
        Route("/tools/notify_file_opened", handle_notify_file_opened, methods=["POST"]),
        Route("/health", handle_health, methods=["GET"]),
    ]

    try:
        # Get the MCP SSE ASGI app and mount it alongside REST routes
        sse_app = mcp.sse_app()
        app = Starlette(
            routes=[
                *rest_routes,
                Mount("/", app=sse_app),
            ],
        )
    except AttributeError:
        # Fallback: if sse_app() is not available, run MCP directly
        logger.warning("FastMCP.sse_app() not available, falling back to mcp.run()")
        try:
            mcp.run(transport="sse")
        except (KeyboardInterrupt, SystemExit):
            pass
        return

    try:
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=settings.mcp_server_port,
            log_level="info",
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except SystemExit:
        pass
    except Exception as e:
        logger.error(
            "MCP Server crashed with unhandled exception",
            error=str(e),
            exc_info=True,
        )
        raise


if __name__ == "__main__":
    main()
