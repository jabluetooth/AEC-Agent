"""
FastMCP server for AEC Agent.

Exposes tools for AutoCAD and Revit automation via MCP protocol.
Uses SSE transport for RDP environment compatibility.
"""

import asyncio
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP
import structlog

from aec_agent.config.settings import get_settings
from aec_agent.mcp.concurrency import ToolLock
from aec_agent.cache.sqlite_cache import CacheManager

logger = structlog.get_logger(__name__)

# Global instances
settings = get_settings()
tool_lock = ToolLock(max_concurrent=settings.max_concurrent_tools)
cache_manager: CacheManager = None


@asynccontextmanager
async def lifespan(app):
    """
    Server lifespan manager.

    Initializes cache and cleans up on shutdown.
    """
    global cache_manager

    logger.info("Starting AEC Agent MCP Server",
                port=settings.mcp_server_port,
                environment=settings.environment.value)

    # Initialize cache
    cache_manager = CacheManager(settings.cache_dir)
    await cache_manager.initialize()

    yield

    # Cleanup
    logger.info("Shutting down AEC Agent MCP Server")
    if cache_manager:
        await cache_manager.close()


# Create MCP server with lifespan and port configuration
mcp = FastMCP(
    "AEC Agent",
    lifespan=lifespan,
    port=settings.mcp_server_port,
)


def get_cache() -> CacheManager:
    """Get the cache manager instance."""
    return cache_manager


def get_lock() -> ToolLock:
    """Get the tool lock instance."""
    return tool_lock
