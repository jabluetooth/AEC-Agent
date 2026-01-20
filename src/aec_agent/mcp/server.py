"""
FastMCP server for AEC Agent.

Exposes tools for AutoCAD and Revit automation via MCP protocol.
Uses SSE transport for RDP environment compatibility.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

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

# Database and semantic services (optional, initialized if configured)
database_pool = None
embedding_service = None
sync_manager = None


@asynccontextmanager
async def lifespan(app):
    """
    Server lifespan manager.

    Initializes cache, database, and semantic services.
    Cleans up on shutdown.
    """
    global cache_manager, database_pool, embedding_service, sync_manager

    logger.info("Starting AEC Agent MCP Server",
                port=settings.mcp_server_port,
                environment=settings.environment.value)

    # Initialize SQLite cache (always available)
    cache_manager = CacheManager(settings.cache_dir)
    await cache_manager.initialize()

    # Initialize PostgreSQL pool (if configured)
    if settings.has_database:
        try:
            from aec_agent.db.connection import initialize_database_pool
            database_pool = await initialize_database_pool(
                settings.database_url,
                settings.database_pool_size,
                settings.database_pool_max_overflow,
            )
            logger.info("PostgreSQL database pool initialized")

            # Initialize embedding service (if sentence-transformers available)
            try:
                from aec_agent.semantic.embeddings import initialize_embedding_service
                embedding_service = initialize_embedding_service(settings.embedding_model)
                logger.info(
                    "Embedding service initialized",
                    model=settings.embedding_model,
                    dimension=settings.embedding_dimension
                )
            except Exception as e:
                logger.warning(
                    "Embedding service not available",
                    error=str(e),
                    hint="Install sentence-transformers for semantic search"
                )

            # Initialize sync manager
            if database_pool:
                from aec_agent.extraction.sync_manager import SyncManager
                sync_manager = SyncManager(database_pool, embedding_service)
                logger.info("Sync manager initialized")

        except Exception as e:
            logger.warning(
                "PostgreSQL database not available",
                error=str(e),
                hint="Set DATABASE_URL for semantic search features"
            )
    else:
        logger.info(
            "PostgreSQL not configured",
            hint="Set DATABASE_URL environment variable for semantic search"
        )

    yield

    # Cleanup
    logger.info("Shutting down AEC Agent MCP Server")

    if sync_manager:
        sync_manager = None

    if embedding_service:
        from aec_agent.semantic.embeddings import close_embedding_service
        close_embedding_service()
        embedding_service = None

    if database_pool:
        from aec_agent.db.connection import close_database_pool
        await close_database_pool()
        database_pool = None

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


def get_lock_timeout() -> float:
    """Get the configured tool lock timeout."""
    return settings.tool_lock_timeout
