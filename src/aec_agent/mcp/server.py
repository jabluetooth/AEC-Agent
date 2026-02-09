"""
FastMCP server for AEC Agent.

Exposes tools for AutoCAD and Revit automation via MCP protocol.
Uses SSE transport for RDP environment compatibility.
"""

import asyncio
from contextlib import asynccontextmanager

import structlog
from mcp.server.fastmcp import FastMCP

from aec_agent.cache.sqlite_cache import CacheManager
from aec_agent.config.settings import get_settings
from aec_agent.mcp.concurrency import ToolLock

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


def get_sync_manager():
    """Get the sync manager instance."""
    return sync_manager


def get_database_pool():
    """Get the database pool instance."""
    return database_pool


def get_embedding_service():
    """Get the embedding service instance."""
    return embedding_service


# Debounce tracking for background syncs
_sync_debounce: dict[str, float] = {}
_sync_debounce_ms: int = 2000  # 2 seconds


async def trigger_background_sync(source: str, file_path: str, force: bool = False) -> dict:
    """
    Trigger a background sync for a file.

    This is called by sidecar event hooks when files are opened/saved.
    Uses debouncing to avoid excessive syncs.

    Args:
        source: 'autocad' or 'revit'
        file_path: Path to the file
        force: Force sync even if recently synced

    Returns:
        Status dict with sync result
    """
    import time

    global _sync_debounce

    if not sync_manager:
        logger.warning("Sync manager not available, skipping background sync")
        return {"status": "skipped", "reason": "sync_manager_not_available"}

    # Debounce check
    cache_key = f"{source}:{file_path}"
    now = time.time()

    if not force and cache_key in _sync_debounce:
        elapsed_ms = (now - _sync_debounce[cache_key]) * 1000
        if elapsed_ms < _sync_debounce_ms:
            logger.debug(
                "Sync debounced",
                source=source,
                file_path=file_path,
                wait_ms=_sync_debounce_ms - elapsed_ms
            )
            return {"status": "debounced", "wait_ms": _sync_debounce_ms - elapsed_ms}

    _sync_debounce[cache_key] = now

    # Run sync in background
    async def do_sync():
        try:
            result = await sync_manager.trigger_full_sync(source, file_path, force=force)
            logger.info(
                "Background sync completed",
                source=source,
                file_path=file_path,
                elements=result.elements_extracted,
                duration_ms=result.duration_ms
            )
        except Exception as e:
            logger.error(
                "Background sync failed",
                source=source,
                file_path=file_path,
                error=str(e)
            )

    asyncio.create_task(do_sync())

    return {"status": "queued", "source": source, "file_path": file_path}
