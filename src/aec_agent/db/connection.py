"""
PostgreSQL connection pool with PostGIS and pgvector support.

Provides async database connectivity for spatial and semantic queries.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional, List, Any, AsyncIterator

import structlog

logger = structlog.get_logger(__name__)

# Optional import - gracefully handle missing asyncpg
try:
    import asyncpg
    from asyncpg import Pool, Connection, Record
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    Pool = None
    Connection = None
    Record = None


class DatabasePoolError(Exception):
    """Database pool operation error."""
    pass


class DatabasePool:
    """
    Async PostgreSQL connection pool.

    Supports PostGIS spatial queries and pgvector semantic search.
    Follows the same patterns as CacheManager for consistency.
    """

    def __init__(
        self,
        database_url: str,
        min_size: int = 2,
        max_size: int = 10,
    ):
        """
        Initialize database pool.

        Args:
            database_url: PostgreSQL connection string
            min_size: Minimum pool connections
            max_size: Maximum pool connections
        """
        if not ASYNCPG_AVAILABLE:
            raise DatabasePoolError(
                "asyncpg is not installed. Install with: pip install asyncpg"
            )

        # asyncpg needs plain postgresql:// (not postgresql+asyncpg://)
        if database_url.startswith("postgresql+asyncpg://"):
            database_url = database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
        self._database_url = database_url
        self._min_size = min_size
        self._max_size = max_size
        self._pool: Optional[Pool] = None
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize the connection pool.

        Creates the pool and verifies PostGIS/pgvector extensions.
        """
        if self._initialized:
            return

        logger.info(
            "Initializing database pool",
            min_size=self._min_size,
            max_size=self._max_size
        )

        try:
            self._pool = await asyncpg.create_pool(
                self._database_url,
                min_size=self._min_size,
                max_size=self._max_size,
                command_timeout=60,
                # Custom type codecs for PostGIS and pgvector
                init=self._init_connection,
            )

            # Verify extensions are available
            await self._verify_extensions()

            self._initialized = True
            logger.info("Database pool initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize database pool", error=str(e))
            raise DatabasePoolError(f"Failed to initialize pool: {e}") from e

    async def _init_connection(self, conn: Connection) -> None:
        """
        Initialize a new connection with custom type handling.

        Args:
            conn: The connection to initialize
        """
        # Register vector type codec for pgvector
        await conn.set_type_codec(
            'vector',
            encoder=self._encode_vector,
            decoder=self._decode_vector,
            schema='public',
            format='text',
        )

    @staticmethod
    def _encode_vector(value: List[float]) -> str:
        """Encode Python list to pgvector format."""
        return f"[{','.join(str(v) for v in value)}]"

    @staticmethod
    def _decode_vector(value: str) -> List[float]:
        """Decode pgvector format to Python list."""
        # Format: [0.1,0.2,0.3,...]
        if value.startswith('[') and value.endswith(']'):
            value = value[1:-1]
        return [float(v) for v in value.split(',') if v]

    async def _verify_extensions(self) -> None:
        """Verify PostGIS and pgvector extensions are installed."""
        # Use pool directly (not self.acquire) because _initialized is not yet True
        async with self._pool.acquire() as conn:
            # Check PostGIS
            result = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'postgis')"
            )
            if not result:
                logger.warning("PostGIS extension not installed - spatial queries disabled")
            else:
                logger.info("PostGIS extension verified")

            # Check pgvector
            result = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
            )
            if not result:
                logger.warning("pgvector extension not installed - semantic search disabled")
            else:
                logger.info("pgvector extension verified")

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._initialized = False
            logger.info("Database pool closed")

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[Connection]:
        """
        Acquire a connection from the pool.

        Yields:
            Database connection

        Example:
            async with pool.acquire() as conn:
                await conn.execute("SELECT 1")
        """
        if not self._initialized or not self._pool:
            raise DatabasePoolError("Database pool not initialized")

        async with self._pool.acquire() as conn:
            yield conn

    async def execute(self, query: str, *args: Any) -> str:
        """
        Execute a query and return status.

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            Query status string
        """
        async with self.acquire() as conn:
            return await conn.execute(query, *args)

    async def executemany(self, query: str, args: List[tuple]) -> None:
        """
        Execute a query multiple times with different parameters.

        Args:
            query: SQL query
            args: List of parameter tuples
        """
        async with self.acquire() as conn:
            await conn.executemany(query, args)

    async def fetch(self, query: str, *args: Any) -> List[Record]:
        """
        Execute a query and return all rows.

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            List of records
        """
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> Optional[Record]:
        """
        Execute a query and return first row.

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            First record or None
        """
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        """
        Execute a query and return first column of first row.

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            Value or None
        """
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Connection]:
        """
        Execute queries in a transaction.

        Yields:
            Connection with active transaction

        Example:
            async with pool.transaction() as conn:
                await conn.execute("INSERT ...")
                await conn.execute("UPDATE ...")
        """
        async with self.acquire() as conn:
            async with conn.transaction():
                yield conn

    @property
    def is_initialized(self) -> bool:
        """Check if pool is initialized."""
        return self._initialized

    def get_pool_stats(self) -> dict:
        """
        Get pool statistics.

        Returns:
            Dict with pool size, free connections, etc.
        """
        if not self._pool:
            return {"status": "not_initialized"}

        return {
            "status": "initialized",
            "size": self._pool.get_size(),
            "min_size": self._pool.get_min_size(),
            "max_size": self._pool.get_max_size(),
            "free_size": self._pool.get_idle_size(),
        }


# Global pool instance (lazy initialized)
_database_pool: Optional[DatabasePool] = None


async def get_database_pool() -> Optional[DatabasePool]:
    """
    Get the global database pool instance.

    Returns:
        DatabasePool if configured, None otherwise
    """
    return _database_pool


async def initialize_database_pool(
    database_url: str,
    pool_size: int = 5,
    max_overflow: int = 10,
) -> DatabasePool:
    """
    Initialize the global database pool.

    Args:
        database_url: PostgreSQL connection string
        pool_size: Base pool size
        max_overflow: Additional connections beyond pool_size

    Returns:
        Initialized DatabasePool
    """
    global _database_pool

    if _database_pool is not None:
        return _database_pool

    _database_pool = DatabasePool(
        database_url=database_url,
        min_size=2,
        max_size=pool_size + max_overflow,
    )
    await _database_pool.initialize()

    return _database_pool


async def close_database_pool() -> None:
    """Close the global database pool."""
    global _database_pool

    if _database_pool:
        await _database_pool.close()
        _database_pool = None
