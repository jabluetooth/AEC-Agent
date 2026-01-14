# Phase 4: Python MCP Middleware (The Intelligence Layer)

**Objective:** Create the semantic bridge that translates natural language into geometric CAD commands, managing concurrency, caching, and sidecar communication.

**Prerequisites:**
- Python 3.9-3.12 (3.11 recommended)
- FastMCP >= 0.1.0
- httpx >= 0.27.0
- tenacity >= 8.2.0
- structlog >= 24.1.0
- aiosqlite >= 0.19.0

## 1. Project Structure

```
src/aec_agent/
├── __init__.py
├── cli.py                    # CLI entry points
├── server.py                 # MCP server (main entry)
├── config/
│   ├── __init__.py
│   └── settings.py           # Pydantic settings
├── mcp/
│   ├── __init__.py
│   ├── server.py             # FastMCP server setup
│   ├── concurrency.py        # asyncio.Lock manager
│   ├── sidecar_client.py     # HTTP client for sidecars
│   └── tools/
│       ├── __init__.py
│       ├── base.py           # Base tool utilities
│       ├── autocad.py        # AutoCAD tools
│       ├── revit.py          # Revit tools
│       └── common.py         # Shared tools (health, cache)
├── cache/
│   ├── __init__.py
│   └── sqlite_cache.py       # SQLite cache manager
└── utils/
    ├── __init__.py
    ├── logging.py            # Structured logging setup
    └── version_checker.py    # Compatibility checks
```

## 2. Server Development

### 2.1 FastMCP Server Setup

Create `src/aec_agent/mcp/server.py`:

```python
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
    await cache_manager.close()


# Create MCP server with lifespan
mcp = FastMCP(
    "AEC Agent",
    description="AI-powered AutoCAD and Revit automation via natural language",
    lifespan=lifespan
)


def get_cache() -> CacheManager:
    """Get the cache manager instance."""
    return cache_manager


def get_lock() -> ToolLock:
    """Get the tool lock instance."""
    return tool_lock
```

### 2.2 Main Server Entry Point

Create `src/aec_agent/server.py`:

```python
"""
AEC Agent MCP Server entry point.

Run with: python -m aec_agent.server
"""

import structlog

from aec_agent.config.settings import get_settings
from aec_agent.utils.logging import setup_logging
from aec_agent.mcp.server import mcp

# Import tools to register them
from aec_agent.mcp.tools import common, autocad, revit  # noqa: F401

logger = structlog.get_logger(__name__)


def main():
    """Run the MCP server."""
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_format)

    logger.info(
        "Starting AEC Agent MCP Server",
        environment=settings.environment.value,
        llm_provider=settings.llm_provider.value,
        port=settings.mcp_server_port,
    )

    # Run with SSE transport for RDP compatibility
    mcp.run(transport="sse")


if __name__ == "__main__":
    main()
```

## 3. Concurrency Control (The Foreman)

### 3.1 The Problem

LLMs may emit multiple tool calls simultaneously:
- "Draw wall from (0,0) to (10,0)"
- "Draw wall from (10,0) to (10,5)"
- "Create layer 'Walls'"

CAD applications (AutoCAD/Revit) are single-threaded (STA). Concurrent API calls cause crashes or data corruption.

### 3.2 Solution: asyncio.Lock with Semaphore

Create `src/aec_agent/mcp/concurrency.py`:

```python
"""
Concurrency control for MCP tool execution.

Ensures CAD operations are executed one at a time to prevent
race conditions in single-threaded CAD applications.
"""

import asyncio
import functools
import time
from typing import Callable, TypeVar, ParamSpec

import structlog

logger = structlog.get_logger(__name__)

P = ParamSpec('P')
T = TypeVar('T')


class ToolLock:
    """
    Manages concurrent tool execution.

    Uses asyncio.Semaphore to limit concurrent CAD operations.
    Default is 1 (serial execution) for CAD safety.
    """

    def __init__(self, max_concurrent: int = 1):
        """
        Initialize tool lock.

        Args:
            max_concurrent: Maximum concurrent operations (default 1)
        """
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_concurrent = max_concurrent
        self._active_count = 0
        self._total_executed = 0
        self._total_wait_time = 0.0

    async def acquire(self) -> float:
        """
        Acquire the lock, waiting if necessary.

        Returns:
            Time spent waiting for lock (seconds)
        """
        start = time.monotonic()
        await self._semaphore.acquire()
        wait_time = time.monotonic() - start

        self._active_count += 1
        self._total_wait_time += wait_time

        if wait_time > 0.1:  # Log if waited more than 100ms
            logger.debug("Tool lock acquired after wait", wait_time_ms=round(wait_time * 1000))

        return wait_time

    def release(self):
        """Release the lock."""
        self._active_count -= 1
        self._total_executed += 1
        self._semaphore.release()

    @property
    def stats(self) -> dict:
        """Get lock statistics."""
        return {
            "max_concurrent": self._max_concurrent,
            "active_count": self._active_count,
            "total_executed": self._total_executed,
            "avg_wait_time_ms": round(
                (self._total_wait_time / self._total_executed * 1000)
                if self._total_executed > 0 else 0,
                2
            )
        }


def with_tool_lock(lock: ToolLock):
    """
    Decorator to execute tool with concurrency lock.

    Usage:
        @mcp.tool()
        @with_tool_lock(tool_lock)
        async def my_tool(...):
            ...

    Args:
        lock: ToolLock instance to use

    Returns:
        Decorated function
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            wait_time = await lock.acquire()
            try:
                logger.debug(
                    "Executing tool",
                    tool=func.__name__,
                    wait_time_ms=round(wait_time * 1000)
                )
                return await func(*args, **kwargs)
            finally:
                lock.release()
        return wrapper
    return decorator


class CircuitBreaker:
    """
    Circuit breaker for sidecar connections.

    Prevents repeated calls to a failing sidecar, allowing
    time for recovery before retrying.

    States:
        - CLOSED: Normal operation, requests pass through
        - OPEN: Failing, requests immediately rejected
        - HALF_OPEN: Testing if service recovered
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            half_open_max_calls: Calls allowed in half-open state
        """
        self._state = self.CLOSED
        self._failure_count = 0
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_calls = half_open_max_calls
        self._half_open_calls = 0
        self._last_failure_time = 0.0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> str:
        """Get current circuit state."""
        return self._state

    async def can_execute(self) -> bool:
        """
        Check if a call can be executed.

        Returns:
            True if call is allowed, False otherwise
        """
        async with self._lock:
            if self._state == self.CLOSED:
                return True

            if self._state == self.OPEN:
                # Check if recovery timeout has passed
                if time.monotonic() - self._last_failure_time >= self._recovery_timeout:
                    self._state = self.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info("Circuit breaker entering half-open state")
                    return True
                return False

            # HALF_OPEN state
            if self._half_open_calls < self._half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False

    async def record_success(self):
        """Record a successful call."""
        async with self._lock:
            if self._state == self.HALF_OPEN:
                self._state = self.CLOSED
                self._failure_count = 0
                logger.info("Circuit breaker closed (service recovered)")
            elif self._state == self.CLOSED:
                self._failure_count = 0

    async def record_failure(self):
        """Record a failed call."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == self.HALF_OPEN:
                self._state = self.OPEN
                logger.warning("Circuit breaker opened (half-open test failed)")
            elif self._failure_count >= self._failure_threshold:
                self._state = self.OPEN
                logger.warning(
                    "Circuit breaker opened",
                    failures=self._failure_count,
                    threshold=self._failure_threshold
                )
```

## 4. Sidecar Client

Create `src/aec_agent/mcp/sidecar_client.py`:

```python
"""
HTTP client for communicating with AutoCAD/Revit sidecars.

Handles:
- Connection pooling
- Retry logic with exponential backoff
- Circuit breaker for fault tolerance
- Session token authentication
"""

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import structlog

from aec_agent.config.settings import get_settings
from aec_agent.mcp.concurrency import CircuitBreaker

logger = structlog.get_logger(__name__)

# Global settings
settings = get_settings()

# Configure timeouts
SIDECAR_TIMEOUT = httpx.Timeout(
    settings.sidecar_read_timeout,
    connect=settings.sidecar_connect_timeout
)

# Reusable HTTP client (connection pooling)
_client: httpx.AsyncClient = None

# Circuit breakers per sidecar type
_circuit_breakers = {
    "autocad": CircuitBreaker(),
    "revit": CircuitBreaker(),
}


async def get_client() -> httpx.AsyncClient:
    """Get or create the HTTP client."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=SIDECAR_TIMEOUT)
    return _client


async def close_client():
    """Close the HTTP client."""
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None


def get_sidecar_url(endpoint: str) -> str:
    """
    Build sidecar URL from endpoint.

    Args:
        endpoint: API endpoint (e.g., "/mcp/walls/create")

    Returns:
        Full URL (e.g., "http://127.0.0.1:20001/mcp/walls/create")
    """
    port = settings.mcp_listener_port
    if not port:
        raise ValueError("MCP_LISTENER_PORT not configured")
    return f"http://127.0.0.1:{port}{endpoint}"


def get_auth_headers() -> dict:
    """Get authentication headers for sidecar requests."""
    token = settings.session_token
    if not token:
        logger.warning("SESSION_TOKEN not configured, requests may be rejected")
        return {}
    return {"X-Session-Token": token}


class SidecarError(Exception):
    """Base exception for sidecar errors."""

    def __init__(self, code: int, message: str, details: str = None):
        self.code = code
        self.message = message
        self.details = details
        super().__init__(f"[{code}] {message}")


class SidecarTimeoutError(SidecarError):
    """Sidecar request timed out."""

    def __init__(self, details: str = None):
        super().__init__(5004, "Sidecar timeout", details)


class SidecarConnectionError(SidecarError):
    """Cannot connect to sidecar."""

    def __init__(self, details: str = None):
        super().__init__(5005, "Sidecar connection failed", details)


class SidecarCircuitOpenError(SidecarError):
    """Circuit breaker is open."""

    def __init__(self):
        super().__init__(5006, "Sidecar circuit breaker open - service temporarily unavailable")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    reraise=True
)
async def _call_sidecar_with_retry(
    method: str,
    url: str,
    headers: dict,
    json_data: dict = None
) -> httpx.Response:
    """
    Make HTTP request with retry logic.

    Args:
        method: HTTP method (GET, POST)
        url: Full URL
        headers: Request headers
        json_data: JSON body for POST requests

    Returns:
        httpx.Response object

    Raises:
        httpx.TimeoutException: Request timed out
        httpx.ConnectError: Connection failed
        httpx.HTTPStatusError: Non-2xx response
    """
    client = await get_client()

    if method.upper() == "GET":
        response = await client.get(url, headers=headers)
    else:
        response = await client.post(url, headers=headers, json=json_data)

    response.raise_for_status()
    return response


async def call_sidecar(
    endpoint: str,
    method: str = "POST",
    payload: dict = None,
    sidecar_type: str = "revit"
) -> dict:
    """
    Call a sidecar endpoint with full error handling.

    Args:
        endpoint: API endpoint (e.g., "/mcp/walls/create")
        method: HTTP method (default POST)
        payload: Request body for POST
        sidecar_type: "autocad" or "revit" (for circuit breaker)

    Returns:
        Response JSON as dict

    Raises:
        SidecarError: On any failure
    """
    circuit_breaker = _circuit_breakers.get(sidecar_type)

    # Check circuit breaker
    if circuit_breaker and not await circuit_breaker.can_execute():
        raise SidecarCircuitOpenError()

    url = get_sidecar_url(endpoint)
    headers = get_auth_headers()

    logger.debug(
        "Calling sidecar",
        endpoint=endpoint,
        method=method,
        sidecar_type=sidecar_type
    )

    try:
        response = await _call_sidecar_with_retry(
            method=method,
            url=url,
            headers=headers,
            json_data=payload
        )

        result = response.json()

        # Record success
        if circuit_breaker:
            await circuit_breaker.record_success()

        logger.debug("Sidecar response", success=result.get("success", False))
        return result

    except httpx.TimeoutException as e:
        if circuit_breaker:
            await circuit_breaker.record_failure()
        logger.error("Sidecar timeout", endpoint=endpoint, error=str(e))
        raise SidecarTimeoutError(str(e))

    except httpx.ConnectError as e:
        if circuit_breaker:
            await circuit_breaker.record_failure()
        logger.error("Sidecar connection failed", endpoint=endpoint, error=str(e))
        raise SidecarConnectionError(str(e))

    except httpx.HTTPStatusError as e:
        if circuit_breaker:
            await circuit_breaker.record_failure()
        logger.error(
            "Sidecar HTTP error",
            endpoint=endpoint,
            status_code=e.response.status_code,
            error=str(e)
        )
        raise SidecarError(
            code=e.response.status_code,
            message=f"Sidecar returned {e.response.status_code}",
            details=str(e)
        )


async def check_sidecar_health(sidecar_type: str = "revit") -> dict:
    """
    Check if sidecar is healthy.

    Args:
        sidecar_type: "autocad" or "revit"

    Returns:
        Health status dict
    """
    try:
        result = await call_sidecar(
            endpoint="/health",
            method="GET",
            sidecar_type=sidecar_type
        )
        return {
            "healthy": result.get("success", False),
            "sidecar_type": sidecar_type,
            "details": result.get("data", {})
        }
    except SidecarError as e:
        return {
            "healthy": False,
            "sidecar_type": sidecar_type,
            "error": {"code": e.code, "message": e.message}
        }
```

## 5. SQLite Cache Integration

Create `src/aec_agent/cache/sqlite_cache.py`:

```python
"""
SQLite cache for AEC Agent.

Provides fast read access to Revit model metadata without
blocking the CAD application's UI thread.
"""

import aiosqlite
import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime

import structlog

logger = structlog.get_logger(__name__)


class CacheManager:
    """
    Async SQLite cache manager.

    Caches element metadata from Revit/AutoCAD for fast queries.
    """

    def __init__(self, cache_dir: Path):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.cache_dir / "aec_cache.sqlite"
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self):
        """Initialize database connection and schema."""
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row

        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS levels (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                elevation_m REAL,
                source TEXT,  -- 'revit' or 'autocad'
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS rooms (
                id INTEGER PRIMARY KEY,
                name TEXT,
                number TEXT,
                level_id INTEGER,
                area_sqm REAL,
                source TEXT,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS layers (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                color INTEGER,
                is_on INTEGER DEFAULT 1,
                is_frozen INTEGER DEFAULT 0,
                source TEXT,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_levels_name ON levels(name);
            CREATE INDEX IF NOT EXISTS idx_rooms_level ON rooms(level_id);
            CREATE INDEX IF NOT EXISTS idx_layers_name ON layers(name);
        """)

        await self._db.commit()
        logger.info("Cache database initialized", path=str(self._db_path))

    async def close(self):
        """Close database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    # =========================================================================
    # Level Operations
    # =========================================================================

    async def get_level_by_name(self, name: str) -> Optional[dict]:
        """
        Get level by name.

        Args:
            name: Level name

        Returns:
            Level dict or None
        """
        cursor = await self._db.execute(
            "SELECT * FROM levels WHERE name = ?",
            (name,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_level_by_id(self, level_id: int) -> Optional[dict]:
        """Get level by ID."""
        cursor = await self._db.execute(
            "SELECT * FROM levels WHERE id = ?",
            (level_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_all_levels(self) -> List[dict]:
        """Get all levels sorted by elevation."""
        cursor = await self._db.execute(
            "SELECT * FROM levels ORDER BY elevation_m ASC"
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def upsert_level(
        self,
        level_id: int,
        name: str,
        elevation_m: float,
        source: str = "revit"
    ):
        """Insert or update a level."""
        now = datetime.utcnow().isoformat()
        await self._db.execute(
            """
            INSERT OR REPLACE INTO levels (id, name, elevation_m, source, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (level_id, name, elevation_m, source, now)
        )
        await self._db.commit()

    # =========================================================================
    # Room Operations
    # =========================================================================

    async def get_all_rooms(self) -> List[dict]:
        """Get all rooms."""
        cursor = await self._db.execute("SELECT * FROM rooms")
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_rooms_by_level(self, level_id: int) -> List[dict]:
        """Get rooms on a specific level."""
        cursor = await self._db.execute(
            "SELECT * FROM rooms WHERE level_id = ?",
            (level_id,)
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def upsert_room(
        self,
        room_id: int,
        name: str,
        number: str,
        level_id: int,
        area_sqm: float,
        source: str = "revit"
    ):
        """Insert or update a room."""
        now = datetime.utcnow().isoformat()
        await self._db.execute(
            """
            INSERT OR REPLACE INTO rooms
            (id, name, number, level_id, area_sqm, source, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (room_id, name, number, level_id, area_sqm, source, now)
        )
        await self._db.commit()

    # =========================================================================
    # Layer Operations (AutoCAD)
    # =========================================================================

    async def get_layer_by_name(self, name: str) -> Optional[dict]:
        """Get layer by name."""
        cursor = await self._db.execute(
            "SELECT * FROM layers WHERE name = ?",
            (name,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_all_layers(self) -> List[dict]:
        """Get all layers."""
        cursor = await self._db.execute(
            "SELECT * FROM layers ORDER BY name ASC"
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def upsert_layer(
        self,
        layer_id: int,
        name: str,
        color: int = 7,
        is_on: bool = True,
        is_frozen: bool = False,
        source: str = "autocad"
    ):
        """Insert or update a layer."""
        now = datetime.utcnow().isoformat()
        await self._db.execute(
            """
            INSERT OR REPLACE INTO layers
            (id, name, color, is_on, is_frozen, source, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (layer_id, name, color, int(is_on), int(is_frozen), source, now)
        )
        await self._db.commit()

    # =========================================================================
    # Metadata Operations
    # =========================================================================

    async def get_metadata(self, key: str) -> Optional[str]:
        """Get metadata value by key."""
        cursor = await self._db.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (key,)
        )
        row = await cursor.fetchone()
        return row["value"] if row else None

    async def set_metadata(self, key: str, value: str):
        """Set metadata value."""
        now = datetime.utcnow().isoformat()
        await self._db.execute(
            """
            INSERT OR REPLACE INTO metadata (key, value, updated_at)
            VALUES (?, ?, ?)
            """,
            (key, value, now)
        )
        await self._db.commit()

    async def get_last_sync_time(self, category: str) -> Optional[str]:
        """Get last sync time for a category."""
        return await self.get_metadata(f"{category}_last_sync")

    async def set_last_sync_time(self, category: str):
        """Set last sync time to now."""
        await self.set_metadata(
            f"{category}_last_sync",
            datetime.utcnow().isoformat()
        )

    # =========================================================================
    # Bulk Operations
    # =========================================================================

    async def clear_category(self, category: str):
        """Clear all data for a category."""
        table_map = {
            "levels": "levels",
            "rooms": "rooms",
            "layers": "layers",
        }
        table = table_map.get(category)
        if table:
            await self._db.execute(f"DELETE FROM {table}")
            await self._db.commit()
            logger.info("Cache cleared", category=category)

    async def clear_all(self):
        """Clear all cached data."""
        for table in ["levels", "rooms", "layers", "metadata"]:
            await self._db.execute(f"DELETE FROM {table}")
        await self._db.commit()
        logger.info("All cache cleared")
```

## 6. MCP Tools Implementation

### 6.1 Base Tool Utilities

Create `src/aec_agent/mcp/tools/base.py`:

```python
"""
Base utilities for MCP tools.
"""

from aec_agent.mcp.server import get_lock, get_cache
from aec_agent.mcp.sidecar_client import (
    call_sidecar,
    SidecarError,
    SidecarTimeoutError,
    SidecarConnectionError,
    SidecarCircuitOpenError,
)


def error_result(code: int, message: str, details: str = None) -> dict:
    """Create standardized error result."""
    result = {
        "success": False,
        "error": {
            "code": code,
            "message": message
        }
    }
    if details:
        result["error"]["details"] = details
    return result


def success_result(data: dict = None, message: str = None) -> dict:
    """Create standardized success result."""
    result = {"success": True}
    if data:
        result["data"] = data
    if message:
        result["message"] = message
    return result


# Error codes
class ErrorCode:
    # 4xxx - Client errors
    MISSING_CONFIG = 4001
    INVALID_PARAMS = 4002
    ELEMENT_NOT_FOUND = 4003
    INVALID_OPERATION = 4004

    # 5xxx - Server/Sidecar errors
    INTERNAL_ERROR = 5001
    SIDECAR_ERROR = 5002
    TRANSACTION_FAILED = 5003
    TIMEOUT = 5004
    CONNECTION_FAILED = 5005
    CIRCUIT_OPEN = 5006
```

### 6.2 Common Tools

Create `src/aec_agent/mcp/tools/common.py`:

```python
"""
Common MCP tools shared across AutoCAD and Revit.
"""

from aec_agent.mcp.server import mcp, get_lock, get_cache
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
```

### 6.3 Revit Tools

Create `src/aec_agent/mcp/tools/revit.py`:

```python
"""
MCP tools for Revit automation.
"""

from typing import Optional

from aec_agent.mcp.server import mcp, get_lock, get_cache
from aec_agent.mcp.concurrency import with_tool_lock
from aec_agent.mcp.sidecar_client import call_sidecar, SidecarError
from aec_agent.config.settings import get_settings
from .base import success_result, error_result, ErrorCode

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Level Operations
# =============================================================================

@mcp.tool()
async def revit_list_levels() -> dict:
    """
    List all levels in the current Revit document.

    Returns:
        List of levels with id, name, and elevation in meters
    """
    try:
        result = await call_sidecar(
            endpoint="/mcp/levels",
            method="GET",
            sidecar_type="revit"
        )
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)


@mcp.tool()
@with_tool_lock(get_lock())
async def revit_create_level(name: str, elevation: float) -> dict:
    """
    Create a new level in Revit.

    Args:
        name: Name for the new level (e.g., "Level 3", "Roof")
        elevation: Elevation in meters above ground level

    Returns:
        Created level details including id

    Example:
        revit_create_level("Level 2", 3.5) - Creates level at 3.5m height
    """
    if not name or not name.strip():
        return error_result(ErrorCode.INVALID_PARAMS, "Level name is required")

    if not isinstance(elevation, (int, float)):
        return error_result(ErrorCode.INVALID_PARAMS, "Elevation must be a number")

    try:
        result = await call_sidecar(
            endpoint="/mcp/levels/create",
            method="POST",
            payload={"name": name.strip(), "elevation": float(elevation)},
            sidecar_type="revit"
        )
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)


# =============================================================================
# Wall Operations
# =============================================================================

@mcp.tool()
async def revit_list_walls() -> dict:
    """
    List all walls in the current Revit document.

    Returns:
        List of walls with id, type, level, and length
    """
    try:
        result = await call_sidecar(
            endpoint="/mcp/walls",
            method="GET",
            sidecar_type="revit"
        )
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)


@mcp.tool()
@with_tool_lock(get_lock())
async def revit_create_wall(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    level_name: str,
    height: float = 3.0
) -> dict:
    """
    Create a wall in Revit between two points.

    Args:
        start_x: Start X coordinate in meters
        start_y: Start Y coordinate in meters
        end_x: End X coordinate in meters
        end_y: End Y coordinate in meters
        level_name: Name of the level to place the wall on
        height: Wall height in meters (default 3.0)

    Returns:
        Created wall details including id

    Example:
        revit_create_wall(0, 0, 10, 0, "Level 1", 3.0) - Creates 10m wall on Level 1
    """
    # Validate coordinates
    for coord_name, coord_val in [
        ("start_x", start_x), ("start_y", start_y),
        ("end_x", end_x), ("end_y", end_y), ("height", height)
    ]:
        if not isinstance(coord_val, (int, float)):
            return error_result(
                ErrorCode.INVALID_PARAMS,
                f"{coord_name} must be a number"
            )

    if not level_name or not level_name.strip():
        return error_result(ErrorCode.INVALID_PARAMS, "level_name is required")

    # Check if start and end are different
    if start_x == end_x and start_y == end_y:
        return error_result(
            ErrorCode.INVALID_PARAMS,
            "Start and end points cannot be the same"
        )

    # Look up level ID from cache
    cache = get_cache()
    level = await cache.get_level_by_name(level_name.strip())

    if not level:
        # Try fetching from sidecar
        levels_result = await call_sidecar(
            endpoint="/mcp/levels",
            method="GET",
            sidecar_type="revit"
        )
        if levels_result.get("success"):
            for lvl in levels_result.get("data", {}).get("levels", []):
                if lvl["name"].lower() == level_name.strip().lower():
                    level = lvl
                    break

    if not level:
        return error_result(
            ErrorCode.ELEMENT_NOT_FOUND,
            f"Level '{level_name}' not found"
        )

    try:
        result = await call_sidecar(
            endpoint="/mcp/walls/create",
            method="POST",
            payload={
                "start": {"x": float(start_x), "y": float(start_y)},
                "end": {"x": float(end_x), "y": float(end_y)},
                "level_id": level["id"],
                "height": float(height)
            },
            sidecar_type="revit"
        )
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)


# =============================================================================
# Room Operations
# =============================================================================

@mcp.tool()
async def revit_list_rooms(use_cache: bool = True) -> dict:
    """
    List all rooms in the current Revit document.

    Args:
        use_cache: If True, uses cached data for faster response.
                  If False, queries Revit directly (slower but current).

    Returns:
        List of rooms with id, name, number, level, and area
    """
    endpoint = "/mcp/rooms/cached" if use_cache else "/mcp/rooms"

    try:
        result = await call_sidecar(
            endpoint=endpoint,
            method="GET",
            sidecar_type="revit"
        )
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)


# =============================================================================
# Document Operations
# =============================================================================

@mcp.tool()
async def revit_get_document_status() -> dict:
    """
    Get current Revit document information.

    Returns:
        Document title, path, modification state, and worksharing info
    """
    try:
        result = await call_sidecar(
            endpoint="/mcp/status",
            method="GET",
            sidecar_type="revit"
        )
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)


@mcp.tool()
@with_tool_lock(get_lock())
async def revit_delete_element(element_id: int) -> dict:
    """
    Delete an element from Revit by its ID.

    Args:
        element_id: The Revit element ID to delete

    Returns:
        Success confirmation or error

    Warning:
        This operation cannot be undone via the API. Use with caution.
    """
    if not isinstance(element_id, int):
        return error_result(ErrorCode.INVALID_PARAMS, "element_id must be an integer")

    try:
        result = await call_sidecar(
            endpoint="/mcp/elements/delete",
            method="POST",
            payload={"element_id": element_id},
            sidecar_type="revit"
        )
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
```

### 6.4 AutoCAD Tools

Create `src/aec_agent/mcp/tools/autocad.py`:

```python
"""
MCP tools for AutoCAD automation.
"""

from typing import Optional, List

from aec_agent.mcp.server import mcp, get_lock, get_cache
from aec_agent.mcp.concurrency import with_tool_lock
from aec_agent.mcp.sidecar_client import call_sidecar, SidecarError
from .base import success_result, error_result, ErrorCode

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Layer Operations
# =============================================================================

@mcp.tool()
async def autocad_list_layers() -> dict:
    """
    List all layers in the current AutoCAD drawing.

    Returns:
        List of layers with name, color, and visibility state
    """
    try:
        result = await call_sidecar(
            endpoint="/layers",
            method="GET",
            sidecar_type="autocad"
        )
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)


@mcp.tool()
@with_tool_lock(get_lock())
async def autocad_create_layer(name: str, color: int = 7) -> dict:
    """
    Create a new layer in AutoCAD.

    Args:
        name: Layer name (e.g., "Walls", "Dimensions")
        color: AutoCAD color index 1-255 (default 7=white)
               Common colors: 1=red, 2=yellow, 3=green, 4=cyan, 5=blue, 6=magenta

    Returns:
        Created layer details

    Example:
        autocad_create_layer("Electrical", 1) - Creates red "Electrical" layer
    """
    if not name or not name.strip():
        return error_result(ErrorCode.INVALID_PARAMS, "Layer name is required")

    if not isinstance(color, int) or not (1 <= color <= 255):
        return error_result(
            ErrorCode.INVALID_PARAMS,
            "Color must be an integer between 1 and 255"
        )

    try:
        result = await call_sidecar(
            endpoint="/layers/create",
            method="POST",
            payload={"name": name.strip(), "color": color},
            sidecar_type="autocad"
        )
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)


@mcp.tool()
@with_tool_lock(get_lock())
async def autocad_set_layer_state(
    name: str,
    is_on: Optional[bool] = None,
    is_frozen: Optional[bool] = None
) -> dict:
    """
    Change layer visibility state in AutoCAD.

    Args:
        name: Layer name
        is_on: Set layer on (True) or off (False)
        is_frozen: Set layer frozen (True) or thawed (False)

    Returns:
        Updated layer state

    Example:
        autocad_set_layer_state("Construction", is_frozen=True) - Freeze layer
    """
    if not name or not name.strip():
        return error_result(ErrorCode.INVALID_PARAMS, "Layer name is required")

    if is_on is None and is_frozen is None:
        return error_result(
            ErrorCode.INVALID_PARAMS,
            "At least one of is_on or is_frozen must be specified"
        )

    payload = {"name": name.strip()}
    if is_on is not None:
        payload["is_on"] = bool(is_on)
    if is_frozen is not None:
        payload["is_frozen"] = bool(is_frozen)

    try:
        result = await call_sidecar(
            endpoint="/layers/state",
            method="POST",
            payload=payload,
            sidecar_type="autocad"
        )
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)


# =============================================================================
# Drawing Operations
# =============================================================================

@mcp.tool()
@with_tool_lock(get_lock())
async def autocad_draw_line(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    layer: Optional[str] = None
) -> dict:
    """
    Draw a line in AutoCAD.

    Args:
        start_x: Start X coordinate
        start_y: Start Y coordinate
        end_x: End X coordinate
        end_y: End Y coordinate
        layer: Layer name to draw on (optional, uses current if not specified)

    Returns:
        Created line entity details

    Example:
        autocad_draw_line(0, 0, 100, 100, "Construction") - Draw diagonal line
    """
    # Validate coordinates
    for coord_name, coord_val in [
        ("start_x", start_x), ("start_y", start_y),
        ("end_x", end_x), ("end_y", end_y)
    ]:
        if not isinstance(coord_val, (int, float)):
            return error_result(
                ErrorCode.INVALID_PARAMS,
                f"{coord_name} must be a number"
            )

    payload = {
        "start": {"x": float(start_x), "y": float(start_y)},
        "end": {"x": float(end_x), "y": float(end_y)}
    }

    if layer:
        payload["layer"] = layer.strip()

    try:
        result = await call_sidecar(
            endpoint="/draw/line",
            method="POST",
            payload=payload,
            sidecar_type="autocad"
        )
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)


@mcp.tool()
@with_tool_lock(get_lock())
async def autocad_draw_circle(
    center_x: float,
    center_y: float,
    radius: float,
    layer: Optional[str] = None
) -> dict:
    """
    Draw a circle in AutoCAD.

    Args:
        center_x: Center X coordinate
        center_y: Center Y coordinate
        radius: Circle radius
        layer: Layer name to draw on (optional)

    Returns:
        Created circle entity details

    Example:
        autocad_draw_circle(50, 50, 25) - Draw circle with radius 25 at (50,50)
    """
    for coord_name, coord_val in [
        ("center_x", center_x), ("center_y", center_y), ("radius", radius)
    ]:
        if not isinstance(coord_val, (int, float)):
            return error_result(
                ErrorCode.INVALID_PARAMS,
                f"{coord_name} must be a number"
            )

    if radius <= 0:
        return error_result(ErrorCode.INVALID_PARAMS, "Radius must be positive")

    payload = {
        "center": {"x": float(center_x), "y": float(center_y)},
        "radius": float(radius)
    }

    if layer:
        payload["layer"] = layer.strip()

    try:
        result = await call_sidecar(
            endpoint="/draw/circle",
            method="POST",
            payload=payload,
            sidecar_type="autocad"
        )
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)


@mcp.tool()
@with_tool_lock(get_lock())
async def autocad_draw_rectangle(
    corner1_x: float,
    corner1_y: float,
    corner2_x: float,
    corner2_y: float,
    layer: Optional[str] = None
) -> dict:
    """
    Draw a rectangle in AutoCAD defined by two corner points.

    Args:
        corner1_x: First corner X coordinate
        corner1_y: First corner Y coordinate
        corner2_x: Opposite corner X coordinate
        corner2_y: Opposite corner Y coordinate
        layer: Layer name to draw on (optional)

    Returns:
        Created rectangle (polyline) entity details

    Example:
        autocad_draw_rectangle(0, 0, 100, 50) - Draw 100x50 rectangle
    """
    for coord_name, coord_val in [
        ("corner1_x", corner1_x), ("corner1_y", corner1_y),
        ("corner2_x", corner2_x), ("corner2_y", corner2_y)
    ]:
        if not isinstance(coord_val, (int, float)):
            return error_result(
                ErrorCode.INVALID_PARAMS,
                f"{coord_name} must be a number"
            )

    payload = {
        "corner1": {"x": float(corner1_x), "y": float(corner1_y)},
        "corner2": {"x": float(corner2_x), "y": float(corner2_y)}
    }

    if layer:
        payload["layer"] = layer.strip()

    try:
        result = await call_sidecar(
            endpoint="/draw/rectangle",
            method="POST",
            payload=payload,
            sidecar_type="autocad"
        )
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)


# =============================================================================
# Query Operations
# =============================================================================

@mcp.tool()
async def autocad_get_drawing_info() -> dict:
    """
    Get current AutoCAD drawing information.

    Returns:
        Drawing name, path, and statistics
    """
    try:
        result = await call_sidecar(
            endpoint="/drawing/info",
            method="GET",
            sidecar_type="autocad"
        )
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)


@mcp.tool()
async def autocad_count_entities(layer: Optional[str] = None) -> dict:
    """
    Count entities in the drawing, optionally filtered by layer.

    Args:
        layer: Layer name to filter by (optional, counts all if not specified)

    Returns:
        Entity counts by type
    """
    endpoint = "/entities/count"
    if layer:
        endpoint += f"?layer={layer}"

    try:
        result = await call_sidecar(
            endpoint=endpoint,
            method="GET",
            sidecar_type="autocad"
        )
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
```

### 6.5 Tools Package Init

Create `src/aec_agent/mcp/tools/__init__.py`:

```python
"""
MCP Tools for AEC Agent.

This package contains all MCP tools organized by target application:
- common: Shared tools (health, status, cache)
- autocad: AutoCAD-specific tools
- revit: Revit-specific tools
"""

from . import common
from . import autocad
from . import revit

__all__ = ["common", "autocad", "revit"]
```

## 7. Structured Logging

Create `src/aec_agent/utils/logging.py`:

```python
"""
Structured logging configuration for AEC Agent.
"""

import sys
import logging

import structlog
from structlog.typing import Processor

from aec_agent.config.settings import LogFormat


def setup_logging(level: str = "INFO", log_format: LogFormat = LogFormat.JSON):
    """
    Configure structured logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_format: Output format (JSON or TEXT)
    """
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    # Shared processors
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.ExtraAdder(),
    ]

    if log_format == LogFormat.JSON:
        # JSON output for production
        renderer = structlog.processors.JSONRenderer()
    else:
        # Pretty console output for development
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure formatter for stdlib logging
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
```

## 8. Running the Server

### 8.1 Development

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Set environment variables
$env:MCP_LISTENER_PORT = "20001"
$env:SESSION_TOKEN = "dev-token-123"

# Run server
python -m aec_agent.server
```

### 8.2 Production (Windows Service)

```powershell
# Using NSSM
nssm install AECAgentMCP "C:\path\to\venv\Scripts\python.exe" "-m aec_agent.server"
nssm set AECAgentMCP AppDirectory "C:\path\to\aec-agent"
nssm set AECAgentMCP AppEnvironmentExtra "MCP_LISTENER_PORT=20001" "SESSION_TOKEN=prod-token"
nssm start AECAgentMCP
```

## 9. Testing

### 9.1 Test Script

Create `tests/test_mcp_server.py`:

```python
"""
Tests for MCP server and tools.
"""

import pytest
from unittest.mock import AsyncMock, patch

from aec_agent.mcp.concurrency import ToolLock, CircuitBreaker
from aec_agent.mcp.tools.base import error_result, success_result


class TestToolLock:
    """Tests for ToolLock concurrency control."""

    @pytest.mark.asyncio
    async def test_acquire_release(self):
        lock = ToolLock(max_concurrent=1)

        wait_time = await lock.acquire()
        assert wait_time >= 0
        assert lock.stats["active_count"] == 1

        lock.release()
        assert lock.stats["active_count"] == 0
        assert lock.stats["total_executed"] == 1


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    @pytest.mark.asyncio
    async def test_starts_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitBreaker.CLOSED
        assert await cb.can_execute() is True

    @pytest.mark.asyncio
    async def test_opens_after_failures(self):
        cb = CircuitBreaker(failure_threshold=3)

        for _ in range(3):
            await cb.record_failure()

        assert cb.state == CircuitBreaker.OPEN
        assert await cb.can_execute() is False


class TestResponseHelpers:
    """Tests for response helper functions."""

    def test_error_result(self):
        result = error_result(4002, "Invalid params", "Missing name")

        assert result["success"] is False
        assert result["error"]["code"] == 4002
        assert result["error"]["message"] == "Invalid params"
        assert result["error"]["details"] == "Missing name"

    def test_success_result(self):
        result = success_result({"id": 123}, "Created successfully")

        assert result["success"] is True
        assert result["data"]["id"] == 123
        assert result["message"] == "Created successfully"
```

## 10. Error Codes Reference

| Code | Name | Description |
|------|------|-------------|
| 4001 | MISSING_CONFIG | Required configuration (port, token) not set |
| 4002 | INVALID_PARAMS | Invalid or missing parameters |
| 4003 | ELEMENT_NOT_FOUND | Requested element doesn't exist |
| 4004 | INVALID_OPERATION | Operation not allowed (e.g., no document open) |
| 5001 | INTERNAL_ERROR | Unexpected server error |
| 5002 | SIDECAR_ERROR | Sidecar returned an error |
| 5003 | TRANSACTION_FAILED | CAD transaction failed |
| 5004 | TIMEOUT | Sidecar request timed out |
| 5005 | CONNECTION_FAILED | Cannot connect to sidecar |
| 5006 | CIRCUIT_OPEN | Circuit breaker open, service unavailable |

## 11. Security Checklist

- [ ] `SESSION_TOKEN` validated on all sidecar calls
- [ ] Connection uses `127.0.0.1` only (localhost)
- [ ] No sensitive data in logs
- [ ] Circuit breaker prevents cascading failures
- [ ] Timeouts configured (120s read, 5s connect)
- [ ] Input validation on all tool parameters
