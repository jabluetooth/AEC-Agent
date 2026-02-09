"""
Concurrency control for MCP tool execution.

Ensures CAD operations are executed one at a time to prevent
race conditions in single-threaded CAD applications.
"""

import asyncio
import functools
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

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
        self._semaphore: asyncio.Semaphore | None = None  # Lazy init
        self._max_concurrent = max_concurrent
        self._active_count = 0
        self._total_executed = 0
        self._total_wait_time = 0.0

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create the semaphore (lazy initialization)."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
        return self._semaphore

    async def acquire(self, timeout: float = 120.0) -> float:
        """
        Acquire the lock, waiting if necessary.

        Args:
            timeout: Maximum time to wait for lock (seconds). Default 120s.

        Returns:
            Time spent waiting for lock (seconds)

        Raises:
            asyncio.TimeoutError: If lock cannot be acquired within timeout
        """
        start = time.monotonic()
        try:
            await asyncio.wait_for(self._get_semaphore().acquire(), timeout=timeout)
        except TimeoutError:
            logger.error(
                "Tool lock acquisition timed out",
                timeout_seconds=timeout,
                active_count=self._active_count,
                total_executed=self._total_executed
            )
            raise TimeoutError(
                f"Could not acquire tool lock within {timeout}s. "
                f"Another operation may be stuck. Active: {self._active_count}"
            )
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
        self._get_semaphore().release()

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


def with_tool_lock(lock: ToolLock, timeout: float = None):
    """
    Decorator to execute tool with concurrency lock.

    Usage:
        @mcp.tool()
        @with_tool_lock(tool_lock)
        async def my_tool(...):
            ...

    Args:
        lock: ToolLock instance to use
        timeout: Maximum time to wait for lock (seconds).
                 If None, uses TOOL_LOCK_TIMEOUT setting (default 120s).

    Returns:
        Decorated function
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Get timeout from settings if not explicitly provided
            from aec_agent.config.settings import get_settings
            effective_timeout = timeout if timeout is not None else get_settings().tool_lock_timeout

            try:
                wait_time = await lock.acquire(timeout=effective_timeout)
            except TimeoutError as e:
                # Return error response instead of raising exception
                logger.warning(
                    "Tool lock timeout - returning error to client",
                    tool=func.__name__,
                    timeout_seconds=effective_timeout
                )
                return {
                    "success": False,
                    "error": {
                        "code": 5007,
                        "message": "Tool lock acquisition timed out",
                        "details": str(e)
                    }
                }
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
        self._lock: asyncio.Lock | None = None  # Lazy init

    def _get_lock(self) -> asyncio.Lock:
        """Get or create the lock (lazy initialization)."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

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
        async with self._get_lock():
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
        async with self._get_lock():
            if self._state == self.HALF_OPEN:
                self._state = self.CLOSED
                self._failure_count = 0
                logger.info("Circuit breaker closed (service recovered)")
            elif self._state == self.CLOSED:
                self._failure_count = 0

    async def record_failure(self):
        """Record a failed call."""
        async with self._get_lock():
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
