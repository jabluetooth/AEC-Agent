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
