"""
MCP (Model Context Protocol) module for AEC Agent.

This package contains:
- server: FastMCP server setup and lifecycle
- concurrency: Tool lock and circuit breaker
- sidecar_client: HTTP client for CAD sidecars
- tools: MCP tool definitions
"""

from .server import mcp, get_cache, get_lock

__all__ = ["mcp", "get_cache", "get_lock"]
