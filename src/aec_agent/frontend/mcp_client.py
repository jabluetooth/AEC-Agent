"""
MCP Client for the Frontend.

Connects to the local FastMCP server via SSE to discover and use tools.
Provides an async interface for the agent to call MCP tools.
"""

import os
import json
from typing import Any, Optional
from dataclasses import dataclass, field

import httpx
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from aec_agent.config.settings import get_settings

logger = structlog.get_logger(__name__)


@dataclass
class Tool:
    """Represents an MCP tool."""

    name: str
    description: str
    input_schema: dict[str, Any]

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic tool use format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class ToolResult:
    """Result from calling an MCP tool."""

    success: bool
    data: Optional[dict[str, Any]] = None
    error: Optional[dict[str, Any]] = None
    raw_content: Optional[str] = None

    def to_message_content(self) -> str:
        """Convert result to a string for LLM context."""
        if self.success and self.data:
            return json.dumps(self.data, indent=2)
        elif self.error:
            return f"Error: {self.error.get('message', 'Unknown error')}"
        elif self.raw_content:
            return self.raw_content
        return "No result returned"


class MCPClientError(Exception):
    """Base exception for MCP client errors."""

    pass


class MCPConnectionError(MCPClientError):
    """Error connecting to MCP server."""

    pass


class MCPToolError(MCPClientError):
    """Error executing MCP tool."""

    pass


class MCPClient:
    """
    Client for communicating with the MCP server via HTTP/SSE.

    Uses the JSON-RPC over HTTP approach for tool calls,
    which is simpler for our synchronous tool-calling needs.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        session_token: Optional[str] = None,
    ):
        """
        Initialize the MCP client.

        Args:
            base_url: Base URL of the MCP server. Defaults to env var.
            session_token: Authentication token. Defaults to env var.
        """
        settings = get_settings()
        port = int(os.environ.get("MCP_SERVER_PORT", settings.mcp_server_port))
        self.base_url = base_url or f"http://127.0.0.1:{port}"
        self.session_token = session_token or os.environ.get("SESSION_TOKEN", "")

        self._tools: dict[str, Tool] = {}
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "MCPClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """
        Connect to the MCP server and discover available tools.

        Raises:
            MCPConnectionError: If connection fails.
        """
        logger.info("Connecting to MCP server", base_url=self.base_url)

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(120.0, connect=5.0),
            headers={"X-Session-Token": self.session_token},
        )

        # Discover tools
        await self._discover_tools()

        logger.info(
            "Connected to MCP server",
            tool_count=len(self._tools),
            tools=list(self._tools.keys()),
        )

    async def close(self) -> None:
        """Close the connection to the MCP server."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(httpx.RequestError),
    )
    async def _discover_tools(self) -> None:
        """
        Discover available tools from the MCP server.

        Uses the MCP tools/list endpoint.
        """
        if not self._client:
            raise MCPClientError("Client not connected")

        try:
            # MCP uses JSON-RPC format
            response = await self._client.post(
                "/mcp/v1/tools/list",
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            )
            response.raise_for_status()

            result = response.json()

            # Handle JSON-RPC response
            if "error" in result:
                raise MCPToolError(f"Tool discovery failed: {result['error']}")

            tools_data = result.get("result", {}).get("tools", [])

            self._tools.clear()
            for tool_data in tools_data:
                tool = Tool(
                    name=tool_data["name"],
                    description=tool_data.get("description", ""),
                    input_schema=tool_data.get("inputSchema", {}),
                )
                self._tools[tool.name] = tool

        except httpx.HTTPStatusError as e:
            # Fallback: Try the SSE endpoint to get tools
            logger.warning(
                "JSON-RPC tools/list failed, trying SSE fallback",
                status=e.response.status_code,
            )
            await self._discover_tools_sse()

    async def _discover_tools_sse(self) -> None:
        """Fallback tool discovery via SSE messages endpoint."""
        # For now, we'll initialize with empty tools and discover on first call
        # FastMCP may use different endpoints
        logger.warning("SSE tool discovery not implemented, using empty tools list")
        self._tools.clear()

    def get_tools(self) -> list[Tool]:
        """Get list of available tools."""
        return list(self._tools.values())

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a specific tool by name."""
        return self._tools.get(name)

    def get_tools_for_openai(self) -> list[dict[str, Any]]:
        """Get tools in OpenAI function calling format."""
        return [tool.to_openai_format() for tool in self._tools.values()]

    def get_tools_for_anthropic(self) -> list[dict[str, Any]]:
        """Get tools in Anthropic tool use format."""
        return [tool.to_anthropic_format() for tool in self._tools.values()]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(httpx.RequestError),
    )
    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """
        Call an MCP tool.

        Args:
            name: Name of the tool to call.
            arguments: Arguments to pass to the tool.

        Returns:
            ToolResult containing the result or error.
        """
        if not self._client:
            raise MCPClientError("Client not connected")

        logger.info("Calling MCP tool", tool=name, arguments=arguments)

        try:
            # MCP JSON-RPC call
            response = await self._client.post(
                "/mcp/v1/tools/call",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": name,
                        "arguments": arguments,
                    },
                },
            )
            response.raise_for_status()

            result = response.json()

            # Handle JSON-RPC response
            if "error" in result:
                error_data = result["error"]
                return ToolResult(
                    success=False,
                    error={
                        "code": error_data.get("code", -1),
                        "message": error_data.get("message", "Unknown error"),
                    },
                )

            # Extract content from result
            content = result.get("result", {}).get("content", [])
            if content:
                # MCP returns content as a list of content blocks
                first_content = content[0]
                if first_content.get("type") == "text":
                    text = first_content.get("text", "")
                    # Try to parse as JSON
                    try:
                        data = json.loads(text)
                        return ToolResult(
                            success=data.get("success", True),
                            data=data.get("data"),
                            error=data.get("error"),
                            raw_content=text,
                        )
                    except json.JSONDecodeError:
                        return ToolResult(
                            success=True,
                            raw_content=text,
                        )

            return ToolResult(success=True)

        except httpx.HTTPStatusError as e:
            logger.error(
                "Tool call HTTP error",
                tool=name,
                status=e.response.status_code,
                response=e.response.text[:500],
            )
            return ToolResult(
                success=False,
                error={
                    "code": e.response.status_code,
                    "message": f"HTTP error: {e.response.status_code}",
                },
            )

        except httpx.RequestError as e:
            logger.error("Tool call request error", tool=name, error=str(e))
            raise

    async def ping(self) -> bool:
        """
        Check if the MCP server is responding.

        Returns:
            True if server is healthy, False otherwise.
        """
        try:
            result = await self.call_tool("ping", {})
            return result.success
        except Exception as e:
            logger.error("Ping failed", error=str(e))
            return False
