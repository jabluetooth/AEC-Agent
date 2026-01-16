"""
MCP Client for the Frontend.

Connects to the local FastMCP server via SSE to discover and use tools.
Provides an async interface for the agent to call MCP tools.
"""

import os
import json
from typing import Any, Optional
from dataclasses import dataclass
from contextlib import AsyncExitStack

import structlog
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp import types as mcp_types

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
    Client for communicating with the MCP server via SSE.

    Uses the official MCP Python client library to properly handle
    the SSE transport protocol.
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
        self._session: Optional[ClientSession] = None
        self._exit_stack: Optional[AsyncExitStack] = None

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
        sse_url = f"{self.base_url}/sse"
        logger.info("Connecting to MCP server via SSE", url=sse_url)

        try:
            # Set up async exit stack for proper resource management
            self._exit_stack = AsyncExitStack()
            await self._exit_stack.__aenter__()

            # Build headers for authentication
            headers = {}
            if self.session_token:
                headers["X-Session-Token"] = self.session_token

            # Connect to MCP server via SSE
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                sse_client(
                    url=sse_url,
                    headers=headers,
                    timeout=5.0,
                    sse_read_timeout=300.0,  # 5 minutes for long operations
                )
            )

            # Create and initialize session
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

            # Initialize the MCP session
            init_result = await self._session.initialize()
            logger.info(
                "MCP session initialized",
                server_name=init_result.serverInfo.name if init_result.serverInfo else "unknown",
                protocol_version=init_result.protocolVersion,
            )

            # Discover tools
            await self._discover_tools()

            logger.info(
                "Connected to MCP server",
                tool_count=len(self._tools),
                tools=list(self._tools.keys()),
            )

        except Exception as e:
            logger.error("Failed to connect to MCP server", error=str(e))
            # Clean up on failure
            if self._exit_stack:
                await self._exit_stack.aclose()
                self._exit_stack = None
            raise MCPConnectionError(f"Failed to connect: {e}") from e

    async def close(self) -> None:
        """Close the connection to the MCP server."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
        self._session = None

    async def _discover_tools(self) -> None:
        """Discover available tools from the MCP server."""
        if not self._session:
            raise MCPClientError("Client not connected")

        try:
            result = await self._session.list_tools()

            self._tools.clear()
            for tool in result.tools:
                self._tools[tool.name] = Tool(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema if tool.inputSchema else {},
                )

            logger.debug("Discovered tools", count=len(self._tools))

        except Exception as e:
            logger.error("Tool discovery failed", error=str(e))
            raise MCPToolError(f"Tool discovery failed: {e}") from e

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
        if not self._session:
            raise MCPClientError("Client not connected")

        logger.info("Calling MCP tool", tool=name, arguments=arguments)

        try:
            result = await self._session.call_tool(name, arguments)

            # Process the content blocks from the result
            raw_content = ""
            data = None
            is_error = result.isError if hasattr(result, 'isError') else False

            for content_block in result.content:
                if isinstance(content_block, mcp_types.TextContent):
                    raw_content += content_block.text
                elif hasattr(content_block, 'text'):
                    raw_content += content_block.text

            # Try to parse raw content as JSON
            if raw_content:
                try:
                    data = json.loads(raw_content)
                    # Check if the parsed data indicates success/failure
                    if isinstance(data, dict):
                        success = data.get("success", not is_error)
                        return ToolResult(
                            success=success,
                            data=data.get("data"),
                            error=data.get("error"),
                            raw_content=raw_content,
                        )
                except json.JSONDecodeError:
                    pass

            return ToolResult(
                success=not is_error,
                data=data,
                raw_content=raw_content if raw_content else None,
            )

        except Exception as e:
            logger.error("Tool call failed", tool=name, error=str(e))
            return ToolResult(
                success=False,
                error={
                    "code": -1,
                    "message": str(e),
                },
            )

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
