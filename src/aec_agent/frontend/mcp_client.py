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


def _compress_description(desc: str) -> str:
    """Compress tool description by removing examples and extra whitespace."""
    if not desc:
        return desc

    lines = desc.split('\n')
    compressed = []
    skip_section = False

    for line in lines:
        line_lower = line.strip().lower()
        # Skip example sections
        if line_lower.startswith('example:') or line_lower.startswith('examples:'):
            skip_section = True
            continue
        # Resume on new section (Args, Returns, etc.)
        if skip_section and line_lower and (line_lower.startswith('args:') or
            line_lower.startswith('returns:') or line_lower.startswith('warning:')):
            skip_section = False
        if skip_section:
            continue
        # Keep non-empty lines, compress whitespace
        if line.strip():
            compressed.append(line.strip())

    return ' '.join(compressed)


def _compress_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Compress JSON schema by removing descriptions from properties."""
    if not schema:
        return schema

    result = {}
    for key, value in schema.items():
        if key == 'properties' and isinstance(value, dict):
            # Keep property definitions but remove verbose descriptions
            compressed_props = {}
            for prop_name, prop_def in value.items():
                if isinstance(prop_def, dict):
                    # Keep type and essential fields, shorten description
                    compressed_prop = {k: v for k, v in prop_def.items()
                                       if k in ('type', 'default', 'enum', 'items', 'minimum', 'maximum')}
                    if 'description' in prop_def:
                        # Keep first sentence only
                        desc = prop_def['description']
                        first_sentence = desc.split('.')[0] if '.' in desc else desc
                        if len(first_sentence) < 100:
                            compressed_prop['description'] = first_sentence
                    compressed_props[prop_name] = compressed_prop
                else:
                    compressed_props[prop_name] = prop_def
            result[key] = compressed_props
        elif isinstance(value, dict):
            result[key] = _compress_schema(value)
        else:
            result[key] = value

    return result


@dataclass
class Tool:
    """Represents an MCP tool."""

    name: str
    description: str
    input_schema: dict[str, Any]

    def to_openai_format(
        self,
        compress: bool = False,
        compression_mode: str = "standard"
    ) -> dict[str, Any]:
        """Convert to OpenAI function calling format.

        Args:
            compress: If True, compress description and schema to reduce tokens.
            compression_mode: Compression level (full, standard, minimal, ultra).
        """
        if compress:
            from aec_agent.frontend.tool_optimization import (
                get_optimized_description,
                get_optimized_schema,
            )
            desc = get_optimized_description(self.name, self.description, compression_mode)
            schema = get_optimized_schema(self.input_schema, compression_mode)
        else:
            desc = self.description
            schema = self.input_schema

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": desc,
                "parameters": schema,
            },
        }

    def to_anthropic_format(
        self,
        compress: bool = False,
        compression_mode: str = "standard"
    ) -> dict[str, Any]:
        """Convert to Anthropic tool use format.

        Args:
            compress: If True, compress description and schema to reduce tokens.
            compression_mode: Compression level (full, standard, minimal, ultra).
        """
        if compress:
            from aec_agent.frontend.tool_optimization import (
                get_optimized_description,
                get_optimized_schema,
            )
            desc = get_optimized_description(self.name, self.description, compression_mode)
            schema = get_optimized_schema(self.input_schema, compression_mode)
        else:
            desc = self.description
            schema = self.input_schema

        return {
            "name": self.name,
            "description": desc,
            "input_schema": schema,
        }


@dataclass
class ToolResult:
    """Result from calling an MCP tool."""

    success: bool
    data: Optional[dict[str, Any]] = None
    error: Optional[dict[str, Any]] = None
    raw_content: Optional[str] = None

    def to_message_content(self, max_length: Optional[int] = None, field_preset: Optional[str] = None) -> str:
        """Convert result to a compact string for LLM context.

        Args:
            max_length: Maximum length of the result string. If None, uses settings.
            field_preset: Field filtering preset (minimal, standard, full). If None, uses settings.
        """
        from aec_agent.config.settings import get_settings
        settings = get_settings()

        if max_length is None:
            max_length = settings.max_tool_result_chars

        if field_preset is None:
            field_preset = settings.result_field_preset

        if self.success and self.data:
            # Filter fields before serialization to reduce tokens
            filtered_data = _filter_result_fields(self.data, field_preset)
            # Compact JSON (no indent) to save tokens
            content = json.dumps(filtered_data, separators=(',', ':'))
        elif self.error:
            return f"Error: {self.error.get('message', 'Unknown error')}"
        elif self.raw_content:
            content = self.raw_content
        else:
            return "No result"

        # Truncate if too long
        if len(content) > max_length:
            return self._smart_truncate(content, max_length)
        return content

    def _smart_truncate(self, content: str, max_length: int) -> str:
        """Smart truncation that preserves structure when possible."""
        if len(content) <= max_length:
            return content

        # Try to parse as JSON for smarter truncation
        try:
            data = json.loads(content)

            # If it's a list, truncate items
            if isinstance(data, list) and len(data) > 0:
                items_shown = 0
                truncated_list = []
                running_length = 2  # for []

                for item in data:
                    item_str = json.dumps(item, separators=(',', ':'))
                    if running_length + len(item_str) + 1 < max_length - 50:
                        truncated_list.append(item)
                        running_length += len(item_str) + 1
                        items_shown += 1
                    else:
                        break

                remaining = len(data) - items_shown
                if remaining > 0:
                    result = json.dumps(truncated_list, separators=(',', ':'))
                    return f"{result[:-1]}]...[{remaining} more items]"
                return json.dumps(truncated_list, separators=(',', ':'))

            # If it's a dict with a list field, truncate that
            if isinstance(data, dict):
                for key in ['elements', 'items', 'results', 'data']:
                    if key in data and isinstance(data[key], list):
                        original_len = len(data[key])
                        # Keep first 5 items
                        data[key] = data[key][:5]
                        truncated = json.dumps(data, separators=(',', ':'))
                        if len(truncated) < max_length:
                            return f"{truncated}...[showing 5 of {original_len}]"

        except (json.JSONDecodeError, TypeError):
            pass

        # Fall back to simple truncation
        return content[:max_length - 20] + "...[truncated]"


# =============================================================================
# Result Field Filtering (Phase 2 optimization)
# =============================================================================
# Reduces output tokens by returning only relevant fields

RESULT_FIELD_PRESETS = {
    "minimal": {"id", "name", "type"},
    "standard": {"id", "name", "type", "layer", "category", "level", "location",
                 "source_id", "entity_type", "description"},
    "full": None,  # All fields
}

# Fields that should always be preserved regardless of preset
PRESERVED_FIELDS = {"elements", "count", "message", "success", "error", "data"}


def _filter_result_fields(data: Any, preset: str = "standard") -> Any:
    """
    Filter result data to include only specified fields.

    Args:
        data: The data to filter (dict, list, or other)
        preset: Field preset name (minimal, standard, full)

    Returns:
        Filtered data with only allowed fields
    """
    allowed = RESULT_FIELD_PRESETS.get(preset)
    if allowed is None:
        return data

    if isinstance(data, dict):
        filtered = {}
        for k, v in data.items():
            # Always preserve structural fields
            if k in PRESERVED_FIELDS:
                if k == "elements" and isinstance(v, list):
                    # Recursively filter element lists
                    filtered[k] = [_filter_result_fields(el, preset) for el in v]
                elif k == "data" and isinstance(v, dict):
                    filtered[k] = _filter_result_fields(v, preset)
                else:
                    filtered[k] = v
            elif k in allowed:
                filtered[k] = v
        return filtered

    if isinstance(data, list):
        return [_filter_result_fields(item, preset) for item in data]

    return data


def filter_tool_result(result: "ToolResult", preset: str = "standard") -> "ToolResult":
    """
    Filter a ToolResult's data using the specified preset.

    Args:
        result: The ToolResult to filter
        preset: Field preset name

    Returns:
        New ToolResult with filtered data
    """
    if result.data is None:
        return result

    filtered_data = _filter_result_fields(result.data, preset)
    return ToolResult(
        success=result.success,
        data=filtered_data,
        error=result.error,
        raw_content=result.raw_content,
    )


# All known app-specific tool prefixes.  Tools whose name does NOT start with
# any of these are considered "shared" (e.g. ``ping``, ``find_elements``,
# ``draw_line_between``) and are always included regardless of prefix filter.
_APP_PREFIXES = ("autocad_", "revit_", "raster_")


def _filter_by_prefix(tools: list, filter_prefix: str) -> list:
    """Filter tools by comma-separated prefixes, always keeping shared tools.

    A tool is included if:
    - Its name starts with one of the requested prefixes, OR
    - Its name does NOT start with any known app prefix (shared tool).
    """
    prefixes = tuple(p.strip() for p in filter_prefix.split(",") if p.strip())
    return [
        t for t in tools
        if t.name.startswith(prefixes) or not t.name.startswith(_APP_PREFIXES)
    ]


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

    def get_tools(self, filter_prefix: Optional[str] = None) -> list[Tool]:
        """Get list of available tools.

        Args:
            filter_prefix: If provided, only return tools matching one of these
                          prefixes (comma-separated, e.g. ``"autocad_,raster_"``).
                          Tools without any recognized app prefix (``autocad_``,
                          ``revit_``, ``raster_``) are always included as shared
                          tools (e.g. ``ping``, ``find_elements``).
        """
        tools = list(self._tools.values())
        if filter_prefix:
            tools = _filter_by_prefix(tools, filter_prefix)
        return tools

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a specific tool by name."""
        return self._tools.get(name)

    def get_tools_for_openai(
        self,
        compress: bool = False,
        compression_mode: str = "standard",
        filter_prefix: Optional[str] = None,
        tool_tier: Optional[str] = None,
        include_metadata: bool = True,
    ) -> list[dict[str, Any]]:
        """Get tools in OpenAI function calling format.

        Args:
            compress: If True, compress descriptions to reduce tokens.
            compression_mode: Compression level (full, standard, minimal, ultra).
            filter_prefix: If provided, only include tools with this prefix
                          (e.g., 'autocad_' or 'revit_').
            tool_tier: Tool tier (essential, standard, advanced). If set,
                      only tools in that tier and below are included.
            include_metadata: If False, exclude metadata/semantic tools.
        """
        tools = list(self._tools.values())

        # Filter by tier
        if tool_tier:
            from aec_agent.frontend.tool_optimization import get_tools_for_tier
            allowed_tools = set(get_tools_for_tier(tool_tier))
            tools = [t for t in tools if t.name in allowed_tools]

        # Filter out metadata tools if disabled
        if not include_metadata:
            metadata_tools = {
                "find_elements", "get_nearby_elements", "get_related_elements",
                "resolve_coordinates", "sync_metadata", "draw_line_between",
                "draw_circle_at", "draw_rectangle_around", "get_distance_between"
            }
            tools = [t for t in tools if t.name not in metadata_tools]

        # Filter by prefix (supports comma-separated, includes shared tools)
        if filter_prefix:
            tools = _filter_by_prefix(tools, filter_prefix)

        return [
            tool.to_openai_format(compress=compress, compression_mode=compression_mode)
            for tool in tools
        ]

    def get_tools_for_anthropic(
        self,
        compress: bool = False,
        compression_mode: str = "standard",
        filter_prefix: Optional[str] = None,
        tool_tier: Optional[str] = None,
        include_metadata: bool = True,
    ) -> list[dict[str, Any]]:
        """Get tools in Anthropic tool use format.

        Args:
            compress: If True, compress descriptions to reduce tokens.
            compression_mode: Compression level (full, standard, minimal, ultra).
            filter_prefix: If provided, only include tools with this prefix.
            tool_tier: Tool tier (essential, standard, advanced).
            include_metadata: If False, exclude metadata/semantic tools.
        """
        tools = list(self._tools.values())

        # Filter by tier
        if tool_tier:
            from aec_agent.frontend.tool_optimization import get_tools_for_tier
            allowed_tools = set(get_tools_for_tier(tool_tier))
            tools = [t for t in tools if t.name in allowed_tools]

        # Filter out metadata tools if disabled
        if not include_metadata:
            metadata_tools = {
                "find_elements", "get_nearby_elements", "get_related_elements",
                "resolve_coordinates", "sync_metadata", "draw_line_between",
                "draw_circle_at", "draw_rectangle_around", "get_distance_between"
            }
            tools = [t for t in tools if t.name not in metadata_tools]

        # Filter by prefix (supports comma-separated, includes shared tools)
        if filter_prefix:
            tools = _filter_by_prefix(tools, filter_prefix)

        return [
            tool.to_anthropic_format(compress=compress, compression_mode=compression_mode)
            for tool in tools
        ]

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
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(
                "Tool call failed",
                tool=name,
                error=str(e),
                traceback=error_traceback
            )
            return ToolResult(
                success=False,
                error={
                    "code": -1,
                    "message": str(e),
                    "traceback": error_traceback,
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
