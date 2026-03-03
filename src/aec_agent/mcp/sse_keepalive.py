"""
SSE Keep-Alive Patch for MCP Server.

The MCP library's SSE transport doesn't send keep-alive pings, which causes
client connections to timeout during long operations or idle periods.

This module monkey-patches the MCP library to add keep-alive ping support.
Import this module BEFORE creating the FastMCP instance to apply the patch.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import anyio
from anyio.streams.memory import MemoryObjectSendStream, MemoryObjectReceiveStream
from mcp.server.sse import SseServerTransport
from mcp.shared.session import SessionMessage
from sse_starlette.sse import EventSourceResponse
from starlette.requests import Request
from starlette.types import Receive, Scope, Send

from aec_agent.config.settings import get_settings

logger = logging.getLogger(__name__)

# Store original method for reference
_original_connect_sse = SseServerTransport.connect_sse


@asynccontextmanager
async def patched_connect_sse(self, scope: Scope, receive: Receive, send: Send):
    """
    Patched version of SseServerTransport.connect_sse that adds keep-alive pings.

    This prevents client timeouts during long-running operations by sending
    periodic ping events to keep the SSE connection alive.
    """
    if scope["type"] != "http":
        logger.error("connect_sse received non-HTTP request")
        raise ValueError("connect_sse can only handle HTTP requests")

    # Validate request headers for DNS rebinding protection
    request = Request(scope, receive)
    error_response = await self._security.validate_request(request, is_post=False)
    if error_response:
        await error_response(scope, receive, send)
        raise ValueError("Request validation failed")

    logger.debug("Setting up SSE connection with keep-alive pings")
    read_stream: MemoryObjectReceiveStream[SessionMessage | Exception]
    read_stream_writer: MemoryObjectSendStream[SessionMessage | Exception]

    write_stream: MemoryObjectSendStream[SessionMessage]
    write_stream_reader: MemoryObjectReceiveStream[SessionMessage]

    read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream(0)

    session_id = uuid4()
    self._read_stream_writers[session_id] = read_stream_writer
    logger.debug(f"Created new session with ID: {session_id}")

    # Determine the full path for the message endpoint
    root_path = scope.get("root_path", "")
    from urllib.parse import quote
    full_message_path_for_client = root_path.rstrip("/") + self._endpoint
    client_post_uri_data = f"{quote(full_message_path_for_client)}?session_id={session_id.hex}"

    sse_stream_writer, sse_stream_reader = anyio.create_memory_object_stream[dict[str, Any]](0)

    async def sse_writer():
        logger.debug("Starting SSE writer")
        async with sse_stream_writer, write_stream_reader:
            await sse_stream_writer.send({"event": "endpoint", "data": client_post_uri_data})
            logger.debug(f"Sent endpoint event: {client_post_uri_data}")

            async for session_message in write_stream_reader:
                logger.debug(f"Sending message via SSE: {session_message}")
                await sse_stream_writer.send(
                    {
                        "event": "message",
                        "data": session_message.message.model_dump_json(by_alias=True, exclude_none=True),
                    }
                )

    async with anyio.create_task_group() as tg:

        async def response_wrapper(scope: Scope, receive: Receive, send: Send):
            """
            Wrapper that creates EventSourceResponse WITH ping support.

            The ping parameter sends periodic "ping" events to keep the
            connection alive and prevent client timeouts.
            """
            settings = get_settings()
            # Send ping every 15 seconds to keep connection alive
            ping_interval = getattr(settings, 'sse_ping_interval', 15)

            await EventSourceResponse(
                content=sse_stream_reader,
                data_sender_callable=sse_writer,
                ping=ping_interval,  # KEY FIX: Send ping every N seconds
            )(scope, receive, send)

            await read_stream_writer.aclose()
            await write_stream_reader.aclose()
            logger.debug(f"Client session disconnected {session_id}")

        logger.debug("Starting SSE response task with keep-alive pings")
        tg.start_soon(response_wrapper, scope, receive, send)

        logger.debug("Yielding read and write streams")
        yield (read_stream, write_stream)


def apply_sse_keepalive_patch():
    """
    Apply the SSE keep-alive patch to the MCP library.

    This should be called BEFORE creating any FastMCP instances.
    """
    if SseServerTransport.connect_sse is patched_connect_sse:
        logger.debug("SSE keep-alive patch already applied")
        return

    SseServerTransport.connect_sse = patched_connect_sse
    logger.info("Applied SSE keep-alive patch to MCP library")


def revert_sse_keepalive_patch():
    """
    Revert the SSE keep-alive patch (for testing purposes).
    """
    SseServerTransport.connect_sse = _original_connect_sse
    logger.info("Reverted SSE keep-alive patch")
