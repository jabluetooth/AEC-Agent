"""
AEC Agent MCP Server entry point.

Run with: python -m aec_agent.server
"""

import structlog

from aec_agent.config.settings import get_settings
from aec_agent.utils.logging import setup_logging
from aec_agent.mcp.server import mcp

# Import tools to register them
# Note: We import them after mcp object creation so decorators work
# Using checking to avoid circular imports if needed, but here simple import should work
# if the tools modules import 'mcp' from aec_agent.mcp.server
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
    # FastMCP.run() handles uvicorn internally
    mcp.run(transport="sse")


if __name__ == "__main__":
    main()
