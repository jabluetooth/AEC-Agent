"""
Routes configuration for AEC Agent Revit sidecar.

Handles dynamic port assignment for multi-session environments.
"""

import os
from pyrevit import routes
from pyrevit.coreutils import logger

# Port range for sidecar (matches settings.py)
MIN_PORT = 20000
MAX_PORT = 30000
DEFAULT_PORT = 48884


def get_configured_port():
    # type: () -> int
    """Get port from environment or use default."""
    port_str = os.environ.get("MCP_LISTENER_PORT")

    if not port_str:
        logger.warn(
            "MCP_LISTENER_PORT not set. "
            "Using default port {}. "
            "This may cause conflicts in multi-session environments.".format(DEFAULT_PORT)
        )
        return DEFAULT_PORT

    try:
        port = int(port_str)
        if not (MIN_PORT <= port <= MAX_PORT):
            logger.error(
                "MCP_LISTENER_PORT {} outside valid range ({}-{}). "
                "Using default.".format(port, MIN_PORT, MAX_PORT)
            )
            return DEFAULT_PORT
        return port
    except ValueError:
        logger.error(
            "MCP_LISTENER_PORT '{}' is not a valid integer. "
            "Using default.".format(port_str)
        )
        return DEFAULT_PORT


def get_session_token():
    # type: () -> str
    """Get session token from environment."""
    token = os.environ.get("SESSION_TOKEN", "")
    if not token:
        logger.warn("SESSION_TOKEN not set. Security validation disabled.")
    return token


def configure_routes():
    # type: () -> bool
    """Configure pyRevit Routes with session-specific settings."""
    port = get_configured_port()

    try:
        # pyRevit Routes API - configure before routes are registered
        server = routes.get_routes_server()
        if server:
            server.port = port
            logger.info("AEC Agent Routes configured on port {}".format(port))
            return True
        else:
            logger.warn("Routes server not available")
            return False
    except AttributeError:
        # Fallback for older pyRevit versions
        logger.warn(
            "Could not configure port dynamically. "
            "pyRevit version may not support routes.get_routes_server(). "
            "Routes will use default port."
        )
        return False
    except Exception as e:
        logger.error("Failed to configure Routes: {}".format(e))
        return False
