"""
AEC Agent Revit Sidecar startup script.

This script runs when Revit starts and initializes the AEC Agent sidecar:
1. Configures the Routes API port from environment variables
2. Registers document event hooks for cache synchronization
3. Loads route handlers to expose the Revit API via HTTP

Environment Variables:
    MCP_LISTENER_PORT: Port for HTTP server (default: 48884)
    SESSION_TOKEN: Authentication token for API requests
"""

import sys
import os

# Add lib directory to Python path
extension_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
lib_dir = os.path.join(extension_dir, 'lib')

if lib_dir not in sys.path:
    sys.path.insert(0, lib_dir)

from pyrevit.coreutils import logger

# Banner
logger.info("=" * 60)
logger.info("AEC Agent Revit Sidecar v0.1.0")
logger.info("=" * 60)

# Import and run configuration
try:
    from routes_config import configure_routes, get_configured_port, get_session_token

    port = get_configured_port()
    token = get_session_token()

    logger.info("Configuration:")
    logger.info("  Port: {}".format(port))
    logger.info("  Session Token: {}".format("configured" if token else "NOT SET (development mode)"))

    if configure_routes():
        logger.info("  Routes: configured successfully")
    else:
        logger.warn("  Routes: using default configuration")

except Exception as e:
    logger.error("Failed to configure routes: {}".format(e))

# Register document event hooks
try:
    from event_hooks import register_hooks
    if register_hooks():
        logger.info("  Event Hooks: registered")
    else:
        logger.warn("  Event Hooks: registration failed")
except Exception as e:
    logger.error("Failed to register event hooks: {}".format(e))

# Import route handlers to register them with pyRevit Routes
try:
    import routes_handlers
    logger.info("  Route Handlers: loaded")
except Exception as e:
    logger.error("Failed to load route handlers: {}".format(e))

# Final status
logger.info("=" * 60)
logger.info("AEC Agent Revit Sidecar Ready")
logger.info("=" * 60)
