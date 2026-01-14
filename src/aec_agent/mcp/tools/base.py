"""
Base utilities for MCP tools.
"""

from aec_agent.mcp.server import get_lock, get_cache
from aec_agent.mcp.sidecar_client import (
    call_sidecar,
    SidecarError,
    SidecarTimeoutError,
    SidecarConnectionError,
    SidecarCircuitOpenError,
)


def error_result(code: int, message: str, details: str = None) -> dict:
    """Create standardized error result."""
    result = {
        "success": False,
        "error": {
            "code": code,
            "message": message
        }
    }
    if details:
        result["error"]["details"] = details
    return result


def success_result(data: dict = None, message: str = None) -> dict:
    """Create standardized success result."""
    result = {"success": True}
    if data:
        result["data"] = data
    if message:
        result["message"] = message
    return result


# Error codes
class ErrorCode:
    # 4xxx - Client errors
    MISSING_CONFIG = 4001
    INVALID_PARAMS = 4002
    ELEMENT_NOT_FOUND = 4003
    INVALID_OPERATION = 4004

    # 5xxx - Server/Sidecar errors
    INTERNAL_ERROR = 5001
    SIDECAR_ERROR = 5002
    TRANSACTION_FAILED = 5003
    TIMEOUT = 5004
    CONNECTION_FAILED = 5005
    CIRCUIT_OPEN = 5006
