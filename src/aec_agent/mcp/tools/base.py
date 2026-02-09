"""
Base utilities for MCP tools.
"""

import functools

import structlog

from aec_agent.mcp.sidecar_client import (
    SidecarError,
)

logger = structlog.get_logger(__name__)


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
    LOCK_TIMEOUT = 5007


def safe_tool(func):
    """
    Decorator to catch any unhandled exceptions in tool functions.
    Prevents the MCP server from crashing on unexpected errors.
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except SidecarError as e:
            logger.error(
                "Sidecar error in tool",
                tool=func.__name__,
                error_code=e.code,
                error_message=e.message,
                error_details=e.details
            )
            return error_result(e.code, e.message, e.details)
        except Exception as e:
            logger.error(
                "Unexpected error in tool",
                tool=func.__name__,
                error=str(e),
                exc_info=True
            )
            return error_result(
                ErrorCode.INTERNAL_ERROR,
                f"Unexpected error in {func.__name__}",
                str(e)
            )
    return wrapper
