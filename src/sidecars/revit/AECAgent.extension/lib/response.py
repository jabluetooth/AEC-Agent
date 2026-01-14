"""
Standardized response format for AEC Agent Revit sidecar.

Matches the response structure used by AutoCAD sidecar for consistency.
"""

import functools
import traceback
from pyrevit.coreutils import logger


def success_response(data=None, message=None):
    # type: (object, str) -> dict
    """
    Create a successful response.

    Args:
        data: Optional data payload
        message: Optional success message

    Returns:
        Structured success response dict
    """
    response = {"success": True}
    if data is not None:
        response["data"] = data
    if message:
        response["message"] = message
    return response


def error_response(code, message, details=None):
    # type: (int, str, str) -> dict
    """
    Create an error response.

    Args:
        code: Error code (see ErrorCode class)
        message: Human-readable error message
        details: Optional additional details

    Returns:
        Structured error response dict
    """
    response = {
        "success": False,
        "error": {
            "code": code,
            "message": message
        }
    }
    if details:
        response["error"]["details"] = details
    return response


class ErrorCode:
    """Standard error codes matching AutoCAD sidecar."""

    # 4xxx - Client errors
    UNAUTHORIZED = 4001
    INVALID_PARAMS = 4002
    ELEMENT_NOT_FOUND = 4003
    INVALID_OPERATION = 4004

    # 5xxx - Server errors
    INTERNAL_ERROR = 5001
    REVIT_API_ERROR = 5002
    TRANSACTION_FAILED = 5003
    TIMEOUT = 5004


def safe_handler(func):
    """
    Decorator to catch exceptions and return structured error responses.

    Usage:
        @routes.route('/mcp/endpoint', methods=['POST'])
        @safe_handler
        def my_handler(request):
            ...

    Args:
        func: The route handler function to wrap

    Returns:
        Wrapped function with exception handling
    """
    @functools.wraps(func)
    def wrapper(request, *args, **kwargs):
        try:
            return func(request, *args, **kwargs)
        except Exception as e:
            logger.error("Handler error: {}\n{}".format(e, traceback.format_exc()))
            return error_response(
                ErrorCode.INTERNAL_ERROR,
                "Internal server error",
                details=str(e)
            )
    return wrapper
