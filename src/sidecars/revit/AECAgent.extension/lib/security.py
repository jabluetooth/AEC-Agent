"""
Security utilities for AEC Agent Revit sidecar.

Validates session tokens to prevent unauthorized access.
"""

import os
import functools
from pyrevit.coreutils import logger


def get_session_token():
    # type: () -> str
    """Get the expected session token from environment."""
    return os.environ.get("SESSION_TOKEN", "")


def validate_token(request):
    # type: (object) -> bool
    """
    Validate the session token from request headers.

    Expected header: X-Session-Token: <token>

    Args:
        request: The HTTP request object

    Returns:
        True if token is valid, False otherwise
    """
    expected_token = get_session_token()

    # If no token configured, skip validation (development mode)
    if not expected_token:
        return True

    # Get token from request headers
    headers = getattr(request, 'headers', {})
    if hasattr(headers, 'get'):
        provided_token = headers.get("X-Session-Token", "")
    else:
        provided_token = ""

    if not provided_token:
        logger.warn("Request missing X-Session-Token header")
        return False

    # Constant-time comparison to prevent timing attacks
    if len(provided_token) != len(expected_token):
        return False

    result = 0
    for a, b in zip(provided_token, expected_token):
        result |= ord(a) ^ ord(b)

    return result == 0


def require_auth(func):
    """
    Decorator to require session token authentication.

    Usage:
        @routes.route('/mcp/endpoint', methods=['POST'])
        @require_auth
        def my_handler(request):
            ...

    Args:
        func: The route handler function to wrap

    Returns:
        Wrapped function that validates auth before calling handler
    """
    @functools.wraps(func)
    def wrapper(request, *args, **kwargs):
        if not validate_token(request):
            return {
                "success": False,
                "error": {
                    "code": 4001,
                    "message": "Unauthorized: Invalid or missing session token"
                }
            }
        return func(request, *args, **kwargs)
    return wrapper
