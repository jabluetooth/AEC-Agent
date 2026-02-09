"""
HTTP client for communicating with AutoCAD/Revit sidecars.

Handles:
- Connection pooling
- Retry logic with exponential backoff
- Circuit breaker for fault tolerance
- Session token authentication
"""

import httpx
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from aec_agent.config.settings import get_settings
from aec_agent.mcp.concurrency import CircuitBreaker

logger = structlog.get_logger(__name__)

# Global settings
settings = get_settings()

# Configure timeouts
SIDECAR_TIMEOUT = httpx.Timeout(
    settings.sidecar_read_timeout,
    connect=settings.sidecar_connect_timeout
)

# Reusable HTTP client (connection pooling)
_client: httpx.AsyncClient = None

# Circuit breakers per sidecar type
_circuit_breakers = {
    "autocad": CircuitBreaker(),
    "revit": CircuitBreaker(),
}


async def get_client() -> httpx.AsyncClient:
    """Get or create the HTTP client."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=SIDECAR_TIMEOUT)
    return _client


async def close_client():
    """Close the HTTP client."""
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None


def get_sidecar_url(endpoint: str) -> str:
    """
    Build sidecar URL from endpoint.

    Args:
        endpoint: API endpoint (e.g., "/mcp/walls/create")

    Returns:
        Full URL (e.g., "http://127.0.0.1:20001/mcp/walls/create")
    """
    port = settings.mcp_listener_port
    if not port:
        raise ValueError("MCP_LISTENER_PORT not configured")
    return f"http://127.0.0.1:{port}{endpoint}"


def get_auth_headers(sidecar_type: str = "revit") -> dict:
    """
    Get authentication headers for sidecar requests.

    Args:
        sidecar_type: "autocad" or "revit" - different auth schemes

    Returns:
        Headers dict with appropriate auth header
    """
    token = settings.session_token
    if not token:
        logger.warning("SESSION_TOKEN not configured, requests may be rejected")
        return {}

    # AutoCAD uses Authorization: Bearer, Revit uses X-Session-Token
    if sidecar_type == "autocad":
        return {"Authorization": f"Bearer {token}"}
    else:
        return {"X-Session-Token": token}


class SidecarError(Exception):
    """Base exception for sidecar errors."""

    def __init__(self, code: int, message: str, details: str = None):
        self.code = code
        self.message = message
        self.details = details
        super().__init__(f"[{code}] {message}")


class SidecarTimeoutError(SidecarError):
    """Sidecar request timed out."""

    def __init__(self, details: str = None):
        super().__init__(5004, "Sidecar timeout", details)


class SidecarConnectionError(SidecarError):
    """Cannot connect to sidecar."""

    def __init__(self, details: str = None):
        super().__init__(5005, "Sidecar connection failed", details)


class SidecarCircuitOpenError(SidecarError):
    """Circuit breaker is open."""

    def __init__(self):
        super().__init__(5006, "Sidecar circuit breaker open - service temporarily unavailable")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    reraise=True
)
async def _call_sidecar_with_retry(
    method: str,
    url: str,
    headers: dict,
    json_data: dict = None
) -> httpx.Response:
    """
    Make HTTP request with retry logic.

    Args:
        method: HTTP method (GET, POST)
        url: Full URL
        headers: Request headers
        json_data: JSON body for POST requests

    Returns:
        httpx.Response object

    Raises:
        httpx.TimeoutException: Request timed out
        httpx.ConnectError: Connection failed
        SidecarError: Non-2xx response with parsed error details
    """
    client = await get_client()

    if method.upper() == "GET":
        response = await client.get(url, headers=headers)
    else:
        response = await client.post(url, headers=headers, json=json_data)

    # Parse response body for error details on non-2xx responses
    if response.status_code >= 400:
        error_details = None
        try:
            error_json = response.json()
            # Extract error message from sidecar response format
            if isinstance(error_json, dict):
                error_obj = error_json.get("error", {})
                if isinstance(error_obj, dict):
                    error_details = error_obj.get("details") or error_obj.get("message")
                elif error_json.get("details"):
                    error_details = error_json.get("details")
        except Exception:
            error_details = response.text[:200] if response.text else None

        raise SidecarError(
            code=response.status_code,
            message=f"Sidecar returned {response.status_code}",
            details=error_details
        )

    return response


async def call_sidecar(
    endpoint: str,
    method: str = "POST",
    payload: dict = None,
    sidecar_type: str = "revit"
) -> dict:
    """
    Call a sidecar endpoint with full error handling.

    Args:
        endpoint: API endpoint (e.g., "/mcp/walls/create")
        method: HTTP method (default POST)
        payload: Request body for POST
        sidecar_type: "autocad" or "revit" (for circuit breaker)

    Returns:
        Response JSON as dict

    Raises:
        SidecarError: On any failure
    """
    circuit_breaker = _circuit_breakers.get(sidecar_type)

    # Check circuit breaker
    if circuit_breaker and not await circuit_breaker.can_execute():
        raise SidecarCircuitOpenError()

    url = get_sidecar_url(endpoint)
    headers = get_auth_headers(sidecar_type)

    logger.debug(
        "Calling sidecar",
        endpoint=endpoint,
        method=method,
        sidecar_type=sidecar_type
    )

    try:
        response = await _call_sidecar_with_retry(
            method=method,
            url=url,
            headers=headers,
            json_data=payload
        )

        result = response.json()

        # Record success
        if circuit_breaker:
            await circuit_breaker.record_success()

        logger.debug("Sidecar response", success=result.get("success", False))
        return result

    except httpx.TimeoutException as e:
        if circuit_breaker:
            await circuit_breaker.record_failure()
        logger.error("Sidecar timeout", endpoint=endpoint, error=str(e))
        raise SidecarTimeoutError(str(e))

    except httpx.ConnectError as e:
        if circuit_breaker:
            await circuit_breaker.record_failure()
        logger.error("Sidecar connection failed", endpoint=endpoint, error=str(e))
        raise SidecarConnectionError(str(e))

    except SidecarError as e:
        if circuit_breaker:
            await circuit_breaker.record_failure()
        logger.error(
            "Sidecar HTTP error",
            endpoint=endpoint,
            status_code=e.code,
            error=e.details or e.message
        )
        raise


async def call_autocad_command(
    command: str,
    params: dict = None
) -> dict:
    """
    Call an AutoCAD sidecar command using command-based API format.

    AutoCAD sidecar uses a different API format than Revit:
    - POST to root endpoint with {"command": "...", "params": {...}}
    - Authorization: Bearer token

    Args:
        command: Command name (e.g., "draw_line", "create_layer")
        params: Command parameters

    Returns:
        Response JSON as dict

    Raises:
        SidecarError: On any failure
    """
    circuit_breaker = _circuit_breakers.get("autocad")

    # Check circuit breaker
    if circuit_breaker and not await circuit_breaker.can_execute():
        raise SidecarCircuitOpenError()

    url = get_sidecar_url("/")
    headers = get_auth_headers("autocad")

    payload = {
        "command": command,
        "parameters": params or {}  # Note: C# sidecar expects "parameters", not "params"
    }

    logger.debug(
        "Calling AutoCAD command",
        command=command,
        params=params
    )

    try:
        response = await _call_sidecar_with_retry(
            method="POST",
            url=url,
            headers=headers,
            json_data=payload
        )

        result = response.json()

        # Record success
        if circuit_breaker:
            await circuit_breaker.record_success()

        logger.debug("AutoCAD response", success=result.get("success", False))
        return result

    except httpx.TimeoutException as e:
        if circuit_breaker:
            await circuit_breaker.record_failure()
        logger.error("AutoCAD timeout", command=command, error=str(e))
        raise SidecarTimeoutError(str(e))

    except httpx.ConnectError as e:
        if circuit_breaker:
            await circuit_breaker.record_failure()
        logger.error("AutoCAD connection failed", command=command, error=str(e))
        raise SidecarConnectionError(str(e))

    except SidecarError as e:
        if circuit_breaker:
            await circuit_breaker.record_failure()
        logger.error(
            "AutoCAD HTTP error",
            command=command,
            params=params,
            status_code=e.code,
            error_message=e.message,
            error_details=e.details,
            circuit_state=circuit_breaker.state if circuit_breaker else "none"
        )
        raise


async def check_sidecar_health(sidecar_type: str = "revit") -> dict:
    """
    Check if sidecar is healthy.

    Args:
        sidecar_type: "autocad" or "revit"

    Returns:
        Health status dict
    """
    try:
        result = await call_sidecar(
            endpoint="/health",
            method="GET",
            sidecar_type=sidecar_type
        )
        return {
            "healthy": result.get("success", False),
            "sidecar_type": sidecar_type,
            "details": result.get("data", {})
        }
    except SidecarError as e:
        return {
            "healthy": False,
            "sidecar_type": sidecar_type,
            "error": {"code": e.code, "message": e.message}
        }
