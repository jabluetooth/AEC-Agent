# Phase 3: Revit Sidecar (pyRevit Routes Implementation)

**Objective:** Expose the Revit API via HTTP using pyRevit Routes, enabling the MCP middleware to control Revit through natural language commands.

**Prerequisites:**
- Revit 2021-2025 installed
- pyRevit 4.8.x (or 5.0+ for Revit 2025)
- Python 3.x (bundled with pyRevit)

## 1. Framework Deployment

### 1.1 Install pyRevit

```powershell
# Download and install pyRevit CLI
Invoke-WebRequest -Uri "https://github.com/pyrevitlabs/pyRevit/releases/latest/download/pyRevit_CLI.exe" -OutFile "pyRevit_CLI.exe"

# Install pyRevit for all users (requires admin)
.\pyRevit_CLI.exe install --allusers

# Or install for current user only
.\pyRevit_CLI.exe install
```

### 1.2 Create Extension Directory Structure

```
%APPDATA%\pyRevit\Extensions\AECAgent.extension\
├── extension.json
├── AECAgent.tab\
│   └── AECAgent.panel\
│       └── startup.pushbutton\
│           ├── script.py
│           └── icon.png
└── lib\
    ├── __init__.py
    ├── routes_config.py
    ├── routes_handlers.py
    ├── security.py
    ├── cache_manager.py
    └── response.py
```

### 1.3 Extension Manifest

Create `extension.json`:

```json
{
    "name": "AECAgent",
    "description": "AI-powered Revit automation sidecar for MCP",
    "author": "AEC Team",
    "version": "0.1.0",
    "type": "extension",
    "engine": {
        "version": "2.8.0",
        "clean": true
    },
    "routes": {
        "enabled": true
    }
}
```

## 2. Port Configuration

### 2.1 The Problem

pyRevit Routes defaults to port 48884. In multi-session RDP environments, each user session needs a unique port to avoid conflicts.

### 2.2 Solution: Dynamic Port from Environment

Create `lib/routes_config.py`:

```python
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


def get_configured_port() -> int:
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


def get_session_token() -> str:
    """Get session token from environment."""
    token = os.environ.get("SESSION_TOKEN", "")
    if not token:
        logger.warn("SESSION_TOKEN not set. Security validation disabled.")
    return token


def configure_routes():
    """Configure pyRevit Routes with session-specific settings."""
    port = get_configured_port()

    try:
        # pyRevit Routes API - configure before routes are registered
        routes.get_routes_server().port = port
        logger.info("AEC Agent Routes configured on port {}".format(port))
        return True
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
```

## 3. Security Implementation

Create `lib/security.py`:

```python
"""
Security utilities for AEC Agent Revit sidecar.
Validates session tokens to prevent unauthorized access.
"""

import os
import functools
from pyrevit.coreutils import logger


def get_session_token() -> str:
    """Get the expected session token from environment."""
    return os.environ.get("SESSION_TOKEN", "")


def validate_token(request) -> bool:
    """
    Validate the session token from request headers.

    Expected header: X-Session-Token: <token>
    """
    expected_token = get_session_token()

    # If no token configured, skip validation (development mode)
    if not expected_token:
        return True

    # Get token from request headers
    provided_token = request.headers.get("X-Session-Token", "")

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
```

## 4. Response Format

Create `lib/response.py`:

```python
"""
Standardized response format for AEC Agent Revit sidecar.
Matches the response structure used by AutoCAD sidecar.
"""

import json
import traceback
from pyrevit.coreutils import logger


def success_response(data=None, message=None):
    """Create a successful response."""
    response = {"success": True}
    if data is not None:
        response["data"] = data
    if message:
        response["message"] = message
    return response


def error_response(code, message, details=None):
    """Create an error response."""
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


# Standard error codes (matching AutoCAD sidecar)
class ErrorCode:
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
    """
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
```

## 5. Route Handlers

Create `lib/routes_handlers.py`:

```python
"""
Route handlers for AEC Agent Revit sidecar.
Exposes Revit API operations via HTTP endpoints.
"""

import json
from pyrevit import routes, revit, DB
from pyrevit.coreutils import logger

from security import require_auth
from response import success_response, error_response, safe_handler, ErrorCode


# =============================================================================
# Health & Status Endpoints
# =============================================================================

@routes.route('/health', methods=['GET'])
def health_check(request):
    """Health check endpoint for monitoring."""
    return success_response(
        data={"status": "healthy", "service": "revit-sidecar"},
        message="Revit sidecar is running"
    )


@routes.route('/mcp/status', methods=['GET'])
@require_auth
@safe_handler
def get_status(request):
    """Get current Revit document status."""
    doc = revit.doc

    if not doc:
        return error_response(
            ErrorCode.INVALID_OPERATION,
            "No document open in Revit"
        )

    return success_response(data={
        "document_title": doc.Title,
        "document_path": doc.PathName or "Not saved",
        "is_modified": doc.IsModified,
        "is_workshared": doc.IsWorkshared,
    })


# =============================================================================
# Layer/Level Operations
# =============================================================================

@routes.route('/mcp/levels', methods=['GET'])
@require_auth
@safe_handler
def list_levels(request):
    """List all levels in the current document."""
    doc = revit.doc

    if not doc:
        return error_response(ErrorCode.INVALID_OPERATION, "No document open")

    collector = DB.FilteredElementCollector(doc)
    levels = collector.OfClass(DB.Level).ToElements()

    level_data = []
    for level in levels:
        level_data.append({
            "id": level.Id.IntegerValue,
            "name": level.Name,
            "elevation": level.Elevation,  # In feet (Revit internal units)
            "elevation_m": level.Elevation * 0.3048,  # Convert to meters
        })

    # Sort by elevation
    level_data.sort(key=lambda x: x["elevation"])

    return success_response(data={"levels": level_data, "count": len(level_data)})


@routes.route('/mcp/levels/create', methods=['POST'])
@require_auth
@safe_handler
def create_level(request):
    """
    Create a new level.

    Request body:
        {
            "name": "Level 3",
            "elevation": 10.0  // meters
        }
    """
    doc = revit.doc

    if not doc:
        return error_response(ErrorCode.INVALID_OPERATION, "No document open")

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return error_response(ErrorCode.INVALID_PARAMS, "Invalid JSON body")

    name = body.get("name")
    elevation_m = body.get("elevation")

    if not name:
        return error_response(ErrorCode.INVALID_PARAMS, "Missing 'name' parameter")
    if elevation_m is None:
        return error_response(ErrorCode.INVALID_PARAMS, "Missing 'elevation' parameter")

    # Convert meters to feet (Revit internal units)
    elevation_ft = float(elevation_m) / 0.3048

    with revit.Transaction("AEC Agent: Create Level"):
        new_level = DB.Level.Create(doc, elevation_ft)
        new_level.Name = name

    return success_response(
        data={
            "id": new_level.Id.IntegerValue,
            "name": new_level.Name,
            "elevation_m": elevation_m
        },
        message="Level '{}' created successfully".format(name)
    )


# =============================================================================
# Wall Operations
# =============================================================================

@routes.route('/mcp/walls', methods=['GET'])
@require_auth
@safe_handler
def list_walls(request):
    """List all walls in the current document."""
    doc = revit.doc

    if not doc:
        return error_response(ErrorCode.INVALID_OPERATION, "No document open")

    collector = DB.FilteredElementCollector(doc)
    walls = collector.OfClass(DB.Wall).ToElements()

    wall_data = []
    for wall in walls:
        wall_type = doc.GetElement(wall.GetTypeId())
        wall_data.append({
            "id": wall.Id.IntegerValue,
            "type_name": wall_type.Name if wall_type else "Unknown",
            "level_id": wall.LevelId.IntegerValue if wall.LevelId else None,
            "length_m": wall.get_Parameter(DB.BuiltInParameter.CURVE_ELEM_LENGTH).AsDouble() * 0.3048,
        })

    return success_response(data={"walls": wall_data, "count": len(wall_data)})


@routes.route('/mcp/walls/create', methods=['POST'])
@require_auth
@safe_handler
def create_wall(request):
    """
    Create a new wall.

    Request body:
        {
            "start": {"x": 0, "y": 0},  // meters
            "end": {"x": 10, "y": 0},   // meters
            "level_id": 12345,
            "height": 3.0,              // meters (optional, default 3m)
            "wall_type_id": 67890       // optional, uses default if not specified
        }
    """
    doc = revit.doc

    if not doc:
        return error_response(ErrorCode.INVALID_OPERATION, "No document open")

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return error_response(ErrorCode.INVALID_PARAMS, "Invalid JSON body")

    # Validate required parameters
    start = body.get("start")
    end = body.get("end")
    level_id = body.get("level_id")

    if not all([start, end, level_id]):
        return error_response(
            ErrorCode.INVALID_PARAMS,
            "Missing required parameters: start, end, level_id"
        )

    # Convert meters to feet
    m_to_ft = 1 / 0.3048
    start_pt = DB.XYZ(start["x"] * m_to_ft, start["y"] * m_to_ft, 0)
    end_pt = DB.XYZ(end["x"] * m_to_ft, end["y"] * m_to_ft, 0)

    # Create line
    line = DB.Line.CreateBound(start_pt, end_pt)

    # Get level
    level = doc.GetElement(DB.ElementId(level_id))
    if not level or not isinstance(level, DB.Level):
        return error_response(ErrorCode.ELEMENT_NOT_FOUND, "Level not found")

    # Height (default 3m = ~10ft)
    height = body.get("height", 3.0) * m_to_ft

    with revit.Transaction("AEC Agent: Create Wall"):
        wall = DB.Wall.Create(
            doc,
            line,
            level.Id,
            False  # structural
        )

        # Set height if specified
        height_param = wall.get_Parameter(DB.BuiltInParameter.WALL_USER_HEIGHT_PARAM)
        if height_param:
            height_param.Set(height)

    return success_response(
        data={"id": wall.Id.IntegerValue},
        message="Wall created successfully"
    )


# =============================================================================
# Room Operations
# =============================================================================

@routes.route('/mcp/rooms', methods=['GET'])
@require_auth
@safe_handler
def list_rooms(request):
    """List all rooms in the current document."""
    doc = revit.doc

    if not doc:
        return error_response(ErrorCode.INVALID_OPERATION, "No document open")

    collector = DB.FilteredElementCollector(doc)
    rooms = collector.OfCategory(DB.BuiltInCategory.OST_Rooms).WhereElementIsNotElementType().ToElements()

    room_data = []
    for room in rooms:
        if room.Area > 0:  # Only include placed rooms
            room_data.append({
                "id": room.Id.IntegerValue,
                "name": room.get_Parameter(DB.BuiltInParameter.ROOM_NAME).AsString(),
                "number": room.get_Parameter(DB.BuiltInParameter.ROOM_NUMBER).AsString(),
                "level": room.Level.Name if room.Level else None,
                "area_sqm": room.Area * 0.092903,  # sqft to sqm
            })

    return success_response(data={"rooms": room_data, "count": len(room_data)})
```

## 6. Startup Script

Create `AECAgent.tab/AECAgent.panel/startup.pushbutton/script.py`:

```python
"""
AEC Agent Revit Sidecar startup script.
Initializes routes and logging when Revit starts.
"""

import sys
import os

# Add lib directory to path
extension_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
lib_dir = os.path.join(extension_dir, 'lib')
if lib_dir not in sys.path:
    sys.path.insert(0, lib_dir)

from pyrevit.coreutils import logger
from routes_config import configure_routes, get_configured_port, get_session_token

# Configure routes on startup
logger.info("=" * 50)
logger.info("AEC Agent Revit Sidecar Starting")
logger.info("=" * 50)

port = get_configured_port()
token = get_session_token()

logger.info("Port: {}".format(port))
logger.info("Session Token: {}".format("configured" if token else "NOT SET (insecure)"))

if configure_routes():
    logger.info("Routes configured successfully")
else:
    logger.warn("Routes configuration failed - using defaults")

# Import route handlers to register them
import routes_handlers

logger.info("AEC Agent Revit Sidecar Ready")
logger.info("=" * 50)
```

## 7. SQLite Cache for Read Operations

### 7.1 The Problem

Querying the Revit model for large datasets (e.g., "list all 500 rooms") is slow and blocks the UI thread.

### 7.2 Solution: Cached Resource Pattern

Create `lib/cache_manager.py`:

```python
"""
SQLite cache manager for AEC Agent.
Caches Revit model metadata for fast read operations.
"""

import os
import hashlib
import sqlite3
import json
from pathlib import Path
from datetime import datetime

from pyrevit import revit, DB
from pyrevit.coreutils import logger


def get_cache_dir() -> Path:
    """Get the cache directory path."""
    # Use LOCALAPPDATA for persistence across sessions
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        # Fallback for non-Windows
        local_app_data = os.path.expanduser("~/.local/share")

    cache_dir = Path(local_app_data) / "AECAgent" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_document_hash(doc_path: str) -> str:
    """Generate a unique hash for a document path."""
    # Using SHA256 (more secure than MD5, though MD5 would work for this use case)
    return hashlib.sha256(doc_path.encode('utf-8')).hexdigest()[:16]


def get_cache_path(doc_path: str) -> Path:
    """Get the cache file path for a document."""
    doc_hash = get_document_hash(doc_path)
    return get_cache_dir() / "{}.sqlite".format(doc_hash)


class RevitCache:
    """Manages SQLite cache for a Revit document."""

    def __init__(self, doc):
        self.doc = doc
        self.doc_path = doc.PathName or "untitled_{}".format(doc.Title)
        self.cache_path = get_cache_path(self.doc_path)
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database schema."""
        with sqlite3.connect(str(self.cache_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS elements (
                    id INTEGER PRIMARY KEY,
                    category TEXT,
                    name TEXT,
                    type_name TEXT,
                    level_id INTEGER,
                    data JSON,
                    updated_at TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_elements_category
                ON elements(category)
            """)

            conn.commit()

    def sync_rooms(self):
        """Sync all rooms to cache."""
        collector = DB.FilteredElementCollector(self.doc)
        rooms = collector.OfCategory(
            DB.BuiltInCategory.OST_Rooms
        ).WhereElementIsNotElementType().ToElements()

        now = datetime.utcnow().isoformat()

        with sqlite3.connect(str(self.cache_path)) as conn:
            # Clear existing rooms
            conn.execute("DELETE FROM elements WHERE category = 'rooms'")

            for room in rooms:
                if room.Area > 0:  # Only placed rooms
                    data = {
                        "name": room.get_Parameter(DB.BuiltInParameter.ROOM_NAME).AsString(),
                        "number": room.get_Parameter(DB.BuiltInParameter.ROOM_NUMBER).AsString(),
                        "area_sqm": room.Area * 0.092903,
                        "level": room.Level.Name if room.Level else None,
                    }

                    conn.execute(
                        """
                        INSERT OR REPLACE INTO elements
                        (id, category, name, type_name, level_id, data, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            room.Id.IntegerValue,
                            "rooms",
                            data["name"],
                            None,
                            room.Level.Id.IntegerValue if room.Level else None,
                            json.dumps(data),
                            now
                        )
                    )

            # Update sync timestamp
            conn.execute(
                """
                INSERT OR REPLACE INTO metadata (key, value, updated_at)
                VALUES ('rooms_synced_at', ?, ?)
                """,
                (now, now)
            )

            conn.commit()

        logger.info("Synced {} rooms to cache".format(len([r for r in rooms if r.Area > 0])))

    def get_rooms_from_cache(self) -> list:
        """Get rooms from cache (fast read)."""
        with sqlite3.connect(str(self.cache_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT id, data FROM elements WHERE category = 'rooms'"
            )

            rooms = []
            for row in cursor:
                data = json.loads(row["data"])
                data["id"] = row["id"]
                rooms.append(data)

            return rooms

    def get_cache_age_seconds(self, category: str) -> float:
        """Get the age of cached data in seconds."""
        with sqlite3.connect(str(self.cache_path)) as conn:
            cursor = conn.execute(
                "SELECT value FROM metadata WHERE key = ?",
                ("{}_synced_at".format(category),)
            )
            row = cursor.fetchone()

            if not row:
                return float('inf')  # Never synced

            synced_at = datetime.fromisoformat(row[0])
            age = (datetime.utcnow() - synced_at).total_seconds()
            return age
```

### 7.3 Cache Sync Triggers

Add to `routes_handlers.py`:

```python
# =============================================================================
# Cache Management
# =============================================================================

@routes.route('/mcp/cache/sync', methods=['POST'])
@require_auth
@safe_handler
def sync_cache(request):
    """
    Trigger a cache sync for the current document.

    Request body (optional):
        {
            "categories": ["rooms", "walls", "levels"]  // default: all
        }
    """
    doc = revit.doc

    if not doc:
        return error_response(ErrorCode.INVALID_OPERATION, "No document open")

    from cache_manager import RevitCache
    cache = RevitCache(doc)

    try:
        body = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        body = {}

    categories = body.get("categories", ["rooms"])
    synced = []

    for category in categories:
        if category == "rooms":
            cache.sync_rooms()
            synced.append("rooms")
        # Add more categories as needed

    return success_response(
        data={"synced_categories": synced},
        message="Cache sync completed"
    )


@routes.route('/mcp/rooms/cached', methods=['GET'])
@require_auth
@safe_handler
def list_rooms_cached(request):
    """
    List rooms from cache (fast, <50ms).
    Falls back to live query if cache is stale (>5 min).
    """
    doc = revit.doc

    if not doc:
        return error_response(ErrorCode.INVALID_OPERATION, "No document open")

    from cache_manager import RevitCache
    cache = RevitCache(doc)

    # Check cache age
    cache_age = cache.get_cache_age_seconds("rooms")
    max_age = 300  # 5 minutes

    if cache_age > max_age:
        # Cache is stale, sync first
        cache.sync_rooms()

    rooms = cache.get_rooms_from_cache()

    return success_response(
        data={
            "rooms": rooms,
            "count": len(rooms),
            "from_cache": True,
            "cache_age_seconds": min(cache_age, cache.get_cache_age_seconds("rooms"))
        }
    )
```

## 8. Document Event Hooks (Auto-Sync)

Create `lib/event_hooks.py`:

```python
"""
Revit document event hooks for automatic cache synchronization.
"""

from pyrevit import HOST_APP
from pyrevit.coreutils import logger

from cache_manager import RevitCache


def on_document_opened(sender, args):
    """Sync cache when a document is opened."""
    doc = args.Document
    if doc and doc.PathName:
        logger.info("Document opened: {}".format(doc.Title))
        try:
            cache = RevitCache(doc)
            cache.sync_rooms()
            logger.info("Initial cache sync completed")
        except Exception as e:
            logger.error("Cache sync failed: {}".format(e))


def on_document_saved(sender, args):
    """Sync cache when a document is saved."""
    doc = args.Document
    if doc:
        logger.info("Document saved: {}".format(doc.Title))
        try:
            cache = RevitCache(doc)
            cache.sync_rooms()
            logger.info("Post-save cache sync completed")
        except Exception as e:
            logger.error("Cache sync failed: {}".format(e))


def register_hooks():
    """Register document event hooks."""
    try:
        HOST_APP.app.DocumentOpened += on_document_opened
        HOST_APP.app.DocumentSaved += on_document_saved
        logger.info("Document event hooks registered")
    except Exception as e:
        logger.error("Failed to register event hooks: {}".format(e))


# Register on module load
register_hooks()
```

## 9. Testing

### 9.1 Manual Testing with curl

```bash
# Health check (no auth required)
curl http://localhost:20001/health

# Get status (requires auth)
curl -H "X-Session-Token: your-token-here" http://localhost:20001/mcp/status

# List levels
curl -H "X-Session-Token: your-token-here" http://localhost:20001/mcp/levels

# Create a level
curl -X POST \
  -H "X-Session-Token: your-token-here" \
  -H "Content-Type: application/json" \
  -d '{"name": "Level 5", "elevation": 15.0}' \
  http://localhost:20001/mcp/levels/create

# List rooms (from cache)
curl -H "X-Session-Token: your-token-here" http://localhost:20001/mcp/rooms/cached
```

### 9.2 PowerShell Test Script

Create `Test-RevitSidecar.ps1`:

```powershell
$port = $env:MCP_LISTENER_PORT ?? 20001
$token = $env:SESSION_TOKEN ?? "test-token"
$baseUrl = "http://localhost:$port"

$headers = @{
    "X-Session-Token" = $token
    "Content-Type" = "application/json"
}

Write-Host "Testing Revit Sidecar at $baseUrl" -ForegroundColor Cyan

# Health check
Write-Host "`n[1] Health Check" -ForegroundColor Yellow
Invoke-RestMethod -Uri "$baseUrl/health" -Method GET | ConvertTo-Json

# Status
Write-Host "`n[2] Document Status" -ForegroundColor Yellow
try {
    Invoke-RestMethod -Uri "$baseUrl/mcp/status" -Method GET -Headers $headers | ConvertTo-Json
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}

# List levels
Write-Host "`n[3] List Levels" -ForegroundColor Yellow
try {
    Invoke-RestMethod -Uri "$baseUrl/mcp/levels" -Method GET -Headers $headers | ConvertTo-Json
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}

Write-Host "`nTests complete!" -ForegroundColor Green
```

## 10. Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Routes not responding | pyRevit Routes not enabled | Check `extension.json` has `"routes": {"enabled": true}` |
| Port already in use | Another session using same port | Verify `MCP_LISTENER_PORT` is unique per session |
| 401 Unauthorized | Token mismatch | Check `SESSION_TOKEN` env var matches request header |
| Transaction failed | Document read-only or workshared conflict | Check document permissions and sync with central |
| Cache stale | Events not firing | Manually call `/mcp/cache/sync` endpoint |

## 11. Security Checklist

- [ ] `SESSION_TOKEN` environment variable set
- [ ] Token validated on all `/mcp/*` endpoints
- [ ] No sensitive data in logs
- [ ] Port range restricted to 20000-30000
- [ ] Health endpoint `/health` does not require auth (for monitoring)
- [ ] Error messages don't leak internal details
