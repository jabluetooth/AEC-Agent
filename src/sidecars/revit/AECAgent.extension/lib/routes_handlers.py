"""
Route handlers for AEC Agent Revit sidecar.

Exposes Revit API operations via HTTP endpoints for MCP middleware.
All endpoints return structured JSON responses matching the AutoCAD sidecar format.
"""

import json
from pyrevit import routes, revit, DB
from pyrevit.coreutils import logger

from security import require_auth
from response import success_response, error_response, safe_handler, ErrorCode

# Conversion constants
FEET_TO_METERS = 0.3048
METERS_TO_FEET = 1 / FEET_TO_METERS
SQFT_TO_SQM = 0.092903


# =============================================================================
# Health & Status Endpoints
# =============================================================================

@routes.route('/health', methods=['GET'])
def health_check(request):
    """
    Health check endpoint for monitoring.

    No authentication required - used by load balancers and monitoring tools.

    Returns:
        Success response with service status
    """
    return success_response(
        data={
            "status": "healthy",
            "service": "revit-sidecar",
            "version": "0.1.0"
        },
        message="Revit sidecar is running"
    )


@routes.route('/mcp/status', methods=['GET'])
@require_auth
@safe_handler
def get_status(request):
    """
    Get current Revit document status.

    Returns:
        Document title, path, modification state, and worksharing info
    """
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
        "is_family_document": doc.IsFamilyDocument,
    })


# =============================================================================
# Level Operations
# =============================================================================

@routes.route('/mcp/levels', methods=['GET'])
@require_auth
@safe_handler
def list_levels(request):
    """
    List all levels in the current document.

    Returns:
        List of levels with id, name, and elevation (in meters)
    """
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
            "elevation_ft": level.Elevation,
            "elevation_m": level.Elevation * FEET_TO_METERS,
        })

    # Sort by elevation
    level_data.sort(key=lambda x: x["elevation_ft"])

    return success_response(data={
        "levels": level_data,
        "count": len(level_data)
    })


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

    Returns:
        Created level id, name, and elevation
    """
    doc = revit.doc

    if not doc:
        return error_response(ErrorCode.INVALID_OPERATION, "No document open")

    try:
        body = json.loads(request.body) if request.body else {}
    except (json.JSONDecodeError, ValueError):
        return error_response(ErrorCode.INVALID_PARAMS, "Invalid JSON body")

    name = body.get("name")
    elevation_m = body.get("elevation")

    if not name:
        return error_response(ErrorCode.INVALID_PARAMS, "Missing 'name' parameter")
    if elevation_m is None:
        return error_response(ErrorCode.INVALID_PARAMS, "Missing 'elevation' parameter")

    try:
        elevation_m = float(elevation_m)
    except (ValueError, TypeError):
        return error_response(ErrorCode.INVALID_PARAMS, "'elevation' must be a number")

    # Convert meters to feet (Revit internal units)
    elevation_ft = elevation_m * METERS_TO_FEET

    try:
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
    except Exception as e:
        return error_response(
            ErrorCode.TRANSACTION_FAILED,
            "Failed to create level",
            details=str(e)
        )


# =============================================================================
# Wall Operations
# =============================================================================

@routes.route('/mcp/walls', methods=['GET'])
@require_auth
@safe_handler
def list_walls(request):
    """
    List all walls in the current document.

    Returns:
        List of walls with id, type, level, and length
    """
    doc = revit.doc

    if not doc:
        return error_response(ErrorCode.INVALID_OPERATION, "No document open")

    collector = DB.FilteredElementCollector(doc)
    walls = collector.OfClass(DB.Wall).ToElements()

    wall_data = []
    for wall in walls:
        wall_type = doc.GetElement(wall.GetTypeId())
        length_param = wall.get_Parameter(DB.BuiltInParameter.CURVE_ELEM_LENGTH)

        wall_data.append({
            "id": wall.Id.IntegerValue,
            "type_name": wall_type.Name if wall_type else "Unknown",
            "level_id": wall.LevelId.IntegerValue if wall.LevelId else None,
            "length_m": length_param.AsDouble() * FEET_TO_METERS if length_param else 0,
        })

    return success_response(data={
        "walls": wall_data,
        "count": len(wall_data)
    })


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
            "height": 3.0               // meters (optional, default 3m)
        }

    Returns:
        Created wall id
    """
    doc = revit.doc

    if not doc:
        return error_response(ErrorCode.INVALID_OPERATION, "No document open")

    try:
        body = json.loads(request.body) if request.body else {}
    except (json.JSONDecodeError, ValueError):
        return error_response(ErrorCode.INVALID_PARAMS, "Invalid JSON body")

    # Validate required parameters
    start = body.get("start")
    end = body.get("end")
    level_id = body.get("level_id")

    if not start or not isinstance(start, dict):
        return error_response(ErrorCode.INVALID_PARAMS, "Missing or invalid 'start' parameter")
    if not end or not isinstance(end, dict):
        return error_response(ErrorCode.INVALID_PARAMS, "Missing or invalid 'end' parameter")
    if level_id is None:
        return error_response(ErrorCode.INVALID_PARAMS, "Missing 'level_id' parameter")

    # Validate start/end have x, y
    if "x" not in start or "y" not in start:
        return error_response(ErrorCode.INVALID_PARAMS, "'start' must have 'x' and 'y' properties")
    if "x" not in end or "y" not in end:
        return error_response(ErrorCode.INVALID_PARAMS, "'end' must have 'x' and 'y' properties")

    try:
        # Convert meters to feet
        start_pt = DB.XYZ(
            float(start["x"]) * METERS_TO_FEET,
            float(start["y"]) * METERS_TO_FEET,
            0
        )
        end_pt = DB.XYZ(
            float(end["x"]) * METERS_TO_FEET,
            float(end["y"]) * METERS_TO_FEET,
            0
        )
    except (ValueError, TypeError) as e:
        return error_response(ErrorCode.INVALID_PARAMS, "Invalid coordinate values: {}".format(e))

    # Create line
    line = DB.Line.CreateBound(start_pt, end_pt)

    # Get level
    level = doc.GetElement(DB.ElementId(int(level_id)))
    if not level or not isinstance(level, DB.Level):
        return error_response(ErrorCode.ELEMENT_NOT_FOUND, "Level not found with id {}".format(level_id))

    # Height (default 3m)
    height_m = body.get("height", 3.0)
    try:
        height_ft = float(height_m) * METERS_TO_FEET
    except (ValueError, TypeError):
        height_ft = 3.0 * METERS_TO_FEET

    try:
        with revit.Transaction("AEC Agent: Create Wall"):
            wall = DB.Wall.Create(
                doc,
                line,
                level.Id,
                False  # structural
            )

            # Set height
            height_param = wall.get_Parameter(DB.BuiltInParameter.WALL_USER_HEIGHT_PARAM)
            if height_param:
                height_param.Set(height_ft)

        return success_response(
            data={"id": wall.Id.IntegerValue},
            message="Wall created successfully"
        )
    except Exception as e:
        return error_response(
            ErrorCode.TRANSACTION_FAILED,
            "Failed to create wall",
            details=str(e)
        )


# =============================================================================
# Room Operations
# =============================================================================

@routes.route('/mcp/rooms', methods=['GET'])
@require_auth
@safe_handler
def list_rooms(request):
    """
    List all rooms in the current document (live query).

    Returns:
        List of rooms with id, name, number, level, and area
    """
    doc = revit.doc

    if not doc:
        return error_response(ErrorCode.INVALID_OPERATION, "No document open")

    collector = DB.FilteredElementCollector(doc)
    rooms = collector.OfCategory(
        DB.BuiltInCategory.OST_Rooms
    ).WhereElementIsNotElementType().ToElements()

    room_data = []
    for room in rooms:
        if room.Area > 0:  # Only include placed rooms
            name_param = room.get_Parameter(DB.BuiltInParameter.ROOM_NAME)
            number_param = room.get_Parameter(DB.BuiltInParameter.ROOM_NUMBER)

            room_data.append({
                "id": room.Id.IntegerValue,
                "name": name_param.AsString() if name_param else "",
                "number": number_param.AsString() if number_param else "",
                "level": room.Level.Name if room.Level else None,
                "area_sqm": room.Area * SQFT_TO_SQM,
            })

    return success_response(data={
        "rooms": room_data,
        "count": len(room_data)
    })


@routes.route('/mcp/rooms/cached', methods=['GET'])
@require_auth
@safe_handler
def list_rooms_cached(request):
    """
    List rooms from cache (fast, <50ms).

    Falls back to live query and syncs if cache is stale (>5 min).

    Returns:
        List of rooms with cache metadata
    """
    doc = revit.doc

    if not doc:
        return error_response(ErrorCode.INVALID_OPERATION, "No document open")

    from cache_manager import RevitCache

    cache = RevitCache(doc)
    cache_age = cache.get_cache_age_seconds("rooms")
    max_age = 300  # 5 minutes

    # If cache is stale, sync first
    if cache_age > max_age:
        cache.sync_rooms()
        cache_age = 0

    rooms = cache.get_rooms_from_cache()

    return success_response(
        data={
            "rooms": rooms,
            "count": len(rooms),
            "from_cache": True,
            "cache_age_seconds": round(cache_age, 1)
        }
    )


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

    Returns:
        List of synced categories with counts
    """
    doc = revit.doc

    if not doc:
        return error_response(ErrorCode.INVALID_OPERATION, "No document open")

    from cache_manager import RevitCache

    cache = RevitCache(doc)

    try:
        body = json.loads(request.body) if request.body else {}
    except (json.JSONDecodeError, ValueError):
        body = {}

    categories = body.get("categories", ["rooms", "levels", "walls"])
    synced = {}

    for category in categories:
        if category == "rooms":
            synced["rooms"] = cache.sync_rooms()
        elif category == "levels":
            synced["levels"] = cache.sync_levels()
        elif category == "walls":
            synced["walls"] = cache.sync_walls()

    return success_response(
        data={"synced": synced},
        message="Cache sync completed"
    )


@routes.route('/mcp/cache/status', methods=['GET'])
@require_auth
@safe_handler
def cache_status(request):
    """
    Get cache status for the current document.

    Returns:
        Cache age for each category
    """
    doc = revit.doc

    if not doc:
        return error_response(ErrorCode.INVALID_OPERATION, "No document open")

    from cache_manager import RevitCache

    cache = RevitCache(doc)

    status = {
        "rooms": {
            "age_seconds": round(cache.get_cache_age_seconds("rooms"), 1),
            "is_stale": cache.is_cache_stale("rooms"),
        },
        "levels": {
            "age_seconds": round(cache.get_cache_age_seconds("levels"), 1),
            "is_stale": cache.is_cache_stale("levels"),
        },
        "walls": {
            "age_seconds": round(cache.get_cache_age_seconds("walls"), 1),
            "is_stale": cache.is_cache_stale("walls"),
        },
    }

    return success_response(data=status)


@routes.route('/mcp/cache/clear', methods=['POST'])
@require_auth
@safe_handler
def clear_cache(request):
    """
    Clear the cache for the current document.

    Returns:
        Success confirmation
    """
    doc = revit.doc

    if not doc:
        return error_response(ErrorCode.INVALID_OPERATION, "No document open")

    from cache_manager import RevitCache

    cache = RevitCache(doc)
    cache.clear_cache()

    return success_response(message="Cache cleared successfully")


# =============================================================================
# Element Operations (Generic)
# =============================================================================

@routes.route('/mcp/elements/get', methods=['POST'])
@require_auth
@safe_handler
def get_element(request):
    """
    Get element by ID.

    Request body:
        {
            "element_id": 12345
        }

    Returns:
        Element details including category, type, and basic parameters
    """
    doc = revit.doc

    if not doc:
        return error_response(ErrorCode.INVALID_OPERATION, "No document open")

    try:
        body = json.loads(request.body) if request.body else {}
    except (json.JSONDecodeError, ValueError):
        return error_response(ErrorCode.INVALID_PARAMS, "Invalid JSON body")

    element_id = body.get("element_id")
    if element_id is None:
        return error_response(ErrorCode.INVALID_PARAMS, "Missing 'element_id' parameter")

    try:
        elem_id = DB.ElementId(int(element_id))
    except (ValueError, TypeError):
        return error_response(ErrorCode.INVALID_PARAMS, "'element_id' must be an integer")

    element = doc.GetElement(elem_id)

    if not element:
        return error_response(
            ErrorCode.ELEMENT_NOT_FOUND,
            "Element not found with id {}".format(element_id)
        )

    # Get basic element info
    category_name = element.Category.Name if element.Category else "Unknown"
    type_elem = doc.GetElement(element.GetTypeId())
    type_name = type_elem.Name if type_elem else "Unknown"

    return success_response(data={
        "id": element.Id.IntegerValue,
        "category": category_name,
        "type_name": type_name,
        "name": getattr(element, 'Name', None),
    })


@routes.route('/mcp/elements/delete', methods=['POST'])
@require_auth
@safe_handler
def delete_element(request):
    """
    Delete element by ID.

    Request body:
        {
            "element_id": 12345
        }

    Returns:
        Success confirmation
    """
    doc = revit.doc

    if not doc:
        return error_response(ErrorCode.INVALID_OPERATION, "No document open")

    try:
        body = json.loads(request.body) if request.body else {}
    except (json.JSONDecodeError, ValueError):
        return error_response(ErrorCode.INVALID_PARAMS, "Invalid JSON body")

    element_id = body.get("element_id")
    if element_id is None:
        return error_response(ErrorCode.INVALID_PARAMS, "Missing 'element_id' parameter")

    try:
        elem_id = DB.ElementId(int(element_id))
    except (ValueError, TypeError):
        return error_response(ErrorCode.INVALID_PARAMS, "'element_id' must be an integer")

    element = doc.GetElement(elem_id)
    if not element:
        return error_response(
            ErrorCode.ELEMENT_NOT_FOUND,
            "Element not found with id {}".format(element_id)
        )

    try:
        with revit.Transaction("AEC Agent: Delete Element"):
            doc.Delete(elem_id)

        return success_response(
            message="Element {} deleted successfully".format(element_id)
        )
    except Exception as e:
        return error_response(
            ErrorCode.TRANSACTION_FAILED,
            "Failed to delete element",
            details=str(e)
        )


# Log that routes are registered
logger.info("AEC Agent route handlers loaded")
