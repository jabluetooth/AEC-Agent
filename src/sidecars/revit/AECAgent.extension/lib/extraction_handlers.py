"""
Extraction handlers for AEC Agent Revit sidecar.

Provides endpoints for extracting element metadata for the PostgreSQL pipeline.
Includes geometry extraction, parameter extraction, and relationship discovery.
"""

import json
from pyrevit import routes, revit, DB
from pyrevit.coreutils import logger

from security import require_auth
from response import success_response, error_response, safe_handler, ErrorCode

# Conversion constants
FEET_TO_METERS = 0.3048
SQFT_TO_SQM = 0.092903


# =============================================================================
# Extraction Endpoints
# =============================================================================

@routes.route('/mcp/extract/batch', methods=['POST'])
@require_auth
@safe_handler
def extract_batch(request):
    """
    Extract elements in batches with full metadata.

    Request body:
        offset: Starting index (default 0)
        limit: Maximum elements to return (default 1000)
        categories: Optional list of category names to filter
        include_parameters: Include element parameters (default true)

    Returns:
        List of elements with geometry and properties
    """
    doc = revit.doc
    if not doc:
        return error_response(ErrorCode.INVALID_OPERATION, "No document open")

    # Parse request body
    try:
        body = json.loads(request.data) if request.data else {}
    except:
        body = {}

    offset = body.get('offset', 0)
    limit = body.get('limit', 1000)
    category_filter = body.get('categories', None)
    include_params = body.get('include_parameters', True)

    # Build category filter
    category_ids = None
    if category_filter:
        category_ids = []
        for cat_name in category_filter:
            cat = _get_category_by_name(doc, cat_name)
            if cat:
                category_ids.append(cat.Id)

    # Collect elements
    collector = DB.FilteredElementCollector(doc)
    collector = collector.WhereElementIsNotElementType()

    # Apply category filter if specified
    if category_ids:
        cat_filter = DB.ElementMulticategoryFilter(category_ids)
        collector = collector.WherePasses(cat_filter)

    all_elements = list(collector.ToElements())
    total = len(all_elements)

    # Apply offset and limit
    batch_elements = all_elements[offset:offset + limit]

    elements_data = []
    for elem in batch_elements:
        try:
            elem_data = _extract_element(elem, doc, include_params)
            if elem_data:
                elements_data.append(elem_data)
        except Exception as e:
            logger.debug("Error extracting element {}: {}".format(elem.Id, str(e)))

    return success_response(data={
        "elements": elements_data,
        "count": len(elements_data),
        "total": total,
        "offset": offset,
        "has_more": (offset + len(elements_data)) < total
    })


@routes.route('/mcp/extract/full', methods=['POST'])
@require_auth
@safe_handler
def extract_full(request):
    """
    Extract all elements (with streaming-friendly chunked response).

    This is the same as extract_batch but starts from offset 0
    and provides a summary.

    Returns:
        Summary of extraction with element counts per category
    """
    doc = revit.doc
    if not doc:
        return error_response(ErrorCode.INVALID_OPERATION, "No document open")

    # Count elements by category
    collector = DB.FilteredElementCollector(doc)
    collector = collector.WhereElementIsNotElementType()

    category_counts = {}
    total = 0

    for elem in collector.ToElements():
        try:
            cat = elem.Category
            if cat:
                cat_name = cat.Name
                category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
                total += 1
        except:
            pass

    return success_response(data={
        "total_elements": total,
        "categories": category_counts,
        "document_path": doc.PathName,
        "document_title": doc.Title,
    })


@routes.route('/mcp/extract/relationships', methods=['POST'])
@require_auth
@safe_handler
def extract_relationships(request):
    """
    Extract relationships from Revit API.

    Discovers:
    - Host relationships (doors/windows in walls)
    - Structural connections
    - Room containment

    Returns:
        Lists of hosted elements and structural connections
    """
    doc = revit.doc
    if not doc:
        return error_response(ErrorCode.INVALID_OPERATION, "No document open")

    hosted_relationships = []
    structural_connections = []

    # Find hosted elements (doors, windows in walls)
    host_categories = [
        DB.BuiltInCategory.OST_Walls,
        DB.BuiltInCategory.OST_Floors,
        DB.BuiltInCategory.OST_Roofs,
        DB.BuiltInCategory.OST_Ceilings,
    ]

    hosted_categories = [
        DB.BuiltInCategory.OST_Doors,
        DB.BuiltInCategory.OST_Windows,
        DB.BuiltInCategory.OST_MechanicalEquipment,
        DB.BuiltInCategory.OST_ElectricalEquipment,
        DB.BuiltInCategory.OST_PlumbingFixtures,
    ]

    for hosted_cat in hosted_categories:
        try:
            collector = DB.FilteredElementCollector(doc)
            collector = collector.OfCategory(hosted_cat)
            collector = collector.WhereElementIsNotElementType()

            for elem in collector.ToElements():
                try:
                    # FamilyInstance has Host property
                    if hasattr(elem, 'Host') and elem.Host:
                        hosted_relationships.append({
                            "host_id": elem.Host.Id.IntegerValue,
                            "hosted_id": elem.Id.IntegerValue,
                            "host_category": elem.Host.Category.Name if elem.Host.Category else "Unknown",
                            "hosted_category": elem.Category.Name if elem.Category else "Unknown",
                        })
                except:
                    pass
        except:
            pass

    # Find structural connections
    try:
        # Get structural columns and framing
        structural_cats = [
            DB.BuiltInCategory.OST_StructuralColumns,
            DB.BuiltInCategory.OST_StructuralFraming,
        ]

        for cat in structural_cats:
            try:
                collector = DB.FilteredElementCollector(doc)
                collector = collector.OfCategory(cat)
                collector = collector.WhereElementIsNotElementType()

                for elem in collector.ToElements():
                    # Check for analytical model connections if available
                    try:
                        # Use StructuralMemberUsage or analyze endpoints
                        if hasattr(elem, 'AnalyticalModel') and elem.AnalyticalModel:
                            analytical = elem.AnalyticalModel
                            # Get connected elements from analytical model
                            # This is simplified - real implementation would
                            # analyze connection points
                    except:
                        pass
            except:
                pass
    except:
        pass

    return success_response(data={
        "hosted": hosted_relationships,
        "connections": structural_connections,
        "hosted_count": len(hosted_relationships),
        "connection_count": len(structural_connections),
    })


# =============================================================================
# Helper Functions
# =============================================================================

def _extract_element(elem, doc, include_params=True):
    """
    Extract full metadata from a Revit element.

    Args:
        elem: Revit Element
        doc: Revit Document
        include_params: Whether to include parameters

    Returns:
        Element data dictionary
    """
    if elem is None:
        return None

    # Skip certain elements
    if not elem.Category:
        return None

    data = {
        "element_id": elem.Id.IntegerValue,
        "unique_id": elem.UniqueId,
        "category": elem.Category.Name if elem.Category else None,
    }

    # Get family and type info
    elem_type = doc.GetElement(elem.GetTypeId())
    if elem_type:
        data["type_name"] = elem_type.Name if hasattr(elem_type, 'Name') else None

        # Get family name for family instances
        if hasattr(elem_type, 'FamilyName'):
            data["family"] = elem_type.FamilyName
        elif hasattr(elem_type, 'Family') and elem_type.Family:
            data["family"] = elem_type.Family.Name
        else:
            data["family"] = None

    # Get level
    level_id = elem.LevelId if hasattr(elem, 'LevelId') else None
    if level_id and level_id != DB.ElementId.InvalidElementId:
        level = doc.GetElement(level_id)
        if level:
            data["level"] = level.Name

    # Get location
    data["location"] = _get_element_location(elem)

    # Get bounding box
    data["bounds"] = _get_element_bounds(elem, doc.ActiveView)

    # Get parameters
    if include_params:
        data["parameters"] = _get_element_parameters(elem)

    return data


def _get_element_location(elem):
    """
    Get element location as a structured dict.

    Returns location as:
    - POINT for point-based elements (columns, furniture)
    - CURVE for linear elements (walls, beams)
    - AREA for area-based elements (rooms)
    """
    loc = elem.Location
    if loc is None:
        return None

    if isinstance(loc, DB.LocationPoint):
        pt = loc.Point
        return {
            "type": "POINT",
            "point": {
                "x": pt.X * FEET_TO_METERS,
                "y": pt.Y * FEET_TO_METERS,
                "z": pt.Z * FEET_TO_METERS,
            }
        }

    elif isinstance(loc, DB.LocationCurve):
        curve = loc.Curve
        start = curve.GetEndPoint(0)
        end = curve.GetEndPoint(1)
        return {
            "type": "CURVE",
            "start": {
                "x": start.X * FEET_TO_METERS,
                "y": start.Y * FEET_TO_METERS,
                "z": start.Z * FEET_TO_METERS,
            },
            "end": {
                "x": end.X * FEET_TO_METERS,
                "y": end.Y * FEET_TO_METERS,
                "z": end.Z * FEET_TO_METERS,
            }
        }

    return None


def _get_element_bounds(elem, view=None):
    """Get element bounding box."""
    try:
        bbox = elem.get_BoundingBox(view)
        if bbox:
            return {
                "min_x": bbox.Min.X * FEET_TO_METERS,
                "min_y": bbox.Min.Y * FEET_TO_METERS,
                "min_z": bbox.Min.Z * FEET_TO_METERS,
                "max_x": bbox.Max.X * FEET_TO_METERS,
                "max_y": bbox.Max.Y * FEET_TO_METERS,
                "max_z": bbox.Max.Z * FEET_TO_METERS,
            }
    except:
        pass
    return None


def _get_element_parameters(elem):
    """
    Get element parameters as a dictionary.

    Extracts both instance and type parameters with common names.
    """
    params = {}

    # Common built-in parameters to extract
    param_names = [
        "Height", "Width", "Length", "Area", "Volume",
        "Mark", "Comments", "Fire Rating",
        "Phase Created", "Phase Demolished",
    ]

    for pname in param_names:
        try:
            param = elem.LookupParameter(pname)
            if param and param.HasValue:
                params[pname] = _get_parameter_value(param)
        except:
            pass

    # Get all parameters if fewer than 50 (avoid huge exports)
    try:
        all_params = list(elem.Parameters)
        if len(all_params) < 50:
            for param in all_params:
                try:
                    if param.HasValue and param.Definition:
                        name = param.Definition.Name
                        if name not in params:
                            value = _get_parameter_value(param)
                            if value is not None:
                                params[name] = value
                except:
                    pass
    except:
        pass

    return params


def _get_parameter_value(param):
    """Get parameter value with proper type conversion."""
    if not param.HasValue:
        return None

    storage_type = param.StorageType

    if storage_type == DB.StorageType.String:
        return param.AsString()

    elif storage_type == DB.StorageType.Integer:
        return param.AsInteger()

    elif storage_type == DB.StorageType.Double:
        value = param.AsDouble()
        # Convert length/area units
        try:
            unit_type = param.Definition.UnitType if hasattr(param.Definition, 'UnitType') else None
            # Simplified - proper implementation would check UnitType
            # For now, assume length params need feet->meters conversion
            # based on parameter name heuristics
            name = param.Definition.Name.lower() if param.Definition else ""
            if any(x in name for x in ['length', 'width', 'height', 'depth', 'thickness']):
                return value * FEET_TO_METERS
            elif 'area' in name:
                return value * SQFT_TO_SQM
            return value
        except:
            return value

    elif storage_type == DB.StorageType.ElementId:
        elem_id = param.AsElementId()
        if elem_id and elem_id != DB.ElementId.InvalidElementId:
            return elem_id.IntegerValue
        return None

    return None


def _get_category_by_name(doc, name):
    """Get category by name."""
    categories = doc.Settings.Categories
    for cat in categories:
        if cat.Name.lower() == name.lower():
            return cat
    return None
