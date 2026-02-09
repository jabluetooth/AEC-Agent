"""
MEP-specific MCP tools for HVAC, electrical, plumbing, and low voltage workflows.

These tools provide specialized operations for MEP (Mechanical, Electrical, Plumbing)
engineering workflows, building on the base metadata and drawing tools.
"""

from uuid import UUID

import structlog

from aec_agent.mcp.server import mcp
from aec_agent.mcp.tools.base import (
    ErrorCode,
    error_result,
    safe_tool,
    success_result,
)

logger = structlog.get_logger(__name__)


# MEP-specific error codes
class MEPErrorCode:
    ELEMENT_NOT_FOUND = 7001
    CLEARANCE_VIOLATION = 7002
    SYSTEM_TRACE_FAILED = 7003
    VALIDATION_FAILED = 7004
    CLASH_DETECTED = 7005


# =============================================================================
# Clearance and Spacing Tools
# =============================================================================

@mcp.tool()
@safe_tool
async def check_clearances(
    element_id: str,
    clearance_type: str = "maintenance",
    min_distance: float = 0.6,
) -> dict:
    """
    Check clearance requirements around an MEP element.

    Verifies that the specified element has adequate clearance for
    maintenance access, code compliance, or equipment operation.

    Args:
        element_id: Element to check clearances for (ID or source_id)
        clearance_type: Type of clearance check:
            - "maintenance": Service/maintenance access (default 0.6m)
            - "code": Code-required clearances
            - "operation": Equipment operation clearances
        min_distance: Minimum required clearance in meters (default 0.6)

    Returns:
        Clearance analysis with any violations found
    """
    from aec_agent.mcp.tools.metadata import _get_active_project_id, _get_services

    pool, _ = await _get_services()
    if not pool:
        return error_result(
            ErrorCode.DATABASE_NOT_CONFIGURED,
            "Database not configured for clearance checking"
        )

    project_id = await _get_active_project_id()
    if not project_id:
        return error_result(
            MEPErrorCode.ELEMENT_NOT_FOUND,
            "No active project"
        )

    from aec_agent.db.repository import ElementRepository

    repo = ElementRepository(pool)

    # Resolve element
    element = await repo.get_element_by_source_id(project_id, element_id)
    if not element:
        try:
            element = await repo.get_element(UUID(element_id))
        except ValueError:
            pass

    if not element:
        return error_result(
            MEPErrorCode.ELEMENT_NOT_FOUND,
            f"Element not found: {element_id}"
        )

    # Get nearby elements within clearance distance
    nearby = await repo.get_nearby_elements(element.id, min_distance * 1.5, limit=50)

    violations = []
    for near_element in nearby:
        # Calculate actual distance (simplified - uses centroid)
        if element.centroid and near_element.centroid:
            dx = element.centroid.x - near_element.centroid.x
            dy = element.centroid.y - near_element.centroid.y
            dz = (element.centroid.z or 0) - (near_element.centroid.z or 0)
            distance = (dx**2 + dy**2 + dz**2) ** 0.5

            if distance < min_distance:
                violations.append({
                    "element_id": near_element.source_id,
                    "element_type": near_element.entity_type,
                    "distance": round(distance, 3),
                    "required": min_distance,
                    "shortage": round(min_distance - distance, 3),
                })

    result = {
        "element_id": element.source_id,
        "element_type": element.entity_type,
        "clearance_type": clearance_type,
        "min_distance": min_distance,
        "elements_checked": len(nearby),
        "violations": violations,
        "passed": len(violations) == 0,
    }

    if violations:
        return success_result(
            data=result,
            message=f"Found {len(violations)} clearance violations"
        )
    else:
        return success_result(
            data=result,
            message=f"Clearance check passed ({len(nearby)} elements checked)"
        )


@mcp.tool()
@safe_tool
async def validate_mep_spacing(
    domain: str,
    level: str | None = None,
    check_type: str = "all",
) -> dict:
    """
    Validate MEP element spacing against standards.

    Checks spacing requirements for a specific MEP domain across
    a level or the entire project.

    Args:
        domain: MEP domain (hvac, electrical, plumbing, fire_protection, low_voltage)
        level: Specific level to check or None for all levels
        check_type: Type of checks to perform:
            - "all": All spacing checks
            - "horizontal": Horizontal spacing only
            - "vertical": Vertical spacing only
            - "clearance": Clearance to structure only

    Returns:
        Validation results with any violations
    """
    from aec_agent.frontend.tool_optimization import get_mep_category_filter
    from aec_agent.mcp.tools.metadata import _get_active_project_id, _get_services

    pool, embeddings = await _get_services()
    if not pool:
        return error_result(
            ErrorCode.DATABASE_NOT_CONFIGURED,
            "Database not configured"
        )

    project_id = await _get_active_project_id()
    if not project_id:
        return error_result(
            MEPErrorCode.ELEMENT_NOT_FOUND,
            "No active project"
        )

    # Get category filters for the domain
    category_filter = get_mep_category_filter(domain)
    if not category_filter:
        return error_result(
            ErrorCode.INVALID_PARAMS,
            f"Unknown MEP domain: {domain}"
        )

    from aec_agent.db.repository import ElementRepository

    repo = ElementRepository(pool)

    # Get elements for this domain
    elements = await repo.search_elements(
        project_id=project_id,
        category=category_filter.get("revit_categories", [None])[0] if category_filter.get("revit_categories") else None,
        limit=500,
    )

    # Filter by level if specified
    if level:
        elements = [e for e in elements if e.level and level.lower() in e.level.lower()]

    # Check spacing between elements
    violations = []
    min_spacing = 0.15  # 150mm minimum between MEP elements

    for i, elem1 in enumerate(elements):
        for elem2 in elements[i+1:]:
            if elem1.centroid and elem2.centroid:
                dx = elem1.centroid.x - elem2.centroid.x
                dy = elem1.centroid.y - elem2.centroid.y
                distance = (dx**2 + dy**2) ** 0.5

                if distance < min_spacing:
                    violations.append({
                        "element1": elem1.source_id,
                        "element2": elem2.source_id,
                        "distance": round(distance, 3),
                        "required": min_spacing,
                    })

            # Limit violation count
            if len(violations) >= 50:
                break
        if len(violations) >= 50:
            break

    result = {
        "domain": domain,
        "level": level,
        "check_type": check_type,
        "elements_checked": len(elements),
        "violations_found": len(violations),
        "violations": violations[:20],  # Limit returned violations
        "passed": len(violations) == 0,
    }

    if violations:
        return success_result(
            data=result,
            message=f"Found {len(violations)} spacing violations in {domain}"
        )
    else:
        return success_result(
            data=result,
            message=f"Spacing validation passed for {len(elements)} {domain} elements"
        )


# =============================================================================
# System Tracing Tools
# =============================================================================

@mcp.tool()
@safe_tool
async def trace_system(
    start_element_id: str,
    system_type: str | None = None,
    max_depth: int = 50,
) -> dict:
    """
    Trace an MEP system from a starting element.

    Follows connections from a terminal device (diffuser, outlet, fixture)
    back toward the source equipment, or vice versa.

    Args:
        start_element_id: Starting element (e.g., diffuser, outlet)
        system_type: Type of system to trace (duct, pipe, conduit, cable_tray)
                    If None, auto-detected from element type.
        max_depth: Maximum elements to trace (default 50)

    Returns:
        Connected elements in the system path
    """
    from aec_agent.mcp.tools.metadata import _get_active_project_id, _get_services

    pool, _ = await _get_services()
    if not pool:
        return error_result(
            ErrorCode.DATABASE_NOT_CONFIGURED,
            "Database not configured"
        )

    project_id = await _get_active_project_id()
    if not project_id:
        return error_result(
            MEPErrorCode.ELEMENT_NOT_FOUND,
            "No active project"
        )

    from aec_agent.db.repository import ElementRepository

    repo = ElementRepository(pool)

    # Resolve starting element
    element = await repo.get_element_by_source_id(project_id, start_element_id)
    if not element:
        try:
            element = await repo.get_element(UUID(start_element_id))
        except ValueError:
            pass

    if not element:
        return error_result(
            MEPErrorCode.ELEMENT_NOT_FOUND,
            f"Start element not found: {start_element_id}"
        )

    # Trace connected elements using relationships
    traced_elements = [element]
    visited_ids = {element.id}
    current_elements = [element]

    for depth in range(max_depth):
        if not current_elements:
            break

        next_elements = []
        for curr in current_elements:
            # Get related elements (connected)
            related = await repo.get_related_elements(
                curr.id,
                relation_types=["connected_to", "hosts", "intersects"],
                limit=20,
            )

            for rel_elem in related:
                if rel_elem.id not in visited_ids:
                    visited_ids.add(rel_elem.id)
                    traced_elements.append(rel_elem)
                    next_elements.append(rel_elem)

        current_elements = next_elements

        if len(traced_elements) >= max_depth:
            break

    # Build result
    path = []
    for elem in traced_elements:
        path.append({
            "id": elem.source_id,
            "type": elem.entity_type,
            "category": elem.category,
            "description": elem.description,
            "level": elem.level,
        })

    return success_result(
        data={
            "start_element": start_element_id,
            "system_type": system_type or "auto",
            "path_length": len(path),
            "path": path,
        },
        message=f"Traced {len(path)} elements in system"
    )


# =============================================================================
# Clash Detection Tools
# =============================================================================

@mcp.tool()
@safe_tool
async def find_clashes(
    system1: str | None = None,
    system2: str | None = None,
    tolerance: float = 0.01,
    level: str | None = None,
) -> dict:
    """
    Find clashes between MEP systems.

    Detects geometric intersections between different MEP systems
    that may indicate coordination issues.

    Args:
        system1: First system type (hvac, electrical, plumbing) or None for all
        system2: Second system type or None for all
        tolerance: Clash tolerance in meters (default 0.01 = 10mm)
        level: Specific level to check or None for all

    Returns:
        List of clashing elements with locations
    """
    from aec_agent.frontend.tool_optimization import get_mep_category_filter
    from aec_agent.mcp.tools.metadata import _get_active_project_id, _get_services

    pool, _ = await _get_services()
    if not pool:
        return error_result(
            ErrorCode.DATABASE_NOT_CONFIGURED,
            "Database not configured"
        )

    project_id = await _get_active_project_id()
    if not project_id:
        return error_result(
            MEPErrorCode.ELEMENT_NOT_FOUND,
            "No active project"
        )

    from aec_agent.db.repository import ElementRepository

    repo = ElementRepository(pool)

    # Get elements for each system
    async def get_system_elements(system_name: str | None) -> list:
        if system_name:
            category_filter = get_mep_category_filter(system_name)
            if category_filter and category_filter.get("revit_categories"):
                return await repo.search_elements(
                    project_id=project_id,
                    category=category_filter["revit_categories"][0],
                    limit=200,
                )
        return await repo.search_elements(project_id=project_id, limit=200)

    elements1 = await get_system_elements(system1)
    elements2 = await get_system_elements(system2) if system2 != system1 else elements1

    # Filter by level if specified
    if level:
        elements1 = [e for e in elements1 if e.level and level.lower() in e.level.lower()]
        elements2 = [e for e in elements2 if e.level and level.lower() in e.level.lower()]

    # Find clashes (simplified - checks centroid proximity)
    clashes = []
    for elem1 in elements1:
        for elem2 in elements2:
            if elem1.id == elem2.id:
                continue

            if elem1.centroid and elem2.centroid:
                dx = elem1.centroid.x - elem2.centroid.x
                dy = elem1.centroid.y - elem2.centroid.y
                dz = (elem1.centroid.z or 0) - (elem2.centroid.z or 0)
                distance = (dx**2 + dy**2 + dz**2) ** 0.5

                if distance < tolerance:
                    clashes.append({
                        "element1": {
                            "id": elem1.source_id,
                            "type": elem1.entity_type,
                            "system": system1 or "unknown",
                        },
                        "element2": {
                            "id": elem2.source_id,
                            "type": elem2.entity_type,
                            "system": system2 or "unknown",
                        },
                        "distance": round(distance, 4),
                        "location": {
                            "x": round((elem1.centroid.x + elem2.centroid.x) / 2, 2),
                            "y": round((elem1.centroid.y + elem2.centroid.y) / 2, 2),
                            "z": round(((elem1.centroid.z or 0) + (elem2.centroid.z or 0)) / 2, 2),
                        },
                    })

            if len(clashes) >= 100:
                break
        if len(clashes) >= 100:
            break

    return success_result(
        data={
            "system1": system1 or "all",
            "system2": system2 or "all",
            "tolerance": tolerance,
            "level": level,
            "elements_checked": len(elements1) + len(elements2),
            "clashes_found": len(clashes),
            "clashes": clashes[:20],  # Limit returned clashes
        },
        message=f"Found {len(clashes)} potential clashes"
    )


# =============================================================================
# MEP Summary Tools
# =============================================================================

@mcp.tool()
@safe_tool
async def get_mep_summary(
    domain: str | None = None,
    level: str | None = None,
) -> dict:
    """
    Get a summary of MEP elements in the project.

    Provides counts and statistics for MEP elements, useful for
    quick project overview and validation.

    Args:
        domain: Specific domain (hvac, electrical, plumbing, fire_protection, low_voltage)
                or None for all MEP systems
        level: Specific level or None for all levels

    Returns:
        Summary statistics including element counts by type
    """
    from aec_agent.frontend.tool_optimization import MEP_CATEGORY_FILTERS
    from aec_agent.mcp.tools.metadata import _get_active_project_id, _get_services

    pool, _ = await _get_services()
    if not pool:
        return error_result(
            ErrorCode.DATABASE_NOT_CONFIGURED,
            "Database not configured"
        )

    project_id = await _get_active_project_id()
    if not project_id:
        return error_result(
            MEPErrorCode.ELEMENT_NOT_FOUND,
            "No active project"
        )

    from aec_agent.db.repository import ElementRepository

    repo = ElementRepository(pool)

    # Get all elements
    all_elements = await repo.search_elements(project_id=project_id, limit=5000)

    # Filter by level if specified
    if level:
        all_elements = [e for e in all_elements if e.level and level.lower() in e.level.lower()]

    # Categorize by domain
    domain_counts = dict.fromkeys(MEP_CATEGORY_FILTERS.keys(), 0)
    type_counts = {}

    for elem in all_elements:
        elem_type = elem.entity_type or elem.category or "unknown"
        type_counts[elem_type] = type_counts.get(elem_type, 0) + 1

        # Match to domain
        for domain_name, filters in MEP_CATEGORY_FILTERS.items():
            if domain and domain_name != domain:
                continue

            categories = filters.get("revit_categories", [])
            layers = filters.get("autocad_layers", [])
            entity_types = filters.get("entity_types", [])

            matched = False
            if elem.category and any(c.lower() in elem.category.lower() for c in categories):
                matched = True
            elif elem.layer and any(l.lower() in elem.layer.lower() for l in layers):
                matched = True
            elif elem_type.lower() in [t.lower() for t in entity_types]:
                matched = True

            if matched:
                domain_counts[domain_name] += 1
                break

    # Build summary
    summary = {
        "project_id": str(project_id),
        "level": level,
        "domain_filter": domain,
        "total_elements": len(all_elements),
        "by_domain": {k: v for k, v in domain_counts.items() if v > 0 or k == domain},
        "by_type": dict(sorted(type_counts.items(), key=lambda x: -x[1])[:20]),
    }

    return success_result(
        data=summary,
        message=f"Summary: {len(all_elements)} MEP elements"
    )
