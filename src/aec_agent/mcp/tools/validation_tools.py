"""
MCP tools for MEP domain rule validation.

Exposes the RulesEngine (src/aec_agent/domain/) as MCP tools so the LLM can
validate elements against MEP design rules (clearance, sizing, routing,
access, code) and retrieve pending design suggestions.
"""

from typing import Any
from uuid import UUID

import structlog

from aec_agent.domain import RuleType, get_knowledge_base, get_rules_engine
from aec_agent.mcp.server import get_database_pool, get_embedding_service, mcp
from aec_agent.mcp.tools.base import ErrorCode, error_result, safe_tool, success_result

logger = structlog.get_logger(__name__)


async def _get_engine():
    """Build a RulesEngine wired to the live server-side DB pool/embeddings."""
    db_pool = get_database_pool()
    knowledge_base = await get_knowledge_base(db_pool=db_pool)
    return await get_rules_engine(
        knowledge_base=knowledge_base,
        db_pool=db_pool,
        embedding_service=get_embedding_service(),
    )


@mcp.tool()
@safe_tool
async def validate_elements(
    project_id: str,
    elements: list[dict[str, Any]],
    rule_types: list[str] | None = None,
) -> dict[str, Any]:
    """
    Validate a list of elements against MEP domain rules.

    Args:
        project_id: Project UUID string.
        elements: Element dicts to validate, each with at least an "id" and
            a "properties" dict (e.g. from a prior find_elements call).
        rule_types: Optional filter — one or more of "clearance", "routing",
            "sizing", "priority", "access", "code", "validation". Validates
            against all rule types if omitted.

    Returns:
        Success result with counts and error/warning details, or error result.

    Example:
        >>> result = await validate_elements(project_id, elements, ["clearance"])
        >>> if result["success"]:
        ...     print(result["data"]["errors"], "errors found")
    """
    try:
        project_uuid = UUID(project_id)
    except ValueError:
        return error_result(ErrorCode.INVALID_PARAMS, f"Invalid project_id: {project_id}")

    parsed_rule_types = None
    if rule_types:
        try:
            parsed_rule_types = [RuleType(rt) for rt in rule_types]
        except ValueError as e:
            return error_result(ErrorCode.INVALID_PARAMS, f"Invalid rule_type: {e}")

    engine = await _get_engine()
    summary = await engine.validate_project(
        project_id=project_uuid,
        elements=elements,
        rule_types=parsed_rule_types,
    )
    return success_result(data=summary)


@mcp.tool()
@safe_tool
async def get_suggestions(project_id: str, limit: int = 10) -> dict[str, Any]:
    """
    Get pending design suggestions for a project.

    Args:
        project_id: Project UUID string.
        limit: Maximum number of suggestions to return (default 10).

    Returns:
        Success result with a list of suggestions, or error result.

    Example:
        >>> result = await get_suggestions(project_id, limit=5)
        >>> if result["success"]:
        ...     for s in result["data"]["suggestions"]:
        ...         print(s["suggestion"], s["priority"])
    """
    try:
        project_uuid = UUID(project_id)
    except ValueError:
        return error_result(ErrorCode.INVALID_PARAMS, f"Invalid project_id: {project_id}")

    engine = await _get_engine()
    suggestions = await engine.get_pending_suggestions(project_id=project_uuid, limit=limit)
    return success_result(data={"suggestions": [s.to_dict() for s in suggestions]})
