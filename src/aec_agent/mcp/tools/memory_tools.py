"""
MCP tools for project memory (facts and decisions).

Exposes ProjectMemory (src/aec_agent/memory/) as MCP tools so the LLM can
store and recall project-specific facts (decisions, constraints,
preferences, notes) across a conversation instead of re-deriving them.
"""

from typing import Any
from uuid import UUID

import structlog

from aec_agent.mcp.server import get_database_pool, get_embedding_service, mcp
from aec_agent.mcp.tools.base import ErrorCode, error_result, safe_tool, success_result
from aec_agent.memory import FactType, get_project_memory

logger = structlog.get_logger(__name__)


async def _get_memory():
    """Build a ProjectMemory wired to the live server-side DB pool/embeddings."""
    return await get_project_memory(
        db_pool=get_database_pool(),
        embedding_service=get_embedding_service(),
    )


@mcp.tool()
@safe_tool
async def store_fact(
    project_id: str,
    fact_type: str,
    key: str,
    value: str,
) -> dict[str, Any]:
    """
    Store a project fact for later recall.

    Args:
        project_id: Project UUID string.
        fact_type: One of "decision", "constraint", "preference", "note",
            "standard", "assumption".
        key: Short unique key for this fact within the project/type
            (e.g. "duct_material", "ceiling_height").
        value: The fact's value as text.

    Returns:
        Success result with the stored fact's id, or error result.

    Example:
        >>> result = await store_fact(project_id, "decision", "duct_material", "galvanized steel")
        >>> if result["success"]:
        ...     print("stored as", result["data"]["fact_id"])
    """
    try:
        project_uuid = UUID(project_id)
    except ValueError:
        return error_result(ErrorCode.INVALID_PARAMS, f"Invalid project_id: {project_id}")

    try:
        parsed_fact_type = FactType(fact_type)
    except ValueError:
        return error_result(ErrorCode.INVALID_PARAMS, f"Invalid fact_type: {fact_type}")

    memory = await _get_memory()
    fact = await memory.store_fact(
        project_id=project_uuid,
        fact_type=parsed_fact_type,
        key=key,
        value=value,
    )
    return success_result(data={"fact_id": str(fact.id)})


@mcp.tool()
@safe_tool
async def recall_facts(
    project_id: str,
    query: str | None = None,
    fact_type: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """
    Recall stored project facts, optionally filtered or semantically searched.

    Args:
        project_id: Project UUID string.
        query: Optional free-text query for semantic/keyword search over facts.
        fact_type: Optional filter — one of "decision", "constraint",
            "preference", "note", "standard", "assumption". Ignored if query
            is given.
        limit: Maximum number of facts to return (default 10).

    Returns:
        Success result with a list of matching facts, or error result.

    Example:
        >>> result = await recall_facts(project_id, query="duct material")
        >>> if result["success"]:
        ...     for f in result["data"]["facts"]:
        ...         print(f["key"], f["value"])
    """
    try:
        project_uuid = UUID(project_id)
    except ValueError:
        return error_result(ErrorCode.INVALID_PARAMS, f"Invalid project_id: {project_id}")

    parsed_fact_type = None
    if fact_type:
        try:
            parsed_fact_type = FactType(fact_type)
        except ValueError:
            return error_result(ErrorCode.INVALID_PARAMS, f"Invalid fact_type: {fact_type}")

    memory = await _get_memory()

    if query:
        facts = await memory.search_facts(project_uuid, query, limit)
    else:
        facts = await memory.get_facts(project_uuid, fact_type=parsed_fact_type, limit=limit)

    return success_result(data={"facts": [f.to_dict() for f in facts]})
