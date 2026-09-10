"""
MCP tools for MEP workflow templates.

Exposes the WorkflowExecutor (src/aec_agent/workflows/) as MCP tools so the
LLM can list and run pre-defined multi-step MEP workflows without a
per-step round trip, reducing token usage for common tasks.
"""

from typing import Any
from uuid import UUID

import structlog

from aec_agent.mcp.server import get_database_pool, mcp
from aec_agent.mcp.tools.base import ErrorCode, error_result, safe_tool, success_result
from aec_agent.memory import get_current_user
from aec_agent.workflows import get_workflow_executor

logger = structlog.get_logger(__name__)


@mcp.tool()
@safe_tool
async def list_workflows(domain: str = "mep") -> dict[str, Any]:
    """
    List available workflow templates for a domain.

    Args:
        domain: Workflow domain to filter by (default "mep").

    Returns:
        Success result with a list of workflow summaries, or error result.

    Example:
        >>> result = await list_workflows()
        >>> if result["success"]:
        ...     for wf in result["data"]["workflows"]:
        ...         print(wf["name"], wf["description"])
    """
    executor = await get_workflow_executor(db_pool=get_database_pool())
    templates = executor.list_templates(domain=domain)

    return success_result(
        data={
            "workflows": [
                {
                    "name": t.name,
                    "description": t.description,
                    "required_inputs": t.required_context,
                    "estimated_tokens_saved": t.estimated_tokens,
                }
                for t in templates
            ]
        }
    )


@mcp.tool()
@safe_tool
async def start_workflow(
    workflow_name: str,
    context: dict[str, Any],
    project_id: str | None = None,
) -> dict[str, Any]:
    """
    Start a workflow template and run it to completion.

    Args:
        workflow_name: Name of the workflow template (see list_workflows).
        context: Initial context providing the workflow's required inputs.
        project_id: Optional project UUID string for tracking.

    Returns:
        Success result with the workflow's final status/data, or error result.

    Example:
        >>> result = await start_workflow(
        ...     "route_supply_duct", {"from_element": "AHU-1", "to_element": "VAV-3"}
        ... )
        >>> if result["success"]:
        ...     print(result["data"]["summary"])
    """
    try:
        project_uuid = UUID(project_id) if project_id else None
    except ValueError:
        return error_result(ErrorCode.INVALID_PARAMS, f"Invalid project_id: {project_id}")

    executor = await get_workflow_executor(db_pool=get_database_pool())

    try:
        execution = await executor.start_workflow(
            template_name=workflow_name,
            context=context,
            project_id=project_uuid,
            user_id=get_current_user(),
        )
    except ValueError as e:
        return error_result(ErrorCode.INVALID_PARAMS, str(e))

    result = await executor.run_to_completion(execution.id)
    return success_result(data=result.to_dict())
