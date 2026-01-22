"""
Workflow Executor for running multi-step MEP workflows.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from aec_agent.workflows.models import (
    WorkflowTemplate,
    WorkflowExecution,
    WorkflowStep,
    WorkflowStatus,
    StepStatus,
    StepResult,
    WorkflowResult,
)

logger = logging.getLogger(__name__)


class WorkflowExecutor:
    """
    Executes workflow templates with minimal LLM involvement.

    Workflows run a series of MCP tool calls based on templates,
    passing results between steps. This reduces token usage by
    avoiding per-step LLM interpretation.
    """

    def __init__(self, mcp_client=None, db_pool=None):
        """
        Initialize the workflow executor.

        Args:
            mcp_client: MCP client for tool calls
            db_pool: Optional asyncpg pool for persistence
        """
        self._mcp_client = mcp_client
        self._db_pool = db_pool
        self._templates: dict[str, WorkflowTemplate] = {}
        self._executions: dict[str, WorkflowExecution] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Load workflow templates from database or defaults."""
        if self._initialized:
            return

        if self._db_pool:
            await self._load_templates_from_db()
        else:
            self._load_default_templates()

        self._initialized = True
        logger.info(f"WorkflowExecutor initialized with {len(self._templates)} templates")

    def _load_default_templates(self) -> None:
        """Load default HVAC workflow templates."""
        from aec_agent.workflows.templates import get_default_hvac_workflows
        for template in get_default_hvac_workflows():
            self._templates[template.name] = template

    async def _load_templates_from_db(self) -> None:
        """Load templates from PostgreSQL."""
        try:
            async with self._db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT id, name, domain, subdomain, description,
                           steps, required_context, default_params, estimated_tokens
                    FROM workflow_templates
                    ORDER BY name
                """)
                for row in rows:
                    template = WorkflowTemplate(
                        id=row["id"],
                        name=row["name"],
                        domain=row["domain"],
                        subdomain=row["subdomain"],
                        description=row["description"] or "",
                        steps=[WorkflowStep.from_dict(s) for s in (row["steps"] or [])],
                        required_context=row["required_context"] or [],
                        default_params=row["default_params"] or {},
                        estimated_tokens=row["estimated_tokens"],
                    )
                    self._templates[template.name] = template
        except Exception as e:
            logger.warning(f"Failed to load templates from DB, using defaults: {e}")
            self._load_default_templates()

    # =========================================================================
    # Template Management
    # =========================================================================

    def list_templates(self, domain: Optional[str] = None) -> list[WorkflowTemplate]:
        """List available workflow templates."""
        templates = list(self._templates.values())
        if domain:
            templates = [t for t in templates if t.domain == domain or t.subdomain == domain]
        return templates

    def get_template(self, name: str) -> Optional[WorkflowTemplate]:
        """Get a workflow template by name."""
        return self._templates.get(name)

    async def register_template(self, template: WorkflowTemplate) -> None:
        """Register a new workflow template."""
        self._templates[template.name] = template

        if self._db_pool:
            try:
                async with self._db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO workflow_templates
                            (id, name, domain, subdomain, description,
                             steps, required_context, default_params, estimated_tokens)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        ON CONFLICT (name) DO UPDATE SET
                            description = EXCLUDED.description,
                            steps = EXCLUDED.steps,
                            required_context = EXCLUDED.required_context,
                            default_params = EXCLUDED.default_params,
                            updated_at = NOW()
                    """,
                        template.id, template.name, template.domain, template.subdomain,
                        template.description, [s.to_dict() for s in template.steps],
                        template.required_context, template.default_params,
                        template.estimated_tokens
                    )
            except Exception as e:
                logger.error(f"Failed to save template to DB: {e}")

    # =========================================================================
    # Workflow Execution
    # =========================================================================

    async def start_workflow(
        self,
        template_name: str,
        context: dict[str, Any],
        project_id: Optional[UUID] = None,
        user_id: Optional[str] = None,
    ) -> WorkflowExecution:
        """
        Start a new workflow execution.

        Args:
            template_name: Name of the workflow template
            context: Initial context with required inputs
            project_id: Optional project ID for tracking
            user_id: Optional user ID (Windows username)

        Returns:
            WorkflowExecution object
        """
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Workflow template not found: {template_name}")

        # Validate required context
        missing = template.validate_context(context)
        if missing:
            raise ValueError(f"Missing required context: {', '.join(missing)}")

        # Merge default params with provided context
        full_context = {**template.default_params, **context}

        execution = WorkflowExecution(
            template_id=template.id,
            template_name=template_name,
            project_id=project_id,
            user_id=user_id,
            status=WorkflowStatus.PENDING,
            context=full_context,
        )

        self._executions[str(execution.id)] = execution
        await self._save_execution(execution)

        logger.info(f"Started workflow: {template_name}", extra={
            "execution_id": str(execution.id),
            "project_id": str(project_id) if project_id else None,
        })

        return execution

    async def execute_step(self, execution_id: UUID) -> StepResult:
        """
        Execute the next step in a workflow.

        Args:
            execution_id: ID of the workflow execution

        Returns:
            StepResult for the executed step
        """
        execution = self._executions.get(str(execution_id))
        if not execution:
            raise ValueError(f"Execution not found: {execution_id}")

        template = self.get_template(execution.template_name)
        if not template:
            raise ValueError(f"Template not found: {execution.template_name}")

        if execution.current_step >= len(template.steps):
            raise ValueError("No more steps to execute")

        step = template.steps[execution.current_step]
        execution.status = WorkflowStatus.IN_PROGRESS

        # Check condition
        if not step.should_execute(execution.context):
            result = StepResult(
                step_name=step.name,
                status=StepStatus.SKIPPED,
                data={"reason": "Condition not met"},
            )
            execution.add_step_result(result)
            return result

        # Execute step
        start_time = time.time()
        result = await self._execute_tool_step(step, execution.context)
        result.duration_ms = (time.time() - start_time) * 1000

        execution.add_step_result(result)

        # Check if workflow is complete or failed
        if not result.success and step.on_error == "abort":
            execution.status = WorkflowStatus.FAILED
            execution.error_message = result.error
            execution.completed_at = datetime.utcnow()
        elif execution.current_step >= len(template.steps):
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.utcnow()

        await self._save_execution(execution)
        return result

    async def run_to_completion(
        self,
        execution_id: UUID,
        max_steps: int = 20,
    ) -> WorkflowResult:
        """
        Run a workflow to completion.

        Args:
            execution_id: ID of the workflow execution
            max_steps: Maximum steps to execute (safety limit)

        Returns:
            WorkflowResult with final status and data
        """
        execution = self._executions.get(str(execution_id))
        if not execution:
            raise ValueError(f"Execution not found: {execution_id}")

        template = self.get_template(execution.template_name)
        if not template:
            raise ValueError(f"Template not found: {execution.template_name}")

        start_time = time.time()
        steps_executed = 0

        while not execution.is_complete and steps_executed < max_steps:
            if execution.current_step >= len(template.steps):
                execution.status = WorkflowStatus.COMPLETED
                execution.completed_at = datetime.utcnow()
                break

            await self.execute_step(execution_id)
            steps_executed += 1

        # Build final result
        final_data = {}
        for result in execution.results:
            if result.success:
                final_data[result.step_name] = result.data

        workflow_result = WorkflowResult(
            execution_id=execution.id,
            template_name=execution.template_name,
            status=execution.status,
            steps_completed=len([r for r in execution.results if r.success]),
            steps_total=len(template.steps),
            final_data=final_data,
            summary=self._generate_summary(execution, template),
            error=execution.error_message,
            duration_ms=(time.time() - start_time) * 1000,
        )

        await self._save_execution(execution)
        return workflow_result

    async def _execute_tool_step(
        self,
        step: WorkflowStep,
        context: dict[str, Any],
    ) -> StepResult:
        """Execute a single tool step."""
        if not self._mcp_client:
            return StepResult(
                step_name=step.name,
                status=StepStatus.FAILED,
                error="MCP client not available",
            )

        params = step.get_params(context)
        retries = 0

        while retries <= step.max_retries:
            try:
                logger.debug(f"Executing step: {step.name}", extra={
                    "tool": step.tool,
                    "params": params,
                    "retry": retries,
                })

                result = await asyncio.wait_for(
                    self._mcp_client.call_tool(step.tool, params),
                    timeout=step.timeout_seconds,
                )

                if result.success:
                    return StepResult(
                        step_name=step.name,
                        status=StepStatus.SUCCESS,
                        data=result.data or {},
                        retries=retries,
                    )
                else:
                    if retries >= step.max_retries:
                        return StepResult(
                            step_name=step.name,
                            status=StepStatus.FAILED,
                            error=result.error or "Tool call failed",
                            retries=retries,
                        )
                    retries += 1

            except asyncio.TimeoutError:
                if retries >= step.max_retries:
                    return StepResult(
                        step_name=step.name,
                        status=StepStatus.FAILED,
                        error=f"Step timed out after {step.timeout_seconds}s",
                        retries=retries,
                    )
                retries += 1

            except Exception as e:
                logger.error(f"Step execution error: {e}")
                if retries >= step.max_retries:
                    return StepResult(
                        step_name=step.name,
                        status=StepStatus.FAILED,
                        error=str(e),
                        retries=retries,
                    )
                retries += 1

        return StepResult(
            step_name=step.name,
            status=StepStatus.FAILED,
            error="Max retries exceeded",
            retries=retries,
        )

    def _generate_summary(
        self,
        execution: WorkflowExecution,
        template: WorkflowTemplate,
    ) -> str:
        """Generate a human-readable summary of the workflow results."""
        if execution.status == WorkflowStatus.COMPLETED:
            successful = len([r for r in execution.results if r.success])
            return f"Completed {template.name}: {successful}/{len(template.steps)} steps successful"
        elif execution.status == WorkflowStatus.FAILED:
            return f"Failed {template.name}: {execution.error_message}"
        else:
            return f"{template.name}: {execution.status.value}"

    async def _save_execution(self, execution: WorkflowExecution) -> None:
        """Save execution state to database."""
        if not self._db_pool:
            return

        try:
            async with self._db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO workflow_executions
                        (id, template_id, project_id, user_id, user_session,
                         status, current_step, context, results, error_message,
                         started_at, completed_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ON CONFLICT (id) DO UPDATE SET
                        status = EXCLUDED.status,
                        current_step = EXCLUDED.current_step,
                        context = EXCLUDED.context,
                        results = EXCLUDED.results,
                        error_message = EXCLUDED.error_message,
                        completed_at = EXCLUDED.completed_at
                """,
                    execution.id, execution.template_id, execution.project_id,
                    execution.user_id, execution.user_session, execution.status.value,
                    execution.current_step, execution.context,
                    [r.to_dict() for r in execution.results], execution.error_message,
                    execution.started_at, execution.completed_at
                )
        except Exception as e:
            logger.error(f"Failed to save execution: {e}")

    # =========================================================================
    # Execution Queries
    # =========================================================================

    def get_execution(self, execution_id: UUID) -> Optional[WorkflowExecution]:
        """Get a workflow execution by ID."""
        return self._executions.get(str(execution_id))

    async def get_execution_status(self, execution_id: UUID) -> dict[str, Any]:
        """Get current status of a workflow execution."""
        execution = self.get_execution(execution_id)
        if not execution:
            return {"error": "Execution not found"}

        return {
            "id": str(execution.id),
            "template": execution.template_name,
            "status": execution.status.value,
            "current_step": execution.current_step,
            "is_complete": execution.is_complete,
            "results_count": len(execution.results),
        }


# Global instance management
_executor: Optional[WorkflowExecutor] = None


async def get_workflow_executor(
    mcp_client=None,
    db_pool=None,
) -> WorkflowExecutor:
    """Get or create the global WorkflowExecutor instance."""
    global _executor

    if _executor is None:
        _executor = WorkflowExecutor(mcp_client=mcp_client, db_pool=db_pool)
        await _executor.initialize()

    return _executor


def reset_workflow_executor() -> None:
    """Reset the global executor (for testing)."""
    global _executor
    _executor = None
