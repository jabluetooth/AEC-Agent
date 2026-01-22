"""
Data models for workflow templates and executions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4


class WorkflowStatus(str, Enum):
    """Status of a workflow execution."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    """Status of a workflow step."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """
    A single step in a workflow template.

    Steps can reference results from previous steps using {step_name.field} syntax.
    """
    name: str                           # Unique name within workflow
    tool: str                           # MCP tool to call
    params_template: dict[str, Any] = field(default_factory=dict)  # Params with placeholders
    description: str = ""               # Human-readable description
    condition: Optional[str] = None     # Python expression for conditional execution
    on_error: str = "abort"             # "abort", "skip", "retry"
    max_retries: int = 1
    timeout_seconds: float = 120.0

    def get_params(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Resolve parameter placeholders using context.

        Supports:
        - {variable} - Direct variable substitution
        - {step_name.field} - Reference to previous step result
        - {context.field} - Reference to workflow context
        """
        resolved = {}
        for key, value in self.params_template.items():
            resolved[key] = self._resolve_value(value, context)
        return resolved

    def _resolve_value(self, value: Any, context: dict[str, Any]) -> Any:
        """Recursively resolve placeholders in a value."""
        if isinstance(value, str) and "{" in value:
            try:
                # Simple format string substitution
                return value.format(**context)
            except KeyError:
                # Try nested access for {step.field} syntax
                import re
                pattern = r'\{(\w+)\.(\w+)\}'
                matches = re.findall(pattern, value)
                result = value
                for step_name, field_name in matches:
                    if step_name in context and isinstance(context[step_name], dict):
                        replacement = context[step_name].get(field_name, "")
                        result = result.replace(f"{{{step_name}.{field_name}}}", str(replacement))
                return result
        elif isinstance(value, dict):
            return {k: self._resolve_value(v, context) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._resolve_value(v, context) for v in value]
        return value

    def should_execute(self, context: dict[str, Any]) -> bool:
        """Check if step should execute based on condition."""
        if not self.condition:
            return True
        try:
            return bool(eval(self.condition, {"__builtins__": {}}, context))
        except Exception:
            return True  # Execute if condition evaluation fails

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "name": self.name,
            "tool": self.tool,
            "params_template": self.params_template,
            "description": self.description,
            "condition": self.condition,
            "on_error": self.on_error,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowStep":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            tool=data["tool"],
            params_template=data.get("params_template", {}),
            description=data.get("description", ""),
            condition=data.get("condition"),
            on_error=data.get("on_error", "abort"),
            max_retries=data.get("max_retries", 1),
            timeout_seconds=data.get("timeout_seconds", 120.0),
        )


@dataclass
class WorkflowTemplate:
    """
    A workflow template that defines a reusable multi-step process.
    """
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    domain: str = "mep"
    subdomain: Optional[str] = None
    description: str = ""
    steps: list[WorkflowStep] = field(default_factory=list)
    required_context: list[str] = field(default_factory=list)  # Required input fields
    default_params: dict[str, Any] = field(default_factory=dict)
    estimated_tokens: Optional[int] = None  # Estimated token savings
    created_at: datetime = field(default_factory=datetime.utcnow)

    def validate_context(self, context: dict[str, Any]) -> list[str]:
        """
        Validate that all required context fields are present.

        Returns list of missing field names.
        """
        missing = []
        for field_name in self.required_context:
            if field_name not in context:
                missing.append(field_name)
        return missing

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": str(self.id),
            "name": self.name,
            "domain": self.domain,
            "subdomain": self.subdomain,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "required_context": self.required_context,
            "default_params": self.default_params,
            "estimated_tokens": self.estimated_tokens,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowTemplate":
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]) if isinstance(data.get("id"), str) else data.get("id", uuid4()),
            name=data.get("name", ""),
            domain=data.get("domain", "mep"),
            subdomain=data.get("subdomain"),
            description=data.get("description", ""),
            steps=[WorkflowStep.from_dict(s) for s in data.get("steps", [])],
            required_context=data.get("required_context", []),
            default_params=data.get("default_params", {}),
            estimated_tokens=data.get("estimated_tokens"),
        )


@dataclass
class StepResult:
    """Result of executing a single workflow step."""
    step_name: str
    status: StepStatus
    data: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: float = 0
    retries: int = 0

    @property
    def success(self) -> bool:
        return self.status == StepStatus.SUCCESS

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_name": self.step_name,
            "status": self.status.value,
            "data": self.data,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "retries": self.retries,
        }


@dataclass
class WorkflowExecution:
    """
    A running or completed workflow execution.
    """
    id: UUID = field(default_factory=uuid4)
    template_id: Optional[UUID] = None
    template_name: str = ""
    project_id: Optional[UUID] = None
    user_id: Optional[str] = None
    user_session: Optional[str] = None
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_step: int = 0
    context: dict[str, Any] = field(default_factory=dict)
    results: list[StepResult] = field(default_factory=list)
    error_message: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    @property
    def is_complete(self) -> bool:
        return self.status in (WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED)

    @property
    def is_running(self) -> bool:
        return self.status == WorkflowStatus.IN_PROGRESS

    def add_step_result(self, result: StepResult) -> None:
        """Add a step result and update context."""
        self.results.append(result)
        if result.success:
            # Add step result to context for subsequent steps
            self.context[result.step_name] = result.data
        self.current_step += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "template_id": str(self.template_id) if self.template_id else None,
            "template_name": self.template_name,
            "project_id": str(self.project_id) if self.project_id else None,
            "user_id": self.user_id,
            "status": self.status.value,
            "current_step": self.current_step,
            "context": self.context,
            "results": [r.to_dict() for r in self.results],
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class WorkflowResult:
    """Final result of a completed workflow."""
    execution_id: UUID
    template_name: str
    status: WorkflowStatus
    steps_completed: int
    steps_total: int
    final_data: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    error: Optional[str] = None
    duration_ms: float = 0

    @property
    def success(self) -> bool:
        return self.status == WorkflowStatus.COMPLETED

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": str(self.execution_id),
            "template_name": self.template_name,
            "status": self.status.value,
            "steps_completed": self.steps_completed,
            "steps_total": self.steps_total,
            "final_data": self.final_data,
            "summary": self.summary,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }
