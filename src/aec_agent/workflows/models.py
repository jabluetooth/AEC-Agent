"""
Data models for workflow templates and executions.
"""

import ast
import operator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

# Safe operators for expression evaluation
_SAFE_OPERATORS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.And: lambda a, b: a and b,
    ast.Or: lambda a, b: a or b,
    ast.Not: operator.not_,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}


def _safe_eval_expr(node: ast.AST, context: dict[str, Any]) -> Any:
    """
    Safely evaluate an AST node with restricted operations.

    Only allows: comparisons, boolean ops, attribute access,
    constants, names from context, and subscript access.
    """
    if isinstance(node, ast.Expression):
        return _safe_eval_expr(node.body, context)
    elif isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Name):
        if node.id in context:
            return context[node.id]
        raise NameError(f"Name '{node.id}' is not defined in context")
    elif isinstance(node, ast.Attribute):
        value = _safe_eval_expr(node.value, context)
        if isinstance(value, dict):
            return value.get(node.attr)
        return getattr(value, node.attr, None)
    elif isinstance(node, ast.Subscript):
        value = _safe_eval_expr(node.value, context)
        key = _safe_eval_expr(node.slice, context)
        return value[key]
    elif isinstance(node, ast.Compare):
        left = _safe_eval_expr(node.left, context)
        for op, comparator in zip(node.ops, node.comparators):
            op_func = _SAFE_OPERATORS.get(type(op))
            if op_func is None:
                raise ValueError(f"Unsupported comparison operator: {type(op).__name__}")
            right = _safe_eval_expr(comparator, context)
            if not op_func(left, right):
                return False
            left = right
        return True
    elif isinstance(node, ast.BoolOp):
        op_func = _SAFE_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported boolean operator: {type(node.op).__name__}")
        values = [_safe_eval_expr(v, context) for v in node.values]
        result = values[0]
        for v in values[1:]:
            result = op_func(result, v)
        return result
    elif isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return not _safe_eval_expr(node.operand, context)
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
    elif isinstance(node, ast.IfExp):
        test = _safe_eval_expr(node.test, context)
        return _safe_eval_expr(node.body if test else node.orelse, context)
    elif isinstance(node, (ast.List, ast.Tuple)):
        return [_safe_eval_expr(elt, context) for elt in node.elts]
    elif isinstance(node, ast.Dict):
        return {
            _safe_eval_expr(k, context): _safe_eval_expr(v, context)
            for k, v in zip(node.keys, node.values)
            if k is not None
        }
    else:
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def safe_eval_condition(condition: str, context: dict[str, Any]) -> bool:
    """
    Safely evaluate a condition string.

    Only allows safe operations: comparisons, boolean operations,
    attribute access, and basic data structures.

    Args:
        condition: A Python expression string
        context: Variables available to the expression

    Returns:
        Boolean result of the expression

    Raises:
        ValueError: If the expression contains unsafe operations
    """
    try:
        tree = ast.parse(condition, mode='eval')
        return bool(_safe_eval_expr(tree, context))
    except (SyntaxError, ValueError, NameError, TypeError, KeyError):
        return True  # Default to execute on error


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
    condition: str | None = None     # Python expression for conditional execution
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
        return safe_eval_condition(self.condition, context)

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
    subdomain: str | None = None
    description: str = ""
    steps: list[WorkflowStep] = field(default_factory=list)
    required_context: list[str] = field(default_factory=list)  # Required input fields
    default_params: dict[str, Any] = field(default_factory=dict)
    estimated_tokens: int | None = None  # Estimated token savings
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
    error: str | None = None
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
    template_id: UUID | None = None
    template_name: str = ""
    project_id: UUID | None = None
    user_id: str | None = None
    user_session: str | None = None
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_step: int = 0
    context: dict[str, Any] = field(default_factory=dict)
    results: list[StepResult] = field(default_factory=list)
    error_message: str | None = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

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
    error: str | None = None
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
