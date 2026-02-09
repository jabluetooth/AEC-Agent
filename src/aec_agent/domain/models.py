"""
Data models for MEP domain knowledge.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class RuleType(str, Enum):
    """Types of domain rules."""
    CLEARANCE = "clearance"      # Minimum distances between elements
    ROUTING = "routing"          # Routing priorities and constraints
    SIZING = "sizing"            # Sizing calculations and limits
    PRIORITY = "priority"        # System coordination priorities
    ACCESS = "access"            # Service access requirements
    CODE = "code"                # Building code requirements
    VALIDATION = "validation"    # General validation rules


class RuleSource(str, Enum):
    """Source of a domain rule."""
    CODE = "code"                # Building code requirement
    BEST_PRACTICE = "best_practice"  # Industry best practice
    TEMPLATE = "template"        # Extracted from CAD template
    USER_DEFINED = "user_defined"  # User-created rule
    COMPUTED = "computed"        # System-generated


class ValidationStatus(str, Enum):
    """Status of a validation check."""
    PASS = "pass"
    WARNING = "warning"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class DomainRule:
    """
    A domain rule that encodes MEP design knowledge.

    Rules can be used for validation, suggestions, and automated decision-making.
    """
    id: UUID = field(default_factory=uuid4)
    domain: str = "mep"          # Domain (mep, structural, architectural)
    subdomain: str | None = None  # Subdomain (hvac, electrical, plumbing)
    rule_type: RuleType = RuleType.VALIDATION
    rule_name: str = ""

    # Rule definition
    condition: dict[str, Any] = field(default_factory=dict)  # When rule applies
    action: dict[str, Any] = field(default_factory=dict)     # What to check/do

    # Metadata
    priority: int = 100          # Lower = higher priority
    source: RuleSource = RuleSource.BEST_PRACTICE
    description: str = ""

    # Optional embedding for semantic search
    embedding: list[float] | None = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)

    def matches_context(self, context: dict[str, Any]) -> bool:
        """
        Check if this rule applies to the given context.

        Args:
            context: Dict with keys like 'element_type', 'subdomain', 'action'

        Returns:
            True if rule conditions match the context
        """
        for key, expected in self.condition.items():
            if key not in context:
                continue
            actual = context[key]

            # Handle list conditions (any match)
            if isinstance(expected, list):
                if actual not in expected:
                    return False
            # Handle exact match
            elif actual != expected:
                return False

        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": str(self.id),
            "domain": self.domain,
            "subdomain": self.subdomain,
            "rule_type": self.rule_type.value,
            "rule_name": self.rule_name,
            "condition": self.condition,
            "action": self.action,
            "priority": self.priority,
            "source": self.source.value,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DomainRule":
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]) if isinstance(data.get("id"), str) else data.get("id", uuid4()),
            domain=data.get("domain", "mep"),
            subdomain=data.get("subdomain"),
            rule_type=RuleType(data.get("rule_type", "validation")),
            rule_name=data.get("rule_name", ""),
            condition=data.get("condition", {}),
            action=data.get("action", {}),
            priority=data.get("priority", 100),
            source=RuleSource(data.get("source", "best_practice")),
            description=data.get("description", ""),
        )


@dataclass
class SystemPriority:
    """
    Defines coordination priority for a system type within a project.

    Lower rank = higher priority (gets right-of-way in coordination).
    """
    id: UUID = field(default_factory=uuid4)
    project_id: UUID | None = None
    system_type: str = ""        # e.g., 'gravity_drain', 'hvac_supply', 'electrical'
    priority_rank: int = 50      # 1 = highest priority
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "project_id": str(self.project_id) if self.project_id else None,
            "system_type": self.system_type,
            "priority_rank": self.priority_rank,
            "notes": self.notes,
        }


@dataclass
class ValidationResult:
    """Result of applying a validation rule."""
    rule_id: UUID
    rule_name: str
    status: str  # 'pass', 'warning', 'error'
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    element_id: UUID | None = None

    @property
    def is_error(self) -> bool:
        return self.status == "error"

    @property
    def is_warning(self) -> bool:
        return self.status == "warning"

    @property
    def passed(self) -> bool:
        return self.status == "pass"


@dataclass
class DesignSuggestion:
    """A design suggestion generated by the rules engine."""
    id: UUID = field(default_factory=uuid4)
    project_id: UUID | None = None
    context_element_id: UUID | None = None
    suggestion_type: str = ""    # 'routing', 'sizing', 'clearance', 'optimization'
    suggestion: str = ""
    rationale: str = ""
    priority: int = 50           # Lower = more important
    status: str = "pending"      # 'pending', 'accepted', 'dismissed'
    embedding: list[float] | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "project_id": str(self.project_id) if self.project_id else None,
            "context_element_id": str(self.context_element_id) if self.context_element_id else None,
            "suggestion_type": self.suggestion_type,
            "suggestion": self.suggestion,
            "rationale": self.rationale,
            "priority": self.priority,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
        }
