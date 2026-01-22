"""
Data models for project memory and user preferences.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4


class FactType(str, Enum):
    """Types of project facts that can be stored."""
    DECISION = "decision"       # Design decisions made
    CONSTRAINT = "constraint"   # Project constraints
    PREFERENCE = "preference"   # Project-specific preferences
    NOTE = "note"               # General notes
    STANDARD = "standard"       # Applied standards
    ASSUMPTION = "assumption"   # Design assumptions


class FactSource(str, Enum):
    """Source of a project fact."""
    USER = "user"               # Explicitly stated by user
    INFERRED = "inferred"       # Inferred from context
    IMPORTED = "imported"       # Imported from external source
    TEMPLATE = "template"       # From project template


@dataclass
class ProjectFact:
    """
    A fact about a project that persists across sessions.

    Examples:
    - decision: "HVAC system will be VAV with central AHU"
    - constraint: "Max ceiling height 3.0m on Level 2"
    - preference: "Use rectangular duct over spiral"
    - note: "Client wants extra receptacles in conference rooms"
    """
    id: UUID = field(default_factory=uuid4)
    project_id: UUID = field(default_factory=uuid4)
    fact_type: FactType = FactType.NOTE
    key: str = ""
    value: Any = None
    source: FactSource = FactSource.USER
    confidence: float = 1.0
    expires_at: Optional[datetime] = None
    embedding: Optional[list[float]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_expired(self) -> bool:
        """Check if fact has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    @property
    def is_high_confidence(self) -> bool:
        """Check if fact has high confidence (>= 0.8)."""
        return self.confidence >= 0.8

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": str(self.id),
            "project_id": str(self.project_id),
            "fact_type": self.fact_type.value,
            "key": self.key,
            "value": self.value,
            "source": self.source.value,
            "confidence": self.confidence,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectFact":
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]) if isinstance(data.get("id"), str) else data.get("id", uuid4()),
            project_id=UUID(data["project_id"]) if isinstance(data.get("project_id"), str) else data.get("project_id", uuid4()),
            fact_type=FactType(data.get("fact_type", "note")),
            key=data.get("key", ""),
            value=data.get("value"),
            source=FactSource(data.get("source", "user")),
            confidence=data.get("confidence", 1.0),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.utcnow(),
        )

    def to_context_string(self) -> str:
        """Format fact for LLM context injection."""
        return f"[{self.fact_type.value.upper()}] {self.key}: {self.value}"


@dataclass
class ConversationSummary:
    """
    Summary of a conversation session for long-term memory.

    Reduces token usage by storing compressed conversation history
    instead of full message threads.
    """
    id: UUID = field(default_factory=uuid4)
    project_id: Optional[UUID] = None
    user_id: Optional[str] = None
    user_session: Optional[str] = None
    summary: str = ""
    key_decisions: list[str] = field(default_factory=list)
    key_topics: list[str] = field(default_factory=list)
    embedding: Optional[list[float]] = None
    message_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": str(self.id),
            "project_id": str(self.project_id) if self.project_id else None,
            "user_id": self.user_id,
            "user_session": self.user_session,
            "summary": self.summary,
            "key_decisions": self.key_decisions,
            "key_topics": self.key_topics,
            "message_count": self.message_count,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationSummary":
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]) if isinstance(data.get("id"), str) else data.get("id", uuid4()),
            project_id=UUID(data["project_id"]) if data.get("project_id") else None,
            user_id=data.get("user_id"),
            user_session=data.get("user_session"),
            summary=data.get("summary", ""),
            key_decisions=data.get("key_decisions", []),
            key_topics=data.get("key_topics", []),
            message_count=data.get("message_count", 0),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
        )

    def to_context_string(self) -> str:
        """Format summary for LLM context injection."""
        lines = [f"Previous session summary ({self.message_count} messages):"]
        lines.append(self.summary)
        if self.key_decisions:
            lines.append("\nKey decisions:")
            for decision in self.key_decisions:
                lines.append(f"  - {decision}")
        return "\n".join(lines)


@dataclass
class MemoryContext:
    """
    Aggregated memory context for injection into LLM prompts.

    Combines relevant facts and conversation history into
    a compact context block.
    """
    facts: list[ProjectFact] = field(default_factory=list)
    recent_summaries: list[ConversationSummary] = field(default_factory=list)
    token_budget: int = 500  # Max tokens for context

    def to_context_string(self) -> str:
        """Generate context string for LLM injection."""
        lines = []

        # Group facts by type
        if self.facts:
            lines.append("=== Project Context ===")
            decisions = [f for f in self.facts if f.fact_type == FactType.DECISION]
            constraints = [f for f in self.facts if f.fact_type == FactType.CONSTRAINT]
            preferences = [f for f in self.facts if f.fact_type == FactType.PREFERENCE]

            if decisions:
                lines.append("\nDecisions:")
                for fact in decisions[:5]:  # Limit to 5
                    lines.append(f"  - {fact.key}: {fact.value}")

            if constraints:
                lines.append("\nConstraints:")
                for fact in constraints[:5]:
                    lines.append(f"  - {fact.key}: {fact.value}")

            if preferences:
                lines.append("\nPreferences:")
                for fact in preferences[:5]:
                    lines.append(f"  - {fact.key}: {fact.value}")

        # Add recent conversation context
        if self.recent_summaries:
            lines.append("\n=== Previous Sessions ===")
            for summary in self.recent_summaries[:2]:  # Limit to 2 recent
                lines.append(summary.to_context_string())

        return "\n".join(lines)

    @property
    def estimated_tokens(self) -> int:
        """Estimate token count for context (rough: 4 chars = 1 token)."""
        context = self.to_context_string()
        return len(context) // 4
