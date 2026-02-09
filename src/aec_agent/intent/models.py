"""
Data models for intent classification.
"""

from dataclasses import dataclass, field
from enum import Enum


class MEPDomain(str, Enum):
    """MEP domain categories."""

    HVAC = "hvac"
    ELECTRICAL = "electrical"
    PLUMBING = "plumbing"
    FIRE_PROTECTION = "fire_protection"
    LOW_VOLTAGE = "low_voltage"  # Security, fire alarm, BMS, AV, data/telecom
    GENERAL = "general"

    @classmethod
    def from_string(cls, value: str) -> "MEPDomain":
        """Convert string to MEPDomain, defaulting to GENERAL."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.GENERAL


class MEPAction(str, Enum):
    """Actions that can be performed on MEP elements."""

    QUERY = "query"           # Find, list, search elements
    CREATE = "create"         # Draw, place, add elements
    MODIFY = "modify"         # Edit, update, change elements
    DELETE = "delete"         # Remove, delete elements
    ANALYZE = "analyze"       # Check, validate, report
    ROUTE = "route"           # Route ducts, pipes, conduits
    SCHEDULE = "schedule"     # Generate schedules, counts
    COORDINATE = "coordinate" # Clash detection, coordination

    @classmethod
    def from_string(cls, value: str) -> "MEPAction":
        """Convert string to MEPAction, defaulting to QUERY."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.QUERY


class AppContext(str, Enum):
    """Application context for tool filtering."""

    AUTOCAD = "autocad"
    REVIT = "revit"
    BOTH = "both"

    def get_tool_prefix(self) -> str | None:
        """Get the tool prefix for filtering.

        Note: AutoCAD includes 'raster_' because raster/vectorization tools
        are AutoCAD-specific but use the 'raster_' prefix.
        """
        if self == AppContext.AUTOCAD:
            return "autocad_,raster_"
        elif self == AppContext.REVIT:
            return "revit_"
        return None


@dataclass
class IntentResult:
    """
    Result of intent classification.

    Used to determine which tools to load and how to process the request.
    """

    domain: MEPDomain = MEPDomain.GENERAL
    subdomain: str | None = None  # More specific: "ductwork", "panels", etc.
    action: MEPAction = MEPAction.QUERY
    app_context: AppContext = AppContext.BOTH
    confidence: float = 0.0
    suggested_tools: list[str] = field(default_factory=list)
    keywords_matched: list[str] = field(default_factory=list)

    @property
    def is_mep_specific(self) -> bool:
        """Check if this is a specific MEP intent (not general)."""
        return self.domain != MEPDomain.GENERAL and self.confidence >= 0.5

    @property
    def tool_filter_prefix(self) -> str | None:
        """Get the tool prefix for filtering based on app context."""
        return self.app_context.get_tool_prefix()

    def get_tool_tier(self) -> str:
        """
        Determine the appropriate tool tier based on intent.

        Returns:
            Tool tier: 'essential', 'standard', or 'advanced'
        """
        # Analysis and routing actions need advanced tools
        if self.action in (MEPAction.ANALYZE, MEPAction.ROUTE, MEPAction.COORDINATE):
            return "advanced"

        # Schedule generation needs standard tools
        if self.action == MEPAction.SCHEDULE:
            return "standard"

        # Basic operations can use essential tools
        if self.action in (MEPAction.QUERY, MEPAction.CREATE):
            return "essential"

        return "standard"

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/debugging."""
        return {
            "domain": self.domain.value,
            "subdomain": self.subdomain,
            "action": self.action.value,
            "app_context": self.app_context.value,
            "confidence": self.confidence,
            "suggested_tools": self.suggested_tools,
            "keywords_matched": self.keywords_matched,
            "is_mep_specific": self.is_mep_specific,
            "tool_tier": self.get_tool_tier(),
        }


@dataclass
class PatternMatch:
    """A single pattern match result."""

    pattern: str
    domain: MEPDomain
    subdomain: str | None
    weight: float
    similarity: float = 0.0

    @property
    def score(self) -> float:
        """Combined score from weight and similarity."""
        return self.weight * (1.0 + self.similarity)
