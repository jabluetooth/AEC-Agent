"""
MEP Domain Knowledge module.

Provides domain-specific rules, knowledge base, validation, and suggestions
for MEP design workflows.
"""

from aec_agent.domain.models import (
    DomainRule,
    SystemPriority,
    RuleType,
    RuleSource,
    ValidationStatus,
    ValidationResult,
    DesignSuggestion,
)
from aec_agent.domain.knowledge import MEPKnowledgeBase, get_knowledge_base
from aec_agent.domain.rules_engine import RulesEngine, get_rules_engine
from aec_agent.domain.validators import (
    ClearanceValidator,
    SizingValidator,
    RoutingValidator,
    CoverageValidator,
    ValidationIssue,
    Point3D,
)

__all__ = [
    # Models
    "DomainRule",
    "SystemPriority",
    "RuleType",
    "RuleSource",
    "ValidationStatus",
    "ValidationResult",
    "DesignSuggestion",
    # Knowledge Base
    "MEPKnowledgeBase",
    "get_knowledge_base",
    # Rules Engine
    "RulesEngine",
    "get_rules_engine",
    # Validators
    "ClearanceValidator",
    "SizingValidator",
    "RoutingValidator",
    "CoverageValidator",
    "ValidationIssue",
    "Point3D",
]
