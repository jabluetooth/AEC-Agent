"""
Domain Rules Engine for MEP validation and suggestions.

Evaluates domain rules against elements and generates validation
results and design suggestions.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from aec_agent.domain.models import DomainRule, RuleType, ValidationStatus

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating an element against rules."""
    id: UUID = field(default_factory=uuid4)
    project_id: UUID = field(default_factory=uuid4)
    element_id: UUID | None = None
    rule_id: UUID | None = None
    rule_name: str = ""
    status: ValidationStatus = ValidationStatus.PASS
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    computed_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None

    @property
    def is_error(self) -> bool:
        return self.status == ValidationStatus.ERROR

    @property
    def is_warning(self) -> bool:
        return self.status == ValidationStatus.WARNING

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "project_id": str(self.project_id),
            "element_id": str(self.element_id) if self.element_id else None,
            "rule_id": str(self.rule_id) if self.rule_id else None,
            "rule_name": self.rule_name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "computed_at": self.computed_at.isoformat(),
        }


@dataclass
class DesignSuggestion:
    """A proactive design suggestion based on rule analysis."""
    id: UUID = field(default_factory=uuid4)
    project_id: UUID = field(default_factory=uuid4)
    context_element_id: UUID | None = None
    workflow_execution_id: UUID | None = None
    suggestion_type: str = ""         # routing, sizing, clearance, optimization
    suggestion: str = ""              # The suggestion text
    rationale: str = ""               # Why this is suggested
    priority: int = 50                # 0-100, higher = more important
    status: str = "pending"           # pending, accepted, dismissed
    embedding: list[float] | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    dismissed_at: datetime | None = None
    accepted_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "project_id": str(self.project_id),
            "context_element_id": str(self.context_element_id) if self.context_element_id else None,
            "suggestion_type": self.suggestion_type,
            "suggestion": self.suggestion,
            "rationale": self.rationale,
            "priority": self.priority,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
        }


class RulesEngine:
    """
    Evaluates domain rules against MEP elements.

    Provides validation checking and design suggestions based on
    rules from the knowledge base.
    """

    def __init__(self, knowledge_base=None, db_pool=None, embedding_service=None):
        """
        Initialize the rules engine.

        Args:
            knowledge_base: MEPKnowledgeBase instance
            db_pool: asyncpg connection pool
            embedding_service: Service for generating embeddings
        """
        self._knowledge_base = knowledge_base
        self._db_pool = db_pool
        self._embedding_service = embedding_service
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the rules engine."""
        if self._initialized:
            return

        if self._knowledge_base:
            await self._knowledge_base.initialize()

        self._initialized = True
        logger.info("RulesEngine initialized")

    # =========================================================================
    # Validation
    # =========================================================================

    async def validate_element(
        self,
        project_id: UUID,
        element: dict[str, Any],
        rule_types: list[RuleType] | None = None,
    ) -> list[ValidationResult]:
        """
        Validate an element against applicable rules.

        Args:
            project_id: Project UUID
            element: Element data dict (id, category, properties, etc.)
            rule_types: Optional filter for rule types

        Returns:
            List of ValidationResults
        """
        if not self._knowledge_base:
            return []

        results = []
        element_category = element.get("category", "").lower()

        # Get applicable rules
        context = {
            "element_type": element_category,
            "subdomain": self._infer_subdomain(element_category),
        }
        rules = self._knowledge_base.get_applicable_rules(context)

        if rule_types:
            rules = [r for r in rules if r.rule_type in rule_types]

        # Evaluate each rule
        for rule in rules:
            result = await self._evaluate_rule(project_id, element, rule)
            if result:
                results.append(result)

        # Save results to database
        if self._db_pool:
            await self._save_validation_results(results)

        return results

    async def validate_project(
        self,
        project_id: UUID,
        elements: list[dict[str, Any]],
        rule_types: list[RuleType] | None = None,
    ) -> dict[str, Any]:
        """
        Validate all elements in a project.

        Args:
            project_id: Project UUID
            elements: List of element dicts
            rule_types: Optional filter for rule types

        Returns:
            Summary dict with counts and results
        """
        all_results = []

        for element in elements:
            results = await self.validate_element(project_id, element, rule_types)
            all_results.extend(results)

        # Summarize
        errors = [r for r in all_results if r.is_error]
        warnings = [r for r in all_results if r.is_warning]
        passes = [r for r in all_results if r.status == ValidationStatus.PASS]

        return {
            "project_id": str(project_id),
            "total_elements": len(elements),
            "total_checks": len(all_results),
            "errors": len(errors),
            "warnings": len(warnings),
            "passes": len(passes),
            "error_details": [r.to_dict() for r in errors[:20]],  # Limit details
            "warning_details": [r.to_dict() for r in warnings[:20]],
        }

    async def _evaluate_rule(
        self,
        project_id: UUID,
        element: dict[str, Any],
        rule: DomainRule,
    ) -> ValidationResult | None:
        """Evaluate a single rule against an element."""
        try:
            condition = rule.condition
            element_props = element.get("properties", {})
            element_id = element.get("id")

            # Check if rule applies to this element type
            if "element_type" in condition:
                required_types = condition["element_type"]
                if isinstance(required_types, str):
                    required_types = [required_types]
                element_category = element.get("category", "").lower()
                if not any(t.lower() in element_category for t in required_types):
                    return None

            # Evaluate based on rule type
            if rule.rule_type == RuleType.CLEARANCE:
                return await self._check_clearance(project_id, element, rule)
            elif rule.rule_type == RuleType.SIZING:
                return await self._check_sizing(project_id, element, rule)
            elif rule.rule_type == RuleType.ROUTING:
                return await self._check_routing(project_id, element, rule)
            elif rule.rule_type == RuleType.ACCESS:
                return await self._check_access(project_id, element, rule)

            return None

        except Exception as e:
            logger.error(f"Rule evaluation error: {e}", extra={
                "rule": rule.rule_name,
                "element": element.get("id"),
            })
            return None

    async def _check_clearance(
        self,
        project_id: UUID,
        element: dict[str, Any],
        rule: DomainRule,
    ) -> ValidationResult | None:
        """Check clearance requirements."""
        condition = rule.condition
        action = rule.action

        min_distance = action.get("min_distance", 0)
        near_type = condition.get("near_type", "any")

        # This would normally query nearby elements from database
        # For now, check if element has clearance metadata
        element_props = element.get("properties", {})
        actual_clearance = element_props.get("clearance", element_props.get("distance_to_nearest"))

        if actual_clearance is None:
            return None  # Can't validate without clearance data

        if actual_clearance < min_distance:
            return ValidationResult(
                project_id=project_id,
                element_id=UUID(element["id"]) if element.get("id") else None,
                rule_id=rule.id,
                rule_name=rule.rule_name,
                status=ValidationStatus.ERROR,
                message=action.get("error_message", f"Clearance violation: {actual_clearance:.2f}m < {min_distance}m"),
                details={
                    "required": min_distance,
                    "actual": actual_clearance,
                    "near_type": near_type,
                },
            )

        return ValidationResult(
            project_id=project_id,
            element_id=UUID(element["id"]) if element.get("id") else None,
            rule_id=rule.id,
            rule_name=rule.rule_name,
            status=ValidationStatus.PASS,
            message="Clearance OK",
            details={"clearance": actual_clearance},
        )

    async def _check_sizing(
        self,
        project_id: UUID,
        element: dict[str, Any],
        rule: DomainRule,
    ) -> ValidationResult | None:
        """Check sizing requirements (velocities, etc.)."""
        condition = rule.condition
        action = rule.action
        element_props = element.get("properties", {})

        # Check velocity limits
        max_velocity = action.get("max_velocity")
        if max_velocity:
            actual_velocity = element_props.get("velocity", element_props.get("air_velocity"))
            if actual_velocity and actual_velocity > max_velocity:
                return ValidationResult(
                    project_id=project_id,
                    element_id=UUID(element["id"]) if element.get("id") else None,
                    rule_id=rule.id,
                    rule_name=rule.rule_name,
                    status=ValidationStatus.WARNING if action.get("severity") == "warning" else ValidationStatus.ERROR,
                    message=action.get("error_message", f"Velocity exceeds limit: {actual_velocity} > {max_velocity} {action.get('unit', 'fpm')}"),
                    details={
                        "max_velocity": max_velocity,
                        "actual_velocity": actual_velocity,
                        "unit": action.get("unit", "fpm"),
                    },
                )

        # Check flow requirements
        min_flow = action.get("min_flow")
        if min_flow:
            actual_flow = element_props.get("flow", element_props.get("cfm"))
            if actual_flow and actual_flow < min_flow:
                return ValidationResult(
                    project_id=project_id,
                    element_id=UUID(element["id"]) if element.get("id") else None,
                    rule_id=rule.id,
                    rule_name=rule.rule_name,
                    status=ValidationStatus.WARNING,
                    message=f"Flow below minimum: {actual_flow} < {min_flow}",
                    details={
                        "min_flow": min_flow,
                        "actual_flow": actual_flow,
                    },
                )

        return None  # No issues found or not enough data

    async def _check_routing(
        self,
        project_id: UUID,
        element: dict[str, Any],
        rule: DomainRule,
    ) -> ValidationResult | None:
        """Check routing requirements (slopes, directions, etc.)."""
        action = rule.action
        element_props = element.get("properties", {})

        # Check slope requirements
        min_slope = action.get("min_slope")
        if min_slope:
            actual_slope = element_props.get("slope")
            if actual_slope is not None and actual_slope < min_slope:
                return ValidationResult(
                    project_id=project_id,
                    element_id=UUID(element["id"]) if element.get("id") else None,
                    rule_id=rule.id,
                    rule_name=rule.rule_name,
                    status=ValidationStatus.ERROR,
                    message=action.get("error_message", f"Insufficient slope: {actual_slope}% < {min_slope}%"),
                    details={
                        "min_slope": min_slope,
                        "actual_slope": actual_slope,
                    },
                )

        return None

    async def _check_access(
        self,
        project_id: UUID,
        element: dict[str, Any],
        rule: DomainRule,
    ) -> ValidationResult | None:
        """Check access requirements for maintenance."""
        action = rule.action
        element_props = element.get("properties", {})

        min_clearance = action.get("min_clearance")
        if min_clearance:
            access_clearance = element_props.get("access_clearance", element_props.get("service_clearance"))
            if access_clearance is not None and access_clearance < min_clearance:
                return ValidationResult(
                    project_id=project_id,
                    element_id=UUID(element["id"]) if element.get("id") else None,
                    rule_id=rule.id,
                    rule_name=rule.rule_name,
                    status=ValidationStatus.WARNING,
                    message=action.get("error_message", f"Insufficient access clearance: {access_clearance} < {min_clearance}"),
                    details={
                        "required": min_clearance,
                        "actual": access_clearance,
                    },
                )

        return None

    # =========================================================================
    # Suggestions
    # =========================================================================

    async def generate_suggestions(
        self,
        project_id: UUID,
        element: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> list[DesignSuggestion]:
        """
        Generate design suggestions for an element.

        Args:
            project_id: Project UUID
            element: Element data dict
            context: Optional additional context

        Returns:
            List of DesignSuggestions
        """
        suggestions = []
        element_props = element.get("properties", {})
        element_category = element.get("category", "").lower()

        # Duct sizing optimization suggestions
        if "duct" in element_category:
            suggestions.extend(await self._suggest_duct_optimizations(project_id, element))

        # Routing suggestions
        if context and "nearby_structure" in context:
            suggestions.extend(await self._suggest_routing(project_id, element, context))

        # Equipment suggestions
        if "equipment" in element_category or "ahu" in element_category:
            suggestions.extend(await self._suggest_equipment_improvements(project_id, element))

        # Save suggestions to database
        if self._db_pool:
            await self._save_suggestions(suggestions)

        return suggestions

    async def _suggest_duct_optimizations(
        self,
        project_id: UUID,
        element: dict[str, Any],
    ) -> list[DesignSuggestion]:
        """Generate duct optimization suggestions."""
        suggestions = []
        element_props = element.get("properties", {})

        # Check for high velocity that could cause noise
        velocity = element_props.get("velocity", element_props.get("air_velocity", 0))
        if velocity > 1500:  # Higher velocities may cause noise
            suggestions.append(DesignSuggestion(
                project_id=project_id,
                context_element_id=UUID(element["id"]) if element.get("id") else None,
                suggestion_type="sizing",
                suggestion=f"Consider increasing duct size to reduce velocity from {velocity} fpm",
                rationale="High duct velocities above 1500 fpm can cause noise issues in occupied spaces",
                priority=60,
            ))

        # Check for oversized ducts (low velocity)
        if 0 < velocity < 400:
            suggestions.append(DesignSuggestion(
                project_id=project_id,
                context_element_id=UUID(element["id"]) if element.get("id") else None,
                suggestion_type="sizing",
                suggestion=f"Duct may be oversized - velocity is only {velocity} fpm",
                rationale="Low velocities indicate oversized ducts, increasing material and installation costs",
                priority=40,
            ))

        return suggestions

    async def _suggest_routing(
        self,
        project_id: UUID,
        element: dict[str, Any],
        context: dict[str, Any],
    ) -> list[DesignSuggestion]:
        """Generate routing suggestions based on nearby elements."""
        suggestions = []

        nearby_structure = context.get("nearby_structure", [])
        if nearby_structure:
            suggestions.append(DesignSuggestion(
                project_id=project_id,
                context_element_id=UUID(element["id"]) if element.get("id") else None,
                suggestion_type="routing",
                suggestion="Consider routing parallel to structural beams for cleaner coordination",
                rationale="Parallel routing reduces coordination conflicts and simplifies installation",
                priority=50,
            ))

        return suggestions

    async def _suggest_equipment_improvements(
        self,
        project_id: UUID,
        element: dict[str, Any],
    ) -> list[DesignSuggestion]:
        """Generate equipment improvement suggestions."""
        suggestions = []
        element_props = element.get("properties", {})

        # Check for missing access clearance
        if "access_clearance" not in element_props:
            suggestions.append(DesignSuggestion(
                project_id=project_id,
                context_element_id=UUID(element["id"]) if element.get("id") else None,
                suggestion_type="access",
                suggestion="Verify service access clearance meets manufacturer requirements",
                rationale="Equipment needs adequate access for filter changes and maintenance",
                priority=70,
            ))

        return suggestions

    # =========================================================================
    # Persistence
    # =========================================================================

    async def _save_validation_results(self, results: list[ValidationResult]) -> None:
        """Save validation results to database."""
        if not self._db_pool or not results:
            return

        try:
            async with self._db_pool.acquire() as conn:
                for result in results:
                    await conn.execute("""
                        INSERT INTO validation_results
                            (id, project_id, element_id, rule_id, status,
                             message, details, computed_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                        result.id, result.project_id, result.element_id,
                        result.rule_id, result.status.value, result.message,
                        result.details, result.computed_at
                    )
        except Exception as e:
            logger.error(f"Failed to save validation results: {e}")

    async def _save_suggestions(self, suggestions: list[DesignSuggestion]) -> None:
        """Save suggestions to database."""
        if not self._db_pool or not suggestions:
            return

        try:
            async with self._db_pool.acquire() as conn:
                for suggestion in suggestions:
                    # Generate embedding for suggestion
                    embedding = None
                    if self._embedding_service:
                        try:
                            text = f"{suggestion.suggestion} {suggestion.rationale}"
                            embedding = await self._embedding_service.embed_text(text)
                        except Exception:
                            pass

                    await conn.execute("""
                        INSERT INTO design_suggestions
                            (id, project_id, context_element_id, suggestion_type,
                             suggestion, rationale, priority, status, embedding, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    """,
                        suggestion.id, suggestion.project_id, suggestion.context_element_id,
                        suggestion.suggestion_type, suggestion.suggestion, suggestion.rationale,
                        suggestion.priority, suggestion.status, embedding, suggestion.created_at
                    )
        except Exception as e:
            logger.error(f"Failed to save suggestions: {e}")

    async def get_pending_suggestions(
        self,
        project_id: UUID,
        limit: int = 20,
    ) -> list[DesignSuggestion]:
        """Get pending suggestions for a project."""
        if not self._db_pool:
            return []

        try:
            async with self._db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT id, project_id, context_element_id, suggestion_type,
                           suggestion, rationale, priority, status, created_at
                    FROM design_suggestions
                    WHERE project_id = $1 AND status = 'pending'
                    ORDER BY priority DESC, created_at DESC
                    LIMIT $2
                """, project_id, limit)

                return [
                    DesignSuggestion(
                        id=row["id"],
                        project_id=row["project_id"],
                        context_element_id=row["context_element_id"],
                        suggestion_type=row["suggestion_type"],
                        suggestion=row["suggestion"],
                        rationale=row["rationale"],
                        priority=row["priority"],
                        status=row["status"],
                        created_at=row["created_at"],
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Failed to get suggestions: {e}")
            return []

    async def dismiss_suggestion(self, suggestion_id: UUID) -> bool:
        """Mark a suggestion as dismissed."""
        if not self._db_pool:
            return False

        try:
            async with self._db_pool.acquire() as conn:
                result = await conn.execute("""
                    UPDATE design_suggestions
                    SET status = 'dismissed', dismissed_at = NOW()
                    WHERE id = $1
                """, suggestion_id)
                return "UPDATE 1" in result
        except Exception as e:
            logger.error(f"Failed to dismiss suggestion: {e}")
            return False

    async def accept_suggestion(self, suggestion_id: UUID) -> bool:
        """Mark a suggestion as accepted."""
        if not self._db_pool:
            return False

        try:
            async with self._db_pool.acquire() as conn:
                result = await conn.execute("""
                    UPDATE design_suggestions
                    SET status = 'accepted', accepted_at = NOW()
                    WHERE id = $1
                """, suggestion_id)
                return "UPDATE 1" in result
        except Exception as e:
            logger.error(f"Failed to accept suggestion: {e}")
            return False

    # =========================================================================
    # Helpers
    # =========================================================================

    def _infer_subdomain(self, category: str) -> str:
        """Infer MEP subdomain from element category."""
        category_lower = category.lower()

        hvac_keywords = ["duct", "diffuser", "vav", "ahu", "hvac", "air", "damper", "fan"]
        electrical_keywords = ["conduit", "panel", "circuit", "cable", "receptacle", "switch"]
        plumbing_keywords = ["pipe", "fixture", "drain", "valve", "plumbing", "water"]
        fire_keywords = ["sprinkler", "standpipe", "fire", "alarm", "detector"]

        if any(kw in category_lower for kw in hvac_keywords):
            return "hvac"
        elif any(kw in category_lower for kw in electrical_keywords):
            return "electrical"
        elif any(kw in category_lower for kw in plumbing_keywords):
            return "plumbing"
        elif any(kw in category_lower for kw in fire_keywords):
            return "fire_protection"

        return "mep"


# Global instance management
_engine: RulesEngine | None = None


async def get_rules_engine(
    knowledge_base=None,
    db_pool=None,
    embedding_service=None,
) -> RulesEngine:
    """Get or create the global RulesEngine instance."""
    global _engine

    if _engine is None:
        _engine = RulesEngine(
            knowledge_base=knowledge_base,
            db_pool=db_pool,
            embedding_service=embedding_service
        )
        await _engine.initialize()

    return _engine


def reset_rules_engine() -> None:
    """Reset the global engine (for testing)."""
    global _engine
    _engine = None
