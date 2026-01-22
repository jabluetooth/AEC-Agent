"""
MEP Knowledge Base for domain rules and design knowledge.
"""

import logging
from typing import Any, Optional
from uuid import UUID

from aec_agent.domain.models import DomainRule, SystemPriority, RuleType, RuleSource

logger = logging.getLogger(__name__)


class MEPKnowledgeBase:
    """
    Knowledge base for MEP design rules and best practices.

    Provides methods to query rules, validate designs, and get suggestions.
    Can operate in-memory or with PostgreSQL backend.
    """

    def __init__(self, db_pool=None):
        """
        Initialize the knowledge base.

        Args:
            db_pool: Optional asyncpg connection pool for database storage.
                    If None, operates in-memory only.
        """
        self._db_pool = db_pool
        self._rules_cache: dict[str, DomainRule] = {}
        self._priorities_cache: dict[str, list[SystemPriority]] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Load rules from database or seed with defaults."""
        if self._initialized:
            return

        if self._db_pool:
            await self._load_rules_from_db()
        else:
            self._load_default_rules()

        self._initialized = True
        logger.info(f"MEPKnowledgeBase initialized with {len(self._rules_cache)} rules")

    def _load_default_rules(self) -> None:
        """Load default HVAC rules into memory cache."""
        from aec_agent.domain.seed_data import get_default_hvac_rules
        for rule in get_default_hvac_rules():
            self._rules_cache[str(rule.id)] = rule

    async def _load_rules_from_db(self) -> None:
        """Load rules from PostgreSQL database."""
        try:
            async with self._db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT id, domain, subdomain, rule_type, rule_name,
                           condition, action, priority, source, description
                    FROM domain_rules
                    ORDER BY priority
                """)
                for row in rows:
                    rule = DomainRule(
                        id=row["id"],
                        domain=row["domain"],
                        subdomain=row["subdomain"],
                        rule_type=RuleType(row["rule_type"]),
                        rule_name=row["rule_name"],
                        condition=row["condition"],
                        action=row["action"],
                        priority=row["priority"],
                        source=RuleSource(row["source"]) if row["source"] else RuleSource.BEST_PRACTICE,
                        description=row["description"] or "",
                    )
                    self._rules_cache[str(rule.id)] = rule
        except Exception as e:
            logger.warning(f"Failed to load rules from DB, using defaults: {e}")
            self._load_default_rules()

    # =========================================================================
    # Rule Queries
    # =========================================================================

    def get_all_rules(self) -> list[DomainRule]:
        """Get all rules sorted by priority."""
        return sorted(self._rules_cache.values(), key=lambda r: r.priority)

    def get_rules_by_type(self, rule_type: RuleType) -> list[DomainRule]:
        """Get rules of a specific type."""
        return [r for r in self._rules_cache.values() if r.rule_type == rule_type]

    def get_rules_by_subdomain(self, subdomain: str) -> list[DomainRule]:
        """Get rules for a specific MEP subdomain (hvac, electrical, plumbing)."""
        return [r for r in self._rules_cache.values()
                if r.subdomain == subdomain or r.subdomain is None]

    def get_clearance_rules(
        self,
        element_type: Optional[str] = None,
        near_type: Optional[str] = None,
    ) -> list[DomainRule]:
        """
        Get clearance rules, optionally filtered by element types.

        Args:
            element_type: Type of element to check (e.g., 'duct', 'pipe')
            near_type: Type of nearby element (e.g., 'beam', 'column')

        Returns:
            List of matching clearance rules
        """
        rules = self.get_rules_by_type(RuleType.CLEARANCE)

        if element_type:
            rules = [r for r in rules if self._matches_element_type(r, element_type)]

        if near_type:
            rules = [r for r in rules if self._matches_near_type(r, near_type)]

        return rules

    def _matches_element_type(self, rule: DomainRule, element_type: str) -> bool:
        """Check if rule applies to element type."""
        condition_types = rule.condition.get("element_type", [])
        if not condition_types:
            return True  # Rule applies to all types
        if isinstance(condition_types, str):
            condition_types = [condition_types]
        return element_type.lower() in [t.lower() for t in condition_types]

    def _matches_near_type(self, rule: DomainRule, near_type: str) -> bool:
        """Check if rule applies to nearby element type."""
        near_types = rule.condition.get("near_type", [])
        if not near_types:
            return True
        if isinstance(near_types, str):
            near_types = [near_types]
        return near_type.lower() in [t.lower() for t in near_types]

    def get_sizing_rules(self, system_type: Optional[str] = None) -> list[DomainRule]:
        """Get sizing rules, optionally filtered by system type."""
        rules = self.get_rules_by_type(RuleType.SIZING)
        if system_type:
            rules = [r for r in rules
                     if r.condition.get("system_type") in (None, system_type)]
        return rules

    def get_routing_rules(self, subdomain: Optional[str] = None) -> list[DomainRule]:
        """Get routing priority rules."""
        rules = self.get_rules_by_type(RuleType.ROUTING)
        if subdomain:
            rules = [r for r in rules if r.subdomain in (None, subdomain)]
        return rules

    # =========================================================================
    # System Priorities
    # =========================================================================

    def get_default_priorities(self) -> list[SystemPriority]:
        """Get default system coordination priorities."""
        return [
            SystemPriority(system_type="gravity_drain", priority_rank=1,
                          notes="Gravity systems cannot be easily rerouted"),
            SystemPriority(system_type="sanitary", priority_rank=2,
                          notes="Requires slope for drainage"),
            SystemPriority(system_type="storm_drain", priority_rank=3,
                          notes="Requires slope for drainage"),
            SystemPriority(system_type="plumbing_domestic", priority_rank=10,
                          notes="Pressurized, can route around obstacles"),
            SystemPriority(system_type="fire_protection", priority_rank=15,
                          notes="Code-driven locations"),
            SystemPriority(system_type="hvac_exhaust", priority_rank=20,
                          notes="Often has fixed endpoints"),
            SystemPriority(system_type="hvac_supply", priority_rank=25,
                          notes="Flexible routing"),
            SystemPriority(system_type="hvac_return", priority_rank=30,
                          notes="Most flexible"),
            SystemPriority(system_type="electrical_conduit", priority_rank=40,
                          notes="Very flexible routing"),
            SystemPriority(system_type="cable_tray", priority_rank=45,
                          notes="Flexible routing"),
        ]

    async def get_project_priorities(self, project_id: UUID) -> list[SystemPriority]:
        """Get system priorities for a project, falling back to defaults."""
        cache_key = str(project_id)
        if cache_key in self._priorities_cache:
            return self._priorities_cache[cache_key]

        if self._db_pool:
            try:
                async with self._db_pool.acquire() as conn:
                    rows = await conn.fetch("""
                        SELECT id, project_id, system_type, priority_rank, notes
                        FROM system_priorities
                        WHERE project_id = $1
                        ORDER BY priority_rank
                    """, project_id)

                    if rows:
                        priorities = [
                            SystemPriority(
                                id=row["id"],
                                project_id=row["project_id"],
                                system_type=row["system_type"],
                                priority_rank=row["priority_rank"],
                                notes=row["notes"],
                            )
                            for row in rows
                        ]
                        self._priorities_cache[cache_key] = priorities
                        return priorities
            except Exception as e:
                logger.warning(f"Failed to load project priorities: {e}")

        # Return defaults
        return self.get_default_priorities()

    def get_priority_rank(self, system_type: str) -> int:
        """Get priority rank for a system type (lower = higher priority)."""
        defaults = {p.system_type: p.priority_rank for p in self.get_default_priorities()}
        return defaults.get(system_type, 50)

    # =========================================================================
    # Rule Application
    # =========================================================================

    def get_applicable_rules(self, context: dict[str, Any]) -> list[DomainRule]:
        """
        Get all rules that apply to a given context.

        Args:
            context: Dict with keys like:
                - element_type: 'duct', 'pipe', etc.
                - subdomain: 'hvac', 'plumbing', etc.
                - action: 'create', 'route', 'validate'
                - near_elements: list of nearby element types

        Returns:
            List of applicable rules sorted by priority
        """
        applicable = []
        for rule in self._rules_cache.values():
            if rule.matches_context(context):
                applicable.append(rule)

        return sorted(applicable, key=lambda r: r.priority)

    def get_rule_action(self, rule: DomainRule, context: dict[str, Any]) -> dict[str, Any]:
        """
        Get the action parameters for a rule given context.

        Supports template substitution in action values.
        """
        action = rule.action.copy()

        # Substitute context values in action
        for key, value in action.items():
            if isinstance(value, str) and "{" in value:
                try:
                    action[key] = value.format(**context)
                except KeyError:
                    pass  # Keep original if substitution fails

        return action

    # =========================================================================
    # Rule Management
    # =========================================================================

    async def add_rule(self, rule: DomainRule) -> UUID:
        """Add a new rule to the knowledge base."""
        self._rules_cache[str(rule.id)] = rule

        if self._db_pool:
            try:
                async with self._db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO domain_rules
                            (id, domain, subdomain, rule_type, rule_name,
                             condition, action, priority, source, description)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        ON CONFLICT (domain, subdomain, rule_name) DO UPDATE SET
                            condition = EXCLUDED.condition,
                            action = EXCLUDED.action,
                            priority = EXCLUDED.priority,
                            description = EXCLUDED.description
                    """,
                        rule.id, rule.domain, rule.subdomain, rule.rule_type.value,
                        rule.rule_name, rule.condition, rule.action,
                        rule.priority, rule.source.value, rule.description
                    )
            except Exception as e:
                logger.error(f"Failed to save rule to DB: {e}")

        return rule.id

    async def seed_default_rules(self) -> int:
        """Seed database with default HVAC rules. Returns count of rules added."""
        from aec_agent.domain.seed_data import get_default_hvac_rules

        count = 0
        for rule in get_default_hvac_rules():
            await self.add_rule(rule)
            count += 1

        logger.info(f"Seeded {count} default HVAC rules")
        return count


# Global instance management
_knowledge_base: Optional[MEPKnowledgeBase] = None


async def get_knowledge_base(db_pool=None) -> MEPKnowledgeBase:
    """Get or create the global MEPKnowledgeBase instance."""
    global _knowledge_base

    if _knowledge_base is None:
        _knowledge_base = MEPKnowledgeBase(db_pool=db_pool)
        await _knowledge_base.initialize()

    return _knowledge_base


def reset_knowledge_base() -> None:
    """Reset the global knowledge base (for testing)."""
    global _knowledge_base
    _knowledge_base = None
