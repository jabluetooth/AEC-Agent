"""
Project Memory System for persistent context across sessions.

Stores and retrieves project facts, decisions, and conversation summaries
to reduce token usage by maintaining compressed context.
"""

import logging
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from aec_agent.memory.models import (
    ConversationSummary,
    FactSource,
    FactType,
    MemoryContext,
    ProjectFact,
)

logger = logging.getLogger(__name__)


class ProjectMemory:
    """
    Manages project memory including facts, decisions, and conversation summaries.

    Provides semantic search over stored facts using embeddings,
    and generates compact context for LLM injection.
    """

    def __init__(self, db_pool=None, embedding_service=None):
        """
        Initialize project memory.

        Args:
            db_pool: asyncpg connection pool
            embedding_service: Service for generating embeddings
        """
        self._db_pool = db_pool
        self._embedding_service = embedding_service
        self._cache: dict[str, list[ProjectFact]] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the memory system."""
        if self._initialized:
            return

        self._initialized = True
        logger.info("ProjectMemory initialized")

    # =========================================================================
    # Fact Management
    # =========================================================================

    async def store_fact(
        self,
        project_id: UUID,
        fact_type: FactType,
        key: str,
        value: Any,
        source: FactSource = FactSource.USER,
        confidence: float = 1.0,
        ttl_hours: int | None = None,
    ) -> ProjectFact:
        """
        Store a new fact about a project.

        Args:
            project_id: Project UUID
            fact_type: Type of fact (decision, constraint, preference, note)
            key: Unique key within project/type
            value: Fact value (can be dict, list, or primitive)
            source: Where the fact came from
            confidence: Confidence level 0.0-1.0
            ttl_hours: Optional time-to-live in hours

        Returns:
            Created ProjectFact
        """
        expires_at = None
        if ttl_hours:
            expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)

        fact = ProjectFact(
            project_id=project_id,
            fact_type=fact_type,
            key=key,
            value=value,
            source=source,
            confidence=confidence,
            expires_at=expires_at,
        )

        # Generate embedding for semantic search
        if self._embedding_service:
            try:
                text = f"{key}: {value}"
                fact.embedding = await self._embedding_service.embed_text(text)
            except Exception as e:
                logger.warning(f"Failed to generate fact embedding: {e}")

        # Store in database
        if self._db_pool:
            await self._save_fact_to_db(fact)
        else:
            # In-memory fallback
            cache_key = str(project_id)
            if cache_key not in self._cache:
                self._cache[cache_key] = []
            # Update existing or add new
            existing = next(
                (f for f in self._cache[cache_key]
                 if f.fact_type == fact_type and f.key == key),
                None
            )
            if existing:
                self._cache[cache_key].remove(existing)
            self._cache[cache_key].append(fact)

        logger.debug(f"Stored fact: {fact_type.value}/{key}", extra={
            "project_id": str(project_id),
            "source": source.value,
        })

        return fact

    async def get_facts(
        self,
        project_id: UUID,
        fact_type: FactType | None = None,
        include_expired: bool = False,
        min_confidence: float = 0.0,
        limit: int = 50,
    ) -> list[ProjectFact]:
        """
        Retrieve facts for a project.

        Args:
            project_id: Project UUID
            fact_type: Optional filter by type
            include_expired: Include expired facts
            min_confidence: Minimum confidence threshold
            limit: Max facts to return

        Returns:
            List of matching ProjectFacts
        """
        if self._db_pool:
            return await self._get_facts_from_db(
                project_id, fact_type, include_expired, min_confidence, limit
            )

        # In-memory fallback
        cache_key = str(project_id)
        facts = self._cache.get(cache_key, [])

        # Apply filters
        result = []
        for fact in facts:
            if fact_type and fact.fact_type != fact_type:
                continue
            if not include_expired and fact.is_expired:
                continue
            if fact.confidence < min_confidence:
                continue
            result.append(fact)

        return result[:limit]

    async def search_facts(
        self,
        project_id: UUID,
        query: str,
        limit: int = 10,
    ) -> list[ProjectFact]:
        """
        Semantic search over project facts.

        Args:
            project_id: Project UUID
            query: Search query
            limit: Max results

        Returns:
            List of matching facts ordered by relevance
        """
        if not self._embedding_service or not self._db_pool:
            # Fall back to keyword search
            return await self._keyword_search_facts(project_id, query, limit)

        try:
            query_embedding = await self._embedding_service.embed_text(query)

            async with self._db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT id, project_id, fact_type, key, value, source,
                           confidence, expires_at, created_at, updated_at,
                           1 - (embedding <=> $1::vector) as similarity
                    FROM project_facts
                    WHERE project_id = $2
                      AND embedding IS NOT NULL
                      AND (expires_at IS NULL OR expires_at > NOW())
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3
                """, query_embedding, project_id, limit)

                return [self._row_to_fact(row) for row in rows]

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return await self._keyword_search_facts(project_id, query, limit)

    async def delete_fact(
        self,
        project_id: UUID,
        fact_type: FactType,
        key: str,
    ) -> bool:
        """
        Delete a specific fact.

        Args:
            project_id: Project UUID
            fact_type: Fact type
            key: Fact key

        Returns:
            True if deleted, False if not found
        """
        if self._db_pool:
            async with self._db_pool.acquire() as conn:
                result = await conn.execute("""
                    DELETE FROM project_facts
                    WHERE project_id = $1 AND fact_type = $2 AND key = $3
                """, project_id, fact_type.value, key)
                return "DELETE 1" in result

        # In-memory fallback
        cache_key = str(project_id)
        if cache_key in self._cache:
            original_len = len(self._cache[cache_key])
            self._cache[cache_key] = [
                f for f in self._cache[cache_key]
                if not (f.fact_type == fact_type and f.key == key)
            ]
            return len(self._cache[cache_key]) < original_len

        return False

    # =========================================================================
    # Conversation Summaries
    # =========================================================================

    async def store_summary(
        self,
        project_id: UUID | None,
        user_id: str,
        user_session: str,
        summary: str,
        key_decisions: list[str],
        key_topics: list[str],
        message_count: int,
    ) -> ConversationSummary:
        """
        Store a conversation summary.

        Args:
            project_id: Optional project UUID
            user_id: Windows username
            user_session: Session identifier
            summary: Compressed summary text
            key_decisions: List of key decisions made
            key_topics: List of topics discussed
            message_count: Number of messages summarized

        Returns:
            Created ConversationSummary
        """
        conv_summary = ConversationSummary(
            project_id=project_id,
            user_id=user_id,
            user_session=user_session,
            summary=summary,
            key_decisions=key_decisions,
            key_topics=key_topics,
            message_count=message_count,
        )

        # Generate embedding
        if self._embedding_service:
            try:
                text = f"{summary} {' '.join(key_topics)}"
                conv_summary.embedding = await self._embedding_service.embed_text(text)
            except Exception as e:
                logger.warning(f"Failed to generate summary embedding: {e}")

        # Store in database
        if self._db_pool:
            await self._save_summary_to_db(conv_summary)

        logger.debug("Stored conversation summary", extra={
            "project_id": str(project_id) if project_id else None,
            "user_id": user_id,
            "message_count": message_count,
        })

        return conv_summary

    async def get_recent_summaries(
        self,
        project_id: UUID | None = None,
        user_id: str | None = None,
        limit: int = 5,
    ) -> list[ConversationSummary]:
        """
        Get recent conversation summaries.

        Args:
            project_id: Optional project filter
            user_id: Optional user filter
            limit: Max summaries to return

        Returns:
            List of recent summaries
        """
        if not self._db_pool:
            return []

        try:
            async with self._db_pool.acquire() as conn:
                if project_id and user_id:
                    rows = await conn.fetch("""
                        SELECT id, project_id, user_id, user_session, summary,
                               key_decisions, key_topics, message_count, created_at
                        FROM conversation_summaries
                        WHERE project_id = $1 AND user_id = $2
                        ORDER BY created_at DESC
                        LIMIT $3
                    """, project_id, user_id, limit)
                elif project_id:
                    rows = await conn.fetch("""
                        SELECT id, project_id, user_id, user_session, summary,
                               key_decisions, key_topics, message_count, created_at
                        FROM conversation_summaries
                        WHERE project_id = $1
                        ORDER BY created_at DESC
                        LIMIT $2
                    """, project_id, limit)
                elif user_id:
                    rows = await conn.fetch("""
                        SELECT id, project_id, user_id, user_session, summary,
                               key_decisions, key_topics, message_count, created_at
                        FROM conversation_summaries
                        WHERE user_id = $1
                        ORDER BY created_at DESC
                        LIMIT $2
                    """, user_id, limit)
                else:
                    rows = await conn.fetch("""
                        SELECT id, project_id, user_id, user_session, summary,
                               key_decisions, key_topics, message_count, created_at
                        FROM conversation_summaries
                        ORDER BY created_at DESC
                        LIMIT $1
                    """, limit)

                return [self._row_to_summary(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get summaries: {e}")
            return []

    # =========================================================================
    # Context Generation
    # =========================================================================

    async def get_context(
        self,
        project_id: UUID,
        user_id: str | None = None,
        query: str | None = None,
        token_budget: int = 500,
    ) -> MemoryContext:
        """
        Generate aggregated memory context for LLM injection.

        Args:
            project_id: Project UUID
            user_id: Optional user for personalization
            query: Optional query for relevant fact selection
            token_budget: Max tokens for context

        Returns:
            MemoryContext with relevant facts and summaries
        """
        context = MemoryContext(token_budget=token_budget)

        # Get high-priority facts (decisions, constraints)
        high_priority = await self.get_facts(
            project_id,
            min_confidence=0.7,
            limit=20,
        )

        # If query provided, also get semantically relevant facts
        if query:
            relevant = await self.search_facts(project_id, query, limit=5)
            # Merge without duplicates
            seen_ids = {f.id for f in high_priority}
            for fact in relevant:
                if fact.id not in seen_ids:
                    high_priority.append(fact)

        context.facts = high_priority

        # Get recent conversation summaries
        context.recent_summaries = await self.get_recent_summaries(
            project_id=project_id,
            user_id=user_id,
            limit=2,
        )

        # Trim to token budget if needed
        while context.estimated_tokens > token_budget and context.facts:
            context.facts.pop()

        return context

    # =========================================================================
    # Database Operations
    # =========================================================================

    async def _save_fact_to_db(self, fact: ProjectFact) -> None:
        """Save fact to PostgreSQL."""
        try:
            async with self._db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO project_facts
                        (id, project_id, fact_type, key, value, source,
                         confidence, expires_at, embedding, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (project_id, fact_type, key) DO UPDATE SET
                        value = EXCLUDED.value,
                        source = EXCLUDED.source,
                        confidence = EXCLUDED.confidence,
                        expires_at = EXCLUDED.expires_at,
                        embedding = EXCLUDED.embedding,
                        updated_at = NOW()
                """,
                    fact.id, fact.project_id, fact.fact_type.value, fact.key,
                    fact.value if isinstance(fact.value, dict) else {"value": fact.value},
                    fact.source.value, fact.confidence, fact.expires_at,
                    fact.embedding, fact.created_at, fact.updated_at
                )
        except Exception as e:
            logger.error(f"Failed to save fact: {e}")

    async def _get_facts_from_db(
        self,
        project_id: UUID,
        fact_type: FactType | None,
        include_expired: bool,
        min_confidence: float,
        limit: int,
    ) -> list[ProjectFact]:
        """Get facts from PostgreSQL."""
        try:
            async with self._db_pool.acquire() as conn:
                if fact_type:
                    if include_expired:
                        rows = await conn.fetch("""
                            SELECT id, project_id, fact_type, key, value, source,
                                   confidence, expires_at, created_at, updated_at
                            FROM project_facts
                            WHERE project_id = $1 AND fact_type = $2
                              AND confidence >= $3
                            ORDER BY confidence DESC, created_at DESC
                            LIMIT $4
                        """, project_id, fact_type.value, min_confidence, limit)
                    else:
                        rows = await conn.fetch("""
                            SELECT id, project_id, fact_type, key, value, source,
                                   confidence, expires_at, created_at, updated_at
                            FROM project_facts
                            WHERE project_id = $1 AND fact_type = $2
                              AND confidence >= $3
                              AND (expires_at IS NULL OR expires_at > NOW())
                            ORDER BY confidence DESC, created_at DESC
                            LIMIT $4
                        """, project_id, fact_type.value, min_confidence, limit)
                else:
                    if include_expired:
                        rows = await conn.fetch("""
                            SELECT id, project_id, fact_type, key, value, source,
                                   confidence, expires_at, created_at, updated_at
                            FROM project_facts
                            WHERE project_id = $1 AND confidence >= $2
                            ORDER BY confidence DESC, created_at DESC
                            LIMIT $3
                        """, project_id, min_confidence, limit)
                    else:
                        rows = await conn.fetch("""
                            SELECT id, project_id, fact_type, key, value, source,
                                   confidence, expires_at, created_at, updated_at
                            FROM project_facts
                            WHERE project_id = $1 AND confidence >= $2
                              AND (expires_at IS NULL OR expires_at > NOW())
                            ORDER BY confidence DESC, created_at DESC
                            LIMIT $3
                        """, project_id, min_confidence, limit)

                return [self._row_to_fact(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get facts: {e}")
            return []

    async def _keyword_search_facts(
        self,
        project_id: UUID,
        query: str,
        limit: int,
    ) -> list[ProjectFact]:
        """Fallback keyword search for facts."""
        if not self._db_pool:
            # In-memory search
            cache_key = str(project_id)
            facts = self._cache.get(cache_key, [])
            query_lower = query.lower()
            matches = [
                f for f in facts
                if query_lower in f.key.lower() or query_lower in str(f.value).lower()
            ]
            return matches[:limit]

        try:
            async with self._db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT id, project_id, fact_type, key, value, source,
                           confidence, expires_at, created_at, updated_at
                    FROM project_facts
                    WHERE project_id = $1
                      AND (expires_at IS NULL OR expires_at > NOW())
                      AND (key ILIKE $2 OR value::text ILIKE $2)
                    ORDER BY confidence DESC
                    LIMIT $3
                """, project_id, f"%{query}%", limit)

                return [self._row_to_fact(row) for row in rows]

        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []

    async def _save_summary_to_db(self, summary: ConversationSummary) -> None:
        """Save conversation summary to PostgreSQL."""
        try:
            async with self._db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO conversation_summaries
                        (id, project_id, user_id, user_session, summary,
                         key_decisions, key_topics, embedding, message_count, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                    summary.id, summary.project_id, summary.user_id,
                    summary.user_session, summary.summary, summary.key_decisions,
                    summary.key_topics, summary.embedding, summary.message_count,
                    summary.created_at
                )
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")

    def _row_to_fact(self, row) -> ProjectFact:
        """Convert database row to ProjectFact."""
        value = row["value"]
        if isinstance(value, dict) and "value" in value and len(value) == 1:
            value = value["value"]

        return ProjectFact(
            id=row["id"],
            project_id=row["project_id"],
            fact_type=FactType(row["fact_type"]),
            key=row["key"],
            value=value,
            source=FactSource(row["source"]),
            confidence=row["confidence"],
            expires_at=row["expires_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_summary(self, row) -> ConversationSummary:
        """Convert database row to ConversationSummary."""
        return ConversationSummary(
            id=row["id"],
            project_id=row["project_id"],
            user_id=row["user_id"],
            user_session=row["user_session"],
            summary=row["summary"],
            key_decisions=row["key_decisions"] or [],
            key_topics=row["key_topics"] or [],
            message_count=row["message_count"],
            created_at=row["created_at"],
        )


# Global instance management
_memory: ProjectMemory | None = None


async def get_project_memory(
    db_pool=None,
    embedding_service=None,
) -> ProjectMemory:
    """Get or create the global ProjectMemory instance."""
    global _memory

    if _memory is None:
        _memory = ProjectMemory(db_pool=db_pool, embedding_service=embedding_service)
        await _memory.initialize()

    return _memory


def reset_project_memory() -> None:
    """Reset the global memory (for testing)."""
    global _memory
    _memory = None
