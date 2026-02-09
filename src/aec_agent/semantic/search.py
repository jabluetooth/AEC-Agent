"""
Semantic search for CAD elements.

Combines vector similarity search (pgvector) with spatial queries (PostGIS)
to resolve natural language element references.
"""

from uuid import UUID

import structlog

from aec_agent.db.connection import DatabasePool
from aec_agent.db.models import CentroidInfo, Element
from aec_agent.db.repository import ElementRepository
from aec_agent.semantic.embeddings import EmbeddingService

logger = structlog.get_logger(__name__)


class SemanticSearch:
    """
    Semantic search service for CAD elements.

    Provides:
    - Vector similarity search using pgvector
    - Combined semantic + spatial search
    - Element resolution from natural language
    """

    def __init__(
        self,
        pool: DatabasePool,
        embedding_service: EmbeddingService,
    ):
        """
        Initialize semantic search.

        Args:
            pool: Database connection pool
            embedding_service: Embedding generation service
        """
        self._pool = pool
        self._embeddings = embedding_service
        self._repository = ElementRepository(pool)

    async def search(
        self,
        query: str,
        project_id: UUID | None = None,
        entity_type: str | None = None,
        category: str | None = None,
        layer: str | None = None,
        limit: int = 10,
    ) -> list[Element]:
        """
        Semantic search for elements.

        Args:
            query: Natural language search query
            project_id: Optional project filter
            entity_type: Optional entity type filter
            category: Optional category filter (Revit)
            layer: Optional layer filter (AutoCAD)
            limit: Maximum results

        Returns:
            List of matching elements ordered by similarity
        """
        # Generate query embedding
        query_embedding = self._embeddings.generate_embedding(query)
        if not query_embedding:
            logger.warning("Could not generate query embedding", query=query)
            # Fall back to text search
            return await self._repository.search_elements_text(
                query, project_id, limit
            )

        return await self._repository.search_elements_semantic(
            query_embedding=query_embedding,
            project_id=project_id,
            entity_type=entity_type,
            category=category,
            layer=layer,
            limit=limit,
        )

    async def search_with_spatial_filter(
        self,
        query: str,
        project_id: UUID,
        near_element_id: UUID | None = None,
        near_point: CentroidInfo | None = None,
        within_distance: float = 10.0,
        limit: int = 10,
    ) -> list[Element]:
        """
        Combined semantic + spatial search.

        Args:
            query: Natural language search query
            project_id: Project to search
            near_element_id: Optional element to search near
            near_point: Optional point to search near
            within_distance: Maximum distance in meters
            limit: Maximum results

        Returns:
            List of matching elements
        """
        # Generate query embedding
        query_embedding = self._embeddings.generate_embedding(query)
        if not query_embedding:
            return []

        # Build combined query
        if near_element_id:
            # Get reference element centroid
            ref_element = await self._repository.get_element(near_element_id)
            if ref_element and ref_element.centroid:
                near_point = ref_element.centroid
            else:
                logger.warning(
                    "Reference element has no centroid",
                    element_id=str(near_element_id)
                )

        if near_point:
            query = f"""
                WITH ref AS (
                    SELECT ST_GeomFromText('{near_point.to_wkt_point()}', 0) as point
                )
                SELECT
                    e.id, e.project_id, e.source_id, e.source,
                    e.entity_type, e.layer, e.category, e.family, e.type_name,
                    ST_AsText(e.geom) as geom_wkt,
                    ST_X(e.centroid) as centroid_x,
                    ST_Y(e.centroid) as centroid_y,
                    ST_Z(e.centroid) as centroid_z,
                    e.properties, e.description, e.embedding,
                    e.deleted_at, e.created_at, e.updated_at,
                    e.embedding <=> $1 as semantic_distance,
                    ST_3DDistance(e.centroid, ref.point) as spatial_distance
                FROM elements e, ref
                WHERE e.project_id = $2
                  AND e.deleted_at IS NULL
                  AND e.embedding IS NOT NULL
                  AND ST_3DDWithin(e.centroid, ref.point, $3)
                ORDER BY e.embedding <=> $1
                LIMIT $4
            """
            rows = await self._pool.fetch(
                query, query_embedding, project_id, within_distance, limit
            )
        else:
            # Just semantic search without spatial filter
            return await self.search(query, project_id, limit=limit)

        return [self._repository._row_to_element(dict(row)) for row in rows]

    async def resolve_element(
        self,
        reference: str,
        project_id: UUID,
        context_element_id: UUID | None = None,
    ) -> Element | None:
        """
        Resolve a natural language element reference.

        This is the core function for smart tools - it takes a description
        like "the east wall" or "column C3" and returns the specific element.

        Args:
            reference: Element reference (ID or description)
            project_id: Project to search in
            context_element_id: Optional context element for relative references

        Returns:
            Resolved element or None
        """
        # First, check if it's a direct ID reference
        element = await self._try_resolve_by_id(reference, project_id)
        if element:
            return element

        # Use semantic search
        candidates = await self.search(
            query=reference,
            project_id=project_id,
            limit=5,
        )

        if not candidates:
            logger.debug("No candidates found", reference=reference)
            return None

        # If context element provided, prefer nearby elements
        if context_element_id:
            context = await self._repository.get_element(context_element_id)
            if context and context.centroid:
                # Sort by proximity to context
                nearby = await self._repository.get_nearby_elements(
                    context_element_id, distance=20.0, limit=50
                )
                nearby_ids = {e.id for e in nearby}

                # Prefer candidates that are nearby
                for candidate in candidates:
                    if candidate.id in nearby_ids:
                        return candidate

        # Return top semantic match
        return candidates[0] if candidates else None

    async def _try_resolve_by_id(
        self,
        reference: str,
        project_id: UUID,
    ) -> Element | None:
        """Try to resolve reference as a direct ID."""
        # Clean reference
        ref = reference.strip()

        # Try as source_id (Handle or ElementId)
        element = await self._repository.get_element_by_source_id(
            project_id, ref
        )
        if element:
            return element

        # Try numeric ID (for Revit ElementId)
        try:
            numeric_ref = str(int(ref))
            element = await self._repository.get_element_by_source_id(
                project_id, numeric_ref, "revit"
            )
            if element:
                return element
        except ValueError:
            pass

        # Try as UUID
        try:
            from uuid import UUID as UUIDType
            element_uuid = UUIDType(ref)
            element = await self._repository.get_element(element_uuid)
            if element and element.project_id == project_id:
                return element
        except ValueError:
            pass

        return None


async def resolve_element(
    reference: str,
    project_id: UUID,
    pool: DatabasePool,
    embedding_service: EmbeddingService,
    context_element_id: UUID | None = None,
) -> Element | None:
    """
    Convenience function to resolve an element reference.

    Args:
        reference: Element reference (ID or description)
        project_id: Project to search in
        pool: Database connection pool
        embedding_service: Embedding service
        context_element_id: Optional context element

    Returns:
        Resolved element or None
    """
    search = SemanticSearch(pool, embedding_service)
    return await search.resolve_element(reference, project_id, context_element_id)
