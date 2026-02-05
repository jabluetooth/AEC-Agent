"""
Repository layer for database operations.

Provides CRUD operations for projects, elements, and relationships
with PostGIS spatial and pgvector semantic queries.
"""

from datetime import datetime
from typing import Optional, List, Literal
from uuid import UUID

import structlog

from aec_agent.db.connection import DatabasePool
from aec_agent.db.models import (
    Project,
    Element,
    ElementRelationship,
    BoundsInfo,
    CentroidInfo,
)

logger = structlog.get_logger(__name__)


class ElementRepository:
    """
    Repository for element and project operations.

    Provides methods for:
    - Project CRUD
    - Element CRUD with batch operations
    - Spatial queries (PostGIS)
    - Semantic search (pgvector)
    - Relationship management
    """

    def __init__(self, pool: DatabasePool):
        """
        Initialize repository.

        Args:
            pool: Database connection pool
        """
        self._pool = pool

    # =========================================================================
    # Project Operations
    # =========================================================================

    async def create_project(self, project: Project) -> UUID:
        """
        Create a new project.

        Args:
            project: Project to create

        Returns:
            Created project ID
        """
        query = """
            INSERT INTO projects (id, name, source, file_path, file_hash, metadata)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id
        """
        result = await self._pool.fetchval(
            query,
            project.id,
            project.name,
            project.source,
            project.file_path,
            project.file_hash,
            project.metadata,
        )
        logger.info("Created project", project_id=str(result), name=project.name)
        return result

    async def get_project(self, project_id: UUID) -> Optional[Project]:
        """Get project by ID."""
        query = "SELECT * FROM projects WHERE id = $1"
        row = await self._pool.fetchrow(query, project_id)
        if row:
            return Project(**dict(row))
        return None

    async def get_project_by_file_path(self, file_path: str) -> Optional[Project]:
        """Get project by file path."""
        query = "SELECT * FROM projects WHERE file_path = $1"
        row = await self._pool.fetchrow(query, file_path)
        if row:
            return Project(**dict(row))
        return None

    async def get_project_by_file_hash(self, file_hash: str) -> Optional[Project]:
        """Get project by file hash."""
        query = "SELECT * FROM projects WHERE file_hash = $1"
        row = await self._pool.fetchrow(query, file_hash)
        if row:
            return Project(**dict(row))
        return None

    async def update_project_extraction_time(self, project_id: UUID) -> None:
        """Update project's last extraction timestamp."""
        query = """
            UPDATE projects
            SET extracted_at = NOW(), updated_at = NOW()
            WHERE id = $1
        """
        await self._pool.execute(query, project_id)

    async def update_project_hash(self, project_id: UUID, file_hash: str) -> None:
        """Update project's file hash."""
        query = """
            UPDATE projects
            SET file_hash = $1, updated_at = NOW()
            WHERE id = $1
        """
        await self._pool.execute(query, project_id, file_hash)

    async def delete_project(self, project_id: UUID) -> bool:
        """Delete project and all its elements (cascade)."""
        query = "DELETE FROM projects WHERE id = $1"
        result = await self._pool.execute(query, project_id)
        deleted = result == "DELETE 1"
        if deleted:
            logger.info("Deleted project", project_id=str(project_id))
        return deleted

    # =========================================================================
    # Element Operations
    # =========================================================================

    async def upsert_element(self, element: Element) -> UUID:
        """
        Insert or update an element.

        Args:
            element: Element to upsert

        Returns:
            Element ID
        """
        # Convert geometry to PostGIS format
        geom_sql = f"ST_GeomFromText('{element.geom_wkt}', 0)" if element.geom_wkt else "NULL"
        centroid_sql = (
            f"ST_GeomFromText('{element.centroid.to_wkt_point()}', 0)"
            if element.centroid else "NULL"
        )

        query = f"""
            INSERT INTO elements (
                id, project_id, source_id, source,
                entity_type, layer, category, family, type_name,
                geom, centroid, properties, description, embedding
            )
            VALUES (
                $1, $2, $3, $4,
                $5, $6, $7, $8, $9,
                {geom_sql}, {centroid_sql}, $10, $11, $12
            )
            ON CONFLICT (project_id, source_id, source)
            DO UPDATE SET
                entity_type = EXCLUDED.entity_type,
                layer = EXCLUDED.layer,
                category = EXCLUDED.category,
                family = EXCLUDED.family,
                type_name = EXCLUDED.type_name,
                geom = EXCLUDED.geom,
                centroid = EXCLUDED.centroid,
                properties = EXCLUDED.properties,
                description = EXCLUDED.description,
                embedding = EXCLUDED.embedding,
                deleted_at = NULL,
                updated_at = NOW()
            RETURNING id
        """

        result = await self._pool.fetchval(
            query,
            element.id,
            element.project_id,
            element.source_id,
            element.source,
            element.entity_type,
            element.layer,
            element.category,
            element.family,
            element.type_name,
            element.properties,
            element.description,
            element.embedding,
        )
        return result

    async def upsert_elements_batch(self, elements: List[Element]) -> int:
        """
        Batch upsert elements for performance.

        Args:
            elements: List of elements to upsert

        Returns:
            Number of elements processed
        """
        if not elements:
            return 0

        count = 0
        async with self._pool.transaction() as conn:
            for element in elements:
                await self.upsert_element(element)
                count += 1

        logger.info("Batch upserted elements", count=count)
        return count

    async def get_element(self, element_id: UUID) -> Optional[Element]:
        """Get element by ID."""
        query = """
            SELECT
                id, project_id, source_id, source,
                entity_type, layer, category, family, type_name,
                ST_AsText(geom) as geom_wkt,
                ST_X(centroid) as centroid_x,
                ST_Y(centroid) as centroid_y,
                ST_Z(centroid) as centroid_z,
                properties, description, embedding,
                deleted_at, created_at, updated_at
            FROM elements
            WHERE id = $1 AND deleted_at IS NULL
        """
        row = await self._pool.fetchrow(query, element_id)
        if row:
            return self._row_to_element(dict(row))
        return None

    async def get_element_by_source_id(
        self,
        project_id: UUID,
        source_id: str,
        source: str = None
    ) -> Optional[Element]:
        """Get element by source ID (Handle or ElementId)."""
        if source:
            query = """
                SELECT
                    id, project_id, source_id, source,
                    entity_type, layer, category, family, type_name,
                    ST_AsText(geom) as geom_wkt,
                    ST_X(centroid) as centroid_x,
                    ST_Y(centroid) as centroid_y,
                    ST_Z(centroid) as centroid_z,
                    properties, description, embedding,
                    deleted_at, created_at, updated_at
                FROM elements
                WHERE project_id = $1 AND source_id = $2 AND source = $3 AND deleted_at IS NULL
            """
            row = await self._pool.fetchrow(query, project_id, source_id, source)
        else:
            query = """
                SELECT
                    id, project_id, source_id, source,
                    entity_type, layer, category, family, type_name,
                    ST_AsText(geom) as geom_wkt,
                    ST_X(centroid) as centroid_x,
                    ST_Y(centroid) as centroid_y,
                    ST_Z(centroid) as centroid_z,
                    properties, description, embedding,
                    deleted_at, created_at, updated_at
                FROM elements
                WHERE project_id = $1 AND source_id = $2 AND deleted_at IS NULL
            """
            row = await self._pool.fetchrow(query, project_id, source_id)

        if row:
            return self._row_to_element(dict(row))
        return None

    async def get_elements_by_type(
        self,
        project_id: UUID,
        entity_type: str,
        limit: int = 100
    ) -> List[Element]:
        """Get elements by entity type."""
        query = """
            SELECT
                id, project_id, source_id, source,
                entity_type, layer, category, family, type_name,
                ST_AsText(geom) as geom_wkt,
                ST_X(centroid) as centroid_x,
                ST_Y(centroid) as centroid_y,
                ST_Z(centroid) as centroid_z,
                properties, description, embedding,
                deleted_at, created_at, updated_at
            FROM elements
            WHERE project_id = $1 AND entity_type = $2 AND deleted_at IS NULL
            LIMIT $3
        """
        rows = await self._pool.fetch(query, project_id, entity_type, limit)
        return [self._row_to_element(dict(row)) for row in rows]

    async def soft_delete_element(self, element_id: UUID) -> bool:
        """Soft delete an element."""
        query = """
            UPDATE elements
            SET deleted_at = NOW(), updated_at = NOW()
            WHERE id = $1 AND deleted_at IS NULL
        """
        result = await self._pool.execute(query, element_id)
        return result == "UPDATE 1"

    async def delete_project_elements(self, project_id: UUID) -> int:
        """Delete all elements for a project."""
        query = "DELETE FROM elements WHERE project_id = $1"
        result = await self._pool.execute(query, project_id)
        # Parse "DELETE n" to get count
        count = int(result.split()[-1]) if result.startswith("DELETE") else 0
        logger.info("Deleted project elements", project_id=str(project_id), count=count)
        return count

    # =========================================================================
    # Spatial Queries (PostGIS)
    # =========================================================================

    async def get_nearby_elements(
        self,
        element_id: UUID,
        distance: float = 1.0,
        limit: int = 50
    ) -> List[Element]:
        """
        Find elements near a given element.

        Args:
            element_id: Reference element
            distance: Search radius in meters
            limit: Maximum results

        Returns:
            List of nearby elements with distance
        """
        query = """
            WITH ref AS (
                SELECT project_id, centroid FROM elements WHERE id = $1
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
                ST_3DDistance(e.centroid, ref.centroid) as distance
            FROM elements e, ref
            WHERE e.project_id = ref.project_id
              AND e.id != $1
              AND e.deleted_at IS NULL
              AND ST_3DDWithin(e.centroid, ref.centroid, $2)
            ORDER BY distance
            LIMIT $3
        """
        rows = await self._pool.fetch(query, element_id, distance, limit)
        return [self._row_to_element(dict(row)) for row in rows]

    async def get_elements_in_bbox(
        self,
        project_id: UUID,
        bounds: BoundsInfo,
        limit: int = 100
    ) -> List[Element]:
        """
        Find elements within a bounding box.

        Args:
            project_id: Project to search
            bounds: Bounding box
            limit: Maximum results

        Returns:
            List of elements in the bbox
        """
        bbox_wkt = bounds.to_wkt_polygon()
        query = f"""
            SELECT
                id, project_id, source_id, source,
                entity_type, layer, category, family, type_name,
                ST_AsText(geom) as geom_wkt,
                ST_X(centroid) as centroid_x,
                ST_Y(centroid) as centroid_y,
                ST_Z(centroid) as centroid_z,
                properties, description, embedding,
                deleted_at, created_at, updated_at
            FROM elements
            WHERE project_id = $1
              AND deleted_at IS NULL
              AND ST_Intersects(centroid, ST_GeomFromText('{bbox_wkt}', 0))
            LIMIT $2
        """
        rows = await self._pool.fetch(query, project_id, limit)
        return [self._row_to_element(dict(row)) for row in rows]

    async def get_intersecting_elements(
        self,
        element_id: UUID,
        limit: int = 50
    ) -> List[Element]:
        """Find elements that intersect with a given element."""
        query = """
            WITH ref AS (
                SELECT project_id, geom FROM elements WHERE id = $1
            )
            SELECT
                e.id, e.project_id, e.source_id, e.source,
                e.entity_type, e.layer, e.category, e.family, e.type_name,
                ST_AsText(e.geom) as geom_wkt,
                ST_X(e.centroid) as centroid_x,
                ST_Y(e.centroid) as centroid_y,
                ST_Z(e.centroid) as centroid_z,
                e.properties, e.description, e.embedding,
                e.deleted_at, e.created_at, e.updated_at
            FROM elements e, ref
            WHERE e.project_id = ref.project_id
              AND e.id != $1
              AND e.deleted_at IS NULL
              AND ST_Intersects(e.geom, ref.geom)
            LIMIT $2
        """
        rows = await self._pool.fetch(query, element_id, limit)
        return [self._row_to_element(dict(row)) for row in rows]

    # =========================================================================
    # Semantic Search (pgvector)
    # =========================================================================

    async def search_elements_semantic(
        self,
        query_embedding: List[float],
        project_id: Optional[UUID] = None,
        entity_type: Optional[str] = None,
        category: Optional[str] = None,
        layer: Optional[str] = None,
        limit: int = 10
    ) -> List[Element]:
        """
        Semantic search for elements using vector similarity.

        Args:
            query_embedding: Query vector from sentence-transformers
            project_id: Optional project filter
            entity_type: Optional entity type filter
            category: Optional category filter (Revit)
            layer: Optional layer filter (AutoCAD)
            limit: Maximum results

        Returns:
            List of elements ordered by similarity
        """
        # Build dynamic WHERE clause
        conditions = ["deleted_at IS NULL", "embedding IS NOT NULL"]
        params = [query_embedding]
        param_idx = 2

        if project_id:
            conditions.append(f"project_id = ${param_idx}")
            params.append(project_id)
            param_idx += 1

        if entity_type:
            conditions.append(f"entity_type = ${param_idx}")
            params.append(entity_type)
            param_idx += 1

        if category:
            conditions.append(f"category = ${param_idx}")
            params.append(category)
            param_idx += 1

        if layer:
            conditions.append(f"layer = ${param_idx}")
            params.append(layer)
            param_idx += 1

        params.append(limit)

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT
                id, project_id, source_id, source,
                entity_type, layer, category, family, type_name,
                ST_AsText(geom) as geom_wkt,
                ST_X(centroid) as centroid_x,
                ST_Y(centroid) as centroid_y,
                ST_Z(centroid) as centroid_z,
                properties, description, embedding,
                deleted_at, created_at, updated_at,
                embedding <=> $1 as distance
            FROM elements
            WHERE {where_clause}
            ORDER BY embedding <=> $1
            LIMIT ${param_idx}
        """

        rows = await self._pool.fetch(query, *params)
        return [self._row_to_element(dict(row)) for row in rows]

    async def search_elements_text(
        self,
        search_text: str,
        project_id: Optional[UUID] = None,
        limit: int = 10
    ) -> List[Element]:
        """
        Text search in element descriptions.

        Args:
            search_text: Text to search for
            project_id: Optional project filter
            limit: Maximum results

        Returns:
            List of matching elements
        """
        search_pattern = f"%{search_text.lower()}%"

        if project_id:
            query = """
                SELECT
                    id, project_id, source_id, source,
                    entity_type, layer, category, family, type_name,
                    ST_AsText(geom) as geom_wkt,
                    ST_X(centroid) as centroid_x,
                    ST_Y(centroid) as centroid_y,
                    ST_Z(centroid) as centroid_z,
                    properties, description, embedding,
                    deleted_at, created_at, updated_at
                FROM elements
                WHERE project_id = $1
                  AND deleted_at IS NULL
                  AND (
                      LOWER(description) LIKE $2
                      OR LOWER(entity_type) LIKE $2
                      OR LOWER(layer) LIKE $2
                      OR LOWER(category) LIKE $2
                      OR LOWER(family) LIKE $2
                  )
                LIMIT $3
            """
            rows = await self._pool.fetch(query, project_id, search_pattern, limit)
        else:
            query = """
                SELECT
                    id, project_id, source_id, source,
                    entity_type, layer, category, family, type_name,
                    ST_AsText(geom) as geom_wkt,
                    ST_X(centroid) as centroid_x,
                    ST_Y(centroid) as centroid_y,
                    ST_Z(centroid) as centroid_z,
                    properties, description, embedding,
                    deleted_at, created_at, updated_at
                FROM elements
                WHERE deleted_at IS NULL
                  AND (
                      LOWER(description) LIKE $1
                      OR LOWER(entity_type) LIKE $1
                      OR LOWER(layer) LIKE $1
                      OR LOWER(category) LIKE $1
                      OR LOWER(family) LIKE $1
                  )
                LIMIT $2
            """
            rows = await self._pool.fetch(query, search_pattern, limit)

        return [self._row_to_element(dict(row)) for row in rows]

    # =========================================================================
    # Relationship Operations
    # =========================================================================

    async def create_relationship(self, relationship: ElementRelationship) -> UUID:
        """Create a relationship between elements."""
        query = """
            INSERT INTO element_relationships (
                id, project_id, from_element_id, to_element_id,
                relation_type, distance, confidence, source, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (project_id, from_element_id, to_element_id, relation_type)
            DO UPDATE SET
                distance = EXCLUDED.distance,
                confidence = EXCLUDED.confidence,
                metadata = EXCLUDED.metadata
            RETURNING id
        """
        return await self._pool.fetchval(
            query,
            relationship.id,
            relationship.project_id,
            relationship.from_element_id,
            relationship.to_element_id,
            relationship.relation_type,
            relationship.distance,
            relationship.confidence,
            relationship.source,
            relationship.metadata,
        )

    async def get_related_elements(
        self,
        element_id: UUID,
        relation_types: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[Element]:
        """
        Get elements related to a given element.

        Args:
            element_id: Source element
            relation_types: Optional filter for relationship types
            limit: Maximum results

        Returns:
            List of related elements
        """
        if relation_types:
            query = """
                SELECT DISTINCT
                    e.id, e.project_id, e.source_id, e.source,
                    e.entity_type, e.layer, e.category, e.family, e.type_name,
                    ST_AsText(e.geom) as geom_wkt,
                    ST_X(e.centroid) as centroid_x,
                    ST_Y(e.centroid) as centroid_y,
                    ST_Z(e.centroid) as centroid_z,
                    e.properties, e.description, e.embedding,
                    e.deleted_at, e.created_at, e.updated_at,
                    r.relation_type, r.distance
                FROM element_relationships r
                JOIN elements e ON (
                    (r.from_element_id = $1 AND r.to_element_id = e.id)
                    OR (r.to_element_id = $1 AND r.from_element_id = e.id)
                )
                WHERE r.relation_type = ANY($2)
                  AND e.deleted_at IS NULL
                LIMIT $3
            """
            rows = await self._pool.fetch(query, element_id, relation_types, limit)
        else:
            query = """
                SELECT DISTINCT
                    e.id, e.project_id, e.source_id, e.source,
                    e.entity_type, e.layer, e.category, e.family, e.type_name,
                    ST_AsText(e.geom) as geom_wkt,
                    ST_X(e.centroid) as centroid_x,
                    ST_Y(e.centroid) as centroid_y,
                    ST_Z(e.centroid) as centroid_z,
                    e.properties, e.description, e.embedding,
                    e.deleted_at, e.created_at, e.updated_at,
                    r.relation_type, r.distance
                FROM element_relationships r
                JOIN elements e ON (
                    (r.from_element_id = $1 AND r.to_element_id = e.id)
                    OR (r.to_element_id = $1 AND r.from_element_id = e.id)
                )
                WHERE e.deleted_at IS NULL
                LIMIT $2
            """
            rows = await self._pool.fetch(query, element_id, limit)

        return [self._row_to_element(dict(row)) for row in rows]

    async def delete_project_relationships(self, project_id: UUID) -> int:
        """Delete all relationships for a project."""
        query = "DELETE FROM element_relationships WHERE project_id = $1"
        result = await self._pool.execute(query, project_id)
        count = int(result.split()[-1]) if result.startswith("DELETE") else 0
        return count

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _row_to_element(self, row: dict) -> Element:
        """Convert database row to Element model."""
        # Build centroid if coordinates are present
        centroid = None
        if row.get("centroid_x") is not None:
            centroid = CentroidInfo(
                x=row["centroid_x"],
                y=row["centroid_y"],
                z=row.get("centroid_z", 0.0) or 0.0,
            )

        return Element(
            id=row["id"],
            project_id=row["project_id"],
            source_id=row["source_id"],
            source=row["source"],
            entity_type=row["entity_type"],
            layer=row.get("layer"),
            category=row.get("category"),
            family=row.get("family"),
            type_name=row.get("type_name"),
            geom_wkt=row.get("geom_wkt"),
            centroid=centroid,
            properties=row.get("properties", {}),
            description=row.get("description"),
            embedding=row.get("embedding"),
            deleted_at=row.get("deleted_at"),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )
