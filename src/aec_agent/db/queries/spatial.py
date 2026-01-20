"""
PostGIS spatial queries for AEC Agent.

Provides optimized SQL queries for spatial operations using PostGIS indexes.
Uses SQL instead of Python loops for O(log n) performance with spatial indexes.
"""

from typing import List, Optional
from uuid import UUID

import structlog

from aec_agent.db.connection import DatabasePool
from aec_agent.db.models import Element, CentroidInfo

logger = structlog.get_logger(__name__)


async def compute_intersecting_elements(
    pool: DatabasePool,
    project_id: UUID,
) -> int:
    """
    Compute all intersecting element pairs and store as relationships.

    Uses PostGIS ST_Intersects with GIST index for performance.

    Args:
        pool: Database connection pool
        project_id: Project to process

    Returns:
        Number of relationships created
    """
    query = """
        INSERT INTO element_relationships (
            project_id, from_element_id, to_element_id, relation_type, source
        )
        SELECT DISTINCT
            e1.project_id,
            e1.id,
            e2.id,
            'intersects',
            'computed'
        FROM elements e1
        JOIN elements e2 ON ST_Intersects(e1.geom, e2.geom)
        WHERE e1.project_id = $1
          AND e2.project_id = $1
          AND e1.id < e2.id  -- Avoid duplicates (A,B) and (B,A)
          AND e1.geom IS NOT NULL
          AND e2.geom IS NOT NULL
          AND e1.deleted_at IS NULL
          AND e2.deleted_at IS NULL
        ON CONFLICT (project_id, from_element_id, to_element_id, relation_type)
        DO NOTHING
    """

    result = await pool.execute(query, project_id)
    count = int(result.split()[-1]) if "INSERT" in result else 0
    logger.info("Computed intersecting elements", project_id=str(project_id), count=count)
    return count


async def compute_nearby_elements(
    pool: DatabasePool,
    project_id: UUID,
    distance_threshold: float = 1.0,
) -> int:
    """
    Compute nearby element pairs within distance threshold.

    Uses PostGIS ST_3DDWithin with GIST index for performance.

    Args:
        pool: Database connection pool
        project_id: Project to process
        distance_threshold: Maximum distance in meters

    Returns:
        Number of relationships created
    """
    query = """
        INSERT INTO element_relationships (
            project_id, from_element_id, to_element_id, relation_type, distance, source
        )
        SELECT
            e1.project_id,
            e1.id,
            e2.id,
            'near',
            ST_3DDistance(e1.centroid, e2.centroid),
            'computed'
        FROM elements e1
        JOIN elements e2 ON ST_3DDWithin(e1.centroid, e2.centroid, $2)
        WHERE e1.project_id = $1
          AND e2.project_id = $1
          AND e1.id != e2.id
          AND e1.centroid IS NOT NULL
          AND e2.centroid IS NOT NULL
          AND e1.deleted_at IS NULL
          AND e2.deleted_at IS NULL
        ON CONFLICT (project_id, from_element_id, to_element_id, relation_type)
        DO UPDATE SET distance = EXCLUDED.distance
    """

    result = await pool.execute(query, project_id, distance_threshold)
    count = int(result.split()[-1]) if "INSERT" in result else 0
    logger.info(
        "Computed nearby elements",
        project_id=str(project_id),
        threshold=distance_threshold,
        count=count
    )
    return count


async def find_elements_near_point(
    pool: DatabasePool,
    project_id: UUID,
    point: CentroidInfo,
    radius: float = 1.0,
    entity_type: Optional[str] = None,
    limit: int = 50,
) -> List[dict]:
    """
    Find elements near a specific point.

    Args:
        pool: Database connection pool
        project_id: Project to search
        point: Center point
        radius: Search radius in meters
        entity_type: Optional filter by entity type
        limit: Maximum results

    Returns:
        List of element dicts with distance
    """
    point_wkt = point.to_wkt_point()

    if entity_type:
        query = f"""
            SELECT
                id, source_id, entity_type, layer, category, family, type_name,
                description,
                ST_X(centroid) as x, ST_Y(centroid) as y, ST_Z(centroid) as z,
                ST_3DDistance(centroid, ST_GeomFromText('{point_wkt}', 0)) as distance
            FROM elements
            WHERE project_id = $1
              AND entity_type = $2
              AND deleted_at IS NULL
              AND ST_3DDWithin(centroid, ST_GeomFromText('{point_wkt}', 0), $3)
            ORDER BY distance
            LIMIT $4
        """
        rows = await pool.fetch(query, project_id, entity_type, radius, limit)
    else:
        query = f"""
            SELECT
                id, source_id, entity_type, layer, category, family, type_name,
                description,
                ST_X(centroid) as x, ST_Y(centroid) as y, ST_Z(centroid) as z,
                ST_3DDistance(centroid, ST_GeomFromText('{point_wkt}', 0)) as distance
            FROM elements
            WHERE project_id = $1
              AND deleted_at IS NULL
              AND ST_3DDWithin(centroid, ST_GeomFromText('{point_wkt}', 0), $2)
            ORDER BY distance
            LIMIT $3
        """
        rows = await pool.fetch(query, project_id, radius, limit)

    return [
        {
            "id": str(row["id"]),
            "source_id": row["source_id"],
            "entity_type": row["entity_type"],
            "layer": row["layer"],
            "category": row["category"],
            "family": row["family"],
            "type_name": row["type_name"],
            "description": row["description"],
            "centroid": {"x": row["x"], "y": row["y"], "z": row["z"]},
            "distance": row["distance"],
        }
        for row in rows
    ]


async def find_elements_on_layer(
    pool: DatabasePool,
    project_id: UUID,
    layer_name: str,
    limit: int = 100,
) -> List[dict]:
    """
    Find all elements on a specific layer.

    Args:
        pool: Database connection pool
        project_id: Project to search
        layer_name: Layer name
        limit: Maximum results

    Returns:
        List of element dicts
    """
    query = """
        SELECT
            id, source_id, entity_type, layer, description,
            ST_X(centroid) as x, ST_Y(centroid) as y, ST_Z(centroid) as z
        FROM elements
        WHERE project_id = $1
          AND layer = $2
          AND deleted_at IS NULL
        LIMIT $3
    """
    rows = await pool.fetch(query, project_id, layer_name, limit)

    return [
        {
            "id": str(row["id"]),
            "source_id": row["source_id"],
            "entity_type": row["entity_type"],
            "layer": row["layer"],
            "description": row["description"],
            "centroid": {"x": row["x"], "y": row["y"], "z": row["z"]}
            if row["x"] is not None else None,
        }
        for row in rows
    ]


async def compute_layer_groupings(
    pool: DatabasePool,
    project_id: UUID,
) -> int:
    """
    Create 'on_layer' relationships for elements on the same layer.

    Only creates relationships for layers with <= 100 elements
    to avoid explosion of relationships.

    Args:
        pool: Database connection pool
        project_id: Project to process

    Returns:
        Number of relationships created
    """
    query = """
        WITH layer_counts AS (
            SELECT layer, COUNT(*) as cnt
            FROM elements
            WHERE project_id = $1 AND layer IS NOT NULL AND deleted_at IS NULL
            GROUP BY layer
            HAVING COUNT(*) <= 100  -- Only small layers
        )
        INSERT INTO element_relationships (
            project_id, from_element_id, to_element_id, relation_type, source
        )
        SELECT DISTINCT
            e1.project_id,
            e1.id,
            e2.id,
            'on_layer',
            'computed'
        FROM elements e1
        JOIN elements e2 ON e1.layer = e2.layer
        JOIN layer_counts lc ON e1.layer = lc.layer
        WHERE e1.project_id = $1
          AND e2.project_id = $1
          AND e1.id < e2.id
          AND e1.deleted_at IS NULL
          AND e2.deleted_at IS NULL
        ON CONFLICT (project_id, from_element_id, to_element_id, relation_type)
        DO NOTHING
    """

    result = await pool.execute(query, project_id)
    count = int(result.split()[-1]) if "INSERT" in result else 0
    logger.info("Computed layer groupings", project_id=str(project_id), count=count)
    return count
