"""
Relationship computation and graph queries for AEC Agent.

Provides methods for computing and traversing element relationships.
"""

from typing import Any
from uuid import UUID

import structlog

from aec_agent.db.connection import DatabasePool
from aec_agent.db.queries.spatial import (
    compute_intersecting_elements,
    compute_layer_groupings,
    compute_nearby_elements,
)

logger = structlog.get_logger(__name__)


async def compute_all_relationships(
    pool: DatabasePool,
    project_id: UUID,
    distance_threshold: float = 1.0,
    include_layer_groups: bool = True,
) -> dict[str, int]:
    """
    Compute all relationship types for a project.

    Args:
        pool: Database connection pool
        project_id: Project to process
        distance_threshold: Distance threshold for 'near' relationships
        include_layer_groups: Whether to compute layer groupings

    Returns:
        Dict with counts per relationship type
    """
    results = {}

    # Compute intersections
    results["intersects"] = await compute_intersecting_elements(pool, project_id)

    # Compute nearby elements
    results["near"] = await compute_nearby_elements(
        pool, project_id, distance_threshold
    )

    # Compute layer groupings (optional)
    if include_layer_groups:
        results["on_layer"] = await compute_layer_groupings(pool, project_id)

    total = sum(results.values())
    logger.info(
        "Computed all relationships",
        project_id=str(project_id),
        total=total,
        breakdown=results
    )

    return results


async def recompute_relationships_for_element(
    pool: DatabasePool,
    element_id: UUID,
    distance_threshold: float = 1.0,
) -> int:
    """
    Recompute relationships for a single element.

    Used for incremental updates when an element changes.

    Args:
        pool: Database connection pool
        element_id: Element to recompute relationships for
        distance_threshold: Distance threshold for 'near' relationships

    Returns:
        Number of relationships created/updated
    """
    # First, delete existing computed relationships for this element
    delete_query = """
        DELETE FROM element_relationships
        WHERE (from_element_id = $1 OR to_element_id = $1)
          AND source = 'computed'
    """
    await pool.execute(delete_query, element_id)

    # Get element info
    element_query = """
        SELECT project_id, geom, centroid, layer
        FROM elements
        WHERE id = $1 AND deleted_at IS NULL
    """
    element = await pool.fetchrow(element_query, element_id)
    if not element:
        return 0

    project_id = element["project_id"]
    count = 0

    # Compute intersections with this element
    if element["geom"]:
        intersect_query = """
            INSERT INTO element_relationships (
                project_id, from_element_id, to_element_id, relation_type, source
            )
            SELECT
                $2,
                LEAST($1, e.id),
                GREATEST($1, e.id),
                'intersects',
                'computed'
            FROM elements e
            WHERE e.project_id = $2
              AND e.id != $1
              AND e.geom IS NOT NULL
              AND e.deleted_at IS NULL
              AND ST_Intersects(e.geom, (SELECT geom FROM elements WHERE id = $1))
            ON CONFLICT DO NOTHING
        """
        result = await pool.execute(intersect_query, element_id, project_id)
        count += int(result.split()[-1]) if "INSERT" in result else 0

    # Compute nearby relationships with this element
    if element["centroid"]:
        nearby_query = """
            INSERT INTO element_relationships (
                project_id, from_element_id, to_element_id, relation_type, distance, source
            )
            SELECT
                $2,
                LEAST($1, e.id),
                GREATEST($1, e.id),
                'near',
                ST_3DDistance(e.centroid, (SELECT centroid FROM elements WHERE id = $1)),
                'computed'
            FROM elements e
            WHERE e.project_id = $2
              AND e.id != $1
              AND e.centroid IS NOT NULL
              AND e.deleted_at IS NULL
              AND ST_3DDWithin(
                  e.centroid,
                  (SELECT centroid FROM elements WHERE id = $1),
                  $3
              )
            ON CONFLICT (project_id, from_element_id, to_element_id, relation_type)
            DO UPDATE SET distance = EXCLUDED.distance
        """
        result = await pool.execute(nearby_query, element_id, project_id, distance_threshold)
        count += int(result.split()[-1]) if "INSERT" in result else 0

    logger.debug(
        "Recomputed element relationships",
        element_id=str(element_id),
        count=count
    )
    return count


async def get_element_graph(
    pool: DatabasePool,
    element_id: UUID,
    depth: int = 1,
    relation_types: list[str] | None = None,
) -> dict[str, Any]:
    """
    Get the relationship graph around an element.

    Args:
        pool: Database connection pool
        element_id: Starting element
        depth: How many hops to traverse (1 = direct neighbors)
        relation_types: Optional filter for relationship types

    Returns:
        Graph structure with nodes and edges
    """
    # Get the starting element
    element_query = """
        SELECT id, source_id, entity_type, layer, category, family, type_name, description
        FROM elements
        WHERE id = $1 AND deleted_at IS NULL
    """
    start_element = await pool.fetchrow(element_query, element_id)
    if not start_element:
        return {"nodes": [], "edges": []}

    nodes = {str(element_id): dict(start_element)}
    edges = []
    visited = {element_id}
    frontier = [element_id]

    for _ in range(depth):
        next_frontier = []

        for current_id in frontier:
            # Get relationships from this node
            if relation_types:
                rel_query = """
                    SELECT
                        r.id as rel_id,
                        r.from_element_id,
                        r.to_element_id,
                        r.relation_type,
                        r.distance,
                        e.id, e.source_id, e.entity_type, e.layer, e.category,
                        e.family, e.type_name, e.description
                    FROM element_relationships r
                    JOIN elements e ON (
                        CASE
                            WHEN r.from_element_id = $1 THEN r.to_element_id
                            ELSE r.from_element_id
                        END = e.id
                    )
                    WHERE (r.from_element_id = $1 OR r.to_element_id = $1)
                      AND r.relation_type = ANY($2)
                      AND e.deleted_at IS NULL
                """
                rows = await pool.fetch(rel_query, current_id, relation_types)
            else:
                rel_query = """
                    SELECT
                        r.id as rel_id,
                        r.from_element_id,
                        r.to_element_id,
                        r.relation_type,
                        r.distance,
                        e.id, e.source_id, e.entity_type, e.layer, e.category,
                        e.family, e.type_name, e.description
                    FROM element_relationships r
                    JOIN elements e ON (
                        CASE
                            WHEN r.from_element_id = $1 THEN r.to_element_id
                            ELSE r.from_element_id
                        END = e.id
                    )
                    WHERE (r.from_element_id = $1 OR r.to_element_id = $1)
                      AND e.deleted_at IS NULL
                """
                rows = await pool.fetch(rel_query, current_id)

            for row in rows:
                # Add edge
                edges.append({
                    "id": str(row["rel_id"]),
                    "from": str(row["from_element_id"]),
                    "to": str(row["to_element_id"]),
                    "type": row["relation_type"],
                    "distance": row["distance"],
                })

                # Add node if not visited
                other_id = row["id"]
                if other_id not in visited:
                    visited.add(other_id)
                    next_frontier.append(other_id)
                    nodes[str(other_id)] = {
                        "id": str(row["id"]),
                        "source_id": row["source_id"],
                        "entity_type": row["entity_type"],
                        "layer": row["layer"],
                        "category": row["category"],
                        "family": row["family"],
                        "type_name": row["type_name"],
                        "description": row["description"],
                    }

        frontier = next_frontier

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "root_id": str(element_id),
    }


async def get_relationship_stats(
    pool: DatabasePool,
    project_id: UUID,
) -> dict[str, Any]:
    """
    Get statistics about relationships in a project.

    Args:
        pool: Database connection pool
        project_id: Project to analyze

    Returns:
        Statistics dict
    """
    query = """
        SELECT
            relation_type,
            COUNT(*) as count,
            AVG(distance) as avg_distance,
            MIN(distance) as min_distance,
            MAX(distance) as max_distance
        FROM element_relationships
        WHERE project_id = $1
        GROUP BY relation_type
    """
    rows = await pool.fetch(query, project_id)

    stats = {}
    for row in rows:
        stats[row["relation_type"]] = {
            "count": row["count"],
            "avg_distance": row["avg_distance"],
            "min_distance": row["min_distance"],
            "max_distance": row["max_distance"],
        }

    total_query = """
        SELECT COUNT(*) as total FROM element_relationships WHERE project_id = $1
    """
    total = await pool.fetchval(total_query, project_id)

    return {
        "total_relationships": total,
        "by_type": stats,
    }
