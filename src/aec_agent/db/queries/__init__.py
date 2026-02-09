"""
Database query modules for PostGIS and pgvector operations.
"""

from aec_agent.db.queries.relationships import (
    compute_all_relationships,
    get_element_graph,
    recompute_relationships_for_element,
)
from aec_agent.db.queries.spatial import (
    compute_intersecting_elements,
    compute_nearby_elements,
    find_elements_near_point,
)

__all__ = [
    "compute_intersecting_elements",
    "compute_nearby_elements",
    "find_elements_near_point",
    "compute_all_relationships",
    "get_element_graph",
    "recompute_relationships_for_element",
]
