"""
Database layer for AEC Agent.

Provides PostgreSQL connectivity with PostGIS (spatial) and pgvector (semantic) support.
"""

from aec_agent.db.connection import DatabasePool, get_database_pool
from aec_agent.db.models import (
    BoundsInfo,
    CentroidInfo,
    Element,
    ElementRelationship,
    GeometryInfo,
    Project,
)
from aec_agent.db.repository import ElementRepository

__all__ = [
    "DatabasePool",
    "get_database_pool",
    "Project",
    "Element",
    "ElementRelationship",
    "GeometryInfo",
    "BoundsInfo",
    "CentroidInfo",
    "ElementRepository",
]
