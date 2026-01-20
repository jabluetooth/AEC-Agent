"""
Database models for AEC Agent metadata pipeline.

Pydantic models matching the PostgreSQL schema with PostGIS and pgvector fields.
"""

from datetime import datetime
from typing import Optional, List, Literal, Dict, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class GeometryInfo(BaseModel):
    """Geometry information for spatial operations."""

    type: str = Field(description="Geometry type: POINT, LINE, POLYGON, etc.")
    wkt: str = Field(description="Well-Known Text representation")

    class Config:
        frozen = True


class BoundsInfo(BaseModel):
    """3D bounding box for spatial indexing."""

    min_x: float
    min_y: float
    min_z: float = 0.0
    max_x: float
    max_y: float
    max_z: float = 0.0

    class Config:
        frozen = True

    def to_wkt_polygon(self) -> str:
        """Convert to WKT polygon for PostGIS (2D projection)."""
        return (
            f"POLYGON(("
            f"{self.min_x} {self.min_y}, "
            f"{self.max_x} {self.min_y}, "
            f"{self.max_x} {self.max_y}, "
            f"{self.min_x} {self.max_y}, "
            f"{self.min_x} {self.min_y}"
            f"))"
        )


class CentroidInfo(BaseModel):
    """3D point for centroid/location."""

    x: float
    y: float
    z: float = 0.0

    class Config:
        frozen = True

    def to_wkt_point(self) -> str:
        """Convert to WKT point for PostGIS."""
        return f"POINT Z({self.x} {self.y} {self.z})"


class Project(BaseModel):
    """
    Project (drawing/model) metadata.

    Represents an AutoCAD drawing or Revit model.
    """

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(description="Project/drawing name")
    source: Literal["autocad", "revit"] = Field(description="Source application")
    file_path: Optional[str] = Field(default=None, description="Full file path")
    file_hash: Optional[str] = Field(default=None, description="File hash for change detection")
    extracted_at: Optional[datetime] = Field(default=None, description="Last extraction time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        from_attributes = True


class Element(BaseModel):
    """
    CAD element metadata.

    Represents an entity (AutoCAD) or element (Revit) with geometry and properties.
    """

    id: UUID = Field(default_factory=uuid4)
    project_id: UUID = Field(description="Parent project ID")
    source_id: str = Field(description="Handle (AutoCAD) or ElementId (Revit)")
    source: Literal["autocad", "revit"] = Field(description="Source application")

    # Classification
    entity_type: str = Field(description="LINE, WALL, DOOR, etc.")
    layer: Optional[str] = Field(default=None, description="AutoCAD layer")
    category: Optional[str] = Field(default=None, description="Revit category")
    family: Optional[str] = Field(default=None, description="Revit family")
    type_name: Optional[str] = Field(default=None, description="Revit type")

    # Geometry (stored as WKT strings for PostGIS)
    geom_wkt: Optional[str] = Field(default=None, description="Geometry as WKT")
    bounds: Optional[BoundsInfo] = Field(default=None, description="Bounding box")
    centroid: Optional[CentroidInfo] = Field(default=None, description="Center point")

    # Properties
    properties: Dict[str, Any] = Field(default_factory=dict, description="All parameters/xdata")

    # Semantic search
    description: Optional[str] = Field(default=None, description="Human-readable description")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding")

    # Lifecycle
    deleted_at: Optional[datetime] = Field(default=None, description="Soft delete timestamp")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True

    @field_validator("embedding", mode="before")
    @classmethod
    def validate_embedding(cls, v):
        """Ensure embedding is a list of floats."""
        if v is None:
            return None
        if isinstance(v, str):
            # Handle pgvector string format
            if v.startswith('[') and v.endswith(']'):
                v = v[1:-1]
            return [float(x) for x in v.split(',') if x]
        return v

    def to_search_result(self) -> dict:
        """Convert to a search result dict for API responses."""
        return {
            "id": str(self.id),
            "source_id": self.source_id,
            "source": self.source,
            "entity_type": self.entity_type,
            "layer": self.layer,
            "category": self.category,
            "family": self.family,
            "type_name": self.type_name,
            "description": self.description,
            "centroid": self.centroid.model_dump() if self.centroid else None,
            "bounds": self.bounds.model_dump() if self.bounds else None,
        }


class ElementRelationship(BaseModel):
    """
    Relationship between two elements.

    Captures spatial (intersects, near) and logical (hosts, connected_to) relationships.
    """

    id: UUID = Field(default_factory=uuid4)
    project_id: UUID = Field(description="Parent project ID")
    from_element_id: UUID = Field(description="Source element ID")
    to_element_id: UUID = Field(description="Target element ID")

    relation_type: str = Field(
        description="Relationship type: intersects, near, hosts, connected_to, on_layer, same_block"
    )
    distance: Optional[float] = Field(default=None, description="Distance in meters (if applicable)")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    source: str = Field(
        default="computed",
        description="How relationship was determined: computed, revit_api, user_defined"
    )

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        from_attributes = True


class ExtractionResult(BaseModel):
    """Result of a metadata extraction operation."""

    project_id: UUID
    elements_extracted: int = 0
    elements_updated: int = 0
    elements_deleted: int = 0
    relationships_computed: int = 0
    embeddings_generated: int = 0
    duration_ms: int = 0
    errors: List[str] = Field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if extraction was successful."""
        return len(self.errors) == 0


class SyncStatus(BaseModel):
    """Status of metadata synchronization."""

    project_id: UUID
    last_sync: Optional[datetime] = None
    is_syncing: bool = False
    pending_changes: int = 0
    file_hash: Optional[str] = None
    needs_resync: bool = False
