"""
Unit tests for database models.
"""

import pytest
from uuid import uuid4

from aec_agent.db.models import (
    Project,
    Element,
    ElementRelationship,
    BoundsInfo,
    CentroidInfo,
    ExtractionResult,
)


class TestBoundsInfo:
    """Tests for BoundsInfo model."""

    def test_bounds_info_creation(self):
        bounds = BoundsInfo(
            min_x=0, min_y=0, min_z=0,
            max_x=10, max_y=10, max_z=5
        )
        assert bounds.min_x == 0
        assert bounds.max_x == 10
        assert bounds.max_z == 5

    def test_bounds_to_wkt_polygon(self):
        bounds = BoundsInfo(
            min_x=0, min_y=0, min_z=0,
            max_x=10, max_y=20, max_z=5
        )
        wkt = bounds.to_wkt_polygon()
        assert "POLYGON" in wkt
        assert "0.0 0.0" in wkt
        assert "10.0 0.0" in wkt
        assert "10.0 20.0" in wkt
        assert "0.0 20.0" in wkt


class TestCentroidInfo:
    """Tests for CentroidInfo model."""

    def test_centroid_info_creation(self):
        centroid = CentroidInfo(x=5, y=10, z=2.5)
        assert centroid.x == 5
        assert centroid.y == 10
        assert centroid.z == 2.5

    def test_centroid_default_z(self):
        centroid = CentroidInfo(x=5, y=10)
        assert centroid.z == 0.0

    def test_centroid_to_wkt_point(self):
        centroid = CentroidInfo(x=5.5, y=10.2, z=2.5)
        wkt = centroid.to_wkt_point()
        assert wkt == "POINT Z(5.5 10.2 2.5)"


class TestProject:
    """Tests for Project model."""

    def test_project_creation(self):
        project = Project(
            name="Test Drawing",
            source="autocad",
            file_path="/path/to/drawing.dwg"
        )
        assert project.name == "Test Drawing"
        assert project.source == "autocad"
        assert project.id is not None

    def test_project_with_metadata(self):
        project = Project(
            name="Test",
            source="revit",
            metadata={"units": "meters", "author": "Test"}
        )
        assert project.metadata["units"] == "meters"


class TestElement:
    """Tests for Element model."""

    def test_element_creation(self):
        project_id = uuid4()
        element = Element(
            project_id=project_id,
            source_id="ABC123",
            source="autocad",
            entity_type="LINE",
            layer="WALLS",
        )
        assert element.project_id == project_id
        assert element.source_id == "ABC123"
        assert element.source == "autocad"

    def test_element_with_geometry(self):
        centroid = CentroidInfo(x=5, y=10, z=0)
        bounds = BoundsInfo(
            min_x=0, min_y=0, min_z=0,
            max_x=10, max_y=20, max_z=0
        )

        element = Element(
            project_id=uuid4(),
            source_id="123",
            source="revit",
            entity_type="WALL",
            centroid=centroid,
            bounds=bounds,
        )
        assert element.centroid.x == 5
        assert element.bounds.max_y == 20

    def test_element_to_search_result(self):
        element = Element(
            project_id=uuid4(),
            source_id="456",
            source="revit",
            entity_type="WALL",
            category="Walls",
            family="Basic Wall",
            type_name="Generic - 200mm",
            description="Wall: Basic Wall - Generic - 200mm",
            centroid=CentroidInfo(x=10, y=20, z=0),
        )

        result = element.to_search_result()
        assert result["source_id"] == "456"
        assert result["category"] == "Walls"
        assert result["family"] == "Basic Wall"
        assert result["centroid"]["x"] == 10

    def test_element_embedding_validation(self):
        # Test with list
        element = Element(
            project_id=uuid4(),
            source_id="123",
            source="autocad",
            entity_type="LINE",
            embedding=[0.1, 0.2, 0.3],
        )
        assert len(element.embedding) == 3

        # Test with pgvector string format
        element2 = Element(
            project_id=uuid4(),
            source_id="456",
            source="autocad",
            entity_type="LINE",
            embedding="[0.1,0.2,0.3,0.4]",
        )
        assert len(element2.embedding) == 4
        assert element2.embedding[0] == 0.1


class TestElementRelationship:
    """Tests for ElementRelationship model."""

    def test_relationship_creation(self):
        project_id = uuid4()
        from_id = uuid4()
        to_id = uuid4()

        rel = ElementRelationship(
            project_id=project_id,
            from_element_id=from_id,
            to_element_id=to_id,
            relation_type="intersects",
        )
        assert rel.relation_type == "intersects"
        assert rel.confidence == 1.0
        assert rel.source == "computed"

    def test_relationship_with_distance(self):
        rel = ElementRelationship(
            project_id=uuid4(),
            from_element_id=uuid4(),
            to_element_id=uuid4(),
            relation_type="near",
            distance=2.5,
        )
        assert rel.distance == 2.5


class TestExtractionResult:
    """Tests for ExtractionResult model."""

    def test_extraction_result_success(self):
        result = ExtractionResult(
            project_id=uuid4(),
            elements_extracted=100,
            relationships_computed=50,
        )
        assert result.success is True
        assert result.elements_extracted == 100

    def test_extraction_result_failure(self):
        result = ExtractionResult(
            project_id=uuid4(),
            errors=["Connection failed", "Timeout"],
        )
        assert result.success is False
        assert len(result.errors) == 2
