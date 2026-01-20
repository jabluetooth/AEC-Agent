"""
Unit tests for description generator.
"""

import pytest

from aec_agent.semantic.description_generator import (
    generate_description,
    generate_autocad_description,
    generate_revit_description,
)


class TestAutoCADDescriptionGenerator:
    """Tests for AutoCAD description generation."""

    def test_line_description(self):
        element = {
            "entity_type": "LINE",
            "layer": "WALLS",
            "properties": {"length": 10.5},
        }
        desc = generate_autocad_description(element)
        assert "LINE" in desc
        assert "WALLS" in desc
        assert "10.5" in desc or "10.50" in desc

    def test_circle_description(self):
        element = {
            "entity_type": "CIRCLE",
            "layer": "ELECTRICAL",
            "properties": {"radius": 0.5},
        }
        desc = generate_autocad_description(element)
        assert "CIRCLE" in desc
        assert "ELECTRICAL" in desc
        assert "0.5" in desc or "50" in desc  # Could be formatted as cm

    def test_text_description(self):
        element = {
            "entity_type": "TEXT",
            "layer": "ANNOTATION",
            "properties": {"content": "Room 101"},
        }
        desc = generate_autocad_description(element)
        assert "TEXT" in desc
        assert "Room 101" in desc

    def test_block_description(self):
        element = {
            "entity_type": "INSERT",
            "layer": "FURNITURE",
            "properties": {"block_name": "Chair-01"},
        }
        desc = generate_autocad_description(element)
        assert "INSERT" in desc or "block" in desc.lower()
        assert "Chair-01" in desc

    def test_with_centroid(self):
        element = {
            "entity_type": "POINT",
            "layer": "0",
            "centroid": {"x": 100.5, "y": 200.3},
        }
        desc = generate_autocad_description(element)
        assert "100.5" in desc
        assert "200.3" in desc

    def test_with_color(self):
        element = {
            "entity_type": "LINE",
            "layer": "TEST",
            "properties": {"color": 1},  # Red
        }
        desc = generate_autocad_description(element)
        assert "color 1" in desc


class TestRevitDescriptionGenerator:
    """Tests for Revit description generation."""

    def test_wall_description(self):
        element = {
            "category": "Walls",
            "family": "Basic Wall",
            "type_name": "Generic - 200mm",
            "properties": {
                "Height": 3.0,
                "Fire Rating": "1HR",
            },
        }
        desc = generate_revit_description(element)
        assert "Wall" in desc
        assert "Basic Wall" in desc
        assert "200mm" in desc
        assert "3" in desc or "height" in desc.lower()
        assert "1HR" in desc

    def test_door_description(self):
        element = {
            "category": "Doors",
            "family": "Single-Flush",
            "type_name": "0915x2134mm",
            "properties": {
                "Width": 0.915,
                "Height": 2.134,
            },
        }
        desc = generate_revit_description(element)
        assert "Door" in desc
        assert "Single-Flush" in desc

    def test_room_description(self):
        element = {
            "category": "Rooms",
            "properties": {
                "Name": "Office",
                "Number": "101",
                "Area": 25.5,
            },
        }
        desc = generate_revit_description(element)
        assert "Room" in desc
        assert "Office" in desc
        assert "101" in desc
        assert "25.5" in desc

    def test_column_description(self):
        element = {
            "category": "Structural Columns",
            "family": "M_Rectangular Column",
            "type_name": "450x600mm",
            "properties": {
                "b": 0.45,
                "h": 0.6,
                "level": "Level 1",
            },
        }
        desc = generate_revit_description(element)
        assert "Column" in desc

    def test_with_level(self):
        element = {
            "category": "Walls",
            "family": "Basic Wall",
            "properties": {
                "level": "Level 2",
            },
        }
        desc = generate_revit_description(element)
        assert "Level 2" in desc

    def test_with_mark(self):
        element = {
            "category": "Doors",
            "family": "Single",
            "properties": {
                "Mark": "D-101",
            },
        }
        desc = generate_revit_description(element)
        assert "D-101" in desc


class TestGenerateDescription:
    """Tests for the unified generate_description function."""

    def test_autocad_routing(self):
        element = {
            "entity_type": "LINE",
            "layer": "TEST",
        }
        desc = generate_description(element, "autocad")
        assert "LINE" in desc

    def test_revit_routing(self):
        element = {
            "category": "Walls",
            "family": "Basic Wall",
        }
        desc = generate_description(element, "revit")
        assert "Wall" in desc
