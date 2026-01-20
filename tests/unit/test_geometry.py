"""
Unit tests for geometry conversion utilities.
"""

import pytest
import math

from aec_agent.extraction.geometry import (
    autocad_geometry_to_wkt,
    revit_location_to_wkt,
    compute_centroid,
    compute_bounds,
)
from aec_agent.db.models import CentroidInfo, BoundsInfo


class TestAutoCADGeometryToWKT:
    """Tests for AutoCAD geometry conversion."""

    def test_line_to_wkt(self):
        geometry = {
            "type": "LINE",
            "points": [[0, 0, 0], [10, 20, 0]]
        }
        wkt = autocad_geometry_to_wkt(geometry)
        assert wkt is not None
        assert "LINESTRING Z" in wkt
        assert "0 0 0" in wkt
        assert "10 20 0" in wkt

    def test_point_to_wkt(self):
        geometry = {
            "type": "POINT",
            "points": [[5, 10, 2.5]]
        }
        wkt = autocad_geometry_to_wkt(geometry)
        assert wkt is not None
        assert "POINT Z" in wkt
        assert "5 10 2.5" in wkt

    def test_circle_to_wkt(self):
        geometry = {
            "type": "CIRCLE",
            "center": [100, 200, 0],
            "radius": 50
        }
        wkt = autocad_geometry_to_wkt(geometry)
        assert wkt is not None
        assert "POLYGON Z" in wkt

    def test_polyline_to_wkt(self):
        geometry = {
            "type": "POLYLINE",
            "points": [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]],
            "closed": False
        }
        wkt = autocad_geometry_to_wkt(geometry)
        assert wkt is not None
        assert "LINESTRING Z" in wkt

    def test_closed_polyline_to_wkt(self):
        geometry = {
            "type": "POLYLINE",
            "points": [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]],
            "closed": True
        }
        wkt = autocad_geometry_to_wkt(geometry)
        assert wkt is not None
        assert "POLYGON Z" in wkt

    def test_arc_to_wkt(self):
        geometry = {
            "type": "ARC",
            "center": [0, 0, 0],
            "radius": 10,
            "start_angle": 0,
            "end_angle": 90
        }
        wkt = autocad_geometry_to_wkt(geometry)
        assert wkt is not None
        assert "LINESTRING Z" in wkt

    def test_unsupported_type_returns_none(self):
        geometry = {
            "type": "UNKNOWN_TYPE",
            "data": {}
        }
        wkt = autocad_geometry_to_wkt(geometry)
        assert wkt is None


class TestRevitLocationToWKT:
    """Tests for Revit location conversion."""

    def test_point_location_to_wkt(self):
        location = {
            "type": "POINT",
            "point": {"x": 5.5, "y": 10.2, "z": 0}
        }
        wkt = revit_location_to_wkt(location)
        assert wkt is not None
        assert "POINT Z" in wkt
        assert "5.5 10.2 0" in wkt

    def test_curve_location_to_wkt(self):
        location = {
            "type": "CURVE",
            "start": {"x": 0, "y": 0, "z": 0},
            "end": {"x": 10, "y": 0, "z": 3}
        }
        wkt = revit_location_to_wkt(location)
        assert wkt is not None
        assert "LINESTRING Z" in wkt
        assert "0 0 0" in wkt
        assert "10 0 3" in wkt

    def test_area_location_to_wkt(self):
        location = {
            "type": "AREA",
            "boundary": [
                {"x": 0, "y": 0, "z": 0},
                {"x": 10, "y": 0, "z": 0},
                {"x": 10, "y": 10, "z": 0},
                {"x": 0, "y": 10, "z": 0},
            ]
        }
        wkt = revit_location_to_wkt(location)
        assert wkt is not None
        assert "POLYGON Z" in wkt


class TestComputeCentroid:
    """Tests for centroid computation."""

    def test_autocad_line_centroid(self):
        geometry = {
            "type": "LINE",
            "points": [[0, 0, 0], [10, 20, 0]]
        }
        centroid = compute_centroid(geometry, "autocad")
        assert centroid is not None
        assert centroid.x == 5
        assert centroid.y == 10

    def test_autocad_circle_centroid(self):
        geometry = {
            "type": "CIRCLE",
            "center": [100, 200, 5],
            "radius": 50
        }
        centroid = compute_centroid(geometry, "autocad")
        assert centroid is not None
        assert centroid.x == 100
        assert centroid.y == 200
        assert centroid.z == 5

    def test_revit_point_centroid(self):
        location = {
            "type": "POINT",
            "point": {"x": 5, "y": 10, "z": 2.5}
        }
        centroid = compute_centroid(location, "revit")
        assert centroid is not None
        assert centroid.x == 5
        assert centroid.y == 10
        assert centroid.z == 2.5

    def test_revit_curve_centroid(self):
        location = {
            "type": "CURVE",
            "start": {"x": 0, "y": 0, "z": 0},
            "end": {"x": 10, "y": 20, "z": 0}
        }
        centroid = compute_centroid(location, "revit")
        assert centroid is not None
        assert centroid.x == 5
        assert centroid.y == 10


class TestComputeBounds:
    """Tests for bounds computation."""

    def test_autocad_line_bounds(self):
        geometry = {
            "type": "LINE",
            "points": [[0, 5, 0], [10, 20, 0]]
        }
        bounds = compute_bounds(geometry, "autocad")
        assert bounds is not None
        assert bounds.min_x == 0
        assert bounds.min_y == 5
        assert bounds.max_x == 10
        assert bounds.max_y == 20

    def test_autocad_precomputed_bounds(self):
        geometry = {
            "type": "LINE",
            "bounds": {
                "min_x": -5, "min_y": -5, "min_z": 0,
                "max_x": 15, "max_y": 25, "max_z": 0
            }
        }
        bounds = compute_bounds(geometry, "autocad")
        assert bounds is not None
        assert bounds.min_x == -5
        assert bounds.max_x == 15

    def test_autocad_circle_bounds(self):
        geometry = {
            "type": "CIRCLE",
            "center": [100, 200, 0],
            "radius": 50
        }
        bounds = compute_bounds(geometry, "autocad")
        assert bounds is not None
        assert bounds.min_x == 50
        assert bounds.max_x == 150
        assert bounds.min_y == 150
        assert bounds.max_y == 250

    def test_revit_curve_bounds(self):
        location = {
            "type": "CURVE",
            "start": {"x": 0, "y": 0, "z": 0},
            "end": {"x": 10, "y": 20, "z": 5}
        }
        bounds = compute_bounds(location, "revit")
        assert bounds is not None
        assert bounds.min_x == 0
        assert bounds.max_x == 10
        assert bounds.min_y == 0
        assert bounds.max_y == 20
        assert bounds.max_z == 5
