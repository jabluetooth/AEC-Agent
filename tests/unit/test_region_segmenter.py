"""Unit tests for region_segmenter.py (Phase A)."""

import numpy as np
import pytest

from aec_agent.mcp.tools.region_segmenter import (
    DetectedRegion,
    RegionType,
    SegmentationResult,
    create_region_mask,
    segment_regions,
)


class TestRegionType:
    """Tests for RegionType enum."""

    def test_all_expected_types_exist(self):
        """All expected region types should be defined."""
        expected = [
            "TITLE_BLOCK", "LEGEND", "DRAWING_AREA", "NOTES",
            "SCHEDULE", "REVISION_BLOCK", "SCALE_BAR", "BORDER", "UNKNOWN"
        ]
        for name in expected:
            assert hasattr(RegionType, name), f"Missing RegionType.{name}"

    def test_region_type_values_are_strings(self):
        """Region type values should be lowercase strings."""
        for rtype in RegionType:
            assert isinstance(rtype.value, str)
            assert rtype.value == rtype.value.lower()


class TestDetectedRegion:
    """Tests for DetectedRegion dataclass."""

    def test_basic_creation(self):
        """Should create region with required fields."""
        region = DetectedRegion(
            region_type=RegionType.TITLE_BLOCK,
            bounds=(100, 200, 300, 150),
            confidence=0.85,
        )
        assert region.region_type == RegionType.TITLE_BLOCK
        assert region.bounds == (100, 200, 300, 150)
        assert region.confidence == 0.85

    def test_property_accessors(self):
        """Property accessors should return correct values."""
        region = DetectedRegion(
            region_type=RegionType.DRAWING_AREA,
            bounds=(10, 20, 800, 600),
            confidence=0.9,
        )
        assert region.x == 10
        assert region.y == 20
        assert region.width == 800
        assert region.height == 600

    def test_area_property(self):
        """Area should be width * height."""
        region = DetectedRegion(
            region_type=RegionType.LEGEND,
            bounds=(0, 0, 100, 200),
            confidence=0.7,
        )
        assert region.area == 100 * 200

    def test_center_property(self):
        """Center should be midpoint of bounds."""
        region = DetectedRegion(
            region_type=RegionType.NOTES,
            bounds=(100, 100, 200, 100),
            confidence=0.6,
        )
        assert region.center == (200, 150)  # (100 + 200//2, 100 + 100//2)

    def test_contains_point_inside(self):
        """contains_point should return True for points inside."""
        region = DetectedRegion(
            region_type=RegionType.DRAWING_AREA,
            bounds=(100, 100, 200, 200),
            confidence=0.9,
        )
        # Point clearly inside
        assert region.contains_point(150, 150) is True
        # Point at top-left corner (inclusive)
        assert region.contains_point(100, 100) is True

    def test_contains_point_outside(self):
        """contains_point should return False for points outside."""
        region = DetectedRegion(
            region_type=RegionType.DRAWING_AREA,
            bounds=(100, 100, 200, 200),
            confidence=0.9,
        )
        # Point outside to the left
        assert region.contains_point(50, 150) is False
        # Point outside to the right (edge is exclusive)
        assert region.contains_point(300, 150) is False
        # Point above
        assert region.contains_point(150, 50) is False

    def test_to_slice(self):
        """to_slice should return numpy-compatible slices."""
        region = DetectedRegion(
            region_type=RegionType.TITLE_BLOCK,
            bounds=(50, 100, 200, 150),
            confidence=0.8,
        )
        row_slice, col_slice = region.to_slice()

        assert row_slice == slice(100, 250)  # y to y+height
        assert col_slice == slice(50, 250)   # x to x+width

    def test_text_content_default(self):
        """text_content should default to empty string."""
        region = DetectedRegion(
            region_type=RegionType.NOTES,
            bounds=(0, 0, 100, 100),
            confidence=0.5,
        )
        assert region.text_content == ""

    def test_metadata_default(self):
        """metadata should default to empty dict."""
        region = DetectedRegion(
            region_type=RegionType.SCHEDULE,
            bounds=(0, 0, 100, 100),
            confidence=0.5,
        )
        assert region.metadata == {}


class TestSegmentationResult:
    """Tests for SegmentationResult dataclass."""

    def test_basic_creation(self):
        """Should create result with required fields."""
        result = SegmentationResult(
            regions=[],
            image_width=1000,
            image_height=800,
        )
        assert result.image_width == 1000
        assert result.image_height == 800
        assert result.regions == []

    def test_get_regions_by_type(self):
        """get_regions_by_type should filter correctly."""
        regions = [
            DetectedRegion(RegionType.DRAWING_AREA, (0, 0, 800, 600), 0.9),
            DetectedRegion(RegionType.TITLE_BLOCK, (800, 600, 200, 200), 0.8),
            DetectedRegion(RegionType.NOTES, (0, 600, 300, 200), 0.6),
            DetectedRegion(RegionType.NOTES, (300, 600, 300, 200), 0.7),
        ]
        result = SegmentationResult(
            regions=regions,
            image_width=1000,
            image_height=800,
        )

        # Should find 2 notes regions
        notes = result.get_regions_by_type(RegionType.NOTES)
        assert len(notes) == 2

        # Should find 1 title block
        title_blocks = result.get_regions_by_type(RegionType.TITLE_BLOCK)
        assert len(title_blocks) == 1

        # Should find 0 legends
        legends = result.get_regions_by_type(RegionType.LEGEND)
        assert len(legends) == 0

    def test_get_main_drawing_area_from_property(self):
        """Should return drawing_area property if set."""
        drawing_area = DetectedRegion(RegionType.DRAWING_AREA, (0, 0, 800, 600), 0.9)
        result = SegmentationResult(
            regions=[drawing_area],
            image_width=1000,
            image_height=800,
            drawing_area=drawing_area,
        )

        assert result.get_main_drawing_area() is drawing_area

    def test_get_main_drawing_area_largest(self):
        """Should return largest DRAWING_AREA if property not set."""
        small = DetectedRegion(RegionType.DRAWING_AREA, (0, 0, 100, 100), 0.7)
        large = DetectedRegion(RegionType.DRAWING_AREA, (0, 0, 800, 600), 0.9)

        result = SegmentationResult(
            regions=[small, large],
            image_width=1000,
            image_height=800,
            drawing_area=None,
        )

        main = result.get_main_drawing_area()
        assert main is large  # Larger area


class TestSegmentRegions:
    """Tests for segment_regions function."""

    def test_simple_image(self):
        """Should handle a simple grayscale image."""
        # Create a simple 1000x800 image
        image = np.ones((800, 1000), dtype=np.uint8) * 255

        result = segment_regions(image)

        assert result.image_width == 1000
        assert result.image_height == 800
        assert isinstance(result.regions, list)

    def test_image_with_title_block_region(self):
        """Should detect potential title block in bottom-right."""
        # Create image with dark region in bottom-right (simulating title block)
        image = np.ones((800, 1000), dtype=np.uint8) * 255

        # Add dark rectangle in bottom-right corner
        image[650:780, 700:950] = 50  # Dark region

        # Add some lines (simulating title block borders)
        image[650, 700:950] = 0   # Top border
        image[780, 700:950] = 0   # Bottom border
        image[650:780, 700] = 0   # Left border
        image[650:780, 950] = 0   # Right border

        result = segment_regions(image, detect_title_block=True)

        # Should detect drawing area at minimum
        assert result.drawing_area is not None

    def test_disable_all_detection(self):
        """Should work with all detection disabled."""
        image = np.ones((800, 1000), dtype=np.uint8) * 255

        result = segment_regions(
            image,
            detect_title_block=False,
            detect_legend=False,
            detect_notes=False,
            detect_schedules=False,
        )

        # Should still compute drawing area
        assert result.drawing_area is not None

    def test_color_image_conversion(self):
        """Should handle color images by converting to grayscale."""
        # Create a color image (3 channels)
        image = np.ones((800, 1000, 3), dtype=np.uint8) * 255

        result = segment_regions(image)

        assert result.image_width == 1000
        assert result.image_height == 800


class TestCreateRegionMask:
    """Tests for create_region_mask function."""

    def test_include_single_region(self):
        """Mask should be white only inside included region."""
        regions = [
            DetectedRegion(RegionType.DRAWING_AREA, (100, 100, 200, 200), 0.9),
        ]

        mask = create_region_mask(
            (500, 500),
            regions,
            include_types=[RegionType.DRAWING_AREA],
        )

        # Inside region should be white (255)
        assert mask[150, 150] == 255
        # Outside region should be black (0)
        assert mask[50, 50] == 0
        assert mask[350, 350] == 0

    def test_exclude_single_region(self):
        """Mask should be black only inside excluded region."""
        regions = [
            DetectedRegion(RegionType.TITLE_BLOCK, (100, 100, 200, 200), 0.8),
        ]

        mask = create_region_mask(
            (500, 500),
            regions,
            exclude_types=[RegionType.TITLE_BLOCK],
        )

        # Inside region should be black (0)
        assert mask[150, 150] == 0
        # Outside region should be white (255)
        assert mask[50, 50] == 255
        assert mask[350, 350] == 255

    def test_no_types_specified_all_white(self):
        """Mask should be all white if no types specified."""
        regions = [
            DetectedRegion(RegionType.DRAWING_AREA, (100, 100, 200, 200), 0.9),
        ]

        mask = create_region_mask((500, 500), regions)

        # Should be all white
        assert np.all(mask == 255)

    def test_include_multiple_regions(self):
        """Mask should include all specified region types."""
        regions = [
            DetectedRegion(RegionType.DRAWING_AREA, (0, 0, 100, 100), 0.9),
            DetectedRegion(RegionType.NOTES, (200, 200, 100, 100), 0.7),
            DetectedRegion(RegionType.TITLE_BLOCK, (400, 400, 100, 100), 0.8),
        ]

        mask = create_region_mask(
            (500, 500),
            regions,
            include_types=[RegionType.DRAWING_AREA, RegionType.NOTES],
        )

        # Drawing area and notes should be white
        assert mask[50, 50] == 255    # Drawing area
        assert mask[250, 250] == 255  # Notes
        # Title block should be black
        assert mask[450, 450] == 0

    def test_mask_shape_matches_image(self):
        """Mask shape should match specified image shape."""
        regions = []

        mask = create_region_mask((800, 1000), regions)

        assert mask.shape == (800, 1000)

    def test_mask_dtype_is_uint8(self):
        """Mask should be uint8 for OpenCV compatibility."""
        regions = []

        mask = create_region_mask((100, 100), regions)

        assert mask.dtype == np.uint8


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_small_image(self):
        """Should handle very small images."""
        image = np.ones((50, 50), dtype=np.uint8) * 255

        result = segment_regions(image)

        assert result.image_width == 50
        assert result.image_height == 50

    def test_single_pixel_image(self):
        """Should handle 1x1 pixel image without crashing."""
        image = np.ones((1, 1), dtype=np.uint8) * 255

        result = segment_regions(image)

        assert result.image_width == 1
        assert result.image_height == 1

    def test_empty_regions_list(self):
        """Should handle empty regions list."""
        result = SegmentationResult(
            regions=[],
            image_width=1000,
            image_height=800,
        )

        assert result.get_regions_by_type(RegionType.DRAWING_AREA) == []
        assert result.get_main_drawing_area() is None

    def test_zero_area_region(self):
        """Should handle regions with zero area."""
        region = DetectedRegion(
            region_type=RegionType.UNKNOWN,
            bounds=(100, 100, 0, 0),
            confidence=0.0,
        )
        assert region.area == 0

    def test_negative_coordinates(self):
        """Regions can technically have negative coordinates (edge case)."""
        region = DetectedRegion(
            region_type=RegionType.UNKNOWN,
            bounds=(-10, -10, 50, 50),
            confidence=0.5,
        )
        # Properties should still work
        assert region.x == -10
        assert region.y == -10
        assert region.center == (15, 15)
