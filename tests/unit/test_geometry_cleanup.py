"""
Unit tests for AEC-specific geometry cleanup utilities.

Tests orthogonal snapping, collinear line merging, and dashed line detection.
Part of Phase 2.5: Semantic AEC Vectorization Pipeline.
"""

import pytest
import math

from aec_agent.utils.geometry_cleanup import (
    snap_to_orthogonal,
    merge_collinear_lines,
    MergedLine,
    _is_regular_pattern,
    _detect_gaps_in_cluster,
    lines_to_tuples,
    tuples_to_merged_lines,
)


class TestSnapToOrthogonal:
    """Tests for orthogonal line snapping."""

    def test_exact_horizontal_unchanged(self):
        """Exactly horizontal line should not change."""
        lines = [(0, 0, 100, 0)]
        result = snap_to_orthogonal(lines, angle_tolerance=2.0)
        assert len(result) == 1
        x1, y1, x2, y2 = result[0]
        assert x1 == 0
        assert y1 == 0
        assert x2 == 100
        assert y2 == 0

    def test_exact_vertical_unchanged(self):
        """Exactly vertical line should not change."""
        lines = [(0, 0, 0, 100)]
        result = snap_to_orthogonal(lines, angle_tolerance=2.0)
        assert len(result) == 1
        x1, y1, x2, y2 = result[0]
        assert x1 == 0
        assert y1 == 0
        assert x2 == 0
        assert y2 == 100

    def test_near_horizontal_snaps(self):
        """Line at ~1 degree should snap to horizontal."""
        # Line from (0,0) to (100, 1.75) is about 1 degree off horizontal
        lines = [(0, 0, 100, 1.75)]
        result = snap_to_orthogonal(lines, angle_tolerance=2.0)
        assert len(result) == 1
        x1, y1, x2, y2 = result[0]
        # Start point stays the same
        assert x1 == 0
        assert y1 == 0
        # End point should be snapped to y1 (horizontal)
        assert y2 == pytest.approx(0, abs=0.01)
        # Length should be preserved
        original_length = math.sqrt(100**2 + 1.75**2)
        new_length = abs(x2 - x1)
        assert new_length == pytest.approx(original_length, rel=0.01)

    def test_near_vertical_snaps(self):
        """Line at ~89 degrees should snap to vertical."""
        # Line from (0,0) to (1.5, 100) is about 89 degrees
        lines = [(0, 0, 1.5, 100)]
        result = snap_to_orthogonal(lines, angle_tolerance=2.0)
        assert len(result) == 1
        x1, y1, x2, y2 = result[0]
        # Start point stays the same
        assert x1 == 0
        assert y1 == 0
        # End point should be snapped to x1 (vertical)
        assert x2 == pytest.approx(0, abs=0.01)
        # Length should be preserved
        original_length = math.sqrt(1.5**2 + 100**2)
        new_length = abs(y2 - y1)
        assert new_length == pytest.approx(original_length, rel=0.01)

    def test_diagonal_not_snapped(self):
        """45-degree diagonal should not be snapped."""
        lines = [(0, 0, 100, 100)]
        result = snap_to_orthogonal(lines, angle_tolerance=2.0)
        assert len(result) == 1
        x1, y1, x2, y2 = result[0]
        # Should remain unchanged
        assert x1 == 0
        assert y1 == 0
        assert x2 == 100
        assert y2 == 100

    def test_outside_tolerance_not_snapped(self):
        """Line at 5 degrees should not snap with 2-degree tolerance."""
        # Line at ~5 degrees
        lines = [(0, 0, 100, 8.75)]
        result = snap_to_orthogonal(lines, angle_tolerance=2.0)
        assert len(result) == 1
        x1, y1, x2, y2 = result[0]
        # Should remain unchanged
        assert x2 == 100
        assert y2 == 8.75

    def test_empty_list(self):
        """Empty input should return empty output."""
        result = snap_to_orthogonal([], angle_tolerance=2.0)
        assert result == []

    def test_zero_length_line(self):
        """Zero-length line should remain unchanged."""
        lines = [(5, 5, 5, 5)]
        result = snap_to_orthogonal(lines, angle_tolerance=2.0)
        assert len(result) == 1
        assert result[0] == (5, 5, 5, 5)

    def test_multiple_lines(self):
        """Multiple lines should be processed independently."""
        lines = [
            (0, 0, 100, 1),     # Near horizontal (should snap)
            (0, 0, 100, 100),  # Diagonal (should not snap)
            (0, 0, 1, 100),    # Near vertical (should snap)
        ]
        result = snap_to_orthogonal(lines, angle_tolerance=2.0)
        assert len(result) == 3

        # First line: near horizontal → snapped to horizontal
        assert result[0][3] == pytest.approx(0, abs=1)  # y2 close to y1=0

        # Second line: diagonal → unchanged
        assert result[1] == (0, 0, 100, 100)

        # Third line: near vertical → snapped to vertical
        assert result[2][2] == pytest.approx(0, abs=1)  # x2 close to x1=0

    def test_negative_direction_horizontal(self):
        """Horizontal line going left (180 degrees) should snap."""
        lines = [(100, 0, 0, 1)]  # Near 180 degrees
        result = snap_to_orthogonal(lines, angle_tolerance=2.0)
        assert len(result) == 1
        assert result[0][3] == pytest.approx(0, abs=0.01)  # y2 snapped to y1

    def test_negative_direction_vertical(self):
        """Vertical line going down (270 degrees) should snap."""
        lines = [(0, 100, 1, 0)]  # Near 270 degrees
        result = snap_to_orthogonal(lines, angle_tolerance=2.0)
        assert len(result) == 1
        assert result[0][2] == pytest.approx(0, abs=0.01)  # x2 snapped to x1


class TestMergeCollinearLines:
    """Tests for collinear line merging."""

    def test_single_line_unchanged(self):
        """Single line should be returned as continuous."""
        lines = [(0, 0, 100, 0)]
        continuous, dashed = merge_collinear_lines(lines)
        assert len(continuous) == 1
        assert len(dashed) == 0
        assert continuous[0].start == (0, 0)
        assert continuous[0].end == (100, 0)
        assert continuous[0].linetype == "CONTINUOUS"

    def test_two_collinear_segments_merge(self):
        """Two collinear segments with small gap should merge."""
        lines = [
            (0, 0, 50, 0),    # First segment
            (60, 0, 100, 0),  # Second segment (10-unit gap)
        ]
        continuous, dashed = merge_collinear_lines(
            lines, gap_tolerance=20.0
        )
        assert len(continuous) == 1
        assert continuous[0].start == pytest.approx((0, 0), rel=0.01)
        assert continuous[0].end == pytest.approx((100, 0), rel=0.01)
        assert continuous[0].segment_count == 2

    def test_non_collinear_not_merged(self):
        """Non-collinear lines should not be merged."""
        lines = [
            (0, 0, 100, 0),   # Horizontal
            (0, 50, 100, 50), # Parallel but different track
        ]
        continuous, dashed = merge_collinear_lines(
            lines, distance_tolerance=5.0
        )
        # Should remain as separate lines
        assert len(continuous) == 2

    def test_perpendicular_not_merged(self):
        """Perpendicular lines should not be merged."""
        lines = [
            (0, 0, 100, 0),  # Horizontal
            (50, 0, 50, 100), # Vertical
        ]
        continuous, dashed = merge_collinear_lines(lines)
        assert len(continuous) == 2

    def test_empty_list(self):
        """Empty input should return empty outputs."""
        continuous, dashed = merge_collinear_lines([])
        assert continuous == []
        assert dashed == []

    def test_gap_too_large(self):
        """Segments with gap larger than tolerance should not merge."""
        lines = [
            (0, 0, 40, 0),
            (100, 0, 150, 0),  # 60-unit gap
        ]
        continuous, dashed = merge_collinear_lines(
            lines, gap_tolerance=20.0
        )
        # Should remain as separate lines
        assert len(continuous) == 2

    def test_overlapping_segments_merge(self):
        """Overlapping collinear segments should merge."""
        lines = [
            (0, 0, 60, 0),
            (40, 0, 100, 0),  # Overlaps from 40-60
        ]
        continuous, dashed = merge_collinear_lines(lines)
        assert len(continuous) == 1
        merged = continuous[0]
        # Should span from 0 to 100
        start_x = min(merged.start[0], merged.end[0])
        end_x = max(merged.start[0], merged.end[0])
        assert start_x == pytest.approx(0, abs=1)
        assert end_x == pytest.approx(100, abs=1)


class TestDashedLineDetection:
    """Tests for dashed line pattern detection."""

    def test_regular_gaps_detected_as_dashed(self):
        """Regular gaps should be detected as dashed pattern."""
        # Simulating a dashed line: segments with regular gaps
        gaps = [5.0, 5.0, 5.0, 5.0]  # 4 equal gaps
        assert _is_regular_pattern(gaps, min_gap=2.0, min_count=2) is True

    def test_irregular_gaps_not_dashed(self):
        """Irregular gaps should not be detected as dashed."""
        gaps = [5.0, 10.0, 3.0, 15.0]  # Varying gaps
        assert _is_regular_pattern(gaps, min_gap=2.0) is False

    def test_single_gap_not_dashed(self):
        """Single gap should not qualify as dashed pattern."""
        gaps = [5.0]
        assert _is_regular_pattern(gaps, min_gap=2.0, min_count=2) is False

    def test_tiny_gaps_filtered(self):
        """Gaps smaller than min_gap should be filtered."""
        gaps = [0.5, 0.5, 0.5, 0.5]  # All below min_gap
        assert _is_regular_pattern(gaps, min_gap=2.0, min_count=2) is False

    def test_mixed_tiny_and_regular_gaps(self):
        """Mix of tiny noise gaps and regular gaps."""
        gaps = [0.1, 5.0, 0.2, 5.0, 0.1, 5.0]  # Regular 5.0 gaps with noise
        # Should detect the regular pattern from significant gaps
        assert _is_regular_pattern(gaps, min_gap=2.0, min_count=2) is True


class TestDetectGapsInCluster:
    """Tests for gap detection between collinear segments."""

    def test_no_overlap_gaps(self):
        """Detect gaps between non-overlapping segments."""
        cluster = [
            (0, 0, 10, 0),
            (20, 0, 30, 0),
            (40, 0, 50, 0),
        ]
        # Unit vector along X axis
        gaps = _detect_gaps_in_cluster(cluster, 1.0, 0.0, 0, 0)
        assert len(gaps) == 2
        assert all(g == pytest.approx(10, abs=0.1) for g in gaps)

    def test_overlapping_no_gaps(self):
        """Overlapping segments should have no gaps."""
        cluster = [
            (0, 0, 30, 0),
            (20, 0, 50, 0),
        ]
        gaps = _detect_gaps_in_cluster(cluster, 1.0, 0.0, 0, 0)
        # Overlapping → no positive gaps
        assert all(g <= 0 for g in gaps) or len(gaps) == 0


class TestHelperFunctions:
    """Tests for helper conversion functions."""

    def test_lines_to_tuples(self):
        """Test conversion from DetectedLine objects to tuples."""
        # Create mock DetectedLine-like objects
        from dataclasses import dataclass

        @dataclass
        class MockLine:
            start: tuple
            end: tuple

        lines = [
            MockLine(start=(0, 0), end=(10, 10)),
            MockLine(start=(5, 5), end=(15, 15)),
        ]
        result = lines_to_tuples(lines)
        assert result == [(0, 0, 10, 10), (5, 5, 15, 15)]

    def test_tuples_to_merged_lines(self):
        """Test conversion from tuples to MergedLine objects."""
        tuples = [(0, 0, 10, 10), (5, 5, 15, 15)]
        result = tuples_to_merged_lines(tuples, linetype="DASHED")
        assert len(result) == 2
        assert result[0].start == (0, 0)
        assert result[0].end == (10, 10)
        assert result[0].linetype == "DASHED"
        assert result[1].start == (5, 5)
        assert result[1].end == (15, 15)


class TestMergedLineDataclass:
    """Tests for MergedLine dataclass."""

    def test_default_values(self):
        """Test MergedLine default values."""
        line = MergedLine(start=(0, 0), end=(10, 10))
        assert line.linetype == "CONTINUOUS"
        assert line.segment_count == 1

    def test_custom_values(self):
        """Test MergedLine with custom values."""
        line = MergedLine(
            start=(0, 0),
            end=(100, 0),
            linetype="DASHED",
            segment_count=5,
        )
        assert line.start == (0, 0)
        assert line.end == (100, 0)
        assert line.linetype == "DASHED"
        assert line.segment_count == 5
