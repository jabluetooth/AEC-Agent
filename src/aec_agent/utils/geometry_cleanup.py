"""
AEC-specific Geometric Cleanup Utilities.

This module provides orthogonal snapping and collinear line merging for
engineering drawings where most lines are intended to be exactly horizontal
or vertical. These are deterministic algorithms that don't require AI.

Part of Phase 2.5: Semantic AEC Vectorization Pipeline.

Key Functions:
- snap_to_orthogonal(): Snap near-orthogonal lines to exact 0/90/180/270 degrees
- merge_collinear_lines(): Merge fragmented collinear segments into single lines
"""

import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MergedLine:
    """Result of merging collinear line segments."""
    start: Tuple[float, float]
    end: Tuple[float, float]
    linetype: str = "CONTINUOUS"  # "CONTINUOUS", "DASHED", "HIDDEN", etc.
    segment_count: int = 1  # Number of original segments merged


def snap_to_orthogonal(
    lines: List[Tuple[float, float, float, float]],
    angle_tolerance: float = 2.0,
) -> List[Tuple[float, float, float, float]]:
    """
    Snap near-orthogonal lines to exact 0/90/180/270 degrees.

    Engineering drawings have lines that are INTENDED to be horizontal or
    vertical, but scanning and detection introduce small angular errors.
    This function snaps lines within ±tolerance of orthogonal angles
    to their exact intended orientation while preserving line length.

    Args:
        lines: List of (x1, y1, x2, y2) line endpoints.
        angle_tolerance: Maximum deviation in degrees to snap (default 2.0).
                        Lines within ±tolerance of 0°, 90°, 180°, or 270°
                        will be snapped to exact orthogonal.

    Returns:
        List of snapped line endpoints (x1, y1, x2, y2).

    Example:
        >>> lines = [(0, 0, 100, 1.75)]  # ~1 degree off horizontal
        >>> snapped = snap_to_orthogonal(lines, angle_tolerance=2.0)
        >>> # Result: [(0, 0, 100, 0)] - exactly horizontal
    """
    if not lines:
        return []

    snapped: List[Tuple[float, float, float, float]] = []
    orthogonal_angles = [0, 90, 180, 270, 360]  # 360 for wraparound to 0

    stats = {"total": len(lines), "snapped": 0}

    for x1, y1, x2, y2 in lines:
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)

        if length < 1e-6:
            # Zero-length line, keep as-is
            snapped.append((x1, y1, x2, y2))
            continue

        # Calculate angle in degrees (0-360, CCW from +X axis)
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360

        # Find nearest orthogonal angle
        best_ortho: Optional[int] = None
        best_diff = angle_tolerance + 1  # Start outside tolerance

        for ortho in orthogonal_angles:
            diff = abs(angle - ortho)
            if diff > 180:
                diff = 360 - diff
            if diff < best_diff:
                best_diff = diff
                best_ortho = ortho

        if best_ortho is not None and best_diff <= angle_tolerance:
            # Snap to orthogonal: adjust endpoint to maintain length
            # Normalize 360 to 0
            if best_ortho == 360:
                best_ortho = 0

            if best_ortho == 0:  # Horizontal right
                new_x2 = x1 + length
                new_y2 = y1
            elif best_ortho == 90:  # Vertical up
                new_x2 = x1
                new_y2 = y1 + length
            elif best_ortho == 180:  # Horizontal left
                new_x2 = x1 - length
                new_y2 = y1
            elif best_ortho == 270:  # Vertical down
                new_x2 = x1
                new_y2 = y1 - length
            else:
                new_x2, new_y2 = x2, y2

            snapped.append((x1, y1, new_x2, new_y2))
            stats["snapped"] += 1
        else:
            # Keep original (diagonal line)
            snapped.append((x1, y1, x2, y2))

    logger.debug(
        "Orthogonal snapping complete",
        total=stats["total"],
        snapped=stats["snapped"],
        tolerance=angle_tolerance,
    )

    return snapped


def merge_collinear_lines(
    lines: List[Tuple[float, float, float, float]],
    angle_tolerance: float = 2.0,
    distance_tolerance: float = 5.0,
    gap_tolerance: float = 20.0,
    min_gap_for_dashed: float = 2.0,
) -> Tuple[List[MergedLine], List[MergedLine]]:
    """
    Merge collinear line segments into single lines.

    Vectorization often produces fragmented lines from what should be a
    single continuous line (due to noise, gaps, or dashed patterns).
    This function groups collinear segments and merges them, optionally
    detecting dashed line patterns.

    Args:
        lines: List of (x1, y1, x2, y2) line endpoints.
        angle_tolerance: Maximum angle difference to consider collinear (degrees).
        distance_tolerance: Maximum perpendicular distance from the line to
                           consider collinear (drawing units).
        gap_tolerance: Maximum gap between segments to merge (drawing units).
        min_gap_for_dashed: Minimum gap size to consider for dashed detection.

    Returns:
        Tuple of:
        - continuous_lines: List of MergedLine with linetype="CONTINUOUS"
        - dashed_lines: List of MergedLine with linetype="DASHED"

    Example:
        >>> lines = [
        ...     (0, 0, 50, 0),
        ...     (60, 0, 100, 0),  # 10-unit gap
        ... ]
        >>> continuous, dashed = merge_collinear_lines(lines, gap_tolerance=20.0)
        >>> # Result: one merged line from (0,0) to (100,0)
    """
    if not lines:
        return [], []

    # Group lines by angle (quantized to 1-degree buckets for efficiency)
    # Use direction-agnostic angle (0-180) since line direction doesn't matter
    angle_groups: Dict[int, List[Tuple[float, float, float, float]]] = {}

    for line in lines:
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            continue  # Skip zero-length lines

        # Direction-agnostic angle (0-180)
        angle = math.degrees(math.atan2(dy, dx)) % 180
        bucket = round(angle)
        if bucket not in angle_groups:
            angle_groups[bucket] = []
        angle_groups[bucket].append(line)

    continuous_lines: List[MergedLine] = []
    dashed_lines: List[MergedLine] = []

    for bucket, group in angle_groups.items():
        if len(group) == 1:
            x1, y1, x2, y2 = group[0]
            continuous_lines.append(MergedLine(
                start=(x1, y1),
                end=(x2, y2),
                linetype="CONTINUOUS",
                segment_count=1,
            ))
            continue

        # Within angle group, cluster by perpendicular distance from a reference
        processed = [False] * len(group)

        for i, line_i in enumerate(group):
            if processed[i]:
                continue

            # Start a collinear cluster with line_i
            cluster = [line_i]
            processed[i] = True

            x1_i, y1_i, x2_i, y2_i = line_i
            dx_i = x2_i - x1_i
            dy_i = y2_i - y1_i
            len_i = math.sqrt(dx_i * dx_i + dy_i * dy_i)

            if len_i < 1e-6:
                continue

            # Unit vector along the line
            ux, uy = dx_i / len_i, dy_i / len_i

            # Perpendicular unit vector
            px, py = -uy, ux

            # Perpendicular distance of line_i from origin (use start point)
            perp_dist_i = x1_i * px + y1_i * py

            for j, line_j in enumerate(group):
                if processed[j]:
                    continue

                x1_j, y1_j, x2_j, y2_j = line_j

                # Check perpendicular distance (are they on the same "track"?)
                perp_dist_j = x1_j * px + y1_j * py
                if abs(perp_dist_j - perp_dist_i) > distance_tolerance:
                    continue

                # Project all 4 endpoints onto the line direction to check overlap/gap
                t1_i = 0  # Reference point
                t2_i = len_i
                t1_j = (x1_j - x1_i) * ux + (y1_j - y1_i) * uy
                t2_j = (x2_j - x1_i) * ux + (y2_j - y1_i) * uy

                min_j, max_j = min(t1_j, t2_j), max(t1_j, t2_j)
                min_i, max_i = min(t1_i, t2_i), max(t1_i, t2_i)

                # Check for overlap or close gap
                gap = max(min_j - max_i, min_i - max_j)
                if gap <= gap_tolerance:
                    cluster.append(line_j)
                    processed[j] = True

            # Merge the cluster
            merged = _merge_cluster(
                cluster, ux, uy, x1_i, y1_i,
                min_gap_for_dashed=min_gap_for_dashed,
            )

            if merged.linetype == "DASHED":
                dashed_lines.append(merged)
            else:
                continuous_lines.append(merged)

    logger.debug(
        "Collinear merging complete",
        input_lines=len(lines),
        continuous=len(continuous_lines),
        dashed=len(dashed_lines),
    )

    return continuous_lines, dashed_lines


def _merge_cluster(
    cluster: List[Tuple[float, float, float, float]],
    ux: float,
    uy: float,
    origin_x: float,
    origin_y: float,
    min_gap_for_dashed: float = 2.0,
) -> MergedLine:
    """
    Merge a cluster of collinear lines into a single MergedLine.

    Args:
        cluster: List of collinear line segments.
        ux, uy: Unit vector along the line direction.
        origin_x, origin_y: Reference point for projection.
        min_gap_for_dashed: Minimum gap to consider for dashed pattern detection.

    Returns:
        MergedLine with appropriate linetype.
    """
    if len(cluster) == 1:
        x1, y1, x2, y2 = cluster[0]
        return MergedLine(
            start=(x1, y1),
            end=(x2, y2),
            linetype="CONTINUOUS",
            segment_count=1,
        )

    # Project all endpoints onto the line direction
    all_t: List[float] = []
    for x1, y1, x2, y2 in cluster:
        t1 = (x1 - origin_x) * ux + (y1 - origin_y) * uy
        t2 = (x2 - origin_x) * ux + (y2 - origin_y) * uy
        all_t.extend([t1, t2])

    t_min, t_max = min(all_t), max(all_t)

    # Calculate merged endpoints
    new_x1 = origin_x + t_min * ux
    new_y1 = origin_y + t_min * uy
    new_x2 = origin_x + t_max * ux
    new_y2 = origin_y + t_max * uy

    # Detect dashed pattern by analyzing gaps
    gaps = _detect_gaps_in_cluster(cluster, ux, uy, origin_x, origin_y)
    is_dashed = _is_regular_pattern(gaps, min_gap=min_gap_for_dashed)

    return MergedLine(
        start=(new_x1, new_y1),
        end=(new_x2, new_y2),
        linetype="DASHED" if is_dashed else "CONTINUOUS",
        segment_count=len(cluster),
    )


def _detect_gaps_in_cluster(
    cluster: List[Tuple[float, float, float, float]],
    ux: float,
    uy: float,
    origin_x: float,
    origin_y: float,
) -> List[float]:
    """
    Detect gaps between segments in a collinear cluster.

    Projects all segments onto the line direction and finds gaps between them.
    """
    # Project all segments to 1D intervals
    segments_1d: List[Tuple[float, float]] = []
    for x1, y1, x2, y2 in cluster:
        t1 = (x1 - origin_x) * ux + (y1 - origin_y) * uy
        t2 = (x2 - origin_x) * ux + (y2 - origin_y) * uy
        segments_1d.append((min(t1, t2), max(t1, t2)))

    # Sort by start position
    segments_1d.sort()

    # Find gaps between consecutive segments
    gaps: List[float] = []
    for i in range(len(segments_1d) - 1):
        gap = segments_1d[i + 1][0] - segments_1d[i][1]
        if gap > 0:  # Only count actual gaps
            gaps.append(gap)

    return gaps


def _is_regular_pattern(
    gaps: List[float],
    min_gap: float = 2.0,
    tolerance: float = 0.3,
    min_count: int = 2,
) -> bool:
    """
    Check if gaps form a regular (dashed) pattern.

    A dashed line has gaps of approximately equal size. This function
    checks if the gap sizes are consistent enough to indicate a dashed pattern.

    Args:
        gaps: List of gap sizes.
        min_gap: Minimum gap size to consider (filters out tiny gaps from noise).
        tolerance: Maximum relative deviation from mean to be considered regular.
        min_count: Minimum number of gaps required for dashed detection.

    Returns:
        True if the gaps form a regular dashed pattern.
    """
    # Filter out tiny gaps that are likely noise
    significant_gaps = [g for g in gaps if g >= min_gap]

    if len(significant_gaps) < min_count:
        return False

    mean_gap = sum(significant_gaps) / len(significant_gaps)
    if mean_gap < min_gap:
        return False

    # Check if all gaps are within tolerance of the mean
    for gap in significant_gaps:
        relative_diff = abs(gap - mean_gap) / mean_gap
        if relative_diff > tolerance:
            return False

    return True


def lines_to_tuples(
    lines: List["DetectedLine"],
) -> List[Tuple[float, float, float, float]]:
    """
    Convert DetectedLine objects to (x1, y1, x2, y2) tuples.

    Helper function for integration with image_vectorizer.py.
    """
    return [(l.start[0], l.start[1], l.end[0], l.end[1]) for l in lines]


def tuples_to_merged_lines(
    tuples: List[Tuple[float, float, float, float]],
    linetype: str = "CONTINUOUS",
) -> List[MergedLine]:
    """
    Convert (x1, y1, x2, y2) tuples to MergedLine objects.

    Helper function for integration with image_vectorizer.py.
    """
    return [
        MergedLine(start=(x1, y1), end=(x2, y2), linetype=linetype, segment_count=1)
        for x1, y1, x2, y2 in tuples
    ]
