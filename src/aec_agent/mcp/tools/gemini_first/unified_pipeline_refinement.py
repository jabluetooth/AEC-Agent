"""
Geometry Refinement Pipeline.

Extracted from unified_pipeline.py as part of a pure structural move refactor
(no logic changes). Consolidates all geometric cleanup operations into a
single class with consistent ordering and configuration.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np
import structlog

if TYPE_CHECKING:
    from PIL import Image as PILImage
    from .gemini_understanding import DrawingAnalysis
    from .coordinate_calibration import ScaleCalibration
    from .adaptive_extraction import EntityToCreate

logger = structlog.get_logger(__name__)

from .unified_pipeline import RefinementConfig


class GeometryRefinementPipeline:
    """
    Unified geometry refinement pipeline.

    Consolidates all geometric cleanup operations into a single class
    with consistent ordering and configuration.

    Order of operations (optimized for best results):
    1. Align nearly-parallel lines to H/V/45°
    2. Connect nearby endpoints
    3. Snap to grid
    4. Remove duplicate lines
    """

    def __init__(self, config: RefinementConfig):
        self.config = config

    async def refine(
        self,
        entities: List[Any],
    ) -> Tuple[List[Any], Dict[str, int]]:
        """
        Refine entities with geometric cleanup.

        Args:
            entities: List of EntityToCreate objects

        Returns:
            Tuple of (refined_entities, statistics)
        """
        from .gemini_refinement import (
            align_nearly_parallel_lines,
            connect_nearby_endpoints,
            snap_to_grid,
            remove_duplicate_lines,
        )

        stats = {
            "input_count": len(entities),
            "junctions_snapped": 0,
            "lines_straightened": 0,
            "endpoints_connected": 0,
            "collinear_merged": 0,
            "points_snapped": 0,
            "duplicates_removed": 0,
        }

        current = entities.copy()

        # 0. Junction detection and endpoint snapping (HAWP) - improves connectivity by 15-25%
        if self.config.use_junction_detection and self.config.image_for_junctions is not None:
            try:
                from .neural_junction_detection import (
                    detect_junctions,
                    snap_endpoints_to_junctions,
                    JunctionDetectionConfig,
                )
                from .adaptive_extraction import EntityType

                junc_config = JunctionDetectionConfig(
                    confidence_threshold=self.config.junction_confidence_threshold,
                    snap_distance=self.config.junction_snap_distance_px,
                )

                detection_result = detect_junctions(
                    self.config.image_for_junctions,
                    config=junc_config,
                )

                if detection_result.num_junctions > 0:
                    # Convert entities to line tuples for snapping
                    lines_to_snap = []
                    line_indices = []
                    for idx, e in enumerate(current):
                        etype = e.entity_type.value if hasattr(e.entity_type, 'value') else str(e.entity_type)
                        if etype == "line":
                            start_coords = e.properties.get("start", [0, 0, 0])
                            end_coords = e.properties.get("end", [0, 0, 0])
                            # Convert to Python floats (handle numpy arrays)
                            start = (float(start_coords[0]), float(start_coords[1]))
                            end = (float(end_coords[0]), float(end_coords[1]))
                            lines_to_snap.append((start, end))
                            line_indices.append(idx)

                    # Snap endpoints to detected junctions
                    if lines_to_snap:
                        snapped_lines = snap_endpoints_to_junctions(
                            lines_to_snap,
                            detection_result.junctions,
                            threshold=self.config.junction_snap_distance_px,
                        )

                        # Update entities with snapped coordinates
                        snapped_count = 0
                        for i, (orig, snapped) in enumerate(zip(lines_to_snap, snapped_lines)):
                            # Check if coordinates changed (with tolerance for float comparison)
                            orig_start, orig_end = orig
                            snap_start, snap_end = snapped
                            start_changed = abs(orig_start[0] - snap_start[0]) > 0.01 or abs(orig_start[1] - snap_start[1]) > 0.01
                            end_changed = abs(orig_end[0] - snap_end[0]) > 0.01 or abs(orig_end[1] - snap_end[1]) > 0.01

                            if start_changed or end_changed:
                                entity_idx = line_indices[i]
                                # Preserve Z coordinate from original, update X and Y
                                orig_start_z = float(current[entity_idx].properties.get("start", [0, 0, 0])[2]) if len(current[entity_idx].properties.get("start", [])) > 2 else 0.0
                                orig_end_z = float(current[entity_idx].properties.get("end", [0, 0, 0])[2]) if len(current[entity_idx].properties.get("end", [])) > 2 else 0.0
                                current[entity_idx].properties["start"] = [float(snap_start[0]), float(snap_start[1]), orig_start_z]
                                current[entity_idx].properties["end"] = [float(snap_end[0]), float(snap_end[1]), orig_end_z]
                                snapped_count += 1

                        stats["junctions_snapped"] = snapped_count
                        logger.info(
                            "junction_snapping_complete",
                            junctions_detected=detection_result.num_junctions,
                            endpoints_snapped=snapped_count,
                        )

            except Exception as e:
                logger.warning("Junction detection failed, continuing without", error=str(e))

        # 1. Align nearly-parallel lines
        if self.config.straighten_lines:
            current, adjustments = align_nearly_parallel_lines(
                current,
                angle_tolerance=self.config.straighten_tolerance_deg,
            )
            stats["lines_straightened"] = len(adjustments)

        # 2. Connect nearby endpoints
        if self.config.connect_endpoints:
            current, adjustments = connect_nearby_endpoints(
                current,
                tolerance=self.config.connect_tolerance_px,
            )
            stats["endpoints_connected"] = len(adjustments)

        # 3. Merge collinear lines (reduces fragmentation)
        if self.config.merge_collinear_lines:
            original_count = len(current)
            current = self._merge_collinear_lines(
                current,
                angle_tolerance=self.config.collinear_angle_tolerance_deg,
                gap_tolerance=self.config.collinear_gap_tolerance_px,
            )
            stats["collinear_merged"] = original_count - len(current)

        # 4. Snap to grid
        if self.config.snap_to_grid:
            current, adjustments = snap_to_grid(
                current,
                grid_size=self.config.grid_size_px,
            )
            stats["points_snapped"] = len(adjustments)

        # 5. Remove duplicates
        if self.config.remove_duplicates:
            original_count = len(current)
            current, adjustments = remove_duplicate_lines(
                current,
                tolerance=self.config.duplicate_tolerance_px,
            )
            stats["duplicates_removed"] = original_count - len(current)

        stats["output_count"] = len(current)

        return current, stats

    def _merge_collinear_lines(
        self,
        entities: List[Any],
        angle_tolerance: float = 3.0,
        gap_tolerance: float = 20.0,
    ) -> List[Any]:
        """
        Merge collinear line segments that are close together.

        This reduces fragmentation by combining line segments that:
        1. Are nearly parallel (within angle_tolerance degrees)
        2. Are collinear (lie on the same infinite line)
        3. Have endpoints within gap_tolerance of each other

        Args:
            entities: List of entities
            angle_tolerance: Max angle difference in degrees
            gap_tolerance: Max gap between line endpoints to merge

        Returns:
            List with collinear lines merged
        """
        import math
        from .adaptive_extraction import EntityType

        # Separate lines from other entities
        lines = []
        other = []
        for e in entities:
            etype = e.entity_type.value if hasattr(e.entity_type, 'value') else str(e.entity_type)
            if etype == "line":
                lines.append(e)
            else:
                other.append(e)

        if len(lines) < 2:
            return entities

        def get_line_angle(line):
            """Get angle of line in degrees (0-180)."""
            start = line.properties.get("start", (0, 0))
            end = line.properties.get("end", (0, 0))
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            angle = math.degrees(math.atan2(dy, dx))
            # Normalize to 0-180 range
            if angle < 0:
                angle += 180
            return angle

        def point_to_line_distance(px, py, x1, y1, x2, y2):
            """Calculate perpendicular distance from point to line."""
            dx = x2 - x1
            dy = y2 - y1
            length_sq = dx * dx + dy * dy
            if length_sq == 0:
                return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
            t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

        def can_merge(line1, line2):
            """Check if two lines can be merged."""
            # Check angle similarity
            angle1 = get_line_angle(line1)
            angle2 = get_line_angle(line2)
            angle_diff = abs(angle1 - angle2)
            if angle_diff > 90:
                angle_diff = 180 - angle_diff
            if angle_diff > angle_tolerance:
                return False

            # Get endpoints
            s1 = line1.properties.get("start", (0, 0))
            e1 = line1.properties.get("end", (0, 0))
            s2 = line2.properties.get("start", (0, 0))
            e2 = line2.properties.get("end", (0, 0))

            # Check collinearity (all points near the infinite line)
            for px, py in [s2, e2]:
                dist = point_to_line_distance(px, py, s1[0], s1[1], e1[0], e1[1])
                if dist > gap_tolerance:
                    return False

            # Check gap between segments
            min_gap = min(
                math.sqrt((s1[0] - s2[0]) ** 2 + (s1[1] - s2[1]) ** 2),
                math.sqrt((s1[0] - e2[0]) ** 2 + (s1[1] - e2[1]) ** 2),
                math.sqrt((e1[0] - s2[0]) ** 2 + (e1[1] - s2[1]) ** 2),
                math.sqrt((e1[0] - e2[0]) ** 2 + (e1[1] - e2[1]) ** 2),
            )
            return min_gap <= gap_tolerance

        def merge_two_lines(line1, line2):
            """Merge two collinear lines into one."""
            from copy import deepcopy

            # Get all endpoints
            s1 = line1.properties.get("start", (0, 0))
            e1 = line1.properties.get("end", (0, 0))
            s2 = line2.properties.get("start", (0, 0))
            e2 = line2.properties.get("end", (0, 0))

            all_points = [s1, e1, s2, e2]

            # Find the two points that are furthest apart
            max_dist = 0
            best_start, best_end = s1, e1
            for i, p1 in enumerate(all_points):
                for j, p2 in enumerate(all_points):
                    if i < j:
                        dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                        if dist > max_dist:
                            max_dist = dist
                            best_start, best_end = p1, p2

            # Create merged line
            merged = deepcopy(line1)
            merged.properties["start"] = best_start
            merged.properties["end"] = best_end
            return merged

        # Iteratively merge collinear lines
        merged_lines = lines.copy()
        changed = True
        while changed:
            changed = False
            new_lines = []
            used = set()

            for i, line1 in enumerate(merged_lines):
                if i in used:
                    continue

                merged = line1
                for j, line2 in enumerate(merged_lines):
                    if j <= i or j in used:
                        continue

                    if can_merge(merged, line2):
                        merged = merge_two_lines(merged, line2)
                        used.add(j)
                        changed = True

                new_lines.append(merged)
                used.add(i)

            merged_lines = new_lines

        return other + merged_lines
