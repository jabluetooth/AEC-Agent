"""
Gemini Refinement Pass - AI-Powered Coordinate Adjustment.

This module uses Gemini Vision to review and refine extracted entities,
improving accuracy by:
- Snapping lines to grid/alignment
- Connecting endpoints that should meet
- Removing duplicate/overlapping geometry
- Fixing line intersections
- Adjusting coordinates based on visual context

The refinement pass acts as a "quality control" step after OpenCV extraction,
using Gemini's visual understanding to make intelligent adjustments.

Usage:
    >>> refined = await refine_entities_with_gemini(
    ...     entities=opencv_entities,
    ...     image_path=image_path,
    ...     calibration=calibration,
    ... )
    >>> print(f"Refined {refined.adjustments_made} entities")
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog

from aec_agent.config.settings import get_settings

from .adaptive_extraction import EntityToCreate, EntityType, ExtractionSource
from .coordinate_calibration import ScaleCalibration

logger = structlog.get_logger(__name__)


@dataclass
class RefinementConfig:
    """Configuration for Gemini refinement pass."""

    # What to refine
    snap_to_grid: bool = True
    connect_endpoints: bool = True
    remove_duplicates: bool = True
    fix_intersections: bool = True
    align_parallel_lines: bool = True

    # Thresholds
    snap_tolerance_px: float = 5.0  # Pixels - snap if within this distance
    connection_tolerance_px: float = 10.0  # Pixels - connect endpoints within
    parallel_tolerance_deg: float = 5.0  # Degrees - consider parallel if within (increased for aggressive straightening)

    # Limits
    max_entities_per_batch: int = 50  # Process in batches for large drawings
    max_refinement_iterations: int = 1  # Usually 1 pass is enough


@dataclass
class RefinementAdjustment:
    """A single adjustment made during refinement."""
    entity_index: int
    adjustment_type: str  # snap, connect, remove, align, move
    original_value: Any
    new_value: Any
    reason: str
    confidence: float = 0.9


@dataclass
class RefinementResult:
    """Result of the Gemini refinement pass."""
    refined_entities: List[EntityToCreate]
    adjustments: List[RefinementAdjustment] = field(default_factory=list)

    # Statistics
    entities_input: int = 0
    entities_output: int = 0
    adjustments_made: int = 0
    entities_removed: int = 0
    endpoints_connected: int = 0
    lines_snapped: int = 0

    def to_dict(self) -> dict:
        return {
            "statistics": {
                "entities_input": self.entities_input,
                "entities_output": self.entities_output,
                "adjustments_made": self.adjustments_made,
                "entities_removed": self.entities_removed,
                "endpoints_connected": self.endpoints_connected,
                "lines_snapped": self.lines_snapped,
            },
            "adjustments": [
                {
                    "entity_index": a.entity_index,
                    "type": a.adjustment_type,
                    "reason": a.reason,
                    "confidence": a.confidence,
                }
                for a in self.adjustments[:20]  # Limit for output
            ],
        }


# =============================================================================
# Geometric Refinement (No LLM - Fast)
# =============================================================================

def snap_to_grid(
    entities: List[EntityToCreate],
    grid_size: float = 1.0,
    tolerance: float = 0.1,
) -> Tuple[List[EntityToCreate], List[RefinementAdjustment]]:
    """
    Snap coordinates to a grid for cleaner geometry.

    Args:
        entities: List of entities to process
        grid_size: Grid spacing in DWG units
        tolerance: Snap if within this distance of grid line

    Returns:
        Tuple of (refined entities, adjustments made)
    """
    refined = []
    adjustments = []

    for i, entity in enumerate(entities):
        new_entity = EntityToCreate(
            entity_type=entity.entity_type,
            layer=entity.layer,
            properties=entity.properties.copy(),
            source=entity.source,
            confidence=entity.confidence,
            source_element=entity.source_element,
        )

        if entity.entity_type in (EntityType.LINE, "line"):
            start = list(entity.properties.get("start", (0, 0)))
            end = list(entity.properties.get("end", (0, 0)))

            snapped_start = _snap_point(start, grid_size, tolerance)
            snapped_end = _snap_point(end, grid_size, tolerance)

            if snapped_start != start or snapped_end != end:
                new_entity.properties["start"] = tuple(snapped_start)
                new_entity.properties["end"] = tuple(snapped_end)
                adjustments.append(RefinementAdjustment(
                    entity_index=i,
                    adjustment_type="snap",
                    original_value={"start": start, "end": end},
                    new_value={"start": snapped_start, "end": snapped_end},
                    reason="Snapped to grid",
                ))

        elif entity.entity_type in (EntityType.CIRCLE, "circle"):
            center = list(entity.properties.get("center", (0, 0)))
            snapped_center = _snap_point(center, grid_size, tolerance)

            if snapped_center != center:
                new_entity.properties["center"] = tuple(snapped_center)
                adjustments.append(RefinementAdjustment(
                    entity_index=i,
                    adjustment_type="snap",
                    original_value={"center": center},
                    new_value={"center": snapped_center},
                    reason="Snapped center to grid",
                ))

        refined.append(new_entity)

    return refined, adjustments


def _snap_point(point: List[float], grid_size: float, tolerance: float) -> List[float]:
    """Snap a point to grid if within tolerance."""
    snapped = []
    for coord in point:
        nearest_grid = round(coord / grid_size) * grid_size
        if abs(coord - nearest_grid) <= tolerance:
            snapped.append(nearest_grid)
        else:
            snapped.append(coord)
    return snapped


def connect_nearby_endpoints(
    entities: List[EntityToCreate],
    tolerance: float = 0.5,
) -> Tuple[List[EntityToCreate], List[RefinementAdjustment]]:
    """
    Connect line endpoints that are close but not touching.

    Args:
        entities: List of entities to process
        tolerance: Connect if endpoints within this distance (DWG units)

    Returns:
        Tuple of (refined entities, adjustments made)
    """
    # Extract all line endpoints
    lines = []
    other_entities = []

    for i, entity in enumerate(entities):
        if entity.entity_type in (EntityType.LINE, "line"):
            lines.append((i, entity))
        else:
            other_entities.append(entity)

    if len(lines) < 2:
        return entities, []

    adjustments = []
    refined_lines = []

    # Build endpoint index for fast lookup
    endpoints = []  # (line_idx, is_start, point)
    for i, (orig_idx, line) in enumerate(lines):
        start = np.array(line.properties.get("start", (0, 0)))
        end = np.array(line.properties.get("end", (0, 0)))
        endpoints.append((i, True, start))
        endpoints.append((i, False, end))

    # Find nearby endpoints and connect them
    connected = set()

    for i, (line_idx_a, is_start_a, point_a) in enumerate(endpoints):
        for j, (line_idx_b, is_start_b, point_b) in enumerate(endpoints):
            if i >= j or line_idx_a == line_idx_b:
                continue

            dist = np.linalg.norm(point_a - point_b)

            if 0 < dist <= tolerance:
                # Connect by averaging the points
                midpoint = ((point_a + point_b) / 2).tolist()
                connected.add((line_idx_a, is_start_a, tuple(midpoint)))
                connected.add((line_idx_b, is_start_b, tuple(midpoint)))

    # Apply connections
    connection_map = {}
    for line_idx, is_start, new_point in connected:
        key = (line_idx, is_start)
        connection_map[key] = new_point

    for i, (orig_idx, line) in enumerate(lines):
        new_entity = EntityToCreate(
            entity_type=line.entity_type,
            layer=line.layer,
            properties=line.properties.copy(),
            source=line.source,
            confidence=line.confidence,
            source_element=line.source_element,
        )

        if (i, True) in connection_map:
            old_start = new_entity.properties["start"]
            new_entity.properties["start"] = connection_map[(i, True)]
            adjustments.append(RefinementAdjustment(
                entity_index=orig_idx,
                adjustment_type="connect",
                original_value=old_start,
                new_value=connection_map[(i, True)],
                reason="Connected nearby endpoint",
            ))

        if (i, False) in connection_map:
            old_end = new_entity.properties["end"]
            new_entity.properties["end"] = connection_map[(i, False)]
            adjustments.append(RefinementAdjustment(
                entity_index=orig_idx,
                adjustment_type="connect",
                original_value=old_end,
                new_value=connection_map[(i, False)],
                reason="Connected nearby endpoint",
            ))

        refined_lines.append(new_entity)

    return refined_lines + other_entities, adjustments


def align_nearly_parallel_lines(
    entities: List[EntityToCreate],
    angle_tolerance: float = 2.0,
) -> Tuple[List[EntityToCreate], List[RefinementAdjustment]]:
    """
    Align lines that are nearly horizontal, vertical, or at 45 degrees.

    Args:
        entities: List of entities to process
        angle_tolerance: Align if within this many degrees of cardinal direction

    Returns:
        Tuple of (refined entities, adjustments made)
    """
    refined = []
    adjustments = []

    cardinal_angles = [0, 45, 90, 135, 180]

    for i, entity in enumerate(entities):
        if entity.entity_type not in (EntityType.LINE, "line"):
            refined.append(entity)
            continue

        start = np.array(entity.properties.get("start", (0, 0)))
        end = np.array(entity.properties.get("end", (0, 0)))

        dx = end[0] - start[0]
        dy = end[1] - start[1]

        if abs(dx) < 0.001 and abs(dy) < 0.001:
            refined.append(entity)
            continue

        angle = np.degrees(np.arctan2(dy, dx)) % 180

        # Check if close to a cardinal angle
        closest_cardinal = None
        min_diff = float('inf')

        for cardinal in cardinal_angles:
            diff = abs(angle - cardinal)
            if diff > 90:
                diff = 180 - diff
            if diff < min_diff:
                min_diff = diff
                closest_cardinal = cardinal

        if min_diff <= angle_tolerance and min_diff > 0.01:
            # Align to cardinal angle
            length = np.sqrt(dx*dx + dy*dy)
            new_angle_rad = np.radians(closest_cardinal)

            new_dx = length * np.cos(new_angle_rad)
            new_dy = length * np.sin(new_angle_rad)

            # Keep start, adjust end
            new_end = (start[0] + new_dx, start[1] + new_dy)

            new_entity = EntityToCreate(
                entity_type=entity.entity_type,
                layer=entity.layer,
                properties=entity.properties.copy(),
                source=entity.source,
                confidence=entity.confidence,
                source_element=entity.source_element,
            )
            new_entity.properties["end"] = new_end

            adjustments.append(RefinementAdjustment(
                entity_index=i,
                adjustment_type="align",
                original_value={"angle": angle},
                new_value={"angle": closest_cardinal},
                reason=f"Aligned to {closest_cardinal}°",
            ))

            refined.append(new_entity)
        else:
            refined.append(entity)

    return refined, adjustments


def remove_duplicate_lines(
    entities: List[EntityToCreate],
    tolerance: float = 0.5,
) -> Tuple[List[EntityToCreate], List[RefinementAdjustment]]:
    """
    Remove duplicate or nearly overlapping lines.

    Args:
        entities: List of entities to process
        tolerance: Consider duplicate if endpoints within this distance

    Returns:
        Tuple of (refined entities, adjustments made)
    """
    lines = []
    other_entities = []

    for i, entity in enumerate(entities):
        if entity.entity_type in (EntityType.LINE, "line"):
            lines.append((i, entity))
        else:
            other_entities.append(entity)

    if len(lines) < 2:
        return entities, []

    adjustments = []
    keep_mask = [True] * len(lines)

    for i in range(len(lines)):
        if not keep_mask[i]:
            continue

        idx_i, line_i = lines[i]
        start_i = np.array(line_i.properties.get("start", (0, 0)))
        end_i = np.array(line_i.properties.get("end", (0, 0)))

        for j in range(i + 1, len(lines)):
            if not keep_mask[j]:
                continue

            idx_j, line_j = lines[j]
            start_j = np.array(line_j.properties.get("start", (0, 0)))
            end_j = np.array(line_j.properties.get("end", (0, 0)))

            # Check both orientations
            match_forward = (
                np.linalg.norm(start_i - start_j) < tolerance and
                np.linalg.norm(end_i - end_j) < tolerance
            )
            match_reverse = (
                np.linalg.norm(start_i - end_j) < tolerance and
                np.linalg.norm(end_i - start_j) < tolerance
            )

            if match_forward or match_reverse:
                keep_mask[j] = False
                adjustments.append(RefinementAdjustment(
                    entity_index=idx_j,
                    adjustment_type="remove",
                    original_value={"start": start_j.tolist(), "end": end_j.tolist()},
                    new_value=None,
                    reason="Duplicate of another line",
                ))

    refined = [lines[i][1] for i in range(len(lines)) if keep_mask[i]]
    return refined + other_entities, adjustments


# =============================================================================
# Gemini-Powered Refinement (Slower but Smarter)
# =============================================================================

async def gemini_review_entities(
    entities: List[EntityToCreate],
    image_path: Path,
    calibration: ScaleCalibration,
    batch_size: int = 50,
) -> List[RefinementAdjustment]:
    """
    Use Gemini Vision to review entities and suggest refinements.

    This is a more intelligent review that can catch context-dependent issues
    that geometric analysis alone cannot detect.

    Args:
        entities: Entities to review
        image_path: Path to the original drawing image
        calibration: Coordinate calibration for pixel/DWG conversion
        batch_size: Process entities in batches

    Returns:
        List of suggested adjustments
    """
    from . import gemini_call_with_retry, get_gemini_client
    from google.genai import types

    settings = get_settings()

    if not image_path.exists():
        logger.warning("Image not found for Gemini review", path=str(image_path))
        return []

    # Only review lines for now
    lines = [e for e in entities if e.entity_type in (EntityType.LINE, "line")]

    if not lines:
        return []

    # Encode image
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # Build entity summary for prompt
    entity_summary = []
    for i, line in enumerate(lines[:batch_size]):
        start = line.properties.get("start", (0, 0))
        end = line.properties.get("end", (0, 0))
        entity_summary.append(f"Line {i}: ({start[0]:.1f}, {start[1]:.1f}) to ({end[0]:.1f}, {end[1]:.1f})")

    prompt = f"""You are reviewing extracted line entities from a technical drawing.

The drawing was processed at {calibration.dpi} DPI with scale factor {calibration.scale_factor:.4f}.

Here are the first {len(entity_summary)} lines extracted (in DWG coordinates):
{chr(10).join(entity_summary[:30])}

Looking at the original drawing image, identify any obvious issues:
1. Lines that appear misaligned (should be horizontal/vertical but aren't)
2. Endpoints that should connect but have a small gap
3. Lines that appear to be duplicates
4. Lines that don't match visible elements in the drawing

Return a JSON array of adjustments. Each adjustment should have:
- "line_index": the line number from the list above
- "issue": brief description of the problem
- "suggestion": "snap", "connect", "remove", or "adjust"
- "confidence": 0.0 to 1.0

Only report clear issues. If the extraction looks good, return an empty array [].

Return ONLY valid JSON, no other text."""

    try:
        client = get_gemini_client()

        response = await gemini_call_with_retry(
            client,
            content=[
                prompt,
                types.Part.from_bytes(
                    data=base64.b64decode(image_data),
                    mime_type="image/png"
                ),
            ],
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 4096,
            },
            model_name=settings.gemini_model,
        )

        # Parse JSON response
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            suggestions = json.loads(json_match.group())

            adjustments = []
            for s in suggestions:
                if isinstance(s, dict) and "line_index" in s:
                    adjustments.append(RefinementAdjustment(
                        entity_index=s.get("line_index", 0),
                        adjustment_type=s.get("suggestion", "unknown"),
                        original_value=None,
                        new_value=None,
                        reason=s.get("issue", "Gemini suggestion"),
                        confidence=s.get("confidence", 0.8),
                    ))

            logger.info(
                "gemini_review_complete",
                suggestions=len(adjustments),
            )
            return adjustments

        return []

    except Exception as e:
        logger.warning("gemini_review_failed", error=str(e))
        return []


# =============================================================================
# Main Refinement Function
# =============================================================================

async def refine_entities_with_gemini(
    entities: List[EntityToCreate],
    image_path: Optional[Path] = None,
    calibration: Optional[ScaleCalibration] = None,
    config: Optional[RefinementConfig] = None,
) -> RefinementResult:
    """
    Refine extracted entities using geometric analysis and Gemini review.

    This is the main entry point for the refinement pass. It applies
    multiple refinement strategies in order:

    1. Remove duplicates (fast, geometric)
    2. Connect nearby endpoints (fast, geometric)
    3. Align nearly-parallel lines (fast, geometric)
    4. Snap to grid (fast, geometric)
    5. Gemini review (slower, AI-powered) - if image provided

    Args:
        entities: List of entities to refine
        image_path: Optional path to original image for Gemini review
        calibration: Optional calibration for coordinate conversion
        config: Optional refinement configuration

    Returns:
        RefinementResult with refined entities and statistics

    Example:
        >>> result = await refine_entities_with_gemini(
        ...     entities=opencv_entities,
        ...     image_path=image_path,
        ...     calibration=calibration,
        ... )
        >>> print(f"Made {result.adjustments_made} adjustments")
    """
    config = config or RefinementConfig()

    result = RefinementResult(
        refined_entities=[],
        entities_input=len(entities),
    )

    if not entities:
        return result

    logger.info(
        "refinement_starting",
        entities=len(entities),
        config={
            "snap_to_grid": config.snap_to_grid,
            "connect_endpoints": config.connect_endpoints,
            "remove_duplicates": config.remove_duplicates,
            "align_parallel_lines": config.align_parallel_lines,
        },
    )

    current_entities = entities.copy()
    all_adjustments = []

    # Step 1: Remove duplicates
    if config.remove_duplicates:
        current_entities, adjustments = remove_duplicate_lines(
            current_entities,
            tolerance=config.snap_tolerance_px * 0.01,  # Convert to DWG units approx
        )
        all_adjustments.extend(adjustments)
        result.entities_removed = len([a for a in adjustments if a.adjustment_type == "remove"])

    # Step 2: Connect nearby endpoints
    if config.connect_endpoints:
        current_entities, adjustments = connect_nearby_endpoints(
            current_entities,
            tolerance=config.connection_tolerance_px * 0.01,
        )
        all_adjustments.extend(adjustments)
        result.endpoints_connected = len([a for a in adjustments if a.adjustment_type == "connect"])

    # Step 3: Align nearly-parallel lines
    if config.align_parallel_lines:
        current_entities, adjustments = align_nearly_parallel_lines(
            current_entities,
            angle_tolerance=config.parallel_tolerance_deg,
        )
        all_adjustments.extend(adjustments)

    # Step 4: Snap to grid
    if config.snap_to_grid:
        current_entities, adjustments = snap_to_grid(
            current_entities,
            grid_size=1.0,  # 1 DWG unit
            tolerance=0.1,
        )
        all_adjustments.extend(adjustments)
        result.lines_snapped = len([a for a in adjustments if a.adjustment_type == "snap"])

    # Step 5: Gemini review (if image provided)
    if image_path and calibration:
        gemini_adjustments = await gemini_review_entities(
            current_entities,
            image_path,
            calibration,
            batch_size=config.max_entities_per_batch,
        )
        all_adjustments.extend(gemini_adjustments)

    # Compile results
    result.refined_entities = current_entities
    result.adjustments = all_adjustments
    result.entities_output = len(current_entities)
    result.adjustments_made = len(all_adjustments)

    logger.info(
        "refinement_complete",
        entities_in=result.entities_input,
        entities_out=result.entities_output,
        adjustments=result.adjustments_made,
        removed=result.entities_removed,
        connected=result.endpoints_connected,
        snapped=result.lines_snapped,
    )

    return result
