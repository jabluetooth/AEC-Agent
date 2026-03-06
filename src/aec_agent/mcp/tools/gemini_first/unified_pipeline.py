"""
Unified PDF to AutoCAD Vectorization Pipeline.

This module consolidates all vectorization approaches into a single, configurable
pipeline that replaces the previous fragmented implementations:
- gemini_extract_pdf_entities (removed)
- gemini_vectorize_pdf (removed)
- gemini_complete_pipeline (removed)
- BestPracticesPipeline (refactored to use this)

The unified pipeline provides:
1. Single entry point with feature flags
2. Consistent configuration across all use cases
3. Strategy pattern for extraction methods
4. Unified geometry refinement
5. Optional stages that can be enabled/disabled

Usage:
    from aec_agent.mcp.tools.gemini_first import UnifiedPipeline, PipelineConfig

    config = PipelineConfig(
        extraction_method="hybrid",
        preprocess=True,
        use_symbol_rag=True,
        create_in_autocad=True,
    )
    pipeline = UnifiedPipeline(config)
    result = await pipeline.process(pdf_path="drawing.pdf", page=1)
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


# =============================================================================
# Enums
# =============================================================================

class ExtractionMethod(str, Enum):
    """Extraction method to use."""
    DIRECT = "direct"  # Gemini coordinates only - fast, less accurate
    HYBRID = "hybrid"  # Gemini + OpenCV - balanced
    BEST = "best"  # Best algorithm per entity type - highest quality
    VTRACER = "vtracer"  # Raster-to-vector for scanned drawings


class SymbolMethod(str, Enum):
    """Symbol recognition method."""
    RAG = "rag"  # CLIP embeddings + pgvector (recommended)
    HARDCODED = "hardcoded"  # Legacy SYMBOL_TO_BLOCK mapping
    GEMINI_ONLY = "gemini_only"  # Trust Gemini's classification
    DISABLED = "disabled"  # Skip symbol recognition


class OutputFormat(str, Enum):
    """Output format."""
    ENTITIES = "entities"  # Return EntityToCreate list
    AUTOCAD = "autocad"  # Create in AutoCAD and return stats
    DXF = "dxf"  # Export to DXF file (future)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """
    Unified pipeline configuration.

    All options have sensible defaults for high-quality vectorization.
    Disable stages by setting their flags to False.
    """

    # === Stage 1: PDF Rendering ===
    dpi: int = 300
    auto_dpi: bool = True  # Auto-select based on content complexity
    convert_grayscale: bool = True

    # === Stage 2: Preprocessing (optional) ===
    preprocess: bool = True
    denoise: bool = True
    denoise_strength: int = 5  # Reduced from 10 to preserve fine details (dashed lines)
    deskew: bool = True
    binarize: bool = False  # DISABLED - destroys grayscale info for lineweight detection
    enhance_contrast: bool = True
    super_resolution: bool = False  # Enable for low-DPI scans
    super_resolution_scale: int = 4

    # === Stage 3: Gemini Analysis ===
    gemini_model: str = "gemini-2.5-flash"
    analyze_scale: bool = True
    analyze_layers: bool = True

    # === Stage 4: Extraction ===
    extraction_method: ExtractionMethod = ExtractionMethod.HYBRID
    extract_lines: bool = True
    extract_circles: bool = True
    extract_arcs: bool = True
    extract_text: bool = True
    extract_symbols: bool = True
    extract_dimensions: bool = True
    use_hawp_junctions: bool = True  # Neural junction detection (improves output by 15-25%)

    # === Stage 4b: Curve Detection (ellipses, arcs, bezier) ===
    detect_curves: bool = True  # Enable ellipse/arc/spline detection
    detect_ellipses: bool = True  # Detect ellipses (not circles)
    detect_splines: bool = True  # Fit bezier curves to curved contours
    min_curve_length: int = 20  # Minimum contour length for curve fitting
    curve_fit_tolerance: float = 2.0  # Max deviation for bezier fit (pixels)

    # === Stage 5: Symbol Recognition (optional) ===
    symbol_method: SymbolMethod = SymbolMethod.RAG
    symbol_min_confidence: float = 0.5
    symbol_top_k: int = 3

    # === Stage 6: Geometry Refinement (optional) ===
    refine_geometry: bool = True
    straighten_tolerance_deg: float = 5.0  # Snap to H/V/45° if within tolerance
    connect_tolerance_px: float = 25.0  # Increased from 10 for better line continuity
    grid_size_px: float = 5.0
    remove_duplicates: bool = True
    duplicate_tolerance_px: float = 5.0  # Increased from 2 for better deduplication
    merge_collinear_lines: bool = True  # Merge collinear line segments
    collinear_angle_tolerance_deg: float = 3.0  # Angle tolerance for collinear detection
    collinear_gap_tolerance_px: float = 20.0  # Max gap to merge collinear lines

    # === Stage 7: Validation (optional) ===
    validate: bool = True
    gemini_visual_qa: bool = True
    max_validation_iterations: int = 3

    # === Stage 8: Output ===
    output_format: OutputFormat = OutputFormat.ENTITIES
    create_in_autocad: bool = False
    use_ncs_layers: bool = True

    # === Database ===
    store_to_database: bool = False

    # === Debug ===
    debug_mode: bool = True  # Save debug screenshots at each pipeline stage
    debug_output_dir: Optional[str] = None  # Directory for debug images (auto-created if None)

    def __post_init__(self):
        """Validate configuration."""
        if isinstance(self.extraction_method, str):
            self.extraction_method = ExtractionMethod(self.extraction_method)
        if isinstance(self.symbol_method, str):
            self.symbol_method = SymbolMethod(self.symbol_method)
        if isinstance(self.output_format, str):
            self.output_format = OutputFormat(self.output_format)


# =============================================================================
# Pipeline Result
# =============================================================================

@dataclass
class StageResult:
    """Result from a pipeline stage."""
    stage: str
    success: bool
    duration_ms: float
    data: Any = None
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Result from the unified pipeline."""
    success: bool
    pipeline_id: UUID
    stages: List[StageResult] = field(default_factory=list)

    # Stage outputs
    image_path: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    analysis: Optional[Any] = None  # DrawingAnalysis
    calibration: Optional[Any] = None  # ScaleCalibration
    entities: List[Any] = field(default_factory=list)  # EntityToCreate

    # Statistics
    total_entities: int = 0
    entities_by_type: Dict[str, int] = field(default_factory=dict)
    entities_created: int = 0
    creation_failures: int = 0

    # Validation
    validation_passed: bool = False
    validation_issues: List[str] = field(default_factory=list)

    # Timing
    total_duration_ms: float = 0.0

    # Error handling
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "pipeline_id": str(self.pipeline_id),
            "total_entities": self.total_entities,
            "entities_by_type": self.entities_by_type,
            "entities_created": self.entities_created,
            "creation_failures": self.creation_failures,
            "validation_passed": self.validation_passed,
            "validation_issues": self.validation_issues,
            "total_duration_ms": self.total_duration_ms,
            "stages": [
                {
                    "stage": s.stage,
                    "success": s.success,
                    "duration_ms": s.duration_ms,
                    "error": s.error,
                }
                for s in self.stages
            ],
            "error": self.error,
        }


# =============================================================================
# Geometry Refinement Pipeline
# =============================================================================

@dataclass
class RefinementConfig:
    """Configuration for geometry refinement."""
    # Junction detection (HAWP) - improves connectivity by 15-25%
    use_junction_detection: bool = True
    junction_snap_distance_px: float = 8.0  # Snap endpoints within this distance to junctions
    junction_confidence_threshold: float = 0.5
    # Line straightening
    straighten_lines: bool = True
    straighten_tolerance_deg: float = 5.0
    connect_endpoints: bool = True
    connect_tolerance_px: float = 25.0  # Increased for better continuity
    snap_to_grid: bool = True
    grid_size_px: float = 5.0
    remove_duplicates: bool = True
    duplicate_tolerance_px: float = 5.0  # Increased for better deduplication
    merge_collinear_lines: bool = True
    collinear_angle_tolerance_deg: float = 3.0
    collinear_gap_tolerance_px: float = 20.0
    # Image for junction detection (set by pipeline)
    image_for_junctions: Optional[np.ndarray] = None


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


# =============================================================================
# Extraction Factory
# =============================================================================

class ExtractionFactory:
    """
    Factory for creating extraction strategies.

    Provides a unified interface to different extraction methods:
    - Direct: Gemini coordinates only
    - Hybrid: Gemini + OpenCV
    - Best: Optimal algorithm per entity type
    - VTracer: Raster-to-vector
    """

    @staticmethod
    async def extract(
        method: ExtractionMethod,
        image_path: str,
        analysis: Any,  # DrawingAnalysis
        calibration: Any,  # ScaleCalibration
        config: PipelineConfig,
    ) -> List[Any]:
        """
        Extract entities using the specified method.

        Args:
            method: Extraction method to use
            image_path: Path to the rendered PDF image
            analysis: Gemini drawing analysis
            calibration: Coordinate calibration
            config: Pipeline configuration

        Returns:
            List of EntityToCreate objects
        """
        from .adaptive_extraction import (
            extract_all,
            extract_direct_only,
            hybrid_extract_all,
            HybridExtractionConfig,
        )

        if method == ExtractionMethod.DIRECT:
            # Fast: Gemini coordinates only
            result = await extract_direct_only(
                analysis=analysis,
                calibration=calibration,
            )
            return result.entities

        elif method == ExtractionMethod.HYBRID:
            # Balanced: Gemini + OpenCV
            hybrid_config = HybridExtractionConfig(
                use_opencv_for_lines=True,
                use_opencv_for_circles=True,
                use_yolo_for_symbols=False,
                use_gemini_for_text=config.extract_text,
                use_gemini_for_semantic=True,
            )
            result = await hybrid_extract_all(
                image_path=image_path,
                analysis=analysis,
                calibration=calibration,
                config=hybrid_config,
            )
            return result.entities

        elif method == ExtractionMethod.BEST:
            # Highest quality: Best algorithm per entity type
            hybrid_config = HybridExtractionConfig(
                use_opencv_for_lines=True,
                use_opencv_for_circles=True,
                use_yolo_for_symbols=config.extract_symbols,
                use_gemini_for_text=config.extract_text,
                use_gemini_for_semantic=True,
                merge_opencv_gemini=True,
            )
            result = await hybrid_extract_all(
                image_path=image_path,
                analysis=analysis,
                calibration=calibration,
                config=hybrid_config,
            )
            return result.entities

        elif method == ExtractionMethod.VTRACER:
            # Raster-to-vector for scanned drawings
            from .vtracer_extraction import VTracerExtractor, is_vtracer_available

            if not is_vtracer_available():
                logger.warning("VTracer not available, falling back to hybrid")
                return await ExtractionFactory.extract(
                    ExtractionMethod.HYBRID,
                    image_path,
                    analysis,
                    calibration,
                    config,
                )

            from PIL import Image
            img = Image.open(image_path)
            extractor = VTracerExtractor(np.array(img))
            vtracer_result = extractor.extract()

            # Convert VTracer paths to entities
            entities = extractor.paths_to_entities(
                vtracer_result.paths,
                calibration,
            )
            return entities

        else:
            raise ValueError(f"Unknown extraction method: {method}")


# =============================================================================
# Unified Pipeline
# =============================================================================

class DebugVisualizer:
    """Helper class for saving debug screenshots at each pipeline stage."""

    def __init__(self, output_dir: Path, enabled: bool = True):
        self.output_dir = output_dir
        self.enabled = enabled
        self.step_counter = 0

        if enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info("debug_visualizer_initialized", output_dir=str(output_dir))

    def save_image(self, image: np.ndarray, stage: str, suffix: str = "") -> Optional[str]:
        """Save an image with stage name and step counter."""
        if not self.enabled:
            return None

        try:
            import cv2
            self.step_counter += 1
            filename = f"{self.step_counter:02d}_{stage}"
            if suffix:
                filename += f"_{suffix}"
            filename += ".png"

            filepath = self.output_dir / filename
            cv2.imwrite(str(filepath), image)
            logger.info("debug_image_saved", stage=stage, path=str(filepath))
            return str(filepath)
        except Exception as e:
            logger.warning("debug_image_save_failed", stage=stage, error=str(e))
            return None

    def save_with_overlay(
        self,
        image: np.ndarray,
        stage: str,
        lines: Optional[List] = None,
        circles: Optional[List] = None,
        text_regions: Optional[List] = None,
        junctions: Optional[List] = None,
    ) -> Optional[str]:
        """Save image with detected elements overlaid."""
        if not self.enabled:
            return None

        try:
            import cv2

            # Convert to color if grayscale
            if len(image.shape) == 2:
                vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                vis = image.copy()

            # Draw lines in green
            if lines:
                for line in lines:
                    if hasattr(line, 'start') and hasattr(line, 'end'):
                        start = (int(line.start[0]), int(line.start[1]))
                        end = (int(line.end[0]), int(line.end[1]))
                        # Color based on line type
                        color = (0, 255, 0)  # Green for continuous
                        if hasattr(line, 'line_type'):
                            lt = line.line_type.value if hasattr(line.line_type, 'value') else str(line.line_type)
                            if lt == "dashed":
                                color = (0, 165, 255)  # Orange
                            elif lt == "dotted":
                                color = (255, 0, 255)  # Magenta
                            elif lt == "center":
                                color = (255, 255, 0)  # Cyan
                        cv2.line(vis, start, end, color, 2)

            # Draw circles in blue
            if circles:
                for circle in circles:
                    if hasattr(circle, 'center') and hasattr(circle, 'radius'):
                        center = (int(circle.center[0]), int(circle.center[1]))
                        radius = int(circle.radius)
                        cv2.circle(vis, center, radius, (255, 0, 0), 2)

            # Draw text regions in yellow
            if text_regions:
                for region in text_regions:
                    if isinstance(region, dict) and 'bounds' in region:
                        x1, y1, x2, y2 = [int(v) for v in region['bounds']]
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    elif hasattr(region, 'position'):
                        pos = region.position
                        x, y = int(pos[0]), int(pos[1])
                        cv2.rectangle(vis, (x-5, y-5), (x+50, y+15), (0, 255, 255), 2)

            # Draw junctions in red
            if junctions:
                for junc in junctions:
                    if hasattr(junc, 'position'):
                        pos = (int(junc.position[0]), int(junc.position[1]))
                        cv2.circle(vis, pos, 5, (0, 0, 255), -1)

            return self.save_image(vis, stage, "overlay")
        except Exception as e:
            logger.warning("debug_overlay_save_failed", stage=stage, error=str(e))
            return None


class UnifiedPipeline:
    """
    Unified PDF to AutoCAD vectorization pipeline.

    Consolidates all vectorization workflows into a single, configurable class.

    Stages:
    1. PDF Rendering
    2. Preprocessing (optional)
    3. Gemini Analysis
    4. Calibration
    5. Extraction
    6. Symbol Recognition (optional)
    7. Geometry Refinement (optional)
    8. AutoCAD Creation (optional)
    9. Validation (optional)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration. Uses defaults if not provided.
        """
        self.config = config or PipelineConfig()
        self.pipeline_id = uuid4()
        self.debug: Optional[DebugVisualizer] = None

    async def process(
        self,
        pdf_path: str,
        page: int = 1,
        output_dir: Optional[str] = None,
    ) -> PipelineResult:
        """
        Process a PDF page through the vectorization pipeline.

        Args:
            pdf_path: Path to the PDF file
            page: Page number (1-indexed)
            output_dir: Optional output directory for intermediate files

        Returns:
            PipelineResult with extracted entities and statistics
        """
        import time
        import cv2
        start_time = time.perf_counter()

        result = PipelineResult(
            success=False,
            pipeline_id=self.pipeline_id,
        )

        # Initialize debug visualizer
        if self.config.debug_mode:
            debug_dir = Path(self.config.debug_output_dir) if self.config.debug_output_dir else Path(output_dir or ".") / f"debug_{self.pipeline_id.hex[:8]}"
            self.debug = DebugVisualizer(debug_dir, enabled=True)
            result.debug_dir = str(debug_dir)
            logger.info("debug_mode_enabled", debug_dir=str(debug_dir))
        else:
            self.debug = DebugVisualizer(Path("."), enabled=False)

        try:
            # Stage 1: PDF Rendering
            image_path = await self._stage_render(pdf_path, page, output_dir, result)
            if not image_path:
                return result
            result.image_path = image_path

            # IMPORTANT: Keep original image path for text/linetype/lineweight detection
            original_image_path = image_path

            # Debug: Save rendered image
            if self.debug.enabled:
                img = cv2.imread(image_path)
                if img is not None:
                    self.debug.save_image(img, "01_rendered_pdf")

            # Stage 2: Preprocessing (optional) - only for geometry extraction
            processed_image_path = image_path
            if self.config.preprocess:
                processed_image_path = await self._stage_preprocess(image_path, result)

                # Debug: Save preprocessed image
                if self.debug.enabled:
                    img = cv2.imread(processed_image_path)
                    if img is not None:
                        self.debug.save_image(img, "02_preprocessed")

            # Stage 3: Gemini Analysis - USE ORIGINAL IMAGE for better text detection
            analysis = await self._stage_analyze(original_image_path, result)
            if not analysis:
                return result
            result.analysis = analysis

            # Debug: Save analysis regions overlay
            if self.debug.enabled and analysis:
                img = cv2.imread(original_image_path)
                if img is not None:
                    text_regions = [{"bounds": [t.position[0]-10, t.position[1]-10, t.position[0]+100, t.position[1]+20]} for t in analysis.elements.text]
                    self.debug.save_with_overlay(img, "03_gemini_analysis", text_regions=text_regions)

            # Stage 4: Calibration
            calibration = await self._stage_calibrate(analysis, result)
            if not calibration:
                return result
            result.calibration = calibration

            # Stage 5: Extraction - use preprocessed for geometry, original for details
            entities = await self._stage_extract(processed_image_path, analysis, calibration, result)

            # Stage 5b: Linetype & Lineweight Detection on ORIGINAL image
            if entities:
                entities = await self._stage_detect_line_properties(
                    original_image_path, entities, calibration, result
                )
            if entities is None:
                return result

            # Stage 5c: Curve Detection (ellipses, arcs, bezier splines)
            if self.config.detect_curves:
                curve_entities = await self._stage_detect_curves(
                    original_image_path, calibration, result
                )
                if curve_entities:
                    entities.extend(curve_entities)

            # Debug: Save extraction results overlay (use original for cleaner viz)
            if self.debug.enabled:
                img = cv2.imread(original_image_path)
                if img is not None:
                    # Convert entities to drawable format
                    from .adaptive_extraction import EntityType
                    lines = []
                    circles = []
                    text_regions = []
                    for e in entities:
                        etype = e.entity_type.value if hasattr(e.entity_type, 'value') else str(e.entity_type)
                        props = e.properties
                        if etype == "line":
                            # Create a simple object for visualization
                            class LineVis:
                                pass
                            lv = LineVis()
                            # Convert DWG back to pixels for visualization
                            start = props.get("start", [0,0])
                            end = props.get("end", [0,0])
                            lv.start = calibration.to_pixels(start[0], start[1]) if calibration else (start[0], start[1])
                            lv.end = calibration.to_pixels(end[0], end[1]) if calibration else (end[0], end[1])
                            lv.line_type = type('obj', (object,), {'value': props.get('linetype', 'continuous').lower()})()
                            lines.append(lv)
                        elif etype == "circle":
                            class CircleVis:
                                pass
                            cv = CircleVis()
                            center = props.get("center", [0,0])
                            cv.center = calibration.to_pixels(center[0], center[1]) if calibration else (center[0], center[1])
                            cv.radius = props.get("radius", 10) / calibration.scale_factor if calibration and calibration.scale_factor > 0 else props.get("radius", 10)
                            circles.append(cv)
                        elif etype in ("mtext", "text"):
                            pos = props.get("position", [0,0])
                            px_pos = calibration.to_pixels(pos[0], pos[1]) if calibration else (pos[0], pos[1])
                            text_regions.append({"bounds": [px_pos[0]-10, px_pos[1]-10, px_pos[0]+100, px_pos[1]+20]})

                    self.debug.save_with_overlay(img, "04_extraction", lines=lines, circles=circles, text_regions=text_regions)

                    # Also save line types summary
                    linetype_counts = {}
                    for e in entities:
                        etype = e.entity_type.value if hasattr(e.entity_type, 'value') else str(e.entity_type)
                        if etype == "line":
                            lt = e.properties.get("linetype", "Continuous")
                            linetype_counts[lt] = linetype_counts.get(lt, 0) + 1
                    logger.info("debug_extraction_linetypes", linetypes=linetype_counts)

            # Stage 6: Symbol Recognition (optional)
            if self.config.symbol_method != SymbolMethod.DISABLED:
                entities = await self._stage_symbols(entities, result)

            # Stage 7: Geometry Refinement (optional)
            if self.config.refine_geometry:
                entities = await self._stage_refine(entities, result, processed_image_path)

            # Debug: Save final refined entities (use original for cleaner viz)
            if self.debug.enabled:
                img = cv2.imread(original_image_path)
                if img is not None:
                    lines = []
                    circles = []
                    text_regions = []
                    for e in entities:
                        etype = e.entity_type.value if hasattr(e.entity_type, 'value') else str(e.entity_type)
                        props = e.properties
                        if etype == "line":
                            class LineVis:
                                pass
                            lv = LineVis()
                            start = props.get("start", [0,0])
                            end = props.get("end", [0,0])
                            lv.start = calibration.to_pixels(start[0], start[1]) if calibration else (start[0], start[1])
                            lv.end = calibration.to_pixels(end[0], end[1]) if calibration else (end[0], end[1])
                            lv.line_type = type('obj', (object,), {'value': props.get('linetype', 'continuous').lower()})()
                            lines.append(lv)
                        elif etype == "circle":
                            class CircleVis:
                                pass
                            cv = CircleVis()
                            center = props.get("center", [0,0])
                            cv.center = calibration.to_pixels(center[0], center[1]) if calibration else (center[0], center[1])
                            cv.radius = props.get("radius", 10) / calibration.scale_factor if calibration and calibration.scale_factor > 0 else props.get("radius", 10)
                            circles.append(cv)
                        elif etype in ("mtext", "text"):
                            pos = props.get("position", [0,0])
                            px_pos = calibration.to_pixels(pos[0], pos[1]) if calibration else (pos[0], pos[1])
                            text_regions.append({"bounds": [px_pos[0]-10, px_pos[1]-10, px_pos[0]+100, px_pos[1]+20]})

                    self.debug.save_with_overlay(img, "05_refined", lines=lines, circles=circles, text_regions=text_regions)

            result.entities = entities
            result.total_entities = len(entities)
            result.entities_by_type = self._count_by_type(entities)

            # Stage 8: AutoCAD Creation (optional)
            if self.config.create_in_autocad:
                await self._stage_create(entities, calibration, result)

            # Stage 9: Validation (optional)
            if self.config.validate:
                await self._stage_validate(image_path, entities, result)

            result.success = True

        except Exception as e:
            logger.exception("Pipeline failed", error=str(e))
            result.error = str(e)
            result.success = False

        finally:
            result.total_duration_ms = (time.perf_counter() - start_time) * 1000

        return result

    # =========================================================================
    # Stage Implementations
    # =========================================================================

    async def _stage_render(
        self,
        pdf_path: str,
        page: int,
        output_dir: Optional[str],
        result: PipelineResult,
    ) -> Optional[str]:
        """Stage 1: Render PDF to image."""
        import time
        start = time.perf_counter()

        try:
            from .pdf_intake import render_pdf_high_quality_async

            render_result = await render_pdf_high_quality_async(
                pdf_path=pdf_path,
                page_number=page,
                dpi=self.config.dpi,
                output_dir=output_dir,
            )

            result.stages.append(StageResult(
                stage="render",
                success=True,
                duration_ms=(time.perf_counter() - start) * 1000,
                data={
                    "image_path": render_result.image_path,
                    "width_px": render_result.width_px,
                    "height_px": render_result.height_px,
                },
            ))

            # Store dimensions in result for calibration stage
            result.image_width = render_result.width_px
            result.image_height = render_result.height_px

            return render_result.image_path

        except Exception as e:
            result.stages.append(StageResult(
                stage="render",
                success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(e),
            ))
            result.error = f"Render failed: {e}"
            return None

    async def _stage_preprocess(
        self,
        image_path: str,
        result: PipelineResult,
    ) -> str:
        """Stage 2: Preprocess image."""
        import time
        start = time.perf_counter()

        try:
            from PIL import Image
            from .preprocessing import denoise, deskew, enhance_contrast
            from .binarization import ensemble_binarize

            img = Image.open(image_path)
            img_array = np.array(img)

            # Apply preprocessing steps
            if self.config.denoise:
                img_array = denoise(img_array, strength=self.config.denoise_strength)

            if self.config.deskew:
                deskew_result = deskew(img_array)
                img_array = deskew_result.image

            if self.config.enhance_contrast:
                img_array = enhance_contrast(img_array)

            if self.config.binarize:
                binarize_result = ensemble_binarize(img_array)
                img_array = binarize_result.binary_image

            # Save preprocessed image
            output_path = str(Path(image_path).with_suffix(".preprocessed.png"))
            preprocessed_img = Image.fromarray(img_array)
            preprocessed_img.save(output_path)

            # Log if dimensions changed but DO NOT update result.image_width/height
            # because Gemini analyzes the ORIGINAL image, so calibration must use original dimensions
            new_width, new_height = preprocessed_img.size
            if new_width != result.image_width or new_height != result.image_height:
                logger.info(
                    "preprocess_dimensions_changed",
                    original_size=(result.image_width, result.image_height),
                    preprocessed_size=(new_width, new_height),
                    note="Keeping original dimensions for calibration (Gemini uses original)",
                )
                # DO NOT UPDATE: result.image_width/height must match original for correct coordinate transform

            result.stages.append(StageResult(
                stage="preprocess",
                success=True,
                duration_ms=(time.perf_counter() - start) * 1000,
                data={
                    "output_path": output_path,
                    "width": new_width,
                    "height": new_height,
                },
            ))

            return output_path

        except Exception as e:
            logger.warning("Preprocessing failed, using original", error=str(e))
            result.stages.append(StageResult(
                stage="preprocess",
                success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(e),
            ))
            return image_path

    async def _stage_analyze(
        self,
        image_path: str,
        result: PipelineResult,
    ) -> Optional[Any]:
        """Stage 3: Analyze with Gemini."""
        import time
        start = time.perf_counter()

        try:
            from .gemini_understanding import analyze_drawing

            analysis = await analyze_drawing(
                image_path,
                model=self.config.gemini_model,
            )

            # Log detailed element counts
            if analysis:
                logger.info(
                    "gemini_analysis_elements",
                    drawing_type=analysis.drawing_type,
                    lines=len(analysis.elements.lines),
                    arcs=len(analysis.elements.arcs),
                    circles=len(analysis.elements.circles),
                    text=len(analysis.elements.text),
                    symbols=len(analysis.elements.symbols),
                    dimensions=len(analysis.elements.dimensions),
                )

            result.stages.append(StageResult(
                stage="analyze",
                success=True,
                duration_ms=(time.perf_counter() - start) * 1000,
                data={
                    "drawing_type": analysis.drawing_type if analysis else None,
                    "lines": len(analysis.elements.lines) if analysis else 0,
                    "circles": len(analysis.elements.circles) if analysis else 0,
                    "text": len(analysis.elements.text) if analysis else 0,
                    "symbols": len(analysis.elements.symbols) if analysis else 0,
                },
            ))

            return analysis

        except Exception as e:
            result.stages.append(StageResult(
                stage="analyze",
                success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(e),
            ))
            result.error = f"Analysis failed: {e}"
            return None

    async def _stage_calibrate(
        self,
        analysis: Any,
        result: PipelineResult,
    ) -> Optional[Any]:
        """Stage 4: Calibrate coordinates."""
        import time
        start = time.perf_counter()

        try:
            from .coordinate_calibration import calibrate_from_analysis

            calibration = calibrate_from_analysis(
                analysis=analysis,
                image_width=result.image_width,
                image_height=result.image_height,
                image_dpi=self.config.dpi,
            )

            result.stages.append(StageResult(
                stage="calibrate",
                success=True,
                duration_ms=(time.perf_counter() - start) * 1000,
                data={
                    "scale_factor": calibration.scale_factor if calibration else None,
                    "units": calibration.units if calibration else None,
                },
            ))

            return calibration

        except Exception as e:
            result.stages.append(StageResult(
                stage="calibrate",
                success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(e),
            ))
            result.error = f"Calibration failed: {e}"
            return None

    async def _stage_extract(
        self,
        image_path: str,
        analysis: Any,
        calibration: Any,
        result: PipelineResult,
    ) -> Optional[List[Any]]:
        """Stage 5: Extract entities."""
        import time
        start = time.perf_counter()

        try:
            entities = await ExtractionFactory.extract(
                method=self.config.extraction_method,
                image_path=image_path,
                analysis=analysis,
                calibration=calibration,
                config=self.config,
            )

            # Count entities by type
            entity_counts = {}
            for e in entities:
                etype = e.entity_type.value if hasattr(e.entity_type, 'value') else str(e.entity_type)
                entity_counts[etype] = entity_counts.get(etype, 0) + 1

            logger.info(
                "extraction_entity_counts",
                total=len(entities),
                by_type=entity_counts,
            )

            result.stages.append(StageResult(
                stage="extract",
                success=True,
                duration_ms=(time.perf_counter() - start) * 1000,
                data={
                    "entity_count": len(entities),
                    "by_type": entity_counts,
                },
            ))

            return entities

        except Exception as e:
            result.stages.append(StageResult(
                stage="extract",
                success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(e),
            ))
            result.error = f"Extraction failed: {e}"
            return None

    async def _stage_detect_line_properties(
        self,
        original_image_path: str,
        entities: List[Any],
        calibration: Any,
        result: PipelineResult,
    ) -> List[Any]:
        """
        Stage 5b: Detect linetype and lineweight on ORIGINAL image.

        This must run on the original (non-preprocessed) image because:
        - Binarization destroys grayscale info needed for lineweight
        - Denoising blurs dashed patterns needed for linetype detection
        """
        import time
        start = time.perf_counter()

        try:
            import cv2
            from .linetype_detection import detect_linetype, LinetypeName
            from .thickness_detection import detect_line_thickness, mm_to_lineweight

            # Load original image
            img = cv2.imread(original_image_path)
            if img is None:
                logger.warning("Could not load original image for line properties")
                return entities

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            height, width = gray.shape[:2]

            # Calculate DPI from calibration (needed for lineweight in mm)
            dpi = 300  # Default
            if calibration and calibration.scale_factor > 0:
                # scale_factor is DWG units per pixel
                # For inches: 1 inch = scale_factor * pixels
                if calibration.units == "inches":
                    dpi = int(1.0 / calibration.scale_factor)
                elif calibration.units == "feet":
                    dpi = int(12.0 / calibration.scale_factor)
                elif calibration.units == "mm":
                    dpi = int(25.4 / calibration.scale_factor)
                dpi = max(72, min(1200, dpi))  # Clamp to reasonable range

            linetype_count = 0
            lineweight_count = 0

            for entity in entities:
                etype = entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type)

                if etype == "line":
                    props = entity.properties
                    start_dwg = props.get("start", (0, 0))
                    end_dwg = props.get("end", (0, 0))

                    # Convert to pixels
                    if calibration and hasattr(calibration, 'to_pixels'):
                        start_px = calibration.to_pixels(start_dwg[0], start_dwg[1])
                        end_px = calibration.to_pixels(end_dwg[0], end_dwg[1])
                    else:
                        start_px = start_dwg
                        end_px = end_dwg

                    # Clamp to image bounds
                    x1 = max(0, min(width-1, int(start_px[0])))
                    y1 = max(0, min(height-1, int(start_px[1])))
                    x2 = max(0, min(width-1, int(end_px[0])))
                    y2 = max(0, min(height-1, int(end_px[1])))

                    # Calculate line length in pixels
                    line_length = ((x2-x1)**2 + (y2-y1)**2)**0.5

                    # Only process lines of reasonable length
                    if line_length > 20:
                        # Detect linetype
                        try:
                            linetype_result = detect_linetype(
                                gray, (x1, y1), (x2, y2),
                                sample_width=3,
                            )
                            if linetype_result and linetype_result.linetype != LinetypeName.CONTINUOUS:
                                props["linetype"] = linetype_result.linetype.value
                                props["linetype_confidence"] = linetype_result.confidence
                                linetype_count += 1
                        except Exception as e:
                            logger.debug("linetype_detection_failed", error=str(e))

                        # Detect lineweight (thickness)
                        try:
                            thickness_result = detect_line_thickness(
                                gray, (x1, y1), (x2, y2),
                                dpi=dpi,
                            )
                            if thickness_result:
                                # Convert to AutoCAD lineweight
                                lw = mm_to_lineweight(thickness_result.thickness_mm)
                                if lw > 0:
                                    props["lineweight"] = lw
                                    props["thickness_mm"] = thickness_result.thickness_mm
                                    lineweight_count += 1
                        except Exception as e:
                            logger.debug("lineweight_detection_failed", error=str(e))

            logger.info(
                "line_properties_detected",
                linetypes=linetype_count,
                lineweights=lineweight_count,
                total_lines=sum(1 for e in entities if (e.entity_type.value if hasattr(e.entity_type, 'value') else str(e.entity_type)) == "line"),
            )

            result.stages.append(StageResult(
                stage="line_properties",
                success=True,
                duration_ms=(time.perf_counter() - start) * 1000,
                data={
                    "linetypes_detected": linetype_count,
                    "lineweights_detected": lineweight_count,
                },
            ))

            return entities

        except Exception as e:
            logger.warning("Line properties detection failed", error=str(e))
            result.stages.append(StageResult(
                stage="line_properties",
                success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(e),
            ))
            return entities  # Return entities unchanged on failure

    async def _stage_detect_curves(
        self,
        image_path: str,
        calibration: Any,
        result: PipelineResult,
    ) -> List[Any]:
        """
        Stage 5c: Detect curved elements (ellipses, arcs, bezier splines).

        Finds curved geometry that line/circle detection misses:
        - Ellipses (not circles)
        - Partial arcs
        - Bezier/spline curves from contours
        """
        import time
        import cv2
        start = time.perf_counter()

        try:
            from .adaptive_extraction import EntityToCreate, EntityType, ExtractionSource

            # Load image
            img = cv2.imread(image_path)
            if img is None:
                logger.warning("Could not load image for curve detection")
                return []

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

            # Threshold for contour detection
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

            entities = []
            circle_count = 0
            ellipse_count = 0
            spline_count = 0

            # Detect ellipses and circles from contours
            if self.config.detect_ellipses:
                try:
                    # Find contours for ellipse/circle fitting
                    contours, _ = cv2.findContours(
                        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
                    )

                    for contour in contours:
                        if len(contour) < self.config.min_curve_length:
                            continue

                        # Need at least 5 points for ellipse fitting
                        if len(contour) < 5:
                            continue

                        # Fit ellipse to contour
                        try:
                            ellipse_params = cv2.fitEllipse(contour)
                            (cx, cy), (width, height), angle = ellipse_params

                            # Semi-axes (cv2 returns full width/height)
                            semi_major = max(width, height) / 2
                            semi_minor = min(width, height) / 2

                            if semi_minor < 3:  # Too small
                                continue

                            # Calculate aspect ratio
                            aspect_ratio = semi_major / semi_minor if semi_minor > 0 else float('inf')

                            # Convert center to DWG coordinates
                            center_dwg = calibration.to_dwg(cx, cy)

                            # If aspect ratio < 1.15, it's effectively a CIRCLE
                            if aspect_ratio < 1.15:
                                radius_dwg = calibration.scale_length((semi_major + semi_minor) / 2)

                                entities.append(EntityToCreate(
                                    entity_type=EntityType.CIRCLE,
                                    layer="0",
                                    properties={
                                        "center": center_dwg,
                                        "radius": radius_dwg,
                                    },
                                    source=ExtractionSource.OPENCV,
                                    source_element="contour_circle",
                                ))
                                circle_count += 1
                            else:
                                # It's an ellipse
                                major_axis_dwg = calibration.scale_length(semi_major)
                                minor_axis_dwg = calibration.scale_length(semi_minor)

                                entities.append(EntityToCreate(
                                    entity_type=EntityType.ELLIPSE,
                                    layer="0",
                                    properties={
                                        "center": center_dwg,
                                        "major_axis": major_axis_dwg,
                                        "minor_axis": minor_axis_dwg,
                                        "rotation": angle,
                                        "start_angle": 0.0,
                                        "end_angle": 360.0,
                                    },
                                    source=ExtractionSource.OPENCV,
                                    source_element="contour_ellipse",
                                ))
                                ellipse_count += 1

                        except cv2.error:
                            # fitEllipse can fail on some contours
                            continue

                except Exception as e:
                    logger.debug("ellipse_detection_error", error=str(e))

            # Detect curved contours and fit bezier splines
            if self.config.detect_splines:
                try:
                    from .bezier_fitting import detect_curves, fit_bezier_to_points, BezierFitConfig

                    curve_segments = detect_curves(
                        binary,
                        min_curvature=0.05,  # Lower threshold to catch more curves
                        min_length=self.config.min_curve_length,
                    )

                    fit_config = BezierFitConfig(
                        error_tolerance=self.config.curve_fit_tolerance,
                    )

                    for segment in curve_segments:
                        # Fit bezier curves to the segment
                        curves = fit_bezier_to_points(segment.points, fit_config)

                        for curve in curves:
                            # Convert control points to DWG coordinates
                            control_points_dwg = [
                                calibration.to_dwg(p[0], p[1])
                                for p in curve.control_points
                            ]

                            entities.append(EntityToCreate(
                                entity_type=EntityType.SPLINE,
                                layer="0",
                                properties={
                                    "control_points": control_points_dwg,
                                    "degree": 3,  # Cubic bezier
                                    "is_closed": segment.is_closed,
                                    "fit_error": curve.fit_error,
                                },
                                source=ExtractionSource.OPENCV,
                                source_element="bezier_fitting",
                            ))
                            spline_count += 1

                except Exception as e:
                    logger.debug("spline_detection_error", error=str(e))

            logger.info(
                "curve_detection_complete",
                circles=circle_count,
                ellipses=ellipse_count,
                splines=spline_count,
            )

            result.stages.append(StageResult(
                stage="curve_detection",
                success=True,
                duration_ms=(time.perf_counter() - start) * 1000,
                data={
                    "circles": circle_count,
                    "ellipses": ellipse_count,
                    "splines": spline_count,
                },
            ))

            return entities

        except Exception as e:
            logger.warning("Curve detection failed", error=str(e))
            result.stages.append(StageResult(
                stage="curve_detection",
                success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(e),
            ))
            return []

    async def _stage_symbols(
        self,
        entities: List[Any],
        result: PipelineResult,
    ) -> List[Any]:
        """Stage 6: Symbol recognition."""
        import time
        start = time.perf_counter()

        try:
            if self.config.symbol_method == SymbolMethod.RAG:
                from .symbol_rag import SymbolRAG, is_symbol_rag_available

                if is_symbol_rag_available():
                    # RAG-based symbol recognition
                    from aec_agent.mcp.server import get_database_pool
                    pool = get_database_pool()

                    if pool:
                        rag = SymbolRAG(pool)
                        await rag.initialize()

                        # Enhance block entities with RAG lookup
                        for entity in entities:
                            if hasattr(entity, 'entity_type') and entity.entity_type.value == 'block':
                                # Search for matching symbol
                                if hasattr(entity, 'properties') and 'name' in entity.properties:
                                    matches = await rag.search_by_description(
                                        entity.properties.get('name', ''),
                                        top_k=self.config.symbol_top_k,
                                    )
                                    if matches and matches[0].confidence >= self.config.symbol_min_confidence:
                                        entity.properties['block_name'] = matches[0].block_name

            result.stages.append(StageResult(
                stage="symbols",
                success=True,
                duration_ms=(time.perf_counter() - start) * 1000,
            ))

        except Exception as e:
            logger.warning("Symbol recognition failed", error=str(e))
            result.stages.append(StageResult(
                stage="symbols",
                success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(e),
            ))

        return entities

    async def _stage_refine(
        self,
        entities: List[Any],
        result: PipelineResult,
        image_path: Optional[str] = None,
    ) -> List[Any]:
        """Stage 7: Geometry refinement."""
        import time
        start = time.perf_counter()

        try:
            # Load image for junction detection if enabled
            image_array = None
            if self.config.use_hawp_junctions and image_path:
                try:
                    import cv2
                    image_array = cv2.imread(image_path)
                    if image_array is not None:
                        logger.debug("Image loaded for junction detection", shape=image_array.shape)
                except Exception as e:
                    logger.warning("Failed to load image for junction detection", error=str(e))

            refine_config = RefinementConfig(
                use_junction_detection=self.config.use_hawp_junctions,
                junction_snap_distance_px=8.0,
                junction_confidence_threshold=0.5,
                straighten_lines=True,
                straighten_tolerance_deg=self.config.straighten_tolerance_deg,
                connect_endpoints=True,
                connect_tolerance_px=self.config.connect_tolerance_px,
                snap_to_grid=True,
                grid_size_px=self.config.grid_size_px,
                remove_duplicates=self.config.remove_duplicates,
                duplicate_tolerance_px=self.config.duplicate_tolerance_px,
                merge_collinear_lines=self.config.merge_collinear_lines,
                collinear_angle_tolerance_deg=self.config.collinear_angle_tolerance_deg,
                collinear_gap_tolerance_px=self.config.collinear_gap_tolerance_px,
                image_for_junctions=image_array,
            )

            refinement = GeometryRefinementPipeline(refine_config)
            entities, stats = await refinement.refine(entities)

            result.stages.append(StageResult(
                stage="refine",
                success=True,
                duration_ms=(time.perf_counter() - start) * 1000,
                data=stats,
            ))

        except Exception as e:
            logger.warning("Geometry refinement failed", error=str(e))
            result.stages.append(StageResult(
                stage="refine",
                success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(e),
            ))

        return entities

    async def _stage_create(
        self,
        entities: List[Any],
        calibration: Any,
        result: PipelineResult,
    ) -> None:
        """Stage 8: Create in AutoCAD."""
        import time
        start = time.perf_counter()

        try:
            from .autocad_creation import create_entities_in_autocad
            from .adaptive_extraction import ExtractionResult

            # Wrap entities in ExtractionResult for create_entities_in_autocad
            extraction_result = ExtractionResult(
                entities=entities,
                drawing_type=result.analysis.drawing_type if result.analysis else "",
                calibration_method=calibration.method if calibration else "",
                calibration_confidence=calibration.confidence if calibration else 0.0,
            )

            creation_result = await create_entities_in_autocad(
                extraction_result=extraction_result,
                create_layers=self.config.use_ncs_layers,
            )

            result.entities_created = creation_result.statistics.success_count
            result.creation_failures = creation_result.statistics.failure_count

            result.stages.append(StageResult(
                stage="create",
                success=creation_result.success,
                duration_ms=(time.perf_counter() - start) * 1000,
                data={
                    "created": creation_result.statistics.success_count,
                    "failed": creation_result.statistics.failure_count,
                },
            ))

        except Exception as e:
            result.stages.append(StageResult(
                stage="create",
                success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(e),
            ))

    async def _stage_validate(
        self,
        image_path: str,
        entities: List[Any],
        result: PipelineResult,
    ) -> None:
        """Stage 9: Validation."""
        import time
        start = time.perf_counter()

        try:
            from .validation import validate_extraction

            validation = await validate_extraction(
                image_path=image_path,
                entities=entities,
                use_gemini=self.config.gemini_visual_qa,
            )

            result.validation_passed = validation.passed
            result.validation_issues = validation.issues

            result.stages.append(StageResult(
                stage="validate",
                success=validation.passed,
                duration_ms=(time.perf_counter() - start) * 1000,
                data={
                    "passed": validation.passed,
                    "issue_count": len(validation.issues),
                },
            ))

        except Exception as e:
            result.stages.append(StageResult(
                stage="validate",
                success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(e),
            ))

    # =========================================================================
    # Utilities
    # =========================================================================

    def _count_by_type(self, entities: List[Any]) -> Dict[str, int]:
        """Count entities by type."""
        counts: Dict[str, int] = {}
        for entity in entities:
            if hasattr(entity, 'entity_type'):
                type_name = entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type)
                counts[type_name] = counts.get(type_name, 0) + 1
        return counts


# =============================================================================
# Convenience Functions
# =============================================================================

async def vectorize_pdf(
    pdf_path: str,
    page: int = 1,
    extraction_method: str = "hybrid",
    preprocess: bool = True,
    use_symbol_rag: bool = True,
    refine_geometry: bool = True,
    create_in_autocad: bool = False,
    validate: bool = False,
    **kwargs,
) -> PipelineResult:
    """
    Convenience function to vectorize a PDF page.

    This is the main entry point for PDF vectorization.

    Args:
        pdf_path: Path to the PDF file
        page: Page number (1-indexed)
        extraction_method: "direct", "hybrid", "best", or "vtracer"
        preprocess: Apply image preprocessing
        use_symbol_rag: Use RAG for symbol recognition
        refine_geometry: Apply geometry refinement
        create_in_autocad: Create entities in AutoCAD
        validate: Validate extraction with Gemini
        **kwargs: Additional PipelineConfig options

    Returns:
        PipelineResult with extracted entities and statistics
    """
    config = PipelineConfig(
        extraction_method=ExtractionMethod(extraction_method),
        preprocess=preprocess,
        symbol_method=SymbolMethod.RAG if use_symbol_rag else SymbolMethod.HARDCODED,
        refine_geometry=refine_geometry,
        create_in_autocad=create_in_autocad,
        validate=validate,
        **kwargs,
    )

    pipeline = UnifiedPipeline(config)
    return await pipeline.process(pdf_path=pdf_path, page=page)
