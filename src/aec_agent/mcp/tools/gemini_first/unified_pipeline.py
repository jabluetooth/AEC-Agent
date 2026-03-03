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
    denoise_strength: int = 10
    deskew: bool = True
    binarize: bool = True
    enhance_contrast: bool = True
    super_resolution: bool = False  # Enable for low-DPI scans
    super_resolution_scale: int = 4

    # === Stage 3: Gemini Analysis ===
    gemini_model: str = "gemini-2.0-flash"
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
    use_hawp_junctions: bool = False  # Neural junction detection (slow but accurate)

    # === Stage 5: Symbol Recognition (optional) ===
    symbol_method: SymbolMethod = SymbolMethod.RAG
    symbol_min_confidence: float = 0.5
    symbol_top_k: int = 3

    # === Stage 6: Geometry Refinement (optional) ===
    refine_geometry: bool = True
    straighten_tolerance_deg: float = 5.0  # Snap to H/V/45° if within tolerance
    connect_tolerance_px: float = 10.0
    grid_size_px: float = 5.0
    remove_duplicates: bool = True
    duplicate_tolerance_px: float = 2.0

    # === Stage 7: Validation (optional) ===
    validate: bool = False
    gemini_visual_qa: bool = False
    max_validation_iterations: int = 3

    # === Stage 8: Output ===
    output_format: OutputFormat = OutputFormat.ENTITIES
    create_in_autocad: bool = False
    use_ncs_layers: bool = True

    # === Database ===
    store_to_database: bool = False

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
    straighten_lines: bool = True
    straighten_tolerance_deg: float = 5.0
    connect_endpoints: bool = True
    connect_tolerance_px: float = 10.0
    snap_to_grid: bool = True
    grid_size_px: float = 5.0
    remove_duplicates: bool = True
    duplicate_tolerance_px: float = 2.0


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
            "lines_straightened": 0,
            "endpoints_connected": 0,
            "points_snapped": 0,
            "duplicates_removed": 0,
        }

        current = entities.copy()

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

        # 3. Snap to grid
        if self.config.snap_to_grid:
            current, adjustments = snap_to_grid(
                current,
                grid_size=self.config.grid_size_px,
            )
            stats["points_snapped"] = len(adjustments)

        # 4. Remove duplicates
        if self.config.remove_duplicates:
            original_count = len(current)
            current, adjustments = remove_duplicate_lines(
                current,
                tolerance=self.config.duplicate_tolerance_px,
            )
            stats["duplicates_removed"] = original_count - len(current)

        stats["output_count"] = len(current)

        return current, stats


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
        start_time = time.perf_counter()

        result = PipelineResult(
            success=False,
            pipeline_id=self.pipeline_id,
        )

        try:
            # Stage 1: PDF Rendering
            image_path = await self._stage_render(pdf_path, page, output_dir, result)
            if not image_path:
                return result
            result.image_path = image_path

            # Stage 2: Preprocessing (optional)
            if self.config.preprocess:
                image_path = await self._stage_preprocess(image_path, result)

            # Stage 3: Gemini Analysis
            analysis = await self._stage_analyze(image_path, result)
            if not analysis:
                return result
            result.analysis = analysis

            # Stage 4: Calibration
            calibration = await self._stage_calibrate(analysis, result)
            if not calibration:
                return result
            result.calibration = calibration

            # Stage 5: Extraction
            entities = await self._stage_extract(image_path, analysis, calibration, result)
            if entities is None:
                return result

            # Stage 6: Symbol Recognition (optional)
            if self.config.symbol_method != SymbolMethod.DISABLED:
                entities = await self._stage_symbols(entities, result)

            # Stage 7: Geometry Refinement (optional)
            if self.config.refine_geometry:
                entities = await self._stage_refine(entities, result)

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
                page=page,
                dpi=self.config.dpi,
                output_dir=output_dir,
            )

            result.stages.append(StageResult(
                stage="render",
                success=True,
                duration_ms=(time.perf_counter() - start) * 1000,
                data={"image_path": render_result.image_path},
            ))

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
                img_array = denoise(img_array, h=self.config.denoise_strength)

            if self.config.deskew:
                img_array, _ = deskew(img_array)

            if self.config.enhance_contrast:
                img_array = enhance_contrast(img_array)

            if self.config.binarize:
                binarize_result = ensemble_binarize(img_array)
                img_array = binarize_result.binary_image

            # Save preprocessed image
            output_path = str(Path(image_path).with_suffix(".preprocessed.png"))
            Image.fromarray(img_array).save(output_path)

            result.stages.append(StageResult(
                stage="preprocess",
                success=True,
                duration_ms=(time.perf_counter() - start) * 1000,
                data={"output_path": output_path},
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

            result.stages.append(StageResult(
                stage="analyze",
                success=True,
                duration_ms=(time.perf_counter() - start) * 1000,
                data={
                    "drawing_type": analysis.drawing_type if analysis else None,
                    "element_count": len(analysis.elements.lines) + len(analysis.elements.circles) if analysis else 0,
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
                analysis,
                dpi=self.config.dpi,
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

            result.stages.append(StageResult(
                stage="extract",
                success=True,
                duration_ms=(time.perf_counter() - start) * 1000,
                data={"entity_count": len(entities)},
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
    ) -> List[Any]:
        """Stage 7: Geometry refinement."""
        import time
        start = time.perf_counter()

        try:
            refine_config = RefinementConfig(
                straighten_lines=True,
                straighten_tolerance_deg=self.config.straighten_tolerance_deg,
                connect_endpoints=True,
                connect_tolerance_px=self.config.connect_tolerance_px,
                snap_to_grid=True,
                grid_size_px=self.config.grid_size_px,
                remove_duplicates=self.config.remove_duplicates,
                duplicate_tolerance_px=self.config.duplicate_tolerance_px,
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

            creation_result = await create_entities_in_autocad(
                entities=entities,
                calibration=calibration,
                use_ncs_layers=self.config.use_ncs_layers,
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
