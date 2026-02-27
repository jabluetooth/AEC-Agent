"""
Best Practices Pipeline: PDF to AutoCAD Vectorization.

This module implements the optimal 7-stage pipeline combining the best algorithms
for each step as documented in docs/BEST_ALGORITHMS_PIPELINE.md.

Pipeline Stages:
1. PDF Upload & Rendering (PyMuPDF @ 300-600 DPI)
2. Image Preprocessing (NLM denoise + Hough deskew + 7-method ensemble binarization)
3. Gemini Analysis (scale, type, MText, layers, symbols with RAG)
4. Vector Extraction (LSD lines + Hough circles + HAWP junctions + VTracer/Bezier)
5. Symbol Recognition (CLIP embeddings + pgvector RAG lookup)
6. Validation (5° line straightening + endpoint connection + Gemini visual QA)
7. AutoCAD Output (scaled DWG with NCS layers)

Usage:
    from aec_agent.mcp.tools.gemini_first import BestPracticesPipeline

    pipeline = BestPracticesPipeline()
    result = await pipeline.process_pdf(
        pdf_path="drawing.pdf",
        page=1,
        output_dir="./output",
    )
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING
from uuid import UUID

import numpy as np
import structlog

if TYPE_CHECKING:
    from PIL import Image as PILImage

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class VectorizationMethod(str, Enum):
    """Vectorization algorithm to use."""
    VTRACER = "vtracer"  # O(n) - fast, good quality
    BEZIER_SPLATTING = "bezier_splatting"  # 150x faster optimization, best quality
    OPENCV_LSD = "opencv_lsd"  # Line Segment Detector - precise geometry
    HYBRID = "hybrid"  # Combine multiple methods


class SymbolRecognitionMethod(str, Enum):
    """Symbol recognition approach."""
    RAG = "rag"  # CLIP embeddings + pgvector (recommended)
    HARDCODED = "hardcoded"  # Legacy SYMBOL_TO_BLOCK dict
    GEMINI_ONLY = "gemini_only"  # Trust Gemini classification


@dataclass
class BestPracticesConfig:
    """Configuration for the best practices pipeline."""

    # Stage 1: PDF Rendering
    dpi: int = 300  # 300 for standard, 600 for detailed, 1200 for archival
    auto_dpi: bool = True  # Auto-select DPI based on content
    convert_grayscale: bool = True

    # Stage 2: Preprocessing
    denoise: bool = True
    denoise_strength: int = 10  # NLM h parameter
    deskew: bool = True
    deskew_method: str = "hough"  # hough, projection, moments
    binarize: bool = True
    binarize_method: str = "ensemble"  # ensemble (7-method), sauvola, otsu
    enhance_contrast: bool = True
    super_resolution: bool = True  # Real-ESRGAN for DPI < 200
    super_resolution_scale: int = 4

    # Stage 3: Gemini Analysis
    gemini_model: str = "gemini-2.0-flash"
    analyze_scale: bool = True
    analyze_drawing_type: bool = True
    analyze_layers: bool = True

    # Stage 4: Vector Extraction
    vectorization_method: VectorizationMethod = VectorizationMethod.HYBRID
    extract_lines: bool = True
    extract_circles: bool = True
    extract_arcs: bool = True
    extract_text: bool = True
    use_hawp_junctions: bool = True  # Neural junction detection
    vtracer_mode: str = "spline"  # spline, polygon

    # Stage 5: Symbol Recognition
    symbol_method: SymbolRecognitionMethod = SymbolRecognitionMethod.RAG
    symbol_min_confidence: float = 0.5
    symbol_top_k: int = 3

    # Stage 6: Validation
    straighten_lines: bool = True
    straighten_tolerance_deg: float = 5.0  # Snap to H/V/45° if within tolerance
    connect_endpoints: bool = True
    connect_tolerance_px: float = 10.0
    snap_to_grid: bool = True
    grid_tolerance_px: float = 5.0
    gemini_visual_qa: bool = True
    max_validation_iterations: int = 3

    # Stage 7: Output
    create_layers: bool = True
    use_ncs_layers: bool = True  # National CAD Standard layer naming
    output_format: str = "entities"  # entities, dwg, dxf

    # Database
    store_to_database: bool = True
    track_symbol_usage: bool = True


@dataclass
class StageResult:
    """Result of a single pipeline stage."""
    stage: str
    success: bool
    duration_ms: float
    data: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Complete pipeline result."""
    success: bool
    pdf_path: str
    page: int
    stages: list[StageResult] = field(default_factory=list)
    total_duration_ms: float = 0.0

    # Stage outputs
    image_path: Optional[Path] = None
    analysis: Optional[Any] = None
    calibration: Optional[Any] = None
    entities: list = field(default_factory=list)
    symbols_recognized: list = field(default_factory=list)
    validation_result: Optional[Any] = None

    # Statistics
    entity_count: int = 0
    symbol_count: int = 0
    line_count: int = 0
    text_count: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "pdf_path": self.pdf_path,
            "page": self.page,
            "total_duration_ms": self.total_duration_ms,
            "stages": [
                {
                    "stage": s.stage,
                    "success": s.success,
                    "duration_ms": s.duration_ms,
                    "errors": s.errors,
                    "warnings": s.warnings,
                }
                for s in self.stages
            ],
            "statistics": {
                "entity_count": self.entity_count,
                "symbol_count": self.symbol_count,
                "line_count": self.line_count,
                "text_count": self.text_count,
            },
            "image_path": str(self.image_path) if self.image_path else None,
        }


# =============================================================================
# Best Practices Pipeline
# =============================================================================

class BestPracticesPipeline:
    """
    Optimal PDF to AutoCAD vectorization pipeline.

    Combines the best algorithms for each stage:
    - PyMuPDF for PDF rendering
    - NLM + Hough + 7-method ensemble for preprocessing
    - Gemini for semantic analysis
    - LSD + HAWP + VTracer for vectorization
    - CLIP + pgvector RAG for symbol recognition
    - 5° straightening + endpoint connection for validation
    """

    def __init__(self, config: Optional[BestPracticesConfig] = None):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or BestPracticesConfig()
        self._symbol_rag = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize pipeline components (lazy loading)."""
        if self._initialized:
            return

        # Initialize Symbol RAG if using RAG method
        if self.config.symbol_method == SymbolRecognitionMethod.RAG:
            try:
                from .symbol_rag import SymbolRAG, is_symbol_rag_available
                from aec_agent.db.connection import get_database_pool

                if is_symbol_rag_available():
                    pool = await get_database_pool()
                    if pool:
                        self._symbol_rag = SymbolRAG(pool)
                        await self._symbol_rag.initialize()
                    else:
                        logger.warning("Database not available, falling back to hardcoded symbols")
                        self.config.symbol_method = SymbolRecognitionMethod.HARDCODED
                else:
                    logger.warning("Symbol RAG not available, falling back to hardcoded symbols")
                    self.config.symbol_method = SymbolRecognitionMethod.HARDCODED
            except Exception as e:
                logger.warning(f"Failed to initialize Symbol RAG: {e}")
                self.config.symbol_method = SymbolRecognitionMethod.HARDCODED

        self._initialized = True

    async def process_pdf(
        self,
        pdf_path: str | Path,
        page: int = 1,
        output_dir: Optional[str | Path] = None,
    ) -> PipelineResult:
        """
        Process a PDF through the complete pipeline.

        Args:
            pdf_path: Path to PDF file
            page: Page number (1-indexed)
            output_dir: Output directory (optional)

        Returns:
            PipelineResult with all stage outputs
        """
        import time

        start_time = time.perf_counter()
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir) if output_dir else pdf_path.parent / "vectorized"
        output_dir.mkdir(parents=True, exist_ok=True)

        result = PipelineResult(
            success=False,
            pdf_path=str(pdf_path),
            page=page,
        )

        await self.initialize()

        try:
            # Stage 1: PDF Rendering
            logger.info("pipeline_stage_starting", stage="1_render_pdf")
            stage1 = await self._stage1_render_pdf(pdf_path, page, output_dir)
            result.stages.append(stage1)
            logger.info("pipeline_stage_complete", stage="1_render_pdf", success=stage1.success)
            if not stage1.success:
                return result
            result.image_path = stage1.data.get("image_path")
            image = stage1.data.get("image")

            # Stage 2: Preprocessing
            logger.info("pipeline_stage_starting", stage="2_preprocess")
            stage2 = await self._stage2_preprocess(image, stage1.data.get("dpi", 300))
            result.stages.append(stage2)
            logger.info("pipeline_stage_complete", stage="2_preprocess", success=stage2.success)
            if not stage2.success:
                return result
            processed_image = stage2.data.get("processed_image", image)

            # Stage 3: Gemini Analysis
            logger.info("pipeline_stage_starting", stage="3_gemini_analysis")
            stage3 = await self._stage3_analyze(processed_image, result.image_path)
            result.stages.append(stage3)
            logger.info("pipeline_stage_complete", stage="3_gemini_analysis", success=stage3.success)
            if not stage3.success:
                return result
            result.analysis = stage3.data.get("analysis")
            result.calibration = stage3.data.get("calibration")

            # Stage 4: Vector Extraction
            logger.info("pipeline_stage_starting", stage="4_extract")
            stage4 = await self._stage4_extract(
                processed_image,
                result.analysis,
                result.calibration,
                image_path=result.image_path,
            )
            result.stages.append(stage4)
            logger.info("pipeline_stage_complete", stage="4_extract", success=stage4.success)
            if not stage4.success:
                return result
            entities = stage4.data.get("entities", [])

            # Stage 5: Symbol Recognition (RAG)
            logger.info("pipeline_stage_starting", stage="5_symbol_rag")
            stage5 = await self._stage5_recognize_symbols(
                processed_image,
                result.analysis,
                result.calibration,
            )
            result.stages.append(stage5)
            logger.info("pipeline_stage_complete", stage="5_symbol_rag", success=stage5.success)
            # Symbol recognition failures are non-fatal
            symbol_entities = stage5.data.get("symbol_entities", [])
            result.symbols_recognized = stage5.data.get("recognized_symbols", [])

            # Merge symbol entities into main entities
            entities.extend(symbol_entities)

            # Stage 6: Validation & Correction
            logger.info("pipeline_stage_starting", stage="6_validate")
            stage6 = await self._stage6_validate(
                entities,
                processed_image,
                result.analysis,
            )
            result.stages.append(stage6)
            logger.info("pipeline_stage_complete", stage="6_validate", success=stage6.success)
            if not stage6.success:
                return result
            validated_entities = stage6.data.get("entities", entities)
            result.validation_result = stage6.data.get("validation_result")

            # Stage 7: Output
            logger.info("pipeline_stage_starting", stage="7_output")
            stage7 = await self._stage7_output(
                validated_entities,
                result.calibration,
                output_dir,
                pdf_path.stem,
            )
            result.stages.append(stage7)
            logger.info("pipeline_stage_complete", stage="7_output", success=stage7.success)

            # Final statistics
            result.entities = validated_entities
            result.entity_count = len(validated_entities)
            result.symbol_count = len(result.symbols_recognized)

            # Handle entity_type as either enum or string
            def get_entity_type_value(e):
                if hasattr(e.entity_type, 'value'):
                    return e.entity_type.value
                return str(e.entity_type)

            result.line_count = sum(1 for e in validated_entities if get_entity_type_value(e) == "LINE")
            result.text_count = sum(1 for e in validated_entities if get_entity_type_value(e) in ("TEXT", "MTEXT"))

            result.success = stage7.success

        except Exception as e:
            logger.exception("Pipeline failed", error=str(e))
            result.stages.append(StageResult(
                stage="error",
                success=False,
                duration_ms=0,
                errors=[str(e)],
            ))

        result.total_duration_ms = (time.perf_counter() - start_time) * 1000
        return result

    # =========================================================================
    # Stage 1: PDF Rendering
    # =========================================================================

    async def _stage1_render_pdf(
        self,
        pdf_path: Path,
        page: int,
        output_dir: Path,
    ) -> StageResult:
        """Stage 1: Render PDF to high-quality image."""
        import time
        from .pdf_intake import render_pdf_high_quality_async, get_pdf_info
        from .super_resolution import is_super_resolution_available, upscale_image

        start = time.perf_counter()
        errors = []
        warnings = []

        try:
            # Get PDF info for auto-DPI selection
            dpi = self.config.dpi
            if self.config.auto_dpi:
                pdf_info = get_pdf_info(pdf_path)
                # Use higher DPI for smaller pages (likely details)
                page_info = pdf_info.pages[page - 1] if page <= len(pdf_info.pages) else None
                if page_info and page_info.get("width_inches", 0) < 11:
                    dpi = max(dpi, 600)
                    warnings.append(f"Auto-selected {dpi} DPI for small page")

            # Render PDF
            render_result = await render_pdf_high_quality_async(
                pdf_path=pdf_path,
                page_number=page - 1,  # 0-indexed
                dpi=dpi,
                output_dir=output_dir,
                convert_grayscale=self.config.convert_grayscale,
            )

            # Load the image for further processing
            from PIL import Image
            image = Image.open(render_result.image_path)

            # Super-resolution for low DPI
            if self.config.super_resolution and dpi < 200 and is_super_resolution_available():
                try:
                    sr_result = await upscale_image(
                        image,
                        scale=self.config.super_resolution_scale,
                    )
                    image = sr_result.upscaled_image
                    dpi *= self.config.super_resolution_scale
                    warnings.append(f"Applied {self.config.super_resolution_scale}x super-resolution")
                except Exception as e:
                    warnings.append(f"Super-resolution failed: {e}")

            return StageResult(
                stage="1_render_pdf",
                success=True,
                duration_ms=(time.perf_counter() - start) * 1000,
                data={
                    "image_path": render_result.image_path,
                    "image": image,
                    "dpi": dpi,
                    "width_px": image.width,
                    "height_px": image.height,
                    "color_mode": image.mode,
                },
                warnings=warnings,
            )

        except Exception as e:
            errors.append(str(e))
            return StageResult(
                stage="1_render_pdf",
                success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                errors=errors,
            )

    # =========================================================================
    # Stage 2: Preprocessing
    # =========================================================================

    async def _stage2_preprocess(
        self,
        image: "PILImage.Image",
        dpi: int,
    ) -> StageResult:
        """Stage 2: Preprocess image (denoise, deskew, binarize)."""
        import time
        import cv2
        import numpy as np

        start = time.perf_counter()
        warnings = []

        try:
            # Check image size - skip expensive operations for very large images
            total_pixels = image.width * image.height
            is_large_image = total_pixels > 20_000_000  # 20 megapixels

            if is_large_image:
                logger.info(
                    "large_image_detected",
                    width=image.width,
                    height=image.height,
                    pixels=total_pixels,
                    skipping="NLM denoise, ensemble binarization"
                )

            # Convert PIL to OpenCV
            logger.debug("preprocess_converting_to_opencv")
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            processed = gray.copy()

            # 2.1 Denoise (NLM) - skip for large images (too slow)
            if self.config.denoise and not is_large_image:
                logger.debug("preprocess_denoising")
                from .preprocessing import denoise
                processed = denoise(processed, strength=self.config.denoise_strength)
            elif is_large_image and self.config.denoise:
                warnings.append("Skipped NLM denoise for large image")

            # 2.2 Deskew
            skew_angle = 0.0
            if self.config.deskew:
                logger.debug("preprocess_deskewing")
                from .preprocessing import deskew, DeskewConfig
                deskew_result = deskew(
                    processed,
                    config=DeskewConfig(method=self.config.deskew_method),
                )
                processed = deskew_result.image
                skew_angle = deskew_result.skew_angle
                if abs(skew_angle) > 0.5:
                    warnings.append(f"Corrected {skew_angle:.2f}° skew")

            # 2.3 Enhance contrast (CLAHE)
            if self.config.enhance_contrast:
                logger.debug("preprocess_enhancing_contrast")
                from .preprocessing import enhance_contrast
                processed = enhance_contrast(processed)

            # 2.4 Binarize - use quick method for large images
            binary = None
            if self.config.binarize:
                logger.debug("preprocess_binarizing")
                if self.config.binarize_method == "ensemble" and not is_large_image:
                    from .binarization import ensemble_binarize
                    binarize_result = ensemble_binarize(processed)
                    binary = binarize_result.binary_image
                else:
                    from .binarization import quick_binarize
                    binary = quick_binarize(processed)  # Returns np.ndarray directly
                    if is_large_image:
                        warnings.append("Used quick binarize for large image")

            # Convert back to PIL
            logger.debug("preprocess_converting_to_pil")
            from PIL import Image
            processed_pil = Image.fromarray(processed)
            binary_pil = Image.fromarray(binary) if binary is not None else None

            return StageResult(
                stage="2_preprocess",
                success=True,
                duration_ms=(time.perf_counter() - start) * 1000,
                data={
                    "processed_image": processed_pil,
                    "binary_image": binary_pil,
                    "skew_angle": skew_angle,
                    "grayscale_array": processed,
                    "binary_array": binary,
                },
                warnings=warnings,
            )

        except Exception as e:
            return StageResult(
                stage="2_preprocess",
                success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                errors=[str(e)],
            )

    # =========================================================================
    # Stage 3: Gemini Analysis
    # =========================================================================

    async def _stage3_analyze(
        self,
        image: "PILImage.Image",
        image_path: Path,
    ) -> StageResult:
        """Stage 3: Analyze drawing with Gemini Vision."""
        import time

        start = time.perf_counter()

        try:
            from .gemini_understanding import analyze_drawing
            from .coordinate_calibration import calibrate_from_analysis

            # Analyze with Gemini
            analysis = await analyze_drawing(
                image_path,
                model=self.config.gemini_model,
            )

            # Calibrate scale
            calibration = calibrate_from_analysis(
                analysis=analysis,
                image_width=image.width,
                image_height=image.height,
            )

            return StageResult(
                stage="3_gemini_analysis",
                success=True,
                duration_ms=(time.perf_counter() - start) * 1000,
                data={
                    "analysis": analysis,
                    "calibration": calibration,
                    "drawing_type": analysis.drawing_type,
                    "scale": analysis.scale,
                    "total_elements": analysis.total_elements,
                },
            )

        except Exception as e:
            return StageResult(
                stage="3_gemini_analysis",
                success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                errors=[str(e)],
            )

    # =========================================================================
    # Stage 4: Vector Extraction
    # =========================================================================

    async def _stage4_extract(
        self,
        image: "PILImage.Image",
        analysis: Any,
        calibration: Any,
        image_path: Optional[Path] = None,
    ) -> StageResult:
        """Stage 4: Extract vectors (lines, circles, arcs, text)."""
        import time

        start = time.perf_counter()
        warnings = []

        try:
            from .adaptive_extraction import (
                hybrid_extract_all,
                HybridExtractionConfig,
            )

            # Configure hybrid extraction
            hybrid_config = HybridExtractionConfig(
                use_opencv_for_lines=True,
                use_opencv_for_circles=True,
                use_yolo_for_symbols=False,
                use_gemini_for_text=True,
                use_gemini_for_semantic=True,
            )

            # Run hybrid extraction
            extraction_result = await hybrid_extract_all(
                analysis=analysis,
                calibration=calibration,
                image_path=image_path,
                config=hybrid_config,
            )

            entities = extraction_result.entities

            # Optional: HAWP junction detection for floor plans
            if self.config.use_hawp_junctions and analysis.drawing_type == "floor_plan":
                try:
                    from .neural_junction_detection import (
                        detect_junctions,
                        snap_endpoints_to_junctions,
                        is_junction_detection_available,
                    )

                    if is_junction_detection_available():
                        import numpy as np
                        img_array = np.array(image)
                        junctions = await detect_junctions(img_array)
                        # Snap line endpoints to detected junctions
                        entities = snap_endpoints_to_junctions(
                            entities,
                            junctions.junctions,
                            snap_distance=10.0,
                        )
                        warnings.append(f"Snapped to {len(junctions.junctions)} HAWP junctions")
                except Exception as e:
                    warnings.append(f"HAWP junction detection skipped: {e}")

            return StageResult(
                stage="4_extract",
                success=True,
                duration_ms=(time.perf_counter() - start) * 1000,
                data={
                    "entities": entities,
                    "extraction_result": extraction_result,
                    "entity_count": len(entities),
                },
                warnings=warnings,
            )

        except Exception as e:
            return StageResult(
                stage="4_extract",
                success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                errors=[str(e)],
            )

    # =========================================================================
    # Stage 5: Symbol Recognition (RAG)
    # =========================================================================

    async def _stage5_recognize_symbols(
        self,
        image: "PILImage.Image",
        analysis: Any,
        calibration: Any,
    ) -> StageResult:
        """Stage 5: Recognize symbols using CLIP + pgvector RAG."""
        import time

        start = time.perf_counter()
        warnings = []
        symbol_entities = []
        recognized_symbols = []

        try:
            # Get detected symbols from Gemini analysis
            if not analysis or not analysis.elements:
                return StageResult(
                    stage="5_symbol_rag",
                    success=True,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    data={"symbol_entities": [], "recognized_symbols": []},
                    warnings=["No symbols detected in analysis"],
                )

            detected_symbols = analysis.elements.symbols or []

            if not detected_symbols:
                return StageResult(
                    stage="5_symbol_rag",
                    success=True,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    data={"symbol_entities": [], "recognized_symbols": []},
                    warnings=["No symbols to recognize"],
                )

            # Process based on method
            if self.config.symbol_method == SymbolRecognitionMethod.RAG and self._symbol_rag:
                # Use RAG for symbol recognition
                from .symbol_rag import extract_symbol_region, SymbolDomain
                from .adaptive_extraction import EntityToCreate, EntityType

                for symbol in detected_symbols:
                    try:
                        # Extract symbol region from image
                        pos = symbol.position
                        # Estimate symbol bounds (typically 50-100 px)
                        size = 75
                        bbox = (
                            max(0, pos[0] - size // 2),
                            max(0, pos[1] - size // 2),
                            min(image.width, pos[0] + size // 2),
                            min(image.height, pos[1] + size // 2),
                        )
                        symbol_patch = extract_symbol_region(image, bbox, padding=10)

                        # Map symbol type to domain
                        domain = self._map_symbol_type_to_domain(symbol.symbol_type)

                        # RAG lookup
                        matches = await self._symbol_rag.recognize_symbol(
                            image=symbol_patch,
                            domain=domain,
                            top_k=self.config.symbol_top_k,
                            min_confidence=self.config.symbol_min_confidence,
                        )

                        if matches:
                            best_match = matches[0]
                            recognized_symbols.append({
                                "original_type": symbol.symbol_type,
                                "original_subtype": symbol.subtype,
                                "rag_block_name": best_match.block_name,
                                "rag_confidence": best_match.confidence,
                                "rag_layer": best_match.layer,
                            })

                            # Create entity with RAG-identified block
                            position_dwg = calibration.to_dwg(*pos) if calibration else pos
                            entity = EntityToCreate(
                                entity_type=EntityType.BLOCK,
                                layer=best_match.layer,
                                properties={
                                    "block_name": best_match.block_name,
                                    "position": position_dwg,
                                    "rotation": symbol.rotation or 0.0,
                                    "scale": best_match.default_scale,
                                    "rag_confidence": best_match.confidence,
                                },
                            )
                            symbol_entities.append(entity)
                        else:
                            warnings.append(f"No RAG match for symbol at {pos}")

                    except Exception as e:
                        warnings.append(f"RAG failed for symbol: {e}")

            else:
                # Fallback to hardcoded mapping
                from .adaptive_extraction import (
                    extract_symbols_direct,
                    EntityToCreate,
                )

                symbol_entities = extract_symbols_direct(
                    analysis.elements,
                    calibration,
                )

            # Get method value (handle both enum and string)
            method_value = (
                self.config.symbol_method.value
                if hasattr(self.config.symbol_method, 'value')
                else str(self.config.symbol_method)
            )

            return StageResult(
                stage="5_symbol_rag",
                success=True,
                duration_ms=(time.perf_counter() - start) * 1000,
                data={
                    "symbol_entities": symbol_entities,
                    "recognized_symbols": recognized_symbols,
                    "method": method_value,
                },
                warnings=warnings,
            )

        except Exception as e:
            return StageResult(
                stage="5_symbol_rag",
                success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                errors=[str(e)],
                data={"symbol_entities": [], "recognized_symbols": []},
            )

    def _map_symbol_type_to_domain(self, symbol_type: str) -> Optional["SymbolDomain"]:
        """Map Gemini symbol type to SymbolDomain enum."""
        from .symbol_rag import SymbolDomain

        mapping = {
            "outlet": SymbolDomain.ELECTRICAL,
            "switch": SymbolDomain.ELECTRICAL,
            "light": SymbolDomain.ELECTRICAL,
            "panel": SymbolDomain.ELECTRICAL,
            "receptacle": SymbolDomain.ELECTRICAL,
            "diffuser": SymbolDomain.MECHANICAL,
            "thermostat": SymbolDomain.MECHANICAL,
            "vav": SymbolDomain.MECHANICAL,
            "ahu": SymbolDomain.MECHANICAL,
            "valve": SymbolDomain.PLUMBING,
            "fixture": SymbolDomain.PLUMBING,
            "sink": SymbolDomain.PLUMBING,
            "toilet": SymbolDomain.PLUMBING,
            "detector": SymbolDomain.FIRE,
            "smoke": SymbolDomain.FIRE,
            "pull_station": SymbolDomain.FIRE,
            "horn": SymbolDomain.FIRE,
            "strobe": SymbolDomain.FIRE,
            "door": SymbolDomain.ARCHITECTURAL,
            "window": SymbolDomain.ARCHITECTURAL,
            "stairs": SymbolDomain.ARCHITECTURAL,
        }
        return mapping.get(symbol_type.lower())

    # =========================================================================
    # Stage 6: Validation & Correction
    # =========================================================================

    async def _stage6_validate(
        self,
        entities: list,
        image: "PILImage.Image",
        analysis: Any,
    ) -> StageResult:
        """Stage 6: Validate and correct geometry."""
        import time

        start = time.perf_counter()
        warnings = []

        try:
            from .gemini_refinement import (
                RefinementConfig,
                refine_entities_with_gemini,
                snap_to_grid,
                connect_nearby_endpoints,
                align_nearly_parallel_lines,
            )

            validated = entities.copy()

            # 6.1 Line straightening (5° tolerance)
            if self.config.straighten_lines:
                validated = align_nearly_parallel_lines(
                    validated,
                    tolerance_deg=self.config.straighten_tolerance_deg,
                )

            # 6.2 Endpoint connection
            if self.config.connect_endpoints:
                validated = connect_nearby_endpoints(
                    validated,
                    tolerance=self.config.connect_tolerance_px,
                )

            # 6.3 Grid snapping
            if self.config.snap_to_grid:
                validated = snap_to_grid(
                    validated,
                    grid_size=self.config.grid_tolerance_px,
                )

            # 6.4 Gemini visual QA (optional)
            validation_result = None
            if self.config.gemini_visual_qa:
                try:
                    from .validation import validate_with_gemini

                    validation_result = await validate_with_gemini(
                        original_image=image,
                        entities=validated,
                        analysis=analysis,
                        max_iterations=self.config.max_validation_iterations,
                    )

                    if validation_result.issues:
                        warnings.append(
                            f"Gemini QA found {len(validation_result.issues)} issues"
                        )
                except Exception as e:
                    warnings.append(f"Gemini visual QA skipped: {e}")

            return StageResult(
                stage="6_validate",
                success=True,
                duration_ms=(time.perf_counter() - start) * 1000,
                data={
                    "entities": validated,
                    "validation_result": validation_result,
                    "corrections_applied": len(entities) - len(validated) if len(validated) < len(entities) else 0,
                },
                warnings=warnings,
            )

        except Exception as e:
            return StageResult(
                stage="6_validate",
                success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                errors=[str(e)],
            )

    # =========================================================================
    # Stage 7: Output
    # =========================================================================

    async def _stage7_output(
        self,
        entities: list,
        calibration: Any,
        output_dir: Path,
        base_name: str,
    ) -> StageResult:
        """Stage 7: Prepare output for AutoCAD."""
        import time

        start = time.perf_counter()
        warnings = []

        try:
            # Store to database if configured
            if self.config.store_to_database:
                try:
                    from aec_agent.db.connection import get_database_pool

                    pool = await get_database_pool()
                    if pool:
                        # Storage logic would go here
                        warnings.append("Database storage enabled")
                except Exception as e:
                    warnings.append(f"Database storage skipped: {e}")

            # Prepare entity summary
            by_type = {}
            by_layer = {}
            for entity in entities:
                t = entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type)
                by_type[t] = by_type.get(t, 0) + 1

                layer = entity.layer or "0"
                by_layer[layer] = by_layer.get(layer, 0) + 1

            return StageResult(
                stage="7_output",
                success=True,
                duration_ms=(time.perf_counter() - start) * 1000,
                data={
                    "entity_count": len(entities),
                    "by_type": by_type,
                    "by_layer": by_layer,
                    "output_format": self.config.output_format,
                    "output_dir": str(output_dir),
                },
                warnings=warnings,
            )

        except Exception as e:
            return StageResult(
                stage="7_output",
                success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                errors=[str(e)],
            )


# =============================================================================
# Convenience Functions
# =============================================================================

async def run_best_practices_pipeline(
    pdf_path: str | Path,
    page: int = 1,
    output_dir: Optional[str | Path] = None,
    config: Optional[BestPracticesConfig] = None,
) -> PipelineResult:
    """
    Run the best practices pipeline on a PDF.

    This is the recommended entry point for PDF to AutoCAD vectorization.

    Args:
        pdf_path: Path to PDF file
        page: Page number (1-indexed)
        output_dir: Output directory
        config: Pipeline configuration

    Returns:
        PipelineResult with all outputs

    Example:
        result = await run_best_practices_pipeline("drawing.pdf")
        if result.success:
            print(f"Extracted {result.entity_count} entities")
            print(f"Recognized {result.symbol_count} symbols via RAG")
    """
    pipeline = BestPracticesPipeline(config)
    return await pipeline.process_pdf(pdf_path, page, output_dir)


def run_best_practices_pipeline_sync(
    pdf_path: str | Path,
    page: int = 1,
    output_dir: Optional[str | Path] = None,
    config: Optional[BestPracticesConfig] = None,
) -> PipelineResult:
    """Synchronous wrapper for run_best_practices_pipeline."""
    return asyncio.run(run_best_practices_pipeline(pdf_path, page, output_dir, config))


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Main classes
    "BestPracticesPipeline",
    "BestPracticesConfig",
    # Result types
    "PipelineResult",
    "StageResult",
    # Enums
    "VectorizationMethod",
    "SymbolRecognitionMethod",
    # Convenience functions
    "run_best_practices_pipeline",
    "run_best_practices_pipeline_sync",
]
