"""
Gemini-First PDF to AutoCAD Vectorization Pipeline.

This package implements the 6-phase Gemini-First architecture:
- Phase 1: PDF Intake & Rendering (no preprocessing, preserve quality)
- Phase 2: Gemini Understanding (analyze original image)
- Phase 3: Coordinate Calibration (map pixels to DWG units)
- Phase 4: Adaptive Extraction (direct/guided/selective strategies)
- Phase 5: AutoCAD Entity Creation (draw in DWG)
- Phase 6: Validation & Self-Correction (Gemini verifies)
"""

import asyncio
from typing import Any, Union

import structlog

logger = structlog.get_logger(__name__)

# Singleton client for the package
_gemini_client = None


def get_gemini_client(api_key: str = None):
    """Get or create the Gemini client singleton."""
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        if api_key:
            _gemini_client = genai.Client(api_key=api_key)
        else:
            # Will use GEMINI_API_KEY or GOOGLE_API_KEY env var
            from aec_agent.config.settings import get_settings
            settings = get_settings()
            if not settings.gemini_api_key:
                raise ValueError(
                    "GEMINI_API_KEY not set. Set it in environment or .env file."
                )
            _gemini_client = genai.Client(api_key=settings.gemini_api_key)
    return _gemini_client


async def gemini_call_with_retry(
    client_or_model: Any,
    content: list,
    generation_config: dict,
    max_retries: int = 8,
    base_delay: float = 10.0,
    model_name: str = "gemini-1.5-flash",
    timeout_seconds: float = 120.0,
) -> str:
    """
    Call Gemini API with exponential backoff retry for 429 rate limits.

    Uses the new google.genai SDK.

    Args:
        client_or_model: Gemini Client instance (new SDK) or model name string
        content: Content to send (prompt + image as list)
        generation_config: Generation configuration dict with temperature, max_output_tokens, etc.
        max_retries: Maximum retry attempts (default 8 for resilience against rate limits)
        base_delay: Base delay in seconds (doubles each retry, default 10s)
        model_name: Model name to use (default gemini-2.0-flash)
        timeout_seconds: Timeout for each API call (default 120s)

    Returns:
        Response text from Gemini

    Raises:
        Exception: If all retries exhausted (after ~42 minutes of retrying)
    """
    from google import genai
    from google.genai import types

    # Get or create client
    if isinstance(client_or_model, genai.Client):
        client = client_or_model
    elif isinstance(client_or_model, str):
        # If a string is passed, treat it as model name and get default client
        model_name = client_or_model
        client = get_gemini_client()
    else:
        # Legacy: if old model object passed, get new client
        client = get_gemini_client()

    # Build contents in the new format
    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(types.Part.from_text(text=item))
        elif hasattr(item, 'mode'):  # PIL Image
            # Convert PIL Image to bytes with size optimization
            import io
            from PIL import Image

            img = item
            # Resize if too large (max 4096px on longest side for Gemini)
            max_dimension = 4096
            if max(img.width, img.height) > max_dimension:
                ratio = max_dimension / max(img.width, img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                logger.debug(
                    "image_resized_for_gemini",
                    original_size=(item.width, item.height),
                    new_size=new_size,
                )

            # Save as JPEG for smaller file size (unless it's grayscale/binary)
            buf = io.BytesIO()
            if img.mode in ('L', '1'):
                img.save(buf, format='PNG', optimize=True)
                mime_type = 'image/png'
            else:
                # Convert to RGB if needed (JPEG doesn't support RGBA)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                img.save(buf, format='JPEG', quality=85, optimize=True)
                mime_type = 'image/jpeg'

            logger.debug(
                "image_prepared_for_gemini",
                size_bytes=buf.tell(),
                mime_type=mime_type,
            )

            parts.append(types.Part.from_bytes(
                data=buf.getvalue(),
                mime_type=mime_type
            ))
        else:
            # Assume it's already a Part or can be converted
            parts.append(item)

    contents = [types.Content(role="user", parts=parts)]

    # Build config
    config = types.GenerateContentConfig(
        temperature=generation_config.get("temperature", 0.1),
        max_output_tokens=generation_config.get("max_output_tokens", 8192),
    )

    last_error = None

    for attempt in range(max_retries + 1):
        try:
            # Wrap API call in timeout
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                ),
                timeout=timeout_seconds,
            )
            return response.text
        except asyncio.TimeoutError:
            last_error = TimeoutError(
                f"Gemini API call timed out after {timeout_seconds}s"
            )
            logger.warning(
                "gemini_timeout",
                attempt=attempt + 1,
                timeout_seconds=timeout_seconds,
            )
            # Timeouts are retryable
            if attempt < max_retries:
                await asyncio.sleep(base_delay)
                continue
            raise last_error
        except Exception as e:
            error_str = str(e)
            last_error = e

            # Check if it's a rate limit error (429)
            if "429" in error_str or "Resource exhausted" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt < max_retries:
                    # Exponential backoff with max cap of 5 minutes
                    delay = min(base_delay * (2 ** attempt), 300.0)
                    logger.warning(
                        "gemini_rate_limited",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        delay_seconds=delay,
                        error=error_str[:200],
                    )
                    await asyncio.sleep(delay)
                    continue

            # Non-retryable error or max retries exceeded
            raise

    raise last_error


# Phase 1: PDF Intake & Rendering
from .pdf_intake import (
    PDFInfo,
    PDFRenderResult,
    render_pdf_high_quality,
    render_pdf_high_quality_async,
    render_pdf_page,
    render_all_pages,
    get_pdf_info,
    is_effectively_grayscale,
    compare_with_bitonal,
)

# Phase 2: Gemini Understanding
from .gemini_understanding import (
    DrawingType,
    ExtractionStrategy,
    DrawingAnalysis,
    DrawingElements,
    DrawingAnalyzer,
    DetectedLine,
    DetectedArc,
    DetectedCircle,
    DetectedText,
    DetectedSymbol,
    DetectedDimension,
    CalibrationHint,
    SpecialRegion,
    RegionBounds,
    ExtractionStrategyConfig,
    analyze_drawing,
)

# Phase 3: Coordinate Calibration
from .coordinate_calibration import (
    ScaleCalibration,
    calibrate_from_analysis,
    calibrate_from_dimension,
    calibrate_from_scale_notation,
    calibrate_from_sheet_size,
    calibrate_from_dpi_default,
    calibrate_manual,
    parse_measurement,
    parse_scale_notation,
    parse_sheet_size,
    estimate_drawing_bounds,
    SHEET_SIZES,
)

# Phase 4: Adaptive Extraction
from .adaptive_extraction import (
    ExtractionSource,
    EntityType,
    EntityToCreate,
    RasterCommand,
    ExtractionResult,
    extract_all,
    extract_direct_only,
    direct_extraction,
    guided_rasterization,
    selective_opencv,
    get_layer_for_element_type,
    get_layer_for_text,
    get_layer_for_symbol,
    get_block_name,
    get_vtool_for_path_type,
    get_entities_by_type,
    get_entities_by_layer,
    get_required_layers,
    get_required_blocks,
    ELEMENT_TYPE_TO_LAYER,
    TEXT_TYPE_TO_LAYER,
    SYMBOL_TYPE_TO_LAYER,
    SYMBOL_TO_BLOCK,
    VTOOL_MAPPING,
    # Hybrid extraction (Gemini + OpenCV + YOLO fusion)
    HybridExtractionConfig,
    HybridExtractionResult,
    hybrid_extract_all,
    hybrid_opencv_extraction,
    hybrid_yolo_extraction,
    # Draftsman-like cleanup
    DraftCleanupConfig,
    CleanupStatistics,
    clean_entities_like_draftsman,
)

# Phase 4b: OpenCV Extraction Utilities
from .opencv_extraction import (
    OpenCVExtractor,
    OpenCVExtractionResult,
    ExtractedLine,
    ExtractedCircle,
    ExtractedArc,
    ExtractedPolyline,
    ExtractedContour,
    LineType,
    extract_from_image,
    extract_lines_from_region,
    extract_circles_from_region,
    OPENCV_AVAILABLE,
)

# Phase B: Text/Graphics Separation (Fletcher-Kasturi)
from .text_graphics_separation import (
    SeparationConfig,
    SeparationResult,
    FletcherKasturiSeparator,
    ConnectedComponent,
    TextComponent,
    GraphicsComponent,
    TextLine,
    ComponentType,
    separate_text_from_graphics,
    is_separation_available,
)

# Phase B: Super-Resolution (Real-ESRGAN)
from .super_resolution import (
    SuperResolutionConfig,
    SuperResolutionResult,
    RealESRGANUpscaler,
    upscale_image,
    is_super_resolution_available,
    is_gpu_available,
    REALESRGAN_AVAILABLE,
    TORCH_AVAILABLE,
)

# Phase B: VTracer Extraction (O(n) vectorization)
from .vtracer_extraction import (
    VTracerExtractor,
    VTracerExtractionResult,
    VTracerConfig,
    VTracerColorMode,
    VTracerMode,
    ExtractedPath,
    BezierSegment,
    LineSegment,
    extract_with_vtracer,
    is_vtracer_available,
    VTRACER_AVAILABLE,
)

# Phase 4c: Gemini Refinement (coordinate adjustment)
from .gemini_refinement import (
    RefinementConfig,
    RefinementResult,
    RefinementAdjustment,
    refine_entities_with_gemini,
    snap_to_grid,
    connect_nearby_endpoints,
    align_nearly_parallel_lines,
    remove_duplicate_lines,
)

# Phase 4d: OCR Text Position Anchoring
from .ocr_text_anchoring import (
    OCRTextBox,
    TextAnchorResult,
    TextAnchoringResult,
    extract_text_with_ocr,
    anchor_text_to_ocr,
    anchor_text_positions,
    is_ocr_available,
    TESSERACT_AVAILABLE,
)

# Phase A: Advanced Preprocessing (NEW - from VECTORIZATION_IMPROVEMENT_ROADMAP)
from .binarization import (
    BinarizationMethod,
    BinarizationConfig,
    BinarizationResult,
    ensemble_binarize,
    quick_binarize,
    fast_binarize,
    binarize_otsu,
    binarize_adaptive_gaussian,
    binarize_sauvola,
    binarize_niblack,
)

from .preprocessing import (
    SkewDetectionMethod,
    DeskewConfig,
    DeskewResult,
    PreprocessingConfig,
    PreprocessingResult,
    deskew,
    quick_deskew,
    preprocess_image,
    quick_preprocess,
    denoise,
    enhance_contrast,
    remove_borders,
)

from .simplification import (
    SimplificationMethod,
    SimplificationConfig,
    SimplificationResult,
    simplify_line,
    simplify_polygon,
    simplify_entities,
    rdp_simplify,
    visvalingam_simplify,
    quick_simplify_rdp,
    quick_simplify_visvalingam,
)

# Phase 5: AutoCAD Entity Creation
from .autocad_creation import (
    EntityCreationResult,
    LayerCreationResult,
    CreationStatistics,
    AutoCADCreationResult,
    create_entities_in_autocad,
    create_entities_batch,
    create_single_entity,
    create_layer_if_needed,
    create_line_entity,
    create_arc_entity,
    create_circle_entity,
    create_text_entity,
    create_block_entity,
    create_polyline_entity,
    get_entity_type_stats,
    get_failed_by_type,
    get_color_for_layer,
    LAYER_PREFIX_COLORS,
    ENTITY_CREATORS,
)

# Phase 6: Validation & Self-Correction
from .validation import (
    ValidationStatus,
    IssueType,
    IssueSeverity,
    CorrectionAction,
    ValidationIssue,
    Correction,
    CorrectionResult,
    ValidationResult,
    validate_extraction,
    apply_corrections,
    validate_with_gemini,
    get_critical_issues,
    get_issues_by_type,
    summarize_validation,
)

# Phase C: Advanced Vectorization Techniques
# C.1: Neural Junction Detection (HAWP)
from .neural_junction_detection import (
    JunctionType,
    JunctionDetectionConfig,
    DetectedJunction,
    DetectedWireframeLine,
    JunctionDetectionResult,
    JunctionDetector,
    detect_junctions,
    snap_endpoints_to_junctions,
    is_junction_detection_available,
)

# C.2: Bezier Splatting (differentiable curve fitting)
from .bezier_splatting import (
    CurveType,
    BezierSplattingConfig,
    OptimizedCurve,
    BezierSplattingResult,
    BezierSplattingOptimizer,
    bezier_splat,
    is_bezier_splatting_available,
)

# C.3: LIVE Layered Vectorization
from .live_vectorization import (
    LIVEConfig,
    VectorPath,
    VectorLayer,
    LIVEResult,
    LIVEVectorizer,
    live_vectorize,
    is_live_available,
)

# Phase C: Pipeline Visualizer (for assessment/debugging)
from .pipeline_visualizer import (
    VisualizationResult,
    visualize_pipeline,
    visualize_pipeline_sync,
)

# Phase D: RAG Symbol Recognition
from .symbol_rag import (
    SymbolRAG,
    CLIPEncoder,
    SymbolRAGRepository,
    SymbolMatch,
    SymbolLibraryEntry,
    SymbolRecognitionConfig,
    SymbolDomain,
    is_symbol_rag_available,
    extract_symbol_region,
    image_to_base64,
    base64_to_image,
)

# Best Practices Pipeline (7-stage orchestrator with all optimal algorithms)
from .best_practices_pipeline import (
    BestPracticesPipeline,
    BestPracticesConfig,
    PipelineResult as BestPracticesPipelineResult,
    StageResult as BestPracticesStageResult,
    VectorizationMethod,
    SymbolRecognitionMethod,
    run_best_practices_pipeline,
    run_best_practices_pipeline_sync,
)

# Unified Pipeline (RECOMMENDED - consolidates all vectorization workflows)
from .unified_pipeline import (
    UnifiedPipeline,
    PipelineConfig,
    PipelineResult,
    StageResult,
    ExtractionMethod,
    SymbolMethod,
    OutputFormat,
    RefinementConfig,
    vectorize_pdf,
)
# Extracted from unified_pipeline.py into focused sibling files (pure structural move)
from .unified_pipeline_refinement import GeometryRefinementPipeline
from .unified_pipeline_factory import ExtractionFactory

# Phase A (Scan2CAD Parity): Linetype Detection
from .linetype_detection import (
    LinetypeName,
    LinetypeResult,
    LinetypeConfig,
    detect_linetype,
    detect_linetypes_batch,
    map_to_autocad_linetype,
    parse_linetype_name,
    LINETYPE_PATTERNS,
)

# Phase A (Scan2CAD Parity): Bezier Curve Fitting
from .bezier_fitting import (
    BezierCurve,
    CurveSegment,
    BezierFitConfig,
    fit_bezier_to_points,
    detect_corners,
    detect_curves,
    curves_to_spline_data,
)

# Phase A (Scan2CAD Parity): DXF Export
from .dxf_export import (
    DXFVersion,
    Units as DXFUnits,
    DXFExportConfig,
    DXFExportResult,
    DXFExporter,
    export_to_dxf,
    is_ezdxf_available,
)

# Phase A (Scan2CAD Parity): Line Thickness Detection
from .thickness_detection import (
    ThicknessCategory,
    ThicknessResult,
    ThicknessConfig,
    detect_line_thickness,
    detect_thickness_batch,
    pixels_to_mm,
    mm_to_lineweight,
    categorize_thickness,
    suggest_layer_from_thickness,
    AUTOCAD_LINEWEIGHTS,
)

# Phase B (Scan2CAD Parity): Batch Processing
from .batch_processor import (
    BatchStatus,
    FileStatus,
    FileResult,
    BatchConfig,
    BatchResult,
    BatchProcessor,
    process_batch,
    process_batch_sync,
    get_batch_status,
    cancel_batch,
)

# Phase B (Scan2CAD Parity): Polyline Auto-Join
from .polyline_builder import (
    PolylineConfig,
    PolylineResult,
    EndpointGraph,
    build_polylines,
    auto_join_lines,
)

# Phase C (Scan2CAD Parity): Preview Generation
from .preview_generator import (
    PreviewConfig,
    PreviewResult,
    generate_preview,
    generate_preview_from_pipeline,
    ENTITY_COLORS,
    LAYER_COLORS,
)

# Phase C (Scan2CAD Parity): Conversion Profiles
from .profiles import (
    ProfileType,
    LayerMapping,
    ExtractionSettings,
    ScaleSettings,
    OutputSettings,
    ConversionProfile,
    get_profile,
    get_profile_by_name,
    get_architectural_profile,
    get_mechanical_profile,
    get_electrical_profile,
    get_structural_profile,
    get_civil_profile,
    get_hvac_profile,
    get_plumbing_profile,
    get_fire_alarm_profile,
    get_general_profile,
    register_custom_profile,
    list_profiles,
    auto_detect_profile,
    create_custom_profile,
)

# Phase C (Scan2CAD Parity): Ellipse Detection
from .ellipse_detection import (
    DetectedEllipse,
    EllipseDetectionConfig,
    EllipseDetectionResult,
    detect_ellipses,
    ellipse_to_entity,
    detect_ellipses_in_regions,
    fit_ellipse_to_arcs,
    is_ellipse_detection_available,
)

# Import MCP tools to register them with the server
# (split from the former monolithic mcp_tools.py into focused sibling modules)
from . import mcp_tools_helpers
from . import mcp_tools_rendering
from . import mcp_tools_analysis
from . import mcp_tools_calibration
from . import mcp_tools_extraction
from . import mcp_tools_creation
from . import mcp_tools_hybrid
from . import mcp_tools_symbol
from . import mcp_tools_pipeline
from . import mcp_tools_export
from . import mcp_tools_knowledge

__all__ = [
    # Phase 1: Data classes
    "PDFInfo",
    "PDFRenderResult",
    # Phase 1: Core functions
    "render_pdf_high_quality",
    "render_pdf_high_quality_async",
    "render_pdf_page",
    "render_all_pages",
    "get_pdf_info",
    "is_effectively_grayscale",
    "compare_with_bitonal",
    # Phase 2: Enums
    "DrawingType",
    "ExtractionStrategy",
    # Phase 2: Data classes
    "DrawingAnalysis",
    "DrawingElements",
    "DetectedLine",
    "DetectedArc",
    "DetectedCircle",
    "DetectedText",
    "DetectedSymbol",
    "DetectedDimension",
    "CalibrationHint",
    "SpecialRegion",
    "RegionBounds",
    "ExtractionStrategyConfig",
    # Phase 2: Core class and function
    "DrawingAnalyzer",
    "analyze_drawing",
    # Phase 3: Data class
    "ScaleCalibration",
    # Phase 3: Core functions
    "calibrate_from_analysis",
    "calibrate_from_dimension",
    "calibrate_from_scale_notation",
    "calibrate_from_sheet_size",
    "calibrate_from_dpi_default",
    "calibrate_manual",
    # Phase 3: Parsing helpers
    "parse_measurement",
    "parse_scale_notation",
    "parse_sheet_size",
    "estimate_drawing_bounds",
    # Phase 3: Constants
    "SHEET_SIZES",
    # Phase 4: Enums
    "ExtractionSource",
    "EntityType",
    # Phase 4: Data classes
    "EntityToCreate",
    "RasterCommand",
    "ExtractionResult",
    # Phase 4: Core functions
    "extract_all",
    "extract_direct_only",
    "direct_extraction",
    "guided_rasterization",
    "selective_opencv",
    # Phase 4: Helpers
    "get_layer_for_element_type",
    "get_layer_for_text",
    "get_layer_for_symbol",
    "get_block_name",
    "get_vtool_for_path_type",
    "get_entities_by_type",
    "get_entities_by_layer",
    "get_required_layers",
    "get_required_blocks",
    # Phase 4: Constants
    "ELEMENT_TYPE_TO_LAYER",
    "TEXT_TYPE_TO_LAYER",
    "SYMBOL_TYPE_TO_LAYER",
    "SYMBOL_TO_BLOCK",
    "VTOOL_MAPPING",
    # Phase 4: Hybrid extraction (Gemini + OpenCV + YOLO fusion)
    "HybridExtractionConfig",
    "HybridExtractionResult",
    "hybrid_extract_all",
    "hybrid_opencv_extraction",
    "hybrid_yolo_extraction",
    # Phase 4: Draftsman-like cleanup
    "DraftCleanupConfig",
    "CleanupStatistics",
    "clean_entities_like_draftsman",
    # Phase 4b: OpenCV extraction utilities
    "OpenCVExtractor",
    "OpenCVExtractionResult",
    "ExtractedLine",
    "ExtractedCircle",
    "ExtractedArc",
    "ExtractedPolyline",
    "ExtractedContour",
    "LineType",
    "extract_from_image",
    "extract_lines_from_region",
    "extract_circles_from_region",
    "OPENCV_AVAILABLE",
    # Phase B: Text/Graphics Separation (Fletcher-Kasturi)
    "SeparationConfig",
    "SeparationResult",
    "FletcherKasturiSeparator",
    "ConnectedComponent",
    "TextComponent",
    "GraphicsComponent",
    "TextLine",
    "ComponentType",
    "separate_text_from_graphics",
    "is_separation_available",
    # Phase B: Super-Resolution (Real-ESRGAN)
    "SuperResolutionConfig",
    "SuperResolutionResult",
    "RealESRGANUpscaler",
    "upscale_image",
    "is_super_resolution_available",
    "is_gpu_available",
    "REALESRGAN_AVAILABLE",
    "TORCH_AVAILABLE",
    # Phase B: VTracer extraction (O(n) vectorization)
    "VTracerExtractor",
    "VTracerExtractionResult",
    "VTracerConfig",
    "VTracerColorMode",
    "VTracerMode",
    "ExtractedPath",
    "BezierSegment",
    "LineSegment",
    "extract_with_vtracer",
    "is_vtracer_available",
    "VTRACER_AVAILABLE",
    # Phase 4c: Gemini Refinement
    "RefinementConfig",
    "RefinementResult",
    "RefinementAdjustment",
    "refine_entities_with_gemini",
    "snap_to_grid",
    "connect_nearby_endpoints",
    "align_nearly_parallel_lines",
    "remove_duplicate_lines",
    # Phase 4d: OCR Text Anchoring
    "OCRTextBox",
    "TextAnchorResult",
    "TextAnchoringResult",
    "extract_text_with_ocr",
    "anchor_text_to_ocr",
    "anchor_text_positions",
    "is_ocr_available",
    "TESSERACT_AVAILABLE",
    # Phase A: Binarization (ensemble pixel voting)
    "BinarizationMethod",
    "BinarizationConfig",
    "BinarizationResult",
    "ensemble_binarize",
    "quick_binarize",
    "fast_binarize",
    "binarize_otsu",
    "binarize_adaptive_gaussian",
    "binarize_sauvola",
    "binarize_niblack",
    # Phase A: Preprocessing (deskewing, denoising)
    "SkewDetectionMethod",
    "DeskewConfig",
    "DeskewResult",
    "PreprocessingConfig",
    "PreprocessingResult",
    "deskew",
    "quick_deskew",
    "preprocess_image",
    "quick_preprocess",
    "denoise",
    "enhance_contrast",
    "remove_borders",
    # Phase A: Line Simplification (RDP, Visvalingam-Whyatt)
    "SimplificationMethod",
    "SimplificationConfig",
    "SimplificationResult",
    "simplify_line",
    "simplify_polygon",
    "simplify_entities",
    "rdp_simplify",
    "visvalingam_simplify",
    "quick_simplify_rdp",
    "quick_simplify_visvalingam",
    # Phase 5: Data classes
    "EntityCreationResult",
    "LayerCreationResult",
    "CreationStatistics",
    "AutoCADCreationResult",
    # Phase 5: Core functions
    "create_entities_in_autocad",
    "create_entities_batch",
    "create_single_entity",
    "create_layer_if_needed",
    # Phase 5: Entity creators
    "create_line_entity",
    "create_arc_entity",
    "create_circle_entity",
    "create_text_entity",
    "create_block_entity",
    "create_polyline_entity",
    # Phase 5: Helpers
    "get_entity_type_stats",
    "get_failed_by_type",
    "get_color_for_layer",
    # Phase 5: Constants
    "LAYER_PREFIX_COLORS",
    "ENTITY_CREATORS",
    # Phase 6: Enums
    "ValidationStatus",
    "IssueType",
    "IssueSeverity",
    "CorrectionAction",
    # Phase 6: Data classes
    "ValidationIssue",
    "Correction",
    "CorrectionResult",
    "ValidationResult",
    # Phase 6: Core functions
    "validate_extraction",
    "apply_corrections",
    "validate_with_gemini",
    # Phase 6: Helpers
    "get_critical_issues",
    "get_issues_by_type",
    "summarize_validation",
    # Phase C: Neural Junction Detection (C.1)
    "JunctionType",
    "JunctionDetectionConfig",
    "DetectedJunction",
    "DetectedWireframeLine",
    "JunctionDetectionResult",
    "JunctionDetector",
    "detect_junctions",
    "snap_endpoints_to_junctions",
    "is_junction_detection_available",
    # Phase C: Bezier Splatting (C.2)
    "CurveType",
    "BezierSplattingConfig",
    "OptimizedCurve",
    "BezierSplattingResult",
    "BezierSplattingOptimizer",
    "bezier_splat",
    "is_bezier_splatting_available",
    # Phase C: LIVE Vectorization (C.3)
    "LIVEConfig",
    "VectorPath",
    "VectorLayer",
    "LIVEResult",
    "LIVEVectorizer",
    "live_vectorize",
    "is_live_available",
    # Phase C: Pipeline Visualizer
    "VisualizationResult",
    "visualize_pipeline",
    "visualize_pipeline_sync",
    # Phase D: RAG Symbol Recognition
    "SymbolRAG",
    "CLIPEncoder",
    "SymbolRAGRepository",
    "SymbolMatch",
    "SymbolLibraryEntry",
    "SymbolRecognitionConfig",
    "SymbolDomain",
    "is_symbol_rag_available",
    "extract_symbol_region",
    "image_to_base64",
    "base64_to_image",
    # Best Practices Pipeline (7-stage orchestrator)
    "BestPracticesPipeline",
    "BestPracticesConfig",
    "PipelineResult",
    "StageResult",
    "VectorizationMethod",
    "SymbolRecognitionMethod",
    "run_best_practices_pipeline",
    "run_best_practices_pipeline_sync",
    # Unified Pipeline (RECOMMENDED)
    "UnifiedPipeline",
    "PipelineConfig",
    "PipelineResult",
    "StageResult",
    "ExtractionMethod",
    "SymbolMethod",
    "OutputFormat",
    "RefinementConfig",
    "GeometryRefinementPipeline",
    "ExtractionFactory",
    "vectorize_pdf",
    # Phase A (Scan2CAD Parity): Linetype Detection
    "LinetypeName",
    "LinetypeResult",
    "LinetypeConfig",
    "detect_linetype",
    "detect_linetypes_batch",
    "map_to_autocad_linetype",
    "parse_linetype_name",
    "LINETYPE_PATTERNS",
    # Phase A (Scan2CAD Parity): Bezier Curve Fitting
    "BezierCurve",
    "CurveSegment",
    "BezierFitConfig",
    "fit_bezier_to_points",
    "detect_corners",
    "detect_curves",
    "curves_to_spline_data",
    # Phase A (Scan2CAD Parity): DXF Export
    "DXFVersion",
    "DXFUnits",
    "DXFExportConfig",
    "DXFExportResult",
    "DXFExporter",
    "export_to_dxf",
    "is_ezdxf_available",
    # Phase A (Scan2CAD Parity): Line Thickness Detection
    "ThicknessCategory",
    "ThicknessResult",
    "ThicknessConfig",
    "detect_line_thickness",
    "detect_thickness_batch",
    "pixels_to_mm",
    "mm_to_lineweight",
    "categorize_thickness",
    "suggest_layer_from_thickness",
    "AUTOCAD_LINEWEIGHTS",
    # Phase B (Scan2CAD Parity): Batch Processing
    "BatchStatus",
    "FileStatus",
    "FileResult",
    "BatchConfig",
    "BatchResult",
    "BatchProcessor",
    "process_batch",
    "process_batch_sync",
    "get_batch_status",
    "cancel_batch",
    # Phase B (Scan2CAD Parity): Polyline Auto-Join
    "PolylineConfig",
    "PolylineResult",
    "EndpointGraph",
    "build_polylines",
    "auto_join_lines",
    # Phase C (Scan2CAD Parity): Preview Generation
    "PreviewConfig",
    "PreviewResult",
    "generate_preview",
    "generate_preview_from_pipeline",
    "ENTITY_COLORS",
    "LAYER_COLORS",
    # Phase C (Scan2CAD Parity): Conversion Profiles
    "ProfileType",
    "LayerMapping",
    "ExtractionSettings",
    "ScaleSettings",
    "OutputSettings",
    "ConversionProfile",
    "get_profile",
    "get_profile_by_name",
    "get_architectural_profile",
    "get_mechanical_profile",
    "get_electrical_profile",
    "get_structural_profile",
    "get_civil_profile",
    "get_hvac_profile",
    "get_plumbing_profile",
    "get_fire_alarm_profile",
    "get_general_profile",
    "register_custom_profile",
    "list_profiles",
    "auto_detect_profile",
    "create_custom_profile",
    # Phase C (Scan2CAD Parity): Ellipse Detection
    "DetectedEllipse",
    "EllipseDetectionConfig",
    "EllipseDetectionResult",
    "detect_ellipses",
    "ellipse_to_entity",
    "detect_ellipses_in_regions",
    "fit_ellipse_to_arcs",
    "is_ellipse_detection_available",
    # MCP tools modules (split from the former monolithic mcp_tools.py)
    "mcp_tools_helpers",
    "mcp_tools_rendering",
    "mcp_tools_analysis",
    "mcp_tools_calibration",
    "mcp_tools_extraction",
    "mcp_tools_creation",
    "mcp_tools_hybrid",
    "mcp_tools_symbol",
    "mcp_tools_pipeline",
    "mcp_tools_export",
    "mcp_tools_knowledge",
]
