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
    max_retries: int = 5,
    base_delay: float = 5.0,
    model_name: str = "gemini-2.0-flash",
) -> str:
    """
    Call Gemini API with exponential backoff retry for 429 rate limits.

    Uses the new google.genai SDK.

    Args:
        client_or_model: Gemini Client instance (new SDK) or model name string
        content: Content to send (prompt + image as list)
        generation_config: Generation configuration dict with temperature, max_output_tokens, etc.
        max_retries: Maximum retry attempts (default 3)
        base_delay: Base delay in seconds (doubles each retry)
        model_name: Model name to use (default gemini-2.0-flash)

    Returns:
        Response text from Gemini

    Raises:
        Exception: If all retries exhausted
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
            # Convert PIL Image to bytes
            import io
            buf = io.BytesIO()
            item.save(buf, format='PNG')
            parts.append(types.Part.from_bytes(
                data=buf.getvalue(),
                mime_type='image/png'
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
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            return response.text
        except Exception as e:
            error_str = str(e)
            last_error = e

            # Check if it's a rate limit error (429)
            if "429" in error_str or "Resource exhausted" in error_str:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "gemini_rate_limited",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        delay_seconds=delay,
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

# Import MCP tools to register them with the server
from . import mcp_tools

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
    # MCP tools module
    "mcp_tools",
]
