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
    # MCP tools module
    "mcp_tools",
]
