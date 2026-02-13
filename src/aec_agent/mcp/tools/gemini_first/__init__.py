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
    # MCP tools module
    "mcp_tools",
]
