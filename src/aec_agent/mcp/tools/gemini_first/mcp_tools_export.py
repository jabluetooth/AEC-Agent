"""
MCP Tools for Gemini-First PDF to AutoCAD Pipeline.

This module exposes the Gemini-First pipeline tools as MCP tools:
- Phase 1: PDF Intake & Rendering
- Phase 2: Gemini Understanding (Drawing Analysis)
- Phase 3: Coordinate Calibration (Map Pixels to DWG Units)
- Phase 4: Adaptive Extraction (Direct / Guided / Selective)
- Phase 5: AutoCAD Entity Creation (Draw in DWG)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import structlog

from aec_agent.mcp.server import mcp
from ..base import ErrorCode, error_result, success_result
from .pdf_intake import (
    PDFRenderResult,
    get_pdf_info,
    render_all_pages,
    render_pdf_high_quality,
    render_pdf_high_quality_async,
)
from .gemini_understanding import (
    DrawingAnalysis,
    DrawingAnalyzer,
    analyze_drawing,
)
from .coordinate_calibration import (
    ScaleCalibration,
    calibrate_from_analysis,
    calibrate_manual,
    parse_measurement,
    parse_scale_notation,
    parse_sheet_size,
    estimate_drawing_bounds,
)
from .adaptive_extraction import (
    ExtractionResult,
    EntityToCreate,
    RasterCommand,
    extract_all,
    extract_direct_only,
    get_layer_for_element_type,
    get_block_name,
    get_entities_by_type,
    get_entities_by_layer,
    get_required_layers,
    get_required_blocks,
    # Hybrid extraction (Gemini + OpenCV + YOLO fusion)
    HybridExtractionConfig,
    HybridExtractionResult,
    hybrid_extract_all,
)
from .autocad_creation import (
    AutoCADCreationResult,
    CreationStatistics,
    EntityCreationResult,
    LayerCreationResult,
    create_entities_in_autocad,
    create_entities_batch,
    get_entity_type_stats,
    get_failed_by_type,
)
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


from .mcp_tools_helpers import logger


@mcp.tool()
async def export_entities_to_dxf(
    pdf_path: str,
    output_path: str,
    page: int = 1,
    extraction_method: str = "hybrid",
    dxf_version: str = "R2018",
    units: str = "inches",
) -> dict[str, Any]:
    """
    Extract entities from a PDF and export directly to DXF file.

    Does NOT require AutoCAD - uses ezdxf library for standalone DXF generation.

    Args:
        pdf_path: Path to input PDF file
        output_path: Path for output DXF file
        page: Page number (1-indexed, default 1)
        extraction_method: Extraction method (direct, hybrid, best, vtracer)
        dxf_version: DXF version (R12, R2000, R2004, R2007, R2010, R2013, R2018)
        units: Drawing units (inches, feet, mm, cm, m)

    Returns:
        Export result with entity count and output path
    """
    try:
        from pathlib import Path
        from .dxf_export import (
            export_to_dxf,
            DXFExportConfig,
            DXFVersion,
            Units as DXFUnits,
            is_ezdxf_available,
        )
        from .unified_pipeline import (
            UnifiedPipeline,
            PipelineConfig,
            ExtractionMethod,
        )

        if not is_ezdxf_available():
            return error_result(
                ErrorCode.MODULE_NOT_FOUND,
                "ezdxf library not installed. Run: pip install ezdxf"
            )

        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            return error_result(
                ErrorCode.FILE_NOT_FOUND,
                f"PDF file not found: {pdf_path}"
            )

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Run extraction pipeline
        config = PipelineConfig(
            extraction_method=ExtractionMethod(extraction_method),
            create_in_autocad=False,  # Don't need AutoCAD
        )
        pipeline = UnifiedPipeline(config)
        result = await pipeline.process(
            pdf_path=str(pdf_file),
            page=page - 1,  # Convert to 0-indexed
        )

        if not result.success or not result.entities:
            return error_result(
                ErrorCode.INTERNAL_ERROR,
                f"Extraction failed: {result.error or 'No entities extracted'}"
            )

        # Export to DXF
        try:
            version = DXFVersion(dxf_version)
        except ValueError:
            version = DXFVersion.R2018

        try:
            dxf_units = DXFUnits(units.lower())
        except ValueError:
            dxf_units = DXFUnits.INCHES

        export_config = DXFExportConfig(
            version=version,
            units=dxf_units,
        )

        export_result = export_to_dxf(
            result.entities,
            output_file,
            export_config,
        )

        if export_result.success:
            return success_result(
                data=export_result.to_dict(),
                message=(
                    f"DXF export complete: {export_result.entity_count} entities "
                    f"to {output_path}"
                ),
            )
        else:
            return error_result(
                ErrorCode.INTERNAL_ERROR,
                f"DXF export failed: {', '.join(export_result.errors)}"
            )

    except Exception as e:
        logger.exception("export_dxf_failed", error=str(e))
        return error_result(
            ErrorCode.INTERNAL_ERROR,
            f"DXF export failed: {e}"
        )


@mcp.tool()
async def generate_vectorization_preview(
    pdf_path: str,
    output_path: str,
    page: int = 1,
    extraction_method: str = "hybrid",
    include_legend: bool = True,
    overlay_alpha: float = 0.6,
) -> dict[str, Any]:
    """
    Generate a visual preview of vectorization results.

    Creates an image showing extracted entities overlaid on the original PDF,
    color-coded by entity type. Useful for quality assessment before AutoCAD creation.

    Args:
        pdf_path: Path to input PDF file
        output_path: Path for output preview image (PNG/JPEG)
        page: Page number (1-indexed, default 1)
        extraction_method: Extraction method (direct, hybrid, best, vtracer)
        include_legend: Include entity count legend (default True)
        overlay_alpha: Entity overlay opacity 0-1 (default 0.6)

    Returns:
        Preview result with entity counts and output path
    """
    try:
        from pathlib import Path
        from .preview_generator import generate_preview, PreviewConfig
        from .unified_pipeline import (
            UnifiedPipeline,
            PipelineConfig,
            ExtractionMethod,
        )
        from .pdf_intake import render_pdf_page

        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            return error_result(
                ErrorCode.FILE_NOT_FOUND,
                f"PDF file not found: {pdf_path}"
            )

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Render PDF to image
        render_result = render_pdf_page(str(pdf_file), page - 1, dpi=300)
        if not render_result or not render_result.image:
            return error_result(
                ErrorCode.INTERNAL_ERROR,
                "Failed to render PDF page"
            )

        # Run extraction pipeline
        config = PipelineConfig(
            extraction_method=ExtractionMethod(extraction_method),
            create_in_autocad=False,
        )
        pipeline = UnifiedPipeline(config)
        result = await pipeline.process(
            pdf_path=str(pdf_file),
            page=page - 1,
        )

        if not result.entities:
            return error_result(
                ErrorCode.INTERNAL_ERROR,
                f"Extraction failed: {result.error or 'No entities extracted'}"
            )

        # Generate preview
        preview_config = PreviewConfig(
            include_legend=include_legend,
            overlay_alpha=overlay_alpha,
        )

        preview_result = generate_preview(
            render_result.image,
            result.entities,
            str(output_file),
            preview_config,
        )

        if preview_result.success:
            return success_result(
                data=preview_result.to_dict(),
                message=(
                    f"Preview generated: {preview_result.entity_count} entities "
                    f"visualized in {output_path}"
                ),
            )
        else:
            return error_result(
                ErrorCode.INTERNAL_ERROR,
                f"Preview generation failed: {preview_result.error}"
            )

    except Exception as e:
        logger.exception("generate_preview_failed", error=str(e))
        return error_result(
            ErrorCode.INTERNAL_ERROR,
            f"Preview generation failed: {e}"
        )


@mcp.tool()
async def list_conversion_profiles() -> dict[str, Any]:
    """
    List all available conversion profiles.

    Profiles provide pre-configured settings optimized for different drawing types:
    - ARCHITECTURAL: Floor plans, elevations, sections
    - MECHANICAL: Machine drawings, assemblies
    - ELECTRICAL: Schematic diagrams, wiring diagrams
    - STRUCTURAL: Beam layouts, foundation plans
    - And more...

    Returns:
        List of available profiles with names and descriptions
    """
    try:
        from .profiles import list_profiles

        profiles = list_profiles()

        return success_result(
            data={
                "profiles": profiles,
                "count": len(profiles),
            },
            message=f"Found {len(profiles)} conversion profiles",
        )

    except Exception as e:
        logger.exception("list_profiles_failed", error=str(e))
        return error_result(
            ErrorCode.INTERNAL_ERROR,
            f"Failed to list profiles: {e}"
        )


@mcp.tool()
async def get_conversion_profile(
    profile_name: str,
) -> dict[str, Any]:
    """
    Get details of a specific conversion profile.

    Args:
        profile_name: Profile name (architectural, mechanical, electrical, etc.)

    Returns:
        Profile configuration details
    """
    try:
        from .profiles import get_profile_by_name

        profile = get_profile_by_name(profile_name)

        if profile is None:
            return error_result(
                ErrorCode.INVALID_PARAMETER,
                f"Profile not found: {profile_name}"
            )

        return success_result(
            data=profile.to_dict(),
            message=f"Profile: {profile.name} - {profile.description}",
        )

    except Exception as e:
        logger.exception("get_profile_failed", error=str(e))
        return error_result(
            ErrorCode.INTERNAL_ERROR,
            f"Failed to get profile: {e}"
        )


@mcp.tool()
async def auto_detect_drawing_profile(
    pdf_path: str,
    page: int = 1,
) -> dict[str, Any]:
    """
    Automatically detect the best conversion profile for a drawing.

    Analyzes the PDF with Gemini to determine the drawing type and
    recommends the appropriate conversion profile.

    Args:
        pdf_path: Path to input PDF file
        page: Page number (1-indexed, default 1)

    Returns:
        Recommended profile with confidence and reasoning
    """
    try:
        from pathlib import Path
        from .profiles import auto_detect_profile
        from .gemini_understanding import analyze_drawing
        from .pdf_intake import render_pdf_page

        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            return error_result(
                ErrorCode.FILE_NOT_FOUND,
                f"PDF file not found: {pdf_path}"
            )

        # Render and analyze
        render_result = render_pdf_page(str(pdf_file), page - 1, dpi=150)
        if not render_result or not render_result.image:
            return error_result(
                ErrorCode.INTERNAL_ERROR,
                "Failed to render PDF page"
            )

        analysis = await analyze_drawing(render_result.image)

        # Detect profile
        profile = auto_detect_profile(
            analysis_result=analysis.to_dict() if analysis else None,
            filename=pdf_file.name,
        )

        return success_result(
            data={
                "profile": profile.to_dict(),
                "drawing_type": analysis.drawing_type.value if analysis else "unknown",
                "filename": pdf_file.name,
            },
            message=f"Recommended profile: {profile.name}",
        )

    except Exception as e:
        logger.exception("auto_detect_profile_failed", error=str(e))
        return error_result(
            ErrorCode.INTERNAL_ERROR,
            f"Profile detection failed: {e}"
        )
