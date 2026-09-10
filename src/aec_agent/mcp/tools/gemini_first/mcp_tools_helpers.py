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

logger = structlog.get_logger(__name__)

# Maximum elements to include in tool results (reduces token usage)
MAX_ELEMENTS_IN_RESULT = 10


async def _store_extraction_to_database(
    extraction: HybridExtractionResult,
    pdf_path: Path,
    calibration: "ScaleCalibration",
) -> dict:
    """
    Store extracted entities to PostgreSQL for semantic search and spatial queries.

    Converts EntityToCreate objects to Element models and stores them in the database.

    Args:
        extraction: Hybrid extraction result with entities
        pdf_path: Source PDF file path
        calibration: Scale calibration for coordinate conversion

    Returns:
        Storage summary dict with counts
    """
    from aec_agent.db.connection import get_database_pool
    from aec_agent.db.repository import ElementRepository
    from aec_agent.db.models import Element, Project, CentroidInfo
    from uuid import uuid4
    import hashlib

    try:
        pool = await get_database_pool()
        if pool is None:
            logger.debug("Database not configured, skipping storage")
            return {"stored": 0, "skipped": True, "reason": "database_not_configured"}

        repo = ElementRepository(pool)

        # Create or get project for this PDF
        file_hash = hashlib.md5(str(pdf_path).encode()).hexdigest()
        existing_project = await repo.get_project_by_file_hash(file_hash)

        if existing_project:
            project_id = existing_project.id
            # Clear existing elements for re-extraction
            await repo.delete_project_elements(project_id)
        else:
            project = Project(
                id=uuid4(),
                name=pdf_path.stem,
                source="autocad",  # Gemini extraction targets AutoCAD
                file_path=str(pdf_path),
                file_hash=file_hash,
                metadata={
                    "extraction_method": "gemini_hybrid",
                    "scale_factor": calibration.scale_factor,
                    "units": calibration.units,
                },
            )
            project_id = await repo.create_project(project)

        # Convert and store entities
        elements = []
        for i, entity in enumerate(extraction.entities):
            # Build geometry WKT based on entity type
            geom_wkt = None
            centroid = None

            # Ensure properties is a dict (guard against malformed data)
            props = entity.properties if isinstance(entity.properties, dict) else {}
            entity_type = entity.entity_type.value if hasattr(entity.entity_type, "value") else str(entity.entity_type)

            if entity_type == "LINE":
                start = props.get("start", (0, 0))
                end = props.get("end", (0, 0))
                geom_wkt = f"LINESTRING({start[0]} {start[1]}, {end[0]} {end[1]})"
                centroid = CentroidInfo(
                    x=(start[0] + end[0]) / 2,
                    y=(start[1] + end[1]) / 2,
                )
            elif entity_type == "CIRCLE":
                center = props.get("center", (0, 0))
                radius = props.get("radius", 1)
                # Store circle as a point with radius in properties
                geom_wkt = f"POINT({center[0]} {center[1]})"
                centroid = CentroidInfo(x=center[0], y=center[1])
            elif entity_type == "ARC":
                center = props.get("center", (0, 0))
                geom_wkt = f"POINT({center[0]} {center[1]})"
                centroid = CentroidInfo(x=center[0], y=center[1])
            elif entity_type == "TEXT" or entity_type == "MTEXT":
                position = props.get("position", (0, 0))
                geom_wkt = f"POINT({position[0]} {position[1]})"
                centroid = CentroidInfo(x=position[0], y=position[1])
            elif "position" in props:
                pos = props["position"]
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    geom_wkt = f"POINT({pos[0]} {pos[1]})"
                    centroid = CentroidInfo(x=pos[0], y=pos[1])

            # Build description for semantic search
            description_parts = [entity_type]
            if entity.layer:
                description_parts.append(f"on layer {entity.layer}")
            if props.get("text"):
                description_parts.append(f"text: {props['text'][:50]}")

            element = Element(
                id=uuid4(),
                project_id=project_id,
                source_id=f"gemini_{i}",
                source="autocad",
                entity_type=entity_type,
                layer=entity.layer,
                geom_wkt=geom_wkt,
                centroid=centroid,
                properties=props,
                description=" ".join(description_parts),
            )
            elements.append(element)

        # Batch insert
        if elements:
            count = await repo.upsert_elements_batch(elements)
            logger.info(
                "Stored extraction to database",
                project_id=str(project_id),
                entities_stored=count,
            )
            return {
                "stored": count,
                "project_id": str(project_id),
                "skipped": False,
            }

        return {"stored": 0, "skipped": False, "reason": "no_entities"}

    except Exception as e:
        logger.warning("Failed to store extraction to database", error=str(e))
        return {"stored": 0, "skipped": True, "reason": str(e)}


def _summarize_analysis(analysis: DrawingAnalysis) -> dict:
    """Summarize analysis for compact tool results."""
    elements = analysis.elements
    return {
        "drawing_type": analysis.drawing_type,
        "complexity": analysis.complexity,
        "scale": analysis.scale,
        "units": analysis.units,
        "total_elements": analysis.total_elements,
        "element_counts": {
            "lines": len(elements.lines),
            "arcs": len(elements.arcs),
            "circles": len(elements.circles),
            "text": len(elements.text),
            "symbols": len(elements.symbols),
            "dimensions": len(elements.dimensions),
        },
        "calibration_hints": len(analysis.calibration_hints),
        "extraction_strategy": analysis.extraction_strategy.primary_strategy if analysis.extraction_strategy else None,
    }


def _summarize_extraction(extraction: ExtractionResult) -> dict:
    """Summarize extraction for compact tool results."""
    by_type = {}
    for entity in extraction.entities:
        t = entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type)
        by_type[t] = by_type.get(t, 0) + 1

    return {
        "total_entities": extraction.total_entities,
        "direct_count": extraction.direct_count,
        "guided_count": extraction.guided_count,
        "opencv_count": extraction.opencv_count,
        "by_type": by_type,
        "layers_used": list(set(e.layer for e in extraction.entities))[:10],
    }


def _summarize_creation(creation: AutoCADCreationResult) -> dict:
    """Summarize creation result for compact tool results."""
    stats_dict = creation.statistics.to_dict()
    return {
        "total_entities": creation.statistics.total_entities,
        "success_count": creation.statistics.success_count,
        "failure_count": creation.statistics.failure_count,
        "success_rate": f"{creation.success_rate:.0%}",
        "layers_created": len(creation.layer_results),
        "by_type": stats_dict.get("by_type", {}),
    }


def _summarize_hybrid_extraction(result: HybridExtractionResult) -> dict:
    """Summarize hybrid extraction result for compact tool output."""
    return {
        "total_entities": result.total_entities,
        "by_source": {
            "gemini": result.gemini_entities,
            "opencv": result.opencv_entities,
            "yolo": result.yolo_entities,
        },
        "fusion": {
            "duplicates_merged": result.duplicates_merged,
            "conflicts_resolved": result.conflicts_resolved,
        },
        "by_type": {
            etype: len(elist)
            for etype, elist in get_entities_by_type(result).items()
        },
        "required_layers": get_required_layers(result)[:10],
        "required_blocks": get_required_blocks(result)[:10],
    }


