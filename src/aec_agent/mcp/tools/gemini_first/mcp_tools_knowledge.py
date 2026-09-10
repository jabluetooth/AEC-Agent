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


# =============================================================================
# Phase 3 Knowledge Base Tools
# =============================================================================


@mcp.tool()
async def query_cad_standards(
    element_type: str,
    subtype: Optional[str] = None,
    system: Optional[str] = None,
    discipline: Optional[str] = None,
    size: Optional[str] = None,
) -> dict[str, Any]:
    """
    Query CAD standards for a specific element type.

    Returns layer name, block name, color, linetype, and default attributes
    based on NCS (National CAD Standard) and LA-specific conventions.

    Args:
        element_type: Type of element (e.g., "valve", "diffuser", "outlet", "smoke_detector")
        subtype: Specific subtype (e.g., "gate", "ball", "duplex", "3way")
        system: MEP system (e.g., "domestic_cold_water", "supply_air", "fire_alarm")
        discipline: AEC discipline (e.g., "plumbing", "mechanical", "electrical", "fire")
        size: Element size (e.g., '3/4"', "24x24")

    Returns:
        CAD standards with layer, block_name, color, linetype, attributes
    """
    try:
        from aec_agent.mcp.tools.knowledge_query import (
            query_cad_standards as _query_cad_standards,
        )

        standards = await _query_cad_standards(
            element_type=element_type,
            subtype=subtype,
            system=system,
            discipline=discipline,
            size=size,
        )

        return success_result(
            data={
                "layer": standards.layer,
                "color": standards.color,
                "linetype": standards.linetype,
                "lineweight": standards.lineweight,
                "block_name": standards.block_name,
                "attributes": standards.attributes,
                "discipline": standards.discipline.value if standards.discipline else None,
                "system": standards.system.value if standards.system else None,
                "description": standards.description,
                "source": standards.source,
            },
            message=f"Standards for {element_type}: Layer={standards.layer}, Block={standards.block_name}",
        )

    except Exception as e:
        logger.exception("query_cad_standards_failed", error=str(e))
        return error_result(
            ErrorCode.INTERNAL_ERROR,
            f"Failed to query CAD standards: {e}"
        )


@mcp.tool()
async def query_knowledge_base(
    question: str,
    provider: str = "auto",
) -> dict[str, Any]:
    """
    Query the knowledge base using natural language.

    Uses LLM (Gemini/OpenAI/Anthropic) to answer questions about:
    - CAD layer naming conventions (NCS)
    - MEP element standards and symbols
    - Building codes (CA: CMC, CEC, CPC, CFC)
    - NFPA standards (72, 90A, 101)
    - Design requirements and formulas

    Args:
        question: Natural language question about CAD/MEP standards or codes
        provider: LLM provider ("gemini", "openai", "anthropic", "auto")

    Returns:
        Answer with element classification, recommended layer/block, code references
    """
    try:
        from aec_agent.mcp.tools.knowledge_query import query_with_llm

        result = await query_with_llm(question=question, provider=provider)

        return success_result(
            data={
                "answer": result.answer,
                "element_type": result.element_type,
                "subtype": result.subtype,
                "system": result.system,
                "discipline": result.discipline,
                "recommended_layer": result.recommended_layer,
                "recommended_block": result.recommended_block,
                "recommended_color": result.recommended_color,
                "attributes": result.attributes,
                "code_references": result.code_references,
                "confidence": result.confidence,
                "provider": result.provider,
            },
            message=f"Knowledge query answered via {result.provider}",
        )

    except Exception as e:
        logger.exception("query_knowledge_base_failed", error=str(e))
        return error_result(
            ErrorCode.INTERNAL_ERROR,
            f"Failed to query knowledge base: {e}"
        )


@mcp.tool()
async def get_code_reference(
    element_type: str,
    context: Optional[str] = None,
) -> dict[str, Any]:
    """
    Get building code references for a specific element type.

    Returns relevant California and national code sections for the element,
    including CFC, NFPA 72, CMC, CPC, CEC, and Title 24 references.

    Args:
        element_type: Type of element (e.g., "smoke_detector", "sprinkler", "outlet")
        context: Optional context (e.g., "corridor", "high-rise", "assembly")

    Returns:
        Code references with section numbers and requirements
    """
    try:
        from aec_agent.mcp.tools.knowledge_query import KnowledgeLLM

        llm = KnowledgeLLM()
        result = await llm.get_code_reference(element_type=element_type, context=context)

        return success_result(
            data={
                "answer": result.answer,
                "element_type": result.element_type,
                "code_references": result.code_references,
                "confidence": result.confidence,
                "provider": result.provider,
            },
            message=f"Code references for {element_type}",
        )

    except Exception as e:
        logger.exception("get_code_reference_failed", error=str(e))
        return error_result(
            ErrorCode.INTERNAL_ERROR,
            f"Failed to get code references: {e}"
        )


@mcp.tool()
async def classify_element(
    description: str,
) -> dict[str, Any]:
    """
    Classify an MEP element from a text description.

    Uses LLM to identify element type, subtype, system, and recommended
    CAD standards from a natural language description.

    Args:
        description: Text description of the element
            (e.g., "3/4 inch gate valve on cold water",
                   "24x24 supply air diffuser 200 CFM",
                   "ceiling mounted smoke detector")

    Returns:
        Element classification with type, subtype, system, layer, block
    """
    try:
        from aec_agent.mcp.tools.knowledge_query import KnowledgeLLM

        llm = KnowledgeLLM()
        result = await llm.classify_element(description=description)

        return success_result(
            data={
                "answer": result.answer,
                "element_type": result.element_type,
                "subtype": result.subtype,
                "system": result.system,
                "discipline": result.discipline,
                "recommended_layer": result.recommended_layer,
                "recommended_block": result.recommended_block,
                "recommended_color": result.recommended_color,
                "attributes": result.attributes,
                "confidence": result.confidence,
                "provider": result.provider,
            },
            message=f"Classified as {result.element_type}/{result.subtype}",
        )

    except Exception as e:
        logger.exception("classify_element_failed", error=str(e))
        return error_result(
            ErrorCode.INTERNAL_ERROR,
            f"Failed to classify element: {e}"
        )


@mcp.tool()
async def get_mep_rules(
    subdomain: Optional[str] = None,
    rule_type: Optional[str] = None,
) -> dict[str, Any]:
    """
    Get MEP design rules from the knowledge base.

    Returns domain rules for clearance, sizing, routing, and validation
    based on best practices and code requirements.

    Args:
        subdomain: MEP subdomain filter (e.g., "hvac", "plumbing", "electrical", "fire")
        rule_type: Rule type filter ("clearance", "sizing", "routing", "validation")

    Returns:
        List of applicable rules with conditions and actions
    """
    try:
        from aec_agent.domain.knowledge import get_knowledge_base
        from aec_agent.domain.models import RuleType

        kb = await get_knowledge_base()

        if rule_type:
            try:
                rt = RuleType(rule_type.lower())
                rules = kb.get_rules_by_type(rt)
            except ValueError:
                return error_result(
                    ErrorCode.INVALID_PARAMETER,
                    f"Invalid rule_type: {rule_type}. Use clearance/sizing/routing/validation."
                )
        elif subdomain:
            rules = kb.get_rules_by_subdomain(subdomain.lower())
        else:
            rules = kb.get_all_rules()

        rules_data = [
            {
                "id": str(r.id),
                "rule_name": r.rule_name,
                "rule_type": r.rule_type.value,
                "domain": r.domain,
                "subdomain": r.subdomain,
                "condition": r.condition,
                "action": r.action,
                "priority": r.priority,
                "description": r.description,
            }
            for r in rules[:20]  # Limit to 20 rules
        ]

        return success_result(
            data={
                "rules": rules_data,
                "total_count": len(rules),
                "returned_count": len(rules_data),
            },
            message=f"Found {len(rules)} MEP rules",
        )

    except Exception as e:
        logger.exception("get_mep_rules_failed", error=str(e))
        return error_result(
            ErrorCode.INTERNAL_ERROR,
            f"Failed to get MEP rules: {e}"
        )


@mcp.tool()
async def get_system_priorities() -> dict[str, Any]:
    """
    Get MEP system coordination priorities.

    Returns the priority ranking for different MEP systems used in
    clash detection and routing coordination. Lower rank = higher priority.

    Priority hierarchy:
    1. Gravity drains (cannot be rerouted)
    2. Sanitary/storm (slope requirements)
    3. Fire protection (code-driven)
    4. Plumbing domestic (pressurized)
    5. HVAC exhaust/supply/return
    6. Electrical/cable tray (most flexible)

    Returns:
        System priorities with rank and notes
    """
    try:
        from aec_agent.domain.knowledge import get_knowledge_base

        kb = await get_knowledge_base()
        priorities = kb.get_default_priorities()

        priorities_data = [
            {
                "system_type": p.system_type,
                "priority_rank": p.priority_rank,
                "notes": p.notes,
            }
            for p in priorities
        ]

        return success_result(
            data={"priorities": priorities_data},
            message=f"Found {len(priorities)} system priorities",
        )

    except Exception as e:
        logger.exception("get_system_priorities_failed", error=str(e))
        return error_result(
            ErrorCode.INTERNAL_ERROR,
            f"Failed to get system priorities: {e}"
        )
