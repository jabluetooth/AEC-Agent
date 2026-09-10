"""
MCP Tools for AEC Agent.

This package contains all MCP tools organized by target application:
- common: Shared tools (health, status, cache)
- autocad: AutoCAD-specific tools
- revit: Revit-specific tools
- metadata: Semantic search and spatial query tools
- mep_tools: MEP-specific workflow tools (clearance, clash, trace)
- raster_design_import, raster_design_cleanup, raster_design_vectorize,
  raster_design_pipeline, raster_design_ocr, raster_design_entities,
  raster_design_status: PDF-to-DWG vectorization via AutoCAD Raster Design
  (split from the former monolithic raster_design module)
- gemini_first: Gemini-First PDF to AutoCAD pipeline (quality-preserving)
- workflow_tools: MEP workflow template listing/execution
- validation_tools: MEP domain rule validation and design suggestions
- memory_tools: Project fact storage and recall
"""

from . import (
    autocad,
    common,
    gemini_first,
    memory_tools,
    mep_tools,
    metadata,
    raster_design_cleanup,
    raster_design_entities,
    raster_design_import,
    raster_design_ocr,
    raster_design_pipeline,
    raster_design_status,
    raster_design_vectorize,
    revit,
    validation_tools,
    workflow_tools,
)

__all__ = [
    "common",
    "autocad",
    "revit",
    "metadata",
    "mep_tools",
    "raster_design_import",
    "raster_design_cleanup",
    "raster_design_vectorize",
    "raster_design_pipeline",
    "raster_design_ocr",
    "raster_design_entities",
    "raster_design_status",
    "gemini_first",
    "workflow_tools",
    "validation_tools",
    "memory_tools",
]
