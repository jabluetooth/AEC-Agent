"""
MCP Tools for AEC Agent.

This package contains all MCP tools organized by target application:
- common: Shared tools (health, status, cache)
- autocad: AutoCAD-specific tools
- revit: Revit-specific tools
- metadata: Semantic search and spatial query tools
- mep_tools: MEP-specific workflow tools (clearance, clash, trace)
- raster_design: PDF-to-DWG vectorization via AutoCAD Raster Design
- gemini_first: Gemini-First PDF to AutoCAD pipeline (quality-preserving)
"""

from . import autocad, common, gemini_first, mep_tools, metadata, raster_design, revit

__all__ = [
    "common",
    "autocad",
    "revit",
    "metadata",
    "mep_tools",
    "raster_design",
    "gemini_first",
]
