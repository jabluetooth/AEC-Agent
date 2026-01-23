"""
MCP Tools for AEC Agent.

This package contains all MCP tools organized by target application:
- common: Shared tools (health, status, cache)
- autocad: AutoCAD-specific tools
- revit: Revit-specific tools
- metadata: Semantic search and spatial query tools
- mep_tools: MEP-specific workflow tools (clearance, clash, trace)
"""

from . import common
from . import autocad
from . import revit
from . import metadata
from . import mep_tools

__all__ = ["common", "autocad", "revit", "metadata", "mep_tools"]
