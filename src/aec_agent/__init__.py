"""
AEC Agent - AI-powered AutoCAD and Revit automation.

This package provides the MCP middleware and utilities for bridging
AI reasoning with legacy CAD applications.
"""

__version__ = "0.1.0"
__author__ = "AEC Team"

from aec_agent.config.settings import Settings
from aec_agent.config.versions import VersionMatrix

__all__ = ["Settings", "VersionMatrix", "__version__"]
