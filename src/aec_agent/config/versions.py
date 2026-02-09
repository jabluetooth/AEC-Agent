"""
Version compatibility matrix for AEC Agent.

This module defines supported versions for all software components
and provides validation utilities.
"""

from dataclasses import dataclass, field
from enum import Enum


class SupportStatus(Enum):
    """Support status for software versions."""
    RECOMMENDED = "recommended"
    SUPPORTED = "supported"
    TESTING = "testing"
    DEPRECATED = "deprecated"
    UNSUPPORTED = "unsupported"


@dataclass
class VersionInfo:
    """Version information with support status."""
    version: str
    status: SupportStatus
    notes: str = ""
    end_of_support: str | None = None


@dataclass
class AutoCADVersion:
    """AutoCAD version compatibility information."""
    version: str
    dotnet_framework: str
    objectarx_sdk: str
    status: SupportStatus
    notes: str = ""


@dataclass
class RevitVersion:
    """Revit version compatibility information."""
    version: str
    dotnet_framework: str
    pyrevit_version: str
    status: SupportStatus
    notes: str = ""


@dataclass
class PythonRequirement:
    """Python package requirement."""
    name: str
    min_version: str
    recommended_version: str
    max_version: str | None = None


@dataclass
class PortRange:
    """Network port range configuration."""
    component: str
    default_port: int | None
    range_start: int
    range_end: int
    protocol: str


@dataclass
class VersionMatrix:
    """
    Complete version compatibility matrix for AEC Agent.

    This class contains all version requirements and compatibility
    information for the system components.
    """

    # AutoCAD Versions
    autocad_versions: list[AutoCADVersion] = field(default_factory=lambda: [
        AutoCADVersion(
            version="2021",
            dotnet_framework="4.8",
            objectarx_sdk="2021",
            status=SupportStatus.SUPPORTED,
            notes="LTS version"
        ),
        AutoCADVersion(
            version="2022",
            dotnet_framework="4.8",
            objectarx_sdk="2022",
            status=SupportStatus.SUPPORTED
        ),
        AutoCADVersion(
            version="2023",
            dotnet_framework="4.8",
            objectarx_sdk="2023",
            status=SupportStatus.SUPPORTED
        ),
        AutoCADVersion(
            version="2024",
            dotnet_framework="4.8",
            objectarx_sdk="2024",
            status=SupportStatus.RECOMMENDED,
            notes="Latest stable"
        ),
        AutoCADVersion(
            version="2025",
            dotnet_framework="4.8",
            objectarx_sdk="2025",
            status=SupportStatus.SUPPORTED,
            notes="Current stable"
        ),
        AutoCADVersion(
            version="2026",
            dotnet_framework="8.0",
            objectarx_sdk="2026",
            status=SupportStatus.TESTING,
            notes=".NET 8 migration - latest version"
        ),
    ])

    # Revit Versions
    revit_versions: list[RevitVersion] = field(default_factory=lambda: [
        RevitVersion(
            version="2021",
            dotnet_framework="4.8",
            pyrevit_version="4.8.x",
            status=SupportStatus.DEPRECATED,
            notes="End of support 2024"
        ),
        RevitVersion(
            version="2022",
            dotnet_framework="4.8",
            pyrevit_version="4.8.x",
            status=SupportStatus.SUPPORTED
        ),
        RevitVersion(
            version="2023",
            dotnet_framework="4.8",
            pyrevit_version="4.8.x",
            status=SupportStatus.SUPPORTED
        ),
        RevitVersion(
            version="2024",
            dotnet_framework="4.8",
            pyrevit_version="4.8.x",
            status=SupportStatus.RECOMMENDED
        ),
        RevitVersion(
            version="2025",
            dotnet_framework="8.0",
            pyrevit_version="5.0+",
            status=SupportStatus.TESTING,
            notes=".NET 8 migration required"
        ),
    ])

    # pyRevit Versions
    pyrevit_versions: list[VersionInfo] = field(default_factory=lambda: [
        VersionInfo(
            version="4.8.12",
            status=SupportStatus.RECOMMENDED,
            notes="Revit 2019-2024 support"
        ),
        VersionInfo(
            version="4.8.13+",
            status=SupportStatus.SUPPORTED,
            notes="Revit 2019-2024 support"
        ),
        VersionInfo(
            version="5.0.x",
            status=SupportStatus.TESTING,
            notes="Revit 2025+ only"
        ),
    ])

    # Python Requirements
    python_requirements: list[PythonRequirement] = field(default_factory=lambda: [
        PythonRequirement("python", "3.9", "3.11", "3.12"),
        PythonRequirement("fastmcp", "0.1.0", "latest"),
        PythonRequirement("httpx", "0.24.0", "0.27.x"),
        PythonRequirement("chainlit", "1.0.0", "1.0.x"),
        PythonRequirement("tenacity", "8.2.0", "latest"),
        PythonRequirement("aiosqlite", "0.19.0", "latest"),
        PythonRequirement("pydantic", "2.0.0", "latest"),
    ])

    # Port Ranges
    port_ranges: list[PortRange] = field(default_factory=lambda: [
        PortRange("chainlit_ui", 8000, 8000, 8100, "HTTP"),
        PortRange("mcp_server", 54321, 54000, 55000, "HTTP/SSE"),
        PortRange("autocad_sidecar", None, 20000, 30000, "HTTP"),
        PortRange("revit_sidecar", 48884, 20000, 30000, "HTTP"),
    ])

    # LLM Providers
    llm_providers: dict[str, dict] = field(default_factory=lambda: {
        "openai": {
            "models": ["gpt-4o", "gpt-4-turbo"],
            "recommended": "gpt-4o",
            "context_window": 128000,
            "tool_calling": True,
        },
        "anthropic": {
            "models": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
            "recommended": "claude-3-5-sonnet-20241022",
            "context_window": 200000,
            "tool_calling": True,
        },
        "azure_openai": {
            "models": ["gpt-4o"],
            "recommended": "gpt-4o",
            "context_window": 128000,
            "tool_calling": True,
        },
    })

    # AWS Instance Types
    aws_instances: dict[str, dict] = field(default_factory=lambda: {
        "minimum": {
            "type": "g4dn.2xlarge",
            "vcpu": 8,
            "ram_gb": 32,
            "gpu": "NVIDIA T4",
            "users_per_instance": 2,
        },
        "recommended": {
            "type": "g4dn.4xlarge",
            "vcpu": 16,
            "ram_gb": 64,
            "gpu": "NVIDIA T4",
            "users_per_instance": 4,
        },
    })

    def get_autocad_version(self, version: str) -> AutoCADVersion | None:
        """Get AutoCAD version info by version string."""
        for v in self.autocad_versions:
            if v.version == version:
                return v
        return None

    def get_revit_version(self, version: str) -> RevitVersion | None:
        """Get Revit version info by version string."""
        for v in self.revit_versions:
            if v.version == version:
                return v
        return None

    def get_recommended_autocad(self) -> AutoCADVersion:
        """Get the recommended AutoCAD version."""
        for v in self.autocad_versions:
            if v.status == SupportStatus.RECOMMENDED:
                return v
        return self.autocad_versions[-1]

    def get_recommended_revit(self) -> RevitVersion:
        """Get the recommended Revit version."""
        for v in self.revit_versions:
            if v.status == SupportStatus.RECOMMENDED:
                return v
        return self.revit_versions[-1]

    def get_port_range(self, component: str) -> PortRange | None:
        """Get port range for a component."""
        for p in self.port_ranges:
            if p.component == component:
                return p
        return None

    def is_version_supported(self, component: str, version: str) -> bool:
        """Check if a version is supported for a component."""
        if component == "autocad":
            acad_ver = self.get_autocad_version(version)
            return acad_ver is not None and acad_ver.status in [
                SupportStatus.RECOMMENDED,
                SupportStatus.SUPPORTED,
                SupportStatus.TESTING
            ]
        elif component == "revit":
            revit_ver = self.get_revit_version(version)
            return revit_ver is not None and revit_ver.status in [
                SupportStatus.RECOMMENDED,
                SupportStatus.SUPPORTED,
                SupportStatus.TESTING
            ]
        return False


# Global instance
VERSION_MATRIX = VersionMatrix()
