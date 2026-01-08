"""Unit tests for version compatibility module."""

import pytest

from aec_agent.config.versions import (
    AutoCADVersion,
    RevitVersion,
    SupportStatus,
    VersionMatrix,
    VERSION_MATRIX,
)


class TestVersionMatrix:
    """Tests for VersionMatrix class."""

    def test_version_matrix_has_autocad_versions(self):
        """Verify AutoCAD versions are defined."""
        matrix = VersionMatrix()
        assert len(matrix.autocad_versions) > 0

    def test_version_matrix_has_revit_versions(self):
        """Verify Revit versions are defined."""
        matrix = VersionMatrix()
        assert len(matrix.revit_versions) > 0

    def test_version_matrix_has_python_requirements(self):
        """Verify Python requirements are defined."""
        matrix = VersionMatrix()
        assert len(matrix.python_requirements) > 0

    def test_get_autocad_version_found(self):
        """Test getting an existing AutoCAD version."""
        matrix = VersionMatrix()
        version = matrix.get_autocad_version("2024")

        assert version is not None
        assert version.version == "2024"
        assert version.dotnet_framework == "4.8"

    def test_get_autocad_version_not_found(self):
        """Test getting a non-existent AutoCAD version."""
        matrix = VersionMatrix()
        version = matrix.get_autocad_version("2000")

        assert version is None

    def test_get_revit_version_found(self):
        """Test getting an existing Revit version."""
        matrix = VersionMatrix()
        version = matrix.get_revit_version("2024")

        assert version is not None
        assert version.version == "2024"

    def test_get_revit_version_not_found(self):
        """Test getting a non-existent Revit version."""
        matrix = VersionMatrix()
        version = matrix.get_revit_version("2000")

        assert version is None

    def test_get_recommended_autocad(self):
        """Test getting recommended AutoCAD version."""
        matrix = VersionMatrix()
        recommended = matrix.get_recommended_autocad()

        assert recommended is not None
        assert recommended.status == SupportStatus.RECOMMENDED

    def test_get_recommended_revit(self):
        """Test getting recommended Revit version."""
        matrix = VersionMatrix()
        recommended = matrix.get_recommended_revit()

        assert recommended is not None
        assert recommended.status == SupportStatus.RECOMMENDED

    def test_is_version_supported_autocad(self):
        """Test version support check for AutoCAD."""
        matrix = VersionMatrix()

        assert matrix.is_version_supported("autocad", "2024") is True
        assert matrix.is_version_supported("autocad", "2023") is True
        assert matrix.is_version_supported("autocad", "2000") is False

    def test_is_version_supported_revit(self):
        """Test version support check for Revit."""
        matrix = VersionMatrix()

        assert matrix.is_version_supported("revit", "2024") is True
        assert matrix.is_version_supported("revit", "2023") is True
        assert matrix.is_version_supported("revit", "2000") is False

    def test_port_ranges_defined(self):
        """Verify port ranges are defined."""
        matrix = VersionMatrix()
        assert len(matrix.port_ranges) > 0

    def test_get_port_range(self):
        """Test getting port range for a component."""
        matrix = VersionMatrix()
        chainlit_range = matrix.get_port_range("chainlit_ui")

        assert chainlit_range is not None
        assert chainlit_range.default_port == 8000
        assert chainlit_range.range_start == 8000
        assert chainlit_range.range_end == 8100

    def test_llm_providers_defined(self):
        """Verify LLM providers are defined."""
        matrix = VersionMatrix()

        assert "openai" in matrix.llm_providers
        assert "anthropic" in matrix.llm_providers
        assert matrix.llm_providers["openai"]["tool_calling"] is True

    def test_aws_instances_defined(self):
        """Verify AWS instance specifications are defined."""
        matrix = VersionMatrix()

        assert "minimum" in matrix.aws_instances
        assert "recommended" in matrix.aws_instances
        assert matrix.aws_instances["recommended"]["type"] == "g4dn.4xlarge"


class TestAutoCADVersion:
    """Tests for AutoCADVersion dataclass."""

    def test_autocad_version_creation(self):
        """Test creating an AutoCAD version."""
        version = AutoCADVersion(
            version="2024",
            dotnet_framework="4.8",
            objectarx_sdk="2024",
            status=SupportStatus.RECOMMENDED,
            notes="Latest stable"
        )

        assert version.version == "2024"
        assert version.status == SupportStatus.RECOMMENDED


class TestRevitVersion:
    """Tests for RevitVersion dataclass."""

    def test_revit_version_creation(self):
        """Test creating a Revit version."""
        version = RevitVersion(
            version="2024",
            dotnet_framework="4.8",
            pyrevit_version="4.8.x",
            status=SupportStatus.RECOMMENDED
        )

        assert version.version == "2024"
        assert version.pyrevit_version == "4.8.x"

    def test_revit_2025_requires_dotnet8(self):
        """Verify Revit 2025 requires .NET 8."""
        matrix = VersionMatrix()
        revit_2025 = matrix.get_revit_version("2025")

        assert revit_2025 is not None
        assert revit_2025.dotnet_framework == "8.0"
        assert "5.0" in revit_2025.pyrevit_version


class TestGlobalVersionMatrix:
    """Tests for the global VERSION_MATRIX instance."""

    def test_global_matrix_exists(self):
        """Verify global VERSION_MATRIX is available."""
        assert VERSION_MATRIX is not None
        assert isinstance(VERSION_MATRIX, VersionMatrix)

    def test_global_matrix_has_data(self):
        """Verify global matrix has all required data."""
        assert len(VERSION_MATRIX.autocad_versions) >= 5
        assert len(VERSION_MATRIX.revit_versions) >= 5
        assert len(VERSION_MATRIX.python_requirements) >= 5
