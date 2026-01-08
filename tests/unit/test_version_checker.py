"""Unit tests for version checker module."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from aec_agent.utils.version_checker import (
    CheckResult,
    CheckStatus,
    VersionChecker,
)


class TestCheckResult:
    """Tests for CheckResult dataclass."""

    def test_check_result_creation(self):
        """Test creating a CheckResult."""
        result = CheckResult(
            component="Python",
            status=CheckStatus.PASS,
            current_version="3.11.0",
            required_version="3.9+",
            message="OK"
        )

        assert result.component == "Python"
        assert result.status == CheckStatus.PASS
        assert result.current_version == "3.11.0"


class TestCheckStatus:
    """Tests for CheckStatus enum."""

    def test_check_status_values(self):
        """Test CheckStatus enum values."""
        assert CheckStatus.PASS.value == "pass"
        assert CheckStatus.WARN.value == "warn"
        assert CheckStatus.FAIL.value == "fail"
        assert CheckStatus.SKIP.value == "skip"


class TestVersionChecker:
    """Tests for VersionChecker class."""

    @pytest.fixture
    def checker(self):
        """Create a VersionChecker instance."""
        return VersionChecker()

    def test_check_python_version_current(self, checker):
        """Test Python version check with current Python."""
        result = checker.check_python_version()

        # Should at least not fail with current Python
        assert result.component == "Python"
        assert result.current_version is not None
        # Current test is running on valid Python
        assert result.status in [CheckStatus.PASS, CheckStatus.WARN]

    def test_compare_versions_equal(self, checker):
        """Test version comparison - equal versions."""
        assert checker._compare_versions("1.0.0", "1.0.0") == 0
        assert checker._compare_versions("2.5.3", "2.5.3") == 0

    def test_compare_versions_less_than(self, checker):
        """Test version comparison - less than."""
        assert checker._compare_versions("1.0.0", "2.0.0") == -1
        assert checker._compare_versions("1.5.0", "1.6.0") == -1
        assert checker._compare_versions("1.5.3", "1.5.4") == -1

    def test_compare_versions_greater_than(self, checker):
        """Test version comparison - greater than."""
        assert checker._compare_versions("2.0.0", "1.0.0") == 1
        assert checker._compare_versions("1.6.0", "1.5.0") == 1
        assert checker._compare_versions("1.5.4", "1.5.3") == 1

    def test_compare_versions_different_lengths(self, checker):
        """Test version comparison with different lengths."""
        assert checker._compare_versions("1.0", "1.0.0") == 0
        assert checker._compare_versions("1.0.0", "1.0") == 0
        assert checker._compare_versions("1.0.1", "1.0") == 1
        assert checker._compare_versions("1.0", "1.0.1") == -1

    def test_check_package_installed(self, checker):
        """Test checking an installed package."""
        # pytest should be installed since we're running tests
        result = checker.check_package_version("pytest", "1.0.0")

        assert result.component == "pytest"
        assert result.status == CheckStatus.PASS
        assert result.current_version is not None

    def test_check_package_not_installed(self, checker):
        """Test checking a package that isn't installed."""
        result = checker.check_package_version("nonexistent-package-xyz", "1.0.0")

        assert result.status == CheckStatus.FAIL
        assert result.current_version is None
        assert "Not installed" in result.message

    def test_check_environment_variables_not_set(self, checker):
        """Test environment variable check when not set."""
        with patch.dict(os.environ, {}, clear=True):
            results = checker.check_environment_variables()

            port_result = next(r for r in results if r.component == "MCP_LISTENER_PORT")
            assert port_result.status == CheckStatus.WARN
            assert "GPO" in port_result.message

    def test_check_environment_variables_set(self, checker):
        """Test environment variable check when set correctly."""
        with patch.dict(os.environ, {
            "MCP_LISTENER_PORT": "25000",
            "SESSION_TOKEN": "12345678-1234-1234-1234-123456789012"
        }):
            results = checker.check_environment_variables()

            port_result = next(r for r in results if r.component == "MCP_LISTENER_PORT")
            assert port_result.status == CheckStatus.PASS

            token_result = next(r for r in results if r.component == "SESSION_TOKEN")
            assert token_result.status == CheckStatus.PASS

    def test_check_environment_variables_port_out_of_range(self, checker):
        """Test environment variable check with port out of range."""
        with patch.dict(os.environ, {"MCP_LISTENER_PORT": "50000"}):
            results = checker.check_environment_variables()

            port_result = next(r for r in results if r.component == "MCP_LISTENER_PORT")
            assert port_result.status == CheckStatus.WARN
            assert "outside" in port_result.message.lower()

    def test_check_environment_variables_invalid_port(self, checker):
        """Test environment variable check with invalid port."""
        with patch.dict(os.environ, {"MCP_LISTENER_PORT": "not-a-number"}):
            results = checker.check_environment_variables()

            port_result = next(r for r in results if r.component == "MCP_LISTENER_PORT")
            assert port_result.status == CheckStatus.FAIL

    def test_check_os_windows(self, checker):
        """Test OS check on Windows."""
        with patch('platform.system', return_value='Windows'):
            with patch('platform.version', return_value='10.0.19041'):
                with patch('platform.platform', return_value='Windows-10-10.0.19041'):
                    result = checker.check_os()

                    assert result.component == "Operating System"
                    assert "Windows" in result.current_version

    def test_check_os_non_windows(self, checker):
        """Test OS check on non-Windows."""
        with patch('platform.system', return_value='Linux'):
            with patch('platform.release', return_value='5.4.0'):
                result = checker.check_os()

                assert result.status == CheckStatus.WARN
                assert "Windows required" in result.message

    @patch('platform.system', return_value='Windows')
    def test_check_dotnet_windows(self, mock_system, checker):
        """Test .NET check on Windows with mocked registry."""
        # This test would need to mock winreg which is complex
        # In practice, you'd test this on actual Windows systems
        pass  # Skip for now

    @patch('platform.system', return_value='Linux')
    def test_check_dotnet_non_windows(self, mock_system, checker):
        """Test .NET check on non-Windows."""
        result = checker.check_dotnet()

        assert result.status == CheckStatus.SKIP
        assert "Windows only" in result.message

    def test_run_all_checks(self, checker):
        """Test running all checks."""
        results = checker.run_all_checks()

        assert len(results) > 0
        assert all(isinstance(r, CheckResult) for r in results)

        # Should have at least Python and OS checks
        components = [r.component for r in results]
        assert "Python" in components
        assert "Operating System" in components


class TestVersionCheckerMain:
    """Tests for version checker CLI."""

    def test_main_help(self):
        """Test main function with help."""
        from aec_agent.utils.version_checker import main

        with pytest.raises(SystemExit) as excinfo:
            main(["--help"])

        # Help exits with code 0
        assert excinfo.value.code == 0
