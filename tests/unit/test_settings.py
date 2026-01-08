"""Unit tests for settings module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from aec_agent.config.settings import (
    Environment,
    LogFormat,
    LLMProvider,
    Settings,
    get_settings,
)


class TestSettings:
    """Tests for Settings class."""

    def test_default_settings(self):
        """Test default settings values."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()

            assert settings.environment == Environment.DEVELOPMENT
            assert settings.llm_provider == LLMProvider.OPENAI
            assert settings.chainlit_port == 8000
            assert settings.mcp_server_port == 54321
            assert settings.log_level == "INFO"

    def test_environment_from_env_var(self):
        """Test loading environment from env var."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            settings = Settings()
            assert settings.environment == Environment.PRODUCTION

    def test_llm_provider_from_env_var(self):
        """Test loading LLM provider from env var."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic"}):
            settings = Settings()
            assert settings.llm_provider == LLMProvider.ANTHROPIC

    def test_port_configuration(self):
        """Test port configuration from env vars."""
        with patch.dict(os.environ, {
            "CHAINLIT_PORT": "8080",
            "MCP_SERVER_PORT": "55000",
            "MCP_LISTENER_PORT": "25000"
        }):
            settings = Settings()

            assert settings.chainlit_port == 8080
            assert settings.mcp_server_port == 55000
            assert settings.mcp_listener_port == 25000

    def test_mcp_listener_port_validation(self):
        """Test MCP listener port range validation."""
        # Valid port
        with patch.dict(os.environ, {"MCP_LISTENER_PORT": "25000"}):
            settings = Settings()
            assert settings.mcp_listener_port == 25000

        # Port below range should fail
        with patch.dict(os.environ, {"MCP_LISTENER_PORT": "19999"}):
            with pytest.raises(ValueError):
                Settings()

        # Port above range should fail
        with patch.dict(os.environ, {"MCP_LISTENER_PORT": "30001"}):
            with pytest.raises(ValueError):
                Settings()

    def test_log_level_validation(self):
        """Test log level validation."""
        # Valid log levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            with patch.dict(os.environ, {"LOG_LEVEL": level}):
                settings = Settings()
                assert settings.log_level == level

        # Case insensitive
        with patch.dict(os.environ, {"LOG_LEVEL": "debug"}):
            settings = Settings()
            assert settings.log_level == "DEBUG"

        # Invalid log level
        with patch.dict(os.environ, {"LOG_LEVEL": "INVALID"}):
            with pytest.raises(ValueError):
                Settings()

    def test_is_production(self):
        """Test is_production property."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            settings = Settings()
            assert settings.is_production is True
            assert settings.is_development is False

    def test_is_development(self):
        """Test is_development property."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            settings = Settings()
            assert settings.is_development is True
            assert settings.is_production is False

    def test_get_llm_api_key_openai(self):
        """Test getting OpenAI API key."""
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-test-key"
        }):
            settings = Settings()
            assert settings.get_llm_api_key() == "sk-test-key"

    def test_get_llm_api_key_anthropic(self):
        """Test getting Anthropic API key."""
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "sk-ant-test-key"
        }):
            settings = Settings()
            assert settings.get_llm_api_key() == "sk-ant-test-key"

    def test_get_llm_api_key_missing(self):
        """Test error when API key is missing."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}, clear=True):
            settings = Settings()
            with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
                settings.get_llm_api_key()

    def test_ensure_cache_dir(self, tmp_path):
        """Test cache directory creation."""
        cache_dir = tmp_path / "test_cache"
        with patch.dict(os.environ, {"CACHE_DIR": str(cache_dir)}):
            settings = Settings()
            result = settings.ensure_cache_dir()

            assert result.exists()
            assert result == cache_dir

    def test_sidecar_timeout_settings(self):
        """Test sidecar timeout configuration."""
        with patch.dict(os.environ, {
            "SIDECAR_CONNECT_TIMEOUT": "10",
            "SIDECAR_READ_TIMEOUT": "180"
        }):
            settings = Settings()

            assert settings.sidecar_connect_timeout == 10.0
            assert settings.sidecar_read_timeout == 180.0

    def test_sidecar_retry_settings(self):
        """Test sidecar retry configuration."""
        with patch.dict(os.environ, {
            "SIDECAR_MAX_RETRIES": "5",
            "SIDECAR_RETRY_DELAY": "2.5"
        }):
            settings = Settings()

            assert settings.sidecar_max_retries == 5
            assert settings.sidecar_retry_delay == 2.5


class TestGetSettings:
    """Tests for get_settings function."""

    def test_get_settings_returns_settings(self):
        """Test that get_settings returns a Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_singleton(self):
        """Test that get_settings returns the same instance."""
        # Note: This test may not work as expected due to module caching
        # In a real scenario, you'd need to reset the module state
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2


class TestEnvironmentEnum:
    """Tests for Environment enum."""

    def test_environment_values(self):
        """Test Environment enum values."""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.STAGING.value == "staging"
        assert Environment.PRODUCTION.value == "production"


class TestLogFormatEnum:
    """Tests for LogFormat enum."""

    def test_log_format_values(self):
        """Test LogFormat enum values."""
        assert LogFormat.JSON.value == "json"
        assert LogFormat.TEXT.value == "text"


class TestLLMProviderEnum:
    """Tests for LLMProvider enum."""

    def test_llm_provider_values(self):
        """Test LLMProvider enum values."""
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.AZURE_OPENAI.value == "azure_openai"
