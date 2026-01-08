"""
Application settings using Pydantic Settings.

Loads configuration from environment variables and .env files.
"""

import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogFormat(str, Enum):
    """Log output format."""
    JSON = "json"
    TEXT = "text"


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Environment variables can be set directly or via a .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ==========================================================================
    # Environment
    # ==========================================================================
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment"
    )

    # ==========================================================================
    # LLM Configuration
    # ==========================================================================
    llm_provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="LLM provider to use"
    )

    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )

    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key"
    )

    azure_openai_api_key: Optional[str] = Field(
        default=None,
        description="Azure OpenAI API key"
    )

    azure_openai_endpoint: Optional[str] = Field(
        default=None,
        description="Azure OpenAI endpoint URL"
    )

    azure_openai_deployment: Optional[str] = Field(
        default=None,
        description="Azure OpenAI deployment name"
    )

    # ==========================================================================
    # Port Configuration
    # ==========================================================================
    mcp_listener_port: Optional[int] = Field(
        default=None,
        ge=20000,
        le=30000,
        description="Port for sidecar communication (set by GPO script)"
    )

    session_token: Optional[str] = Field(
        default=None,
        description="Session authentication token (set by GPO script)"
    )

    chainlit_port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="Chainlit UI port"
    )

    mcp_server_port: int = Field(
        default=54321,
        ge=1024,
        le=65535,
        description="MCP server SSE port"
    )

    # ==========================================================================
    # AWS Configuration
    # ==========================================================================
    aws_region: str = Field(
        default="us-east-1",
        description="AWS region"
    )

    aws_access_key_id: Optional[str] = Field(
        default=None,
        description="AWS access key ID"
    )

    aws_secret_access_key: Optional[str] = Field(
        default=None,
        description="AWS secret access key"
    )

    cloudwatch_log_group: str = Field(
        default="/aec-agent/application",
        description="CloudWatch log group name"
    )

    # ==========================================================================
    # Logging Configuration
    # ==========================================================================
    log_level: str = Field(
        default="INFO",
        description="Log level"
    )

    log_format: LogFormat = Field(
        default=LogFormat.JSON,
        description="Log output format"
    )

    enable_request_logging: bool = Field(
        default=True,
        description="Enable detailed request logging"
    )

    # ==========================================================================
    # Cache Configuration
    # ==========================================================================
    cache_dir: Path = Field(
        default_factory=lambda: Path(os.environ.get(
            "LOCALAPPDATA",
            os.path.expanduser("~/.local/share")
        )) / "AECAgent" / "cache",
        description="SQLite cache directory"
    )

    cache_sync_interval: int = Field(
        default=300,
        ge=60,
        description="Cache sync interval in seconds"
    )

    # ==========================================================================
    # Sidecar Configuration
    # ==========================================================================
    sidecar_connect_timeout: float = Field(
        default=5.0,
        ge=1.0,
        le=30.0,
        description="Sidecar connection timeout in seconds"
    )

    sidecar_read_timeout: float = Field(
        default=120.0,
        ge=10.0,
        le=600.0,
        description="Sidecar read timeout in seconds"
    )

    sidecar_max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for sidecar calls"
    )

    sidecar_retry_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Initial retry delay in seconds"
    )

    # ==========================================================================
    # Security Configuration
    # ==========================================================================
    session_token_expiry: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Session token expiry in hours"
    )

    # ==========================================================================
    # Performance Configuration
    # ==========================================================================
    max_concurrent_tools: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Maximum concurrent tool executions"
    )

    memory_warning_threshold: int = Field(
        default=800,
        ge=256,
        le=4096,
        description="Memory warning threshold in MB"
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v = v.upper()
        if v not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v

    @field_validator("cache_dir", mode="before")
    @classmethod
    def expand_cache_dir(cls, v):
        """Expand environment variables in cache dir."""
        if isinstance(v, str):
            v = os.path.expandvars(v)
            return Path(v)
        return v

    def get_llm_api_key(self) -> str:
        """Get the API key for the configured LLM provider."""
        if self.llm_provider == LLMProvider.OPENAI:
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required")
            return self.openai_api_key
        elif self.llm_provider == LLMProvider.ANTHROPIC:
            if not self.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY is required")
            return self.anthropic_api_key
        elif self.llm_provider == LLMProvider.AZURE_OPENAI:
            if not self.azure_openai_api_key:
                raise ValueError("AZURE_OPENAI_API_KEY is required")
            return self.azure_openai_api_key
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")

    def ensure_cache_dir(self) -> Path:
        """Ensure cache directory exists and return path."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT


# Global settings instance (lazy loaded)
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
