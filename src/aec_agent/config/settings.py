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
    HUGGINGFACE = "huggingface"
    GROQ = "groq"


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

    huggingface_api_key: Optional[str] = Field(
        default=None,
        description="Hugging Face API key"
    )

    huggingface_model: str = Field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        description="Hugging Face model to use"
    )

    groq_api_key: Optional[str] = Field(
        default=None,
        description="Groq API key"
    )

    groq_model: str = Field(
        default="moonshotai/kimi-k2-instruct",
        description="Groq model to use"
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

    tool_lock_timeout: float = Field(
        default=120.0,
        ge=10.0,
        le=600.0,
        description="Maximum time to wait for tool lock in seconds"
    )

    memory_warning_threshold: int = Field(
        default=800,
        ge=256,
        le=4096,
        description="Memory warning threshold in MB"
    )

    max_history_messages: int = Field(
        default=20,
        ge=4,
        le=100,
        description="Maximum conversation messages to keep (reduces token usage)"
    )

    compress_tool_schemas: bool = Field(
        default=True,
        description="Compress tool schemas to reduce token usage (auto-enabled for HF/Groq)"
    )

    smart_tool_routing: bool = Field(
        default=True,
        description="Enable smart tool filtering based on user context (AutoCAD vs Revit)"
    )

    # ==========================================================================
    # Tool Optimization Configuration
    # ==========================================================================
    tool_tier: str = Field(
        default="standard",
        description="Tool loading tier: essential (minimal), standard (default), advanced (all)"
    )

    tool_compression_mode: str = Field(
        default="standard",
        description="Tool description compression: full (none), standard, minimal, ultra"
    )

    enable_metadata_tools: bool = Field(
        default=True,
        description="Enable metadata/semantic search tools (requires database)"
    )

    max_tool_result_chars: int = Field(
        default=2000,
        ge=500,
        le=10000,
        description="Maximum characters in tool result (truncation threshold)"
    )

    enable_result_summarization: bool = Field(
        default=False,
        description="Use LLM to summarize long results instead of truncating"
    )

    # ==========================================================================
    # Dynamic Token Optimization (Phase 1)
    # ==========================================================================
    max_context_tokens: int = Field(
        default=8000,
        ge=2000,
        le=128000,
        description="Maximum context tokens before aggressive compression"
    )

    enable_dynamic_compression: bool = Field(
        default=True,
        description="Auto-escalate compression based on context size"
    )

    enable_token_logging: bool = Field(
        default=True,
        description="Log token usage per request for monitoring"
    )

    result_field_preset: str = Field(
        default="standard",
        description="Result field filtering: minimal, standard, full"
    )

    # ==========================================================================
    # Intent Classification Configuration (MEP workflow enhancement)
    # ==========================================================================
    enable_intent_classification: bool = Field(
        default=True,
        description="Enable MEP-aware intent classification for smart tool filtering"
    )

    use_intent_embeddings: bool = Field(
        default=False,
        description="Use embedding-based similarity for intent classification (requires more memory)"
    )

    intent_min_confidence: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for MEP intent classification"
    )

    mep_domain_priority: str = Field(
        default="hvac",
        description="Primary MEP domain focus: hvac, electrical, plumbing, fire_protection, all"
    )

    # ==========================================================================
    # Database Configuration (PostgreSQL with PostGIS + pgvector)
    # ==========================================================================
    database_url: Optional[str] = Field(
        default=None,
        description="PostgreSQL connection string (postgresql://user:pass@host:port/db)"
    )

    database_pool_size: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Database connection pool size"
    )

    database_pool_max_overflow: int = Field(
        default=10,
        ge=0,
        le=50,
        description="Maximum overflow connections beyond pool size"
    )

    # ==========================================================================
    # Embedding Configuration
    # ==========================================================================
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for semantic search"
    )

    embedding_dimension: int = Field(
        default=384,
        ge=64,
        le=4096,
        description="Embedding vector dimension (must match model)"
    )

    # ==========================================================================
    # Extraction & Sync Configuration
    # ==========================================================================
    sync_on_save: bool = Field(
        default=True,
        description="Automatically sync metadata when document is saved"
    )

    sync_debounce_ms: int = Field(
        default=2000,
        ge=100,
        le=10000,
        description="Debounce time for incremental sync in milliseconds"
    )

    relationship_distance_threshold: float = Field(
        default=1.0,
        ge=0.1,
        le=100.0,
        description="Distance threshold in meters for 'near' relationships"
    )

    extraction_batch_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Batch size for entity extraction (prevents memory issues)"
    )

    @property
    def has_database(self) -> bool:
        """Check if PostgreSQL database is configured."""
        return self.database_url is not None

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
        elif self.llm_provider == LLMProvider.HUGGINGFACE:
            if not self.huggingface_api_key:
                raise ValueError("HUGGINGFACE_API_KEY is required")
            return self.huggingface_api_key
        elif self.llm_provider == LLMProvider.GROQ:
            if not self.groq_api_key:
                raise ValueError("GROQ_API_KEY is required")
            return self.groq_api_key
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
