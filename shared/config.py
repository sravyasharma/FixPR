"""
shared/config.py

Centralized configuration management for the AI Code Review Platform.

All environment variables are validated at startup via Pydantic Settings.
No service should call os.getenv() directly — import from here instead.

Architecture role:
- Imported by every layer: API, Worker, Agents, DB, Queue, Tracing
- Loaded once per process via lru_cache singleton
- Fails fast on missing required values (production safety)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Platform-wide settings loaded from environment / .env file.

    Grouped into logical sections:
      - Application
      - GitHub
      - OpenAI / Anthropic
      - Database
      - Redis Queue
      - LangSmith Tracing
      - Static Analysis
      - Security Analysis
      - Worker
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Application
    # ------------------------------------------------------------------ #
    app_env: Literal["development", "staging", "production"] = Field(
        default="development", description="Deployment environment"
    )
    app_name: str = Field(default="github-ai-review-agent")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )
    secret_key: str = Field(description="Used to verify GitHub webhook signatures")

    # ------------------------------------------------------------------ #
    # GitHub
    # ------------------------------------------------------------------ #
    github_token: str = Field(description="Personal access token or GitHub App token")
    github_webhook_secret: str = Field(
        description="HMAC secret for validating webhook payloads"
    )
    github_api_url: str = Field(default="https://api.github.com")
    github_clone_base_dir: str = Field(
        default="/tmp/repos", description="Local directory for cloned repositories"
    )

    # ------------------------------------------------------------------ #
    # LLM Providers
    # ------------------------------------------------------------------ #
    openai_api_key: str = Field(description="OpenAI API key for GPT-4o")
    openai_model: str = Field(default="gpt-4o")
    openai_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    openai_max_tokens: int = Field(default=4096)

    anthropic_api_key: str | None = Field(
        default=None, description="Optional: Anthropic API key for Claude fallback"
    )
    anthropic_model: str = Field(default="claude-sonnet-4-6")

    # Active LLM provider — allows runtime switching
    llm_provider: Literal["openai", "anthropic"] = Field(default="openai")

    # ------------------------------------------------------------------ #
    # Database
    # ------------------------------------------------------------------ #
    database_url: str = Field(
        description="Database connection string; Postgres for production and sqlite+aiosqlite for local/tests"
    )
    db_pool_size: int = Field(default=10)
    db_max_overflow: int = Field(default=20)
    db_pool_timeout: int = Field(default=30)
    db_echo_sql: bool = Field(default=False)

    # ------------------------------------------------------------------ #
    # Redis
    # ------------------------------------------------------------------ #
    redis_url: RedisDsn = Field(
        description="Redis connection string e.g. redis://localhost:6379/0"
    )
    redis_queue_name: str = Field(default="pr_review_jobs")
    redis_result_ttl: int = Field(
        default=86400, description="TTL in seconds for cached results"
    )
    worker_concurrency: int = Field(
        default=4, description="Number of parallel worker coroutines"
    )

    # ------------------------------------------------------------------ #
    # LangSmith Tracing
    # ------------------------------------------------------------------ #
    langchain_tracing_v2: bool = Field(default=False)
    langchain_api_key: str | None = Field(default=None, description="LangSmith API key")
    langchain_project: str = Field(default="github-ai-review-agent")
    langchain_endpoint: str = Field(default="https://api.smith.langchain.com")

    # ------------------------------------------------------------------ #
    # Static Analysis
    # ------------------------------------------------------------------ #
    pylint_enabled: bool = Field(default=True)
    flake8_enabled: bool = Field(default=True)
    mypy_enabled: bool = Field(default=True)
    eslint_enabled: bool = Field(default=True)
    eslint_config_path: str | None = Field(
        default=None, description="Path to custom .eslintrc"
    )

    # ------------------------------------------------------------------ #
    # Security Analysis
    # ------------------------------------------------------------------ #
    bandit_enabled: bool = Field(default=True)
    semgrep_enabled: bool = Field(default=True)
    semgrep_rules: str = Field(
        default="p/owasp-top-ten p/security-audit p/secrets",
        description="Space-separated Semgrep rule packs",
    )
    sarif_output_dir: str = Field(default="/tmp/sarif_output")

    # ------------------------------------------------------------------ #
    # Auto-fix Agent
    # ------------------------------------------------------------------ #
    autofix_enabled: bool = Field(default=True)
    autofix_min_severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(
        default="HIGH",
        description="Minimum severity to trigger auto-fix",
    )
    autofix_run_tests: bool = Field(default=True)
    autofix_run_lint: bool = Field(default=True)

    # ------------------------------------------------------------------ #
    # Review posting
    # ------------------------------------------------------------------ #
    post_review_comments: bool = Field(
        default=True, description="Whether to post findings as PR comments"
    )
    max_comments_per_review: int = Field(
        default=25, description="Cap to avoid spamming PRs"
    )

    @field_validator("database_url", mode="before")
    @classmethod
    def ensure_async_driver(cls, v: str) -> str:
        """Normalize DB URLs for local tests and production Postgres."""
        v = str(v)
        if v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        if v.startswith("postgres://"):
            return v.replace("postgres://", "postgresql+asyncpg://", 1)
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    Use lru_cache so Settings is only instantiated and validated once
    per process, regardless of how many modules import it.

    Usage:
        from shared.config import get_settings
        settings = get_settings()
    """
    return Settings()