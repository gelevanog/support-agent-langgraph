"""Application settings loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from support_agent.rules.policy import PolicyConfig

LLMProvider = Literal["fake", "openai", "anthropic", "openrouter"]
# Anthropic has no embeddings API, hence no "anthropic" option here.
EmbeddingsProvider = Literal["fake", "openai", "openrouter"]


class Settings(BaseSettings):
    """All runtime configuration. Every field maps to an upper-case env variable."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM -----------------------------------------------------------------------------
    llm_provider: LLMProvider = "fake"
    openai_model: str = "gpt-5-mini"
    openai_api_key: SecretStr | None = None
    anthropic_model: str = "claude-sonnet-5"
    anthropic_api_key: SecretStr | None = None
    # OpenRouter: one key for hundreds of models (OpenAI, Anthropic, Google, open weights) through
    # an OpenAI-compatible API. Model ids are "vendor/model".
    openrouter_model: str = "openai/gpt-5.4-mini"
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_max_tokens: int = Field(default=8192, ge=256)
    llm_timeout_seconds: float = Field(default=90.0, gt=0)
    max_research_steps: int = Field(default=4, ge=1, le=10)

    # --- Knowledge-base retrieval ----------------------------------------------------------
    # fake = deterministic hashing embeddings (offline); openai / openrouter = embeddings API.
    embeddings_provider: EmbeddingsProvider = "fake"
    openai_embeddings_model: str = "text-embedding-3-small"
    openrouter_embeddings_model: str = "openai/text-embedding-3-small"
    # pgvector's HNSW index supports up to 2000 dimensions.
    embeddings_dimensions: int = Field(default=1536, ge=64, le=2000)
    # memory = in-process vector index; pgvector = vectors in the Postgres DATABASE_URL.
    kb_backend: Literal["memory", "pgvector"] = "memory"

    # --- Persistence ---------------------------------------------------------------------
    # SQLAlchemy URL. Tickets, audit trail and LangGraph checkpoints share this database.
    # Postgres: postgresql+psycopg://user:pass@host:5432/db (install the `postgres` extra).
    database_url: str = "sqlite+aiosqlite:///./data/support_agent.db"

    # --- Store API (orders, customers, knowledge base) ------------------------------------
    # Empty = use the bundled mock Store API in-process. Set to an URL to call a remote one.
    store_api_url: str | None = None
    store_api_timeout_seconds: float = Field(default=10.0, gt=0)

    # --- Business rules --------------------------------------------------------------------
    policy_refund_window_days: int = Field(default=30, ge=0)
    policy_auto_approve_refund_limit: Decimal = Field(default=Decimal("100.00"), ge=0)
    policy_vip_refunds_require_approval: bool = True
    policy_negative_sentiment_requires_approval: bool = True
    policy_unverified_sender_requires_approval: bool = True
    policy_escalate_negative_complaints: bool = True
    policy_kb_min_score: float = Field(default=0.15, ge=0)

    # --- App -----------------------------------------------------------------------------
    store_name: str = "Brewline Coffee"
    examples_dir: Path = Path("examples")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["console", "json"] = "console"

    # --- Tracing (OpenTelemetry) -------------------------------------------------------------
    # none = off (zero overhead); console = print spans; otlp = export to OTEL_EXPORTER_OTLP_ENDPOINT.
    otel_traces_exporter: Literal["none", "console", "otlp"] = "none"
    otel_service_name: str = "support-autopilot"

    def policy_config(self) -> PolicyConfig:
        return PolicyConfig(
            refund_window_days=self.policy_refund_window_days,
            auto_approve_refund_limit=self.policy_auto_approve_refund_limit,
            vip_refunds_require_approval=self.policy_vip_refunds_require_approval,
            negative_sentiment_requires_approval=self.policy_negative_sentiment_requires_approval,
            unverified_sender_requires_approval=self.policy_unverified_sender_requires_approval,
            escalate_negative_complaints=self.policy_escalate_negative_complaints,
            kb_min_score=self.policy_kb_min_score,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
