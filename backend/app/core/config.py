"""
Typed application settings, loaded from environment variables.

Phase 1 added DATABASE_URL and CORS config. Phase 2 added JWT settings.
Phase 3 added: `environment` (enables a fail-fast production safety check,
see the validator below), document size/page/quota limits, and the local
storage root. Phase 4 adds the optional embedding provider settings.
Later phases extend this same Settings class rather than introducing a
second config mechanism.
"""

from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The dev-only JWT secret default. Named as a module constant (not inlined)
# so the production-safety validator below and the field default can never
# drift out of sync with each other.
_DEV_ONLY_JWT_SECRET = "dev-only-insecure-secret-change-me"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://psi_user:psi_password@db:5432/psi_db"
    backend_cors_origins: str = "http://localhost:5173"

    # --- Phase 2: JWT authentication ---
    jwt_secret_key: str = _DEV_ONLY_JWT_SECRET
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # --- Phase 3: environment + document ingestion limits ---
    # "development" is the default so `docker compose up` works out of the box
    # for this take-home. Set ENVIRONMENT=production to enable the fail-fast
    # check below (e.g. before any real deployment) — nothing else in the
    # codebase currently branches on this value.
    environment: Literal["development", "production"] = "development"

    max_document_size_bytes: int = 10 * 1024 * 1024  # 10 MB
    max_document_pages: int = 20
    max_documents_per_organization: int = 10
    document_storage_root: str = "/app/storage/documents"

    # --- Phase 4: embeddings ---
    # Optional on purpose: the app boots and uploads work without a key
    # (chunks simply stay un-embedded — see app/embeddings/service.py).
    # Only the retrieval search endpoint hard-requires a provider (503 without one).
    # The embedding *dimension* is deliberately not a setting — it is a fixed
    # constant tied to the pgvector column DDL (see app/embeddings/provider.py).
    openai_api_key: str | None = None
    embedding_model: str = "text-embedding-3-small"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.backend_cors_origins.split(",") if origin.strip()]

    @model_validator(mode="after")
    def _require_explicit_jwt_secret_in_production(self) -> "Settings":
        """
        Preserves the security requirement from Phase 2's approval: the
        development JWT secret fallback is acceptable only for local
        development. If ENVIRONMENT=production and no explicit JWT_SECRET_KEY
        was supplied, fail immediately at startup (this validator runs when
        `Settings()` is constructed, i.e. at import time of this module) —
        never silently fall back to a known, checked-in secret in production.
        """
        if self.environment == "production" and self.jwt_secret_key == _DEV_ONLY_JWT_SECRET:
            raise ValueError(
                "JWT_SECRET_KEY must be set explicitly when ENVIRONMENT=production. "
                "Refusing to start with the development default secret."
            )
        return self


settings = Settings()
