"""
Embedding provider abstraction.

One Protocol, one production implementation (OpenAI's text-embedding-3-small,
1536 dimensions — a stable, well-documented, cost-effective choice already
settled in the original architecture design). A future provider swap (a
different API, a locally-hosted model) means one new class implementing the
same `embed()` method — no changes to app/embeddings/service.py or
app/retrieval/service.py, which only ever depend on the Protocol.

EMBEDDING_DIMENSION is a fixed module-level constant, not a runtime Settings
value: pgvector requires a fixed dimension baked into the column's DDL
(Vector(1536) in the DocumentChunk model, matched in the migration), so
"configurable" here would be misleading — changing it for real always means
a new migration and re-embedding every chunk, never just an env var flip.

Tests never exercise OpenAIEmbeddingProvider directly (no network access to
any external embedding API is assumed or required to run this test suite) —
they inject a deterministic fake via the same dependency-override pattern
already used for StorageBackend in Phase 3 (see tests/embedding_fixtures.py).
"""

import logging
from typing import Protocol

from openai import OpenAI, OpenAIError

from app.core.config import settings

logger = logging.getLogger(__name__)

EMBEDDING_DIMENSION = 1536


class EmbeddingProviderError(Exception):
    """
    Raised for any embedding provider failure (auth, rate limit, network,
    malformed response). Callers catch this one exception type, never a
    provider-specific exception — same pattern as StorageError.
    """


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Returns one embedding vector per input text, same order, each of length EMBEDDING_DIMENSION."""
        ...


class OpenAIEmbeddingProvider:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self._client.embeddings.create(model=self._model, input=texts)
        except OpenAIError as exc:
            raise EmbeddingProviderError(f"OpenAI embedding request failed: {exc}") from exc

        # The API preserves input order in `.data`, but each item also carries
        # its own `.index` — sort by that rather than assume ordering holds,
        # since relying on unstated ordering guarantees is exactly the kind
        # of thing that breaks silently later.
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]


_embedding_provider_instance: EmbeddingProvider | None = None
_embedding_provider_resolved: bool = False


def get_embedding_provider() -> EmbeddingProvider | None:
    """
    FastAPI dependency. Returns None if no OPENAI_API_KEY is configured —
    callers decide what "no provider" means for them: the upload pipeline
    treats it as "skip embedding, leave chunks pending" (never a hard
    failure — see app/documents/router.py); the retrieval search endpoint
    treats it as a 503, since search genuinely cannot function without one.

    Cached after first resolution (module-level singleton), same pattern as
    get_storage_backend. Tests override this via app.dependency_overrides.
    """
    global _embedding_provider_instance, _embedding_provider_resolved
    if not _embedding_provider_resolved:
        if settings.openai_api_key:
            _embedding_provider_instance = OpenAIEmbeddingProvider(
                api_key=settings.openai_api_key, model=settings.embedding_model
            )
        else:
            logger.info("OPENAI_API_KEY not configured; embedding provider unavailable")
            _embedding_provider_instance = None
        _embedding_provider_resolved = True
    return _embedding_provider_instance
