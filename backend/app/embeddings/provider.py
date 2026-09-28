"""
Embedding provider abstraction for OpenAI and Gemini.

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

from google import genai
from google.genai import types
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


class GeminiEmbeddingProvider:
    """Gemini Embedding 2 adapter returning separate 1536d vectors per text."""

    def __init__(self, api_key: str, model: str = "gemini-embedding-2", client=None):
        self._client = client or genai.Client(api_key=api_key)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Gemini Embedding 2 aggregates raw multi-part inputs. A list of
        # Content entries instead produces one corresponding vector per text.
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=text)],
            )
            for text in texts
        ]
        try:
            response = self._client.models.embed_content(
                model=self._model,
                contents=contents,
                config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSION),
            )
            return [[float(value) for value in item.values] for item in response.embeddings]
        except Exception as exc:  # SDK errors vary by HTTP/auth failure type
            raise EmbeddingProviderError("Gemini embedding request failed") from exc


_embedding_provider_instance: EmbeddingProvider | None = None
_embedding_provider_resolved: bool = False


def get_embedding_provider() -> EmbeddingProvider | None:
    """
    FastAPI dependency. Returns None if the selected provider key is absent —
    callers decide what "no provider" means for them: the upload pipeline
    treats it as "skip embedding, leave chunks pending" (never a hard
    failure — see app/documents/router.py); the retrieval search endpoint
    treats it as a 503, since search genuinely cannot function without one.

    Cached after first resolution (module-level singleton), same pattern as
    get_storage_backend. Tests override this via app.dependency_overrides.
    """
    global _embedding_provider_instance, _embedding_provider_resolved
    if not _embedding_provider_resolved:
        if settings.embedding_provider == "openai" and settings.openai_api_key:
            _embedding_provider_instance = OpenAIEmbeddingProvider(
                api_key=settings.openai_api_key, model=settings.embedding_model
            )
        elif settings.embedding_provider == "gemini" and (settings.gemini_api_key or "").strip():
            try:
                _embedding_provider_instance = GeminiEmbeddingProvider(
                    api_key=settings.gemini_api_key.strip(),
                    model=settings.gemini_embedding_model,
                )
            except Exception:
                logger.exception("Gemini embedding provider could not be initialized")
                _embedding_provider_instance = None
        else:
            configured_provider = settings.embedding_provider
            if configured_provider not in ("openai", "gemini"):
                logger.error("Unsupported embedding provider configured: %s", configured_provider)
                _embedding_provider_instance = None
                _embedding_provider_resolved = True
                return _embedding_provider_instance
            key_name = "OPENAI_API_KEY" if settings.embedding_provider == "openai" else "GEMINI_API_KEY"
            logger.info("%s not configured; embedding provider unavailable", key_name)
            _embedding_provider_instance = None
        _embedding_provider_resolved = True
    return _embedding_provider_instance
