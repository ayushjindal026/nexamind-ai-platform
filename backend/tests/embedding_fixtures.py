"""
Deterministic, offline embedding providers for tests.

HashingEmbeddingProvider uses the "hashing trick" (feature hashing) — a
real, long-established technique, not random noise: each lowercase word is
hashed into one of EMBEDDING_DIMENSION buckets and counted, then the vector
is L2-normalized. Texts sharing words therefore have genuinely higher cosine
similarity than texts that don't, which is exactly the property retrieval
tests need to assert meaningful ranking — with no network, no model
download, and fully deterministic output.

It is NOT a semantic model (it knows nothing about synonyms), and the tests
say so: they assert word-overlap ranking, not semantic understanding.
Production uses OpenAIEmbeddingProvider; this exists so the test suite never
needs an external API, an API key, or network access.
"""

import hashlib
import math
import re

from app.embeddings.provider import EMBEDDING_DIMENSION, EmbeddingProviderError

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbeddingProvider:
    def __init__(self):
        self.calls: list[list[str]] = []  # record of every embed() call, for assertions

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._embed_one(text) for text in texts]

    @staticmethod
    def _embed_one(text: str) -> list[float]:
        vector = [0.0] * EMBEDDING_DIMENSION
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest, "big") % EMBEDDING_DIMENSION
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            # No tokens at all: return a fixed unit vector rather than a zero
            # vector (cosine distance to a zero vector is undefined in pgvector).
            vector[0] = 1.0
            return vector
        return [v / norm for v in vector]


class FailingEmbeddingProvider:
    """Always raises — simulates provider outage / auth / rate-limit failure."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingProviderError("simulated provider outage")


class MalformedEmbeddingProvider:
    """Returns vectors of the wrong dimension — simulates a misconfigured model."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]
