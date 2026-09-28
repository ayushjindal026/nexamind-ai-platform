"""
Embedding orchestration: turns a Document's extracted pages into
DocumentChunk rows and fills in their embeddings.

Deliberately separate from app/documents/service.py's upload pipeline — that
pipeline (and its exhaustive failure-mode tests) is untouched by Phase 4.
Embedding is an *enhancement* layered on top of a successful upload, never a
precondition of one: a document is `ready` once its text is extracted and
stored, whether or not it has been embedded yet.

Three properties this module guarantees, each covered by tests:

1. Idempotent. Calling chunk_and_embed_document twice never duplicates
   chunks (the UNIQUE(document_id, page_number, chunk_index) constraint is
   the backstop; the existence check below is what avoids ever tripping it).

2. Degrades gracefully. If no provider is configured, or the provider fails
   or returns malformed data, chunks are still created and simply remain
   un-embedded (embedding IS NULL). Nothing is raised to the caller for a
   provider failure — the return value reports how many chunks are pending.
   Retrieval only considers embedded chunks (app/retrieval/service.py), so a
   pending chunk is invisible to search, not an error.

3. Recoverable. embed_pending_chunks_for_organization() picks up anything
   still pending (provider was down, key wasn't configured yet, etc.) and
   embeds it — the "handle documents that have not yet been embedded" path.
   No queue or worker: it's a plain function a caller invokes when it wants.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
import uuid

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.documents.chunking import build_chunk_specs
from app.embeddings.provider import (
    EMBEDDING_DIMENSION,
    EmbeddingProvider,
    EmbeddingProviderError,
)
from app.models.document import Document
from app.models.document_chunk import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingSummary:
    chunk_count: int
    embedded_count: int

    @property
    def pending_count(self) -> int:
        return self.chunk_count - self.embedded_count


def _create_missing_chunks(db: Session, document: Document) -> None:
    existing = {
        (chunk.page_number, chunk.chunk_index)
        for chunk in db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).all()
    }
    for spec in build_chunk_specs(document):
        if (spec.page_number, spec.chunk_index) in existing:
            continue
        db.add(
            DocumentChunk(
                document_id=document.id,
                page_number=spec.page_number,
                chunk_index=spec.chunk_index,
                text=spec.text,
                char_count=len(spec.text),
            )
        )
    db.commit()


def _embed_chunks(
    db: Session, embedding_provider: EmbeddingProvider, chunks: list[DocumentChunk]
) -> int:
    """
    Embeds the given chunks in a single provider call. Returns how many were
    stored. Any provider failure or malformed response results in 0 stored
    and the chunks left un-embedded — never a partial write of bad data.
    """
    if not chunks:
        return 0

    try:
        vectors = embedding_provider.embed([chunk.text for chunk in chunks])
    except EmbeddingProviderError:
        logger.exception("Embedding provider failed; %d chunk(s) left pending", len(chunks))
        return 0

    if len(vectors) != len(chunks) or any(len(vector) != EMBEDDING_DIMENSION for vector in vectors):
        logger.error(
            "Embedding provider returned malformed output (expected %d vectors of dimension %d); "
            "%d chunk(s) left pending",
            len(chunks),
            EMBEDDING_DIMENSION,
            len(chunks),
        )
        return 0

    now = datetime.now(timezone.utc)
    for chunk, vector in zip(chunks, vectors):
        chunk.embedding = vector
        chunk.embedded_at = now
    db.commit()
    return len(chunks)


def chunk_and_embed_document(
    db: Session, embedding_provider: EmbeddingProvider | None, document: Document
) -> EmbeddingSummary:
    """
    Creates any missing chunks for `document`, then embeds whichever chunks
    lack an embedding. `embedding_provider=None` is valid and means "create
    chunks, skip embedding" — see module docstring, property 2.

    `document` must already be organization-verified by the caller (it comes
    from upload_document(), which is always called with the server-derived
    organization_id). This function operates on that one document only.
    """
    _create_missing_chunks(db, document)

    chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document.id)
        .order_by(DocumentChunk.page_number, DocumentChunk.chunk_index)
        .all()
    )
    pending = [chunk for chunk in chunks if chunk.embedding is None]

    if embedding_provider is not None and pending:
        _embed_chunks(db, embedding_provider, pending)

    embedded_count = sum(1 for chunk in chunks if chunk.embedding is not None)
    return EmbeddingSummary(chunk_count=len(chunks), embedded_count=embedded_count)


def embed_pending_chunks_for_organization(
    db: Session,
    embedding_provider: EmbeddingProvider,
    organization_id: uuid.UUID,
    limit: int = 200,
) -> int:
    """
    Recovery path: embeds up to `limit` chunks that belong to
    `organization_id`'s documents and still lack an embedding. Returns the
    number embedded. Organization-scoped by construction — the join through
    Document is the same tenant boundary every other query in the app uses.
    """
    pending = (
        db.query(DocumentChunk)
        .join(Document, Document.id == DocumentChunk.document_id)
        .filter(Document.organization_id == organization_id)
        .filter(DocumentChunk.embedding.is_(None))
        .order_by(DocumentChunk.created_at)
        .limit(limit)
        .all()
    )
    return _embed_chunks(db, embedding_provider, pending)


def get_chunk_counts(db: Session, document_ids: list[uuid.UUID]) -> dict[uuid.UUID, EmbeddingSummary]:
    """
    (total chunks, embedded chunks) per document in ONE grouped query — avoids
    loading chunk rows (and their 1536-float vectors) just to count them.
    Callers pass document ids they have already organization-verified.
    """
    if not document_ids:
        return {}
    rows = (
        db.query(
            DocumentChunk.document_id,
            func.count(DocumentChunk.id),
            func.count(DocumentChunk.embedding),  # COUNT(col) skips NULLs
        )
        .filter(DocumentChunk.document_id.in_(document_ids))
        .group_by(DocumentChunk.document_id)
        .all()
    )
    return {
        document_id: EmbeddingSummary(chunk_count=total, embedded_count=embedded)
        for document_id, total, embedded in rows
    }
