"""
Retrieval: organization-scoped vector similarity search.

This is the retrieval primitive later RAG phases build on (the /api/ask
pipeline will call search_similar_chunks() directly). It does no language
understanding, decision-making, or answer generation.

Tenant isolation: `organization_id` is a REQUIRED positional argument —
there is no code path that searches without one — and every query joins
DocumentChunk -> Document and filters on Document.organization_id. Chunks
carry no organization_id of their own (see models/document_chunk.py); the
document is the single ownership boundary. Callers pass only the
server-derived organization_id (from get_current_membership, or later from
the assistant-token lookup), never anything client-supplied.

Only embedded chunks (embedding IS NOT NULL) of `ready` documents are
searchable. A chunk that hasn't been embedded yet is simply invisible to
search — not an error (see app/embeddings/service.py).

Known limitation, stated plainly: the HNSW index (see the migration) is an
*approximate* index, and the organization filter is applied alongside it.
At this project's scale (a few hundred chunks per tenant at most) Postgres
plans this as an exact scan and results are exact. At much larger scale, a
selective tenant filter combined with an approximate index can return fewer
than `top_k` rows even when more matching chunks exist; the standard
mitigations (pgvector iterative index scans, per-tenant partial indexes, or
partitioning) are production improvements, not needed for this MVP.
"""

import uuid
from dataclasses import dataclass

# pyrefly: ignore [missing-import]
from sqlalchemy.orm import Session

from app.models.document import Document, DocumentStatus
from app.models.document_chunk import DocumentChunk


@dataclass
class ChunkSearchResult:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_name: str
    page_number: int
    chunk_index: int
    text: str
    similarity: float  # cosine similarity in [-1, 1]; 1.0 = identical direction


def search_similar_chunks(
    db: Session,
    organization_id: uuid.UUID,
    query_embedding: list[float],
    top_k: int = 5,
    min_similarity: float | None = None,
) -> list[ChunkSearchResult]:
    """
    Returns up to `top_k` chunks, most similar first. If `min_similarity` is
    given, chunks scoring below it are excluded — this is the relevance
    threshold later phases use to decide "the knowledge base doesn't contain
    an answer" without ever calling an LLM with irrelevant context.
    """
    distance = DocumentChunk.embedding.cosine_distance(query_embedding).label("distance")

    query = (
        db.query(DocumentChunk, Document.original_filename, distance)
        .join(Document, Document.id == DocumentChunk.document_id)
        .filter(Document.organization_id == organization_id)
        .filter(Document.status == DocumentStatus.ready)
        .filter(DocumentChunk.embedding.is_not(None))
    )
    if min_similarity is not None:
        # cosine_distance = 1 - cosine_similarity
        query = query.filter(distance <= 1.0 - min_similarity)

    rows = query.order_by(distance).limit(top_k).all()

    return [
        ChunkSearchResult(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            document_name=filename,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            similarity=1.0 - float(dist),
        )
        for chunk, filename, dist in rows
    ]
