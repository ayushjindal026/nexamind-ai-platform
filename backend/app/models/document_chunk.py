"""
DocumentChunk: the unit that gets embedded and searched.

Distinct from DocumentPage on purpose: DocumentPage is the raw extraction
record (one row per physical page, always present once a document is
`ready`); DocumentChunk is the RAG-facing unit derived from it (see
app/documents/chunking.py). Today they're 1:1 — one chunk per non-empty
page — but keeping them as separate tables means real sub-page chunking
later is a change to chunking.py, not a migration.

Same tenant-ownership convention as DocumentPage: NO organization_id column.
Ownership flows document_chunk -> document -> organization; every retrieval
query joins through Document (see app/retrieval/service.py), which is
already always organization-scoped.

`embedding` is nullable — a chunk can and normally does exist for a moment
(or indefinitely, if the embedding provider isn't configured or a call
fails) without one. Retrieval only ever considers chunks where embedding
IS NOT NULL — see app/retrieval/service.py. `embedded_at` distinguishes
"never attempted" from a future retry policy being able to tell how stale
an embedding is, without overloading `created_at` for that purpose.
"""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Index, Integer, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.embeddings.provider import EMBEDDING_DIMENSION


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "page_number", "chunk_index"),
        # HNSW (not IVFFlat): builds incrementally, so it works on an empty
        # table at migration time and needs no row-count-based tuning (IVFFlat
        # needs existing data to cluster well). vector_cosine_ops matches the
        # `<=>` cosine-distance operator the retrieval query orders by.
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIMENSION), nullable=True)
    embedded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped["Document"] = relationship(back_populates="chunks")  # noqa: F821
