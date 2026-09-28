"""
Document: metadata about an uploaded PDF, never the file bytes themselves.

The raw PDF content lives in exactly one place — the storage backend, at
`storage_key` (see app/documents/storage.py). Nothing in this table, or in
DocumentPage, holds the file itself; `document_pages.text` is *extracted*
data, derived from the file once, not a copy of it.

`storage_key` is always "{organization_id}/{id}.pdf" — both server-generated
UUIDs. `original_filename` is sanitized and kept for display only; it is
never used to construct a path (see app/documents/service.py).
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class DocumentStatus(str, enum.Enum):
    processing = "processing"
    ready = "ready"
    failed = "failed"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, name="document_status", native_enum=True),
        nullable=False,
        default=DocumentStatus.processing,
    )
    failure_reason: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    organization: Mapped["Organization"] = relationship(back_populates="documents")  # noqa: F821
    pages: Mapped[list["DocumentPage"]] = relationship(  # noqa: F821
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentPage.page_number",
    )
    chunks: Mapped[list["DocumentChunk"]] = relationship(  # noqa: F821
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentChunk.page_number, DocumentChunk.chunk_index",
    )
