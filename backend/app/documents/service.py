"""
The document upload pipeline.

Ordering is deliberate and mirrors the original ingestion design exactly:
validate raw bytes (empty / too large / not a PDF by magic bytes) -> open the
PDF -> check page count -> check organization quota -> extract page text ->
create the DB row -> write to storage -> persist pages and finalize status.

Every step before "create the DB row" raises one of the *ValidationError
exceptions below (or InvalidPDFError from pdf_extraction) — the router maps
all of these to 422, and no Document row is created for any of them. Nothing
is rolled back because nothing was written yet.

Every step after the DB row exists is handled explicitly per the Phase 3
design's named failure scenarios (see the two try/except blocks in
upload_document): if storage or the final DB commit fails, this function
attempts to leave the system in a known, recoverable state — a `failed` row,
and a best-effort deletion of any orphaned file — rather than pretending to
offer distributed-transaction guarantees it can't provide without a queue
and a reconciliation worker, which are explicitly out of scope for this MVP.

Known limitation, stated plainly rather than hidden: if the *second*,
simpler "mark this failed" commit also fails (a genuine double database
failure within one request), the row is left in `processing` with no
automatic reconciliation. This is an accepted, documented tradeoff, not an
oversight — see README "Known limitations."
"""

import logging
import uuid

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.documents.pdf_extraction import InvalidPDFError, extract_pages, open_pdf
from app.documents.storage import StorageBackend, StorageError
from app.models.document import Document, DocumentStatus
from app.models.document_page import DocumentPage

logger = logging.getLogger(__name__)

_PDF_MAGIC_BYTES = b"%PDF-"


class EmptyFileError(Exception):
    pass


class FileTooLargeError(Exception):
    pass


class UnsupportedFileTypeError(Exception):
    pass


class PageLimitExceededError(Exception):
    pass


class DocumentQuotaExceededError(Exception):
    pass


def sanitize_filename(filename: str) -> str:
    """
    Basename-only, control-character/null-byte stripped, length-capped.
    This is for DISPLAY purposes only (original_filename is never used to
    build a filesystem path — see storage.py and the storage_key construction
    in upload_document below) — sanitizing it well is a courtesy to the
    dashboard UI, not the security boundary.
    """
    name = filename.replace("\\", "/").split("/")[-1]
    name = "".join(ch for ch in name if ch.isprintable())
    name = name.strip()
    if not name:
        name = "document.pdf"
    return name[:255]


def _validate_upload_bytes(content: bytes) -> None:
    if len(content) == 0:
        raise EmptyFileError()
    if len(content) > settings.max_document_size_bytes:
        raise FileTooLargeError()
    if not content.startswith(_PDF_MAGIC_BYTES):
        raise UnsupportedFileTypeError()


def count_organization_documents(db: Session, organization_id: uuid.UUID) -> int:
    return db.query(Document).filter(Document.organization_id == organization_id).count()


def list_documents(db: Session, organization_id: uuid.UUID) -> list[Document]:
    """Always organization-scoped. Never call without a real organization_id."""
    return (
        db.query(Document)
        .filter(Document.organization_id == organization_id)
        .order_by(Document.created_at.desc())
        .all()
    )


def upload_document(
    db: Session,
    storage: StorageBackend,
    organization_id: uuid.UUID,
    filename: str,
    content: bytes,
) -> Document:
    """
    Raises (caller maps these to 422, no Document row created for any of them):
        EmptyFileError, FileTooLargeError, UnsupportedFileTypeError,
        InvalidPDFError, PageLimitExceededError, DocumentQuotaExceededError

    Raises (caller maps these to 500; a `failed` Document row IS created/left
    behind for these, since they occur after persistence has begun):
        StorageError, SQLAlchemyError
    """
    _validate_upload_bytes(content)

    pdf = open_pdf(content)  # raises InvalidPDFError
    try:
        if pdf.page_count > settings.max_document_pages:
            raise PageLimitExceededError()

        if count_organization_documents(db, organization_id) >= settings.max_documents_per_organization:
            raise DocumentQuotaExceededError()

        pages = extract_pages(pdf)  # raises InvalidPDFError on a corrupt page
        page_count = pdf.page_count
    finally:
        pdf.close()

    document_id = uuid.uuid4()
    storage_key = f"{organization_id}/{document_id}.pdf"

    document = Document(
        id=document_id,
        organization_id=organization_id,
        original_filename=sanitize_filename(filename),
        storage_key=storage_key,
        content_type="application/pdf",
        file_size_bytes=len(content),
        page_count=page_count,
        status=DocumentStatus.processing,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    # --- Named scenario 1: DB insert succeeded, now storage write fails ---
    try:
        storage.save(storage_key, content)
    except StorageError:
        logger.exception("Storage write failed for document %s", document_id)
        document.status = DocumentStatus.failed
        document.failure_reason = "Storage write failed"
        db.commit()
        raise

    # --- Named scenario 2: storage write succeeded, now DB finalize fails ---
    try:
        for page_number, text in pages:
            db.add(
                DocumentPage(
                    document_id=document.id,
                    page_number=page_number,
                    text=text,
                    char_count=len(text),
                )
            )
        document.status = DocumentStatus.ready
        db.commit()
    except SQLAlchemyError:
        logger.exception("Failed to finalize document %s after storage succeeded", document_id)
        db.rollback()

        # Compensating action: the file is now orphaned (DB doesn't know about
        # it in a consistent state) — best-effort delete.
        try:
            storage.delete(storage_key)
        except StorageError:
            logger.exception("Failed to clean up orphaned storage file for document %s", document_id)

        # A second, simpler commit is more likely to succeed than the one that
        # just failed. If it also fails, the row is left at `processing` — a
        # known, documented residual risk (see module docstring), not silently
        # hidden.
        try:
            document.status = DocumentStatus.failed
            document.failure_reason = "Failed to save extracted content"
            db.commit()
        except SQLAlchemyError:
            logger.exception(
                "Document %s left in 'processing' state after a double DB failure", document_id
            )
            db.rollback()
        raise

    return document
