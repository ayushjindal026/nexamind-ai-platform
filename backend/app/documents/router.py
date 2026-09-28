"""
Documents endpoints.

Only POST (upload) and GET (list) exist — per the approved Phase 3 scope,
GET /{id}, DELETE, and update endpoints are deliberately not added yet.

Every *ValidationError-style exception from the service layer maps to 422
with a specific, safe message. StorageError and SQLAlchemyError map to a
single generic 500 message — the real cause is logged server-side (see
service.py's logger.exception calls) but never returned to the client. This
mirrors the same "translate domain exceptions to HTTP" pattern already used
in app/auth/router.py.
"""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_membership
from app.core.config import settings
from app.db.session import get_db
from app.documents.service import (
    DocumentQuotaExceededError,
    EmptyFileError,
    FileTooLargeError,
    PageLimitExceededError,
    UnsupportedFileTypeError,
    list_documents,
    upload_document,
)
from app.documents.pdf_extraction import InvalidPDFError
from app.embeddings.provider import EmbeddingProvider, get_embedding_provider
from app.embeddings.service import EmbeddingSummary, chunk_and_embed_document, get_chunk_counts
from app.documents.storage import StorageBackend, StorageError, get_storage_backend
from app.models.document import Document
from app.models.membership import Membership
from app.schemas.documents import DocumentOut

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])


def _to_document_out(document: Document, summary: EmbeddingSummary | None = None) -> DocumentOut:
    return DocumentOut(
        id=document.id,
        filename=document.original_filename,
        content_type=document.content_type,
        file_size_bytes=document.file_size_bytes,
        page_count=document.page_count,
        status=document.status.value,
        failure_reason=document.failure_reason,
        created_at=document.created_at,
        chunk_count=summary.chunk_count if summary else 0,
        embedded_chunk_count=summary.embedded_count if summary else 0,
    )


@router.post("", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def upload(
    file: UploadFile = File(...),
    membership: Membership = Depends(get_current_membership),
    db: Session = Depends(get_db),
    storage: StorageBackend = Depends(get_storage_backend),
    embedding_provider: EmbeddingProvider | None = Depends(get_embedding_provider),
) -> DocumentOut:
    content = await file.read()

    try:
        document = upload_document(
            db=db,
            storage=storage,
            organization_id=membership.organization_id,
            filename=file.filename or "document.pdf",
            content=content,
        )
    except EmptyFileError:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Uploaded file is empty")
    except FileTooLargeError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"File exceeds the maximum allowed size of {settings.max_document_size_bytes} bytes",
        )
    except UnsupportedFileTypeError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unsupported file type; only PDF files are accepted",
        )
    except InvalidPDFError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="File could not be parsed as a valid PDF",
        )
    except PageLimitExceededError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"PDF exceeds the maximum allowed page count of {settings.max_document_pages}",
        )
    except DocumentQuotaExceededError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Organization has reached its document upload limit of "
                f"{settings.max_documents_per_organization}"
            ),
        )
    except (StorageError, SQLAlchemyError):
        # Real cause is already logged inside upload_document(); never surface
        # filesystem paths, SQL errors, or stack traces to the client.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process document",
        )

    # Phase 4: chunk + embed as a best-effort step AFTER a successful upload.
    # The upload itself has already succeeded and is committed; embedding is
    # an enhancement, so nothing here may turn that success into an error.
    # No provider configured, a provider failure, or even an unexpected
    # exception all leave the document `ready` with un-embedded chunks —
    # retrievable later via embed_pending_chunks_for_organization().
    summary: EmbeddingSummary | None = None
    try:
        summary = chunk_and_embed_document(db, embedding_provider, document)
    except Exception:  # noqa: BLE001 — deliberate: never fail a completed upload over embedding
        logger.exception("Chunking/embedding failed for document %s; left pending", document.id)
        db.rollback()
        try:
            summary = get_chunk_counts(db, [document.id]).get(document.id)
        except Exception:  # noqa: BLE001
            summary = None

    return _to_document_out(document, summary)


@router.get("", response_model=list[DocumentOut])
def list_documents_endpoint(
    membership: Membership = Depends(get_current_membership),
    db: Session = Depends(get_db),
) -> list[DocumentOut]:
    documents = list_documents(db, membership.organization_id)
    counts = get_chunk_counts(db, [document.id for document in documents])
    return [_to_document_out(document, counts.get(document.id)) for document in documents]
