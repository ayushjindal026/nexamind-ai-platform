"""
Embedding pipeline tests (chunk creation + embedding + graceful degradation).

All run against real Postgres with the real pgvector extension. The
embedding provider is always a deterministic offline fake — see
tests/embedding_fixtures.py — so nothing here needs network access.
"""

import uuid

from sqlalchemy import text

from app.embeddings.provider import EMBEDDING_DIMENSION
from app.embeddings.service import (
    chunk_and_embed_document,
    embed_pending_chunks_for_organization,
    get_chunk_counts,
)
from app.main import app
from app.embeddings.provider import get_embedding_provider
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from tests.embedding_fixtures import (
    FailingEmbeddingProvider,
    HashingEmbeddingProvider,
    MalformedEmbeddingProvider,
)
from tests.rag_helpers import GPA_TEXT, HOSTEL_TEXT, LEAVE_TEXT, register_org, upload_pdf


def _chunks(db_session, document_id: str) -> list[DocumentChunk]:
    return (
        db_session.query(DocumentChunk)
        .filter(DocumentChunk.document_id == uuid.UUID(document_id))
        .order_by(DocumentChunk.page_number)
        .all()
    )


# --- upload -> chunk -> embed, through the real HTTP endpoint ---


def test_upload_creates_one_embedded_chunk_per_page(client, tmp_storage, embedding_provider, db_session):
    org = register_org(client, "a@acme-university.com", "Acme University")

    body = upload_pdf(client, org, [GPA_TEXT, LEAVE_TEXT, HOSTEL_TEXT])

    assert body["status"] == "ready"
    assert body["chunk_count"] == 3
    assert body["embedded_chunk_count"] == 3

    chunks = _chunks(db_session, body["id"])
    assert [chunk.page_number for chunk in chunks] == [1, 2, 3]
    assert [chunk.chunk_index for chunk in chunks] == [0, 0, 0]
    assert chunks[0].text.strip() == GPA_TEXT
    for chunk in chunks:
        assert chunk.embedding is not None
        assert len(chunk.embedding) == EMBEDDING_DIMENSION
        assert chunk.embedded_at is not None
        assert chunk.char_count == len(chunk.text)


def test_all_chunks_embedded_in_a_single_provider_call(client, tmp_storage, embedding_provider):
    org = register_org(client, "b@acme-university.com", "Acme University")
    upload_pdf(client, org, [GPA_TEXT, LEAVE_TEXT, HOSTEL_TEXT])

    # One batched call for the whole document, not one call per chunk.
    assert len(embedding_provider.calls) == 1
    assert len(embedding_provider.calls[0]) == 3


def test_blank_page_produces_no_chunk(client, tmp_storage, embedding_provider):
    org = register_org(client, "c@acme-university.com", "Acme University")

    body = upload_pdf(client, org, [GPA_TEXT, ""])

    assert body["page_count"] == 2
    assert body["chunk_count"] == 1  # the blank page has nothing to embed


# --- graceful degradation: "handle documents that have not yet been embedded" ---


def test_upload_without_provider_succeeds_and_leaves_chunks_pending(client, tmp_storage, db_session):
    # No `embedding_provider` fixture -> conftest default (provider=None).
    org = register_org(client, "d@acme-university.com", "Acme University")

    body = upload_pdf(client, org, [GPA_TEXT, LEAVE_TEXT])

    assert body["status"] == "ready"  # document success is independent of embedding
    assert body["chunk_count"] == 2
    assert body["embedded_chunk_count"] == 0
    assert all(chunk.embedding is None for chunk in _chunks(db_session, body["id"]))


def test_provider_failure_does_not_fail_the_upload(client, tmp_storage, db_session):
    app.dependency_overrides[get_embedding_provider] = lambda: FailingEmbeddingProvider()
    org = register_org(client, "e@acme-university.com", "Acme University")

    response_body = upload_pdf(client, org, [GPA_TEXT])

    assert response_body["status"] == "ready"
    assert response_body["chunk_count"] == 1
    assert response_body["embedded_chunk_count"] == 0
    assert _chunks(db_session, response_body["id"])[0].embedding is None


def test_malformed_provider_output_is_never_stored(client, tmp_storage, db_session):
    app.dependency_overrides[get_embedding_provider] = lambda: MalformedEmbeddingProvider()
    org = register_org(client, "f@acme-university.com", "Acme University")

    body = upload_pdf(client, org, [GPA_TEXT])

    assert body["embedded_chunk_count"] == 0
    assert _chunks(db_session, body["id"])[0].embedding is None


def test_list_endpoint_reports_embedding_progress(client, tmp_storage, embedding_provider):
    org = register_org(client, "g@acme-university.com", "Acme University")
    upload_pdf(client, org, [GPA_TEXT, LEAVE_TEXT], filename="embedded.pdf")

    listing = client.get("/api/documents", headers=org.headers).json()

    assert len(listing) == 1
    assert listing[0]["chunk_count"] == 2
    assert listing[0]["embedded_chunk_count"] == 2


# --- service-level behavior ---


def test_chunk_and_embed_is_idempotent(client, tmp_storage, embedding_provider, db_session):
    org = register_org(client, "h@acme-university.com", "Acme University")
    body = upload_pdf(client, org, [GPA_TEXT, LEAVE_TEXT])
    document = db_session.get(Document, uuid.UUID(body["id"]))
    calls_after_upload = len(embedding_provider.calls)

    summary = chunk_and_embed_document(db_session, embedding_provider, document)

    assert summary.chunk_count == 2
    assert summary.embedded_count == 2
    assert len(_chunks(db_session, body["id"])) == 2  # no duplicates
    assert len(embedding_provider.calls) == calls_after_upload  # nothing pending -> no provider call


def test_pending_chunks_are_recovered_later(client, tmp_storage, db_session):
    org = register_org(client, "i@acme-university.com", "Acme University")
    body = upload_pdf(client, org, [GPA_TEXT, LEAVE_TEXT])  # provider=None -> pending
    assert body["embedded_chunk_count"] == 0

    provider = HashingEmbeddingProvider()
    embedded = embed_pending_chunks_for_organization(
        db_session, provider, uuid.UUID(org.organization_id)
    )

    assert embedded == 2
    chunks = _chunks(db_session, body["id"])
    assert all(chunk.embedding is not None and chunk.embedded_at is not None for chunk in chunks)


def test_recovery_with_failing_provider_leaves_chunks_pending(client, tmp_storage, db_session):
    org = register_org(client, "j@acme-university.com", "Acme University")
    body = upload_pdf(client, org, [GPA_TEXT])

    embedded = embed_pending_chunks_for_organization(
        db_session, FailingEmbeddingProvider(), uuid.UUID(org.organization_id)
    )

    assert embedded == 0
    assert _chunks(db_session, body["id"])[0].embedding is None


def test_recovery_is_organization_scoped(client, tmp_storage, db_session):
    org_a = register_org(client, "k@acme-university.com", "Acme University")
    org_b = register_org(client, "l@globaltech-institute.com", "Global Tech Institute")
    doc_a = upload_pdf(client, org_a, [GPA_TEXT], filename="a.pdf")
    doc_b = upload_pdf(client, org_b, [LEAVE_TEXT], filename="b.pdf")

    embedded = embed_pending_chunks_for_organization(
        db_session, HashingEmbeddingProvider(), uuid.UUID(org_a.organization_id)
    )

    assert embedded == 1
    assert _chunks(db_session, doc_a["id"])[0].embedding is not None
    assert _chunks(db_session, doc_b["id"])[0].embedding is None  # org B untouched


def test_get_chunk_counts(client, tmp_storage, embedding_provider, db_session):
    org = register_org(client, "m@acme-university.com", "Acme University")
    body = upload_pdf(client, org, [GPA_TEXT, LEAVE_TEXT, HOSTEL_TEXT])
    document_id = uuid.UUID(body["id"])

    # Un-embed one chunk to get a mixed state.
    chunk = _chunks(db_session, body["id"])[0]
    chunk.embedding = None
    db_session.commit()

    counts = get_chunk_counts(db_session, [document_id])
    assert counts[document_id].chunk_count == 3
    assert counts[document_id].embedded_count == 2
    assert counts[document_id].pending_count == 1
    assert get_chunk_counts(db_session, []) == {}


# --- schema ---


def test_deleting_a_document_cascades_to_its_chunks(client, tmp_storage, embedding_provider, db_session):
    org = register_org(client, "n@acme-university.com", "Acme University")
    body = upload_pdf(client, org, [GPA_TEXT, LEAVE_TEXT])
    document = db_session.get(Document, uuid.UUID(body["id"]))

    db_session.delete(document)
    db_session.commit()

    assert _chunks(db_session, body["id"]) == []


def test_hnsw_cosine_index_exists_on_embedding_column(db_session):
    row = db_session.execute(
        text(
            "SELECT indexdef FROM pg_indexes "
            "WHERE tablename = 'document_chunks' AND indexname = 'ix_document_chunks_embedding_hnsw'"
        )
    ).scalar_one_or_none()

    assert row is not None, "HNSW index missing — was the migration applied?"
    assert "hnsw" in row
    assert "vector_cosine_ops" in row
