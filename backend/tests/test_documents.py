"""
Document upload/list tests.

Uses the same fixture pattern as test_auth.py/test_tenant_isolation.py:
real Postgres, per-test SAVEPOINT rollback. Storage uses a temp-directory
LocalFilesystemStorage per test (via dependency override), never the real
configured storage root — so tests never touch actual application data.
"""

import io
import uuid

import pytest

from app.documents.storage import LocalFilesystemStorage, StorageBackend, StorageError, get_storage_backend
from app.main import app
from tests.pdf_fixtures import make_corrupt_pdf_bytes, make_pdf_bytes


# --- fixtures specific to this file ---


@pytest.fixture()
def tmp_storage(tmp_path):
    storage = LocalFilesystemStorage(tmp_path / "documents")
    app.dependency_overrides[get_storage_backend] = lambda: storage
    yield storage
    app.dependency_overrides.pop(get_storage_backend, None)


def _register_and_get_token(client, email, org_name) -> str:
    response = client.post(
        "/api/auth/register",
        json={"email": email, "password": "correct-horse-battery", "organization_name": org_name},
    )
    assert response.status_code == 201
    return response.json()["access_token"]


def _upload(client, token, content: bytes, filename: str = "test.pdf"):
    return client.post(
        "/api/documents",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": (filename, io.BytesIO(content), "application/pdf")},
    )


# --- happy path ---


def test_authenticated_upload_succeeds(client, tmp_storage):
    token = _register_and_get_token(client, "alice@acme-university.com", "Acme University")

    response = _upload(client, token, make_pdf_bytes(["Hello world"]), filename="Handbook.pdf")

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "Handbook.pdf"
    assert body["content_type"] == "application/pdf"
    assert body["page_count"] == 1
    assert body["status"] == "ready"
    assert body["failure_reason"] is None
    assert body["file_size_bytes"] > 0


def test_document_metadata_persisted_correctly(client, tmp_storage, db_session):
    from app.models.document import Document

    token = _register_and_get_token(client, "meta@acme-university.com", "Acme University")
    content = make_pdf_bytes(["one", "two", "three"])

    response = _upload(client, token, content, filename="ThreePager.pdf")
    document_id = uuid.UUID(response.json()["id"])

    document = db_session.query(Document).filter(Document.id == document_id).one()
    assert document.original_filename == "ThreePager.pdf"
    assert document.page_count == 3
    assert document.file_size_bytes == len(content)
    assert document.status.value == "ready"
    assert len(document.pages) == 3


# --- auth ---


def test_unauthenticated_upload_rejected(client, tmp_storage):
    response = client.post(
        "/api/documents", files={"file": ("test.pdf", io.BytesIO(make_pdf_bytes()), "application/pdf")}
    )
    assert response.status_code == 401


# --- validation ---


def test_unsupported_file_type_rejected(client, tmp_storage):
    token = _register_and_get_token(client, "bob@acme-university.com", "Acme University")
    response = _upload(client, token, b"not a pdf at all, just text", filename="notes.txt")
    assert response.status_code == 422
    assert "PDF" in response.json()["detail"] or "file type" in response.json()["detail"].lower()


def test_oversized_file_rejected(client, tmp_storage, monkeypatch):
    from app.core import config as config_module

    monkeypatch.setattr(config_module.settings, "max_document_size_bytes", 100)
    token = _register_and_get_token(client, "carol@acme-university.com", "Acme University")

    oversized = b"%PDF-1.4\n" + b"0" * 200
    response = _upload(client, token, oversized, filename="big.pdf")

    assert response.status_code == 422
    assert "size" in response.json()["detail"].lower()


def test_empty_file_rejected(client, tmp_storage):
    token = _register_and_get_token(client, "dana@acme-university.com", "Acme University")
    response = _upload(client, token, b"", filename="empty.pdf")
    assert response.status_code == 422
    assert "empty" in response.json()["detail"].lower()


def test_malformed_pdf_rejected_and_no_row_created(client, tmp_storage, db_session):
    from app.models.document import Document

    token = _register_and_get_token(client, "eve@acme-university.com", "Eve's Org")
    response = _upload(client, token, make_corrupt_pdf_bytes(), filename="corrupt.pdf")

    assert response.status_code == 422
    assert "valid PDF" in response.json()["detail"]

    count = db_session.query(Document).filter(Document.original_filename == "corrupt.pdf").count()
    assert count == 0


def test_page_limit_exceeded_rejected(client, tmp_storage, monkeypatch):
    from app.core import config as config_module

    monkeypatch.setattr(config_module.settings, "max_document_pages", 2)
    token = _register_and_get_token(client, "frank@acme-university.com", "Acme University")

    response = _upload(client, token, make_pdf_bytes(["p1", "p2", "p3"]), filename="toolong.pdf")

    assert response.status_code == 422
    assert "page" in response.json()["detail"].lower()


def test_organization_quota_exceeded_rejected(client, tmp_storage, monkeypatch):
    from app.core import config as config_module

    monkeypatch.setattr(config_module.settings, "max_documents_per_organization", 1)
    token = _register_and_get_token(client, "grace@acme-university.com", "Acme University")

    first = _upload(client, token, make_pdf_bytes(), filename="one.pdf")
    assert first.status_code == 201

    second = _upload(client, token, make_pdf_bytes(), filename="two.pdf")
    assert second.status_code == 422
    assert "limit" in second.json()["detail"].lower()


def test_duplicate_filenames_allowed(client, tmp_storage):
    """No uniqueness constraint on filename — storage_key (UUID-based) is the real identity."""
    token = _register_and_get_token(client, "henry@acme-university.com", "Acme University")

    first = _upload(client, token, make_pdf_bytes(["a"]), filename="policy.pdf")
    second = _upload(client, token, make_pdf_bytes(["b"]), filename="policy.pdf")

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["id"] != second.json()["id"]


# --- security ---


def test_malicious_filename_cannot_escape_storage_directory(client, tmp_storage):
    token = _register_and_get_token(client, "ivy@acme-university.com", "Acme University")

    response = _upload(client, token, make_pdf_bytes(), filename="../../etc/passwd.pdf")
    assert response.status_code == 201

    # Sanitized for display: no directory components survive.
    assert response.json()["filename"] == "passwd.pdf"

    # And the actual stored file lives only under the org-scoped temp storage root.
    stored_files = list(tmp_storage.root_dir.rglob("*.pdf"))
    assert len(stored_files) == 1
    assert tmp_storage.root_dir in stored_files[0].resolve().parents


# --- tenant isolation ---


def test_document_belongs_to_authenticated_organization(client, tmp_storage, db_session):
    from app.models.document import Document

    token = _register_and_get_token(client, "jack@acme-university.com", "Acme University")
    response = _upload(client, token, make_pdf_bytes(), filename="mine.pdf")

    document = db_session.query(Document).filter(Document.id == uuid.UUID(response.json()["id"])).one()
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).json()
    assert str(document.organization_id) == me["organization"]["id"]


def test_get_list_isolation_org_b_cannot_see_org_a_document(client, tmp_storage):
    token_a = _register_and_get_token(client, "kate@acme-university.com", "Acme University")
    token_b = _register_and_get_token(client, "leo@globaltech-institute.com", "Global Tech Institute")

    upload_response = _upload(client, token_a, make_pdf_bytes(), filename="AcmeOnly.pdf")
    assert upload_response.status_code == 201

    list_as_b = client.get("/api/documents", headers={"Authorization": f"Bearer {token_b}"})
    assert list_as_b.status_code == 200
    assert list_as_b.json() == []

    list_as_a = client.get("/api/documents", headers={"Authorization": f"Bearer {token_a}"})
    assert list_as_a.status_code == 200
    assert len(list_as_a.json()) == 1
    assert list_as_a.json()[0]["filename"] == "AcmeOnly.pdf"


# --- infrastructure failure modes ---


class _AlwaysFailsOnSave:
    """A StorageBackend that fails on save() — for testing the storage-failure path."""

    def save(self, key: str, content: bytes) -> None:
        raise StorageError("simulated storage failure")

    def read(self, key: str) -> bytes:
        raise StorageError("simulated storage failure")

    def delete(self, key: str) -> None:
        raise StorageError("simulated storage failure")


def test_storage_failure_returns_500_and_marks_document_failed(client, db_session):
    from app.models.document import Document

    app.dependency_overrides[get_storage_backend] = lambda: _AlwaysFailsOnSave()
    try:
        token = _register_and_get_token(client, "mia@acme-university.com", "Acme University")
        response = _upload(client, token, make_pdf_bytes(), filename="willfail.pdf")

        assert response.status_code == 500
        assert response.json() == {"detail": "Failed to process document"}  # no raw exception leaked

        document = (
            db_session.query(Document).filter(Document.original_filename == "willfail.pdf").one()
        )
        assert document.status.value == "failed"
        assert document.failure_reason == "Storage write failed"
    finally:
        app.dependency_overrides.pop(get_storage_backend, None)


def test_cleanup_after_final_db_failure(client, tmp_storage, monkeypatch, db_session):
    """
    Storage succeeds, but the final DB commit (page persistence + status=ready)
    fails. Expect: the orphaned file is deleted from storage, the row is left
    `failed` (not `processing`), and the client gets a generic 500.
    """
    from app.documents import service as service_module

    token = _register_and_get_token(client, "nina@acme-university.com", "Acme University")

    original_commit = db_session.commit
    call_count = {"n": 0}

    def _commit_second_call_fails():
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise service_module.SQLAlchemyError("simulated DB failure on finalize")
        return original_commit()

    monkeypatch.setattr(db_session, "commit", _commit_second_call_fails)

    response = _upload(client, token, make_pdf_bytes(), filename="dbfail.pdf")

    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to process document"}

    stored_files = list(tmp_storage.root_dir.rglob("*.pdf"))
    assert stored_files == []  # cleaned up, not orphaned
