"""
Test fixtures.

Tests run against a real Postgres (the same one the app uses — no SQLite
substitution, since UUID/Enum behavior genuinely differs and this project's
whole point is tenant-isolation correctness, not test speed).

Isolation between tests uses SQLAlchemy's documented pattern for joining a
Session to an external transaction: each test gets a SAVEPOINT-backed session
that is rolled back afterward, so tests never leak data into each other even
though they share one long-lived connection to a real database. See:
https://docs.sqlalchemy.org/en/20/orm/session_transaction.html#joining-a-session-into-an-external-transaction-such-as-for-test-suites
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

import app.models  # noqa: F401  (registers every model on Base.metadata before create_all)
from app.db.base import Base
from app.db.session import engine, get_db
from app.documents.storage import LocalFilesystemStorage, get_storage_backend
from app.embeddings.provider import get_embedding_provider
from app.llm.provider import get_llm_provider
from app.decisions.engine import get_decision_engine
from app.main import app


@pytest.fixture(scope="session", autouse=True)
def _create_schema():
    """
    Ensures tables exist for the test run. checkfirst=True (the default for
    create_all) makes this a no-op against a DB that Alembic has already
    migrated — this does NOT replace Alembic as the source of truth for
    schema changes, it only guarantees tests can run standalone too.
    """
    # pgvector must exist before a vector column can be created. In real
    # deployments the Alembic migration does this; tests can run standalone too.
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture()
def db_session():
    connection = engine.connect()
    trans = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")

    yield session

    session.close()
    trans.rollback()
    connection.close()


@pytest.fixture()
def client(db_session):
    def _get_db_override():
        yield db_session

    app.dependency_overrides[get_db] = _get_db_override
    # Safety default: tests NEVER reach a real embedding API, even if a
    # developer has OPENAI_API_KEY set locally. Tests that exercise embedding
    # opt in explicitly with the `embedding_provider` fixture (test_embeddings.py).
    app.dependency_overrides[get_embedding_provider] = lambda: None
    # LLM calls are also disabled by default; assistant tests explicitly
    # inject deterministic fakes and never contact a paid service.
    app.dependency_overrides[get_llm_provider] = lambda: None
    # Never initialize Laya/model weights in test runs; decision tests opt in
    # to deterministic fakes explicitly.
    app.dependency_overrides[get_decision_engine] = lambda: None
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture()
def tmp_storage(tmp_path):
    """Per-test temp-directory storage — tests never touch the real storage root."""
    storage = LocalFilesystemStorage(tmp_path / "documents")
    app.dependency_overrides[get_storage_backend] = lambda: storage
    yield storage
    app.dependency_overrides.pop(get_storage_backend, None)


@pytest.fixture()
def embedding_provider(client):
    """
    Opts a test into the deterministic offline embedder (see
    tests/embedding_fixtures.py). Depends on `client` so this override is
    applied AFTER the client fixture's safety default (provider=None).
    """
    from tests.embedding_fixtures import HashingEmbeddingProvider

    provider = HashingEmbeddingProvider()
    app.dependency_overrides[get_embedding_provider] = lambda: provider
    return provider
