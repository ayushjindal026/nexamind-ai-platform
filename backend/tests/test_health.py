"""
Basic app-boot tests. Since Phase 2's conftest.py requires a live Postgres
connection for the whole test suite (see tests/conftest.py), the RUN_DB_TESTS
gate from Phase 1 no longer serves a purpose and has been removed — both
tests below now assume the same live-DB fixture as everything else.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_db_endpoint_reports_reachable():
    response = client.get("/health/db")
    assert response.status_code == 200
    assert response.json()["database"] == "reachable"
