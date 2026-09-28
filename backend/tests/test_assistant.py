"""Assistant management and public RAG API tests; providers are deterministic fakes."""

import uuid

import pytest

from app.assistant.service import get_assistant_for_organization
from app.llm.provider import LLMProviderError, get_llm_provider
from app.main import app
from tests.rag_helpers import GPA_TEXT, LEAVE_TEXT, register_org, upload_pdf


class MockLLMProvider:
    def __init__(self, answer="The scholarship requires a minimum GPA of 3.0.", error=False):
        self.answer = answer
        self.error = error
        self.calls = []

    def generate_answer(self, question, context):
        self.calls.append((question, context))
        if self.error:
            raise LLMProviderError("simulated provider outage")
        return self.answer


@pytest.fixture
def llm_provider(client):
    provider = MockLLMProvider()
    app.dependency_overrides[get_llm_provider] = lambda: provider
    return provider


def _create_assistant(client, org):
    response = client.post("/api/assistant", headers=org.headers)
    assert response.status_code == 201, response.text
    return response.json()


def _ask(client, token, question, *, query_token=False):
    url = "/api/ask" + ("?assistant_token=" + token if query_token else "")
    payload = {"question": question}
    if not query_token:
        payload["assistant_token"] = token
    return client.post(url, json=payload)


def test_assistant_management_requires_dashboard_authentication(client):
    assert client.get("/api/assistant").status_code == 401
    assert client.post("/api/assistant").status_code == 401
    assert client.post("/api/assistant/regenerate").status_code == 401


def test_assistant_status_is_limited_to_the_authenticated_organization(client):
    org_a = register_org(client, "assistant-status-a@example.com", "Assistant Status A")
    org_b = register_org(client, "assistant-status-b@example.com", "Assistant Status B")
    _create_assistant(client, org_a)

    status_a = client.get("/api/assistant", headers=org_a.headers).json()
    status_b = client.get("/api/assistant", headers=org_b.headers).json()
    foreign_rotation = client.post("/api/assistant/regenerate", headers=org_b.headers)

    assert status_a["configured"] is True
    assert status_b["configured"] is False
    assert status_b["assistant_id"] is None
    assert foreign_rotation.status_code == 404


def test_create_assistant_returns_one_time_token_and_embed_code(client, db_session):
    org = register_org(client, "assistant-create@example.com", "Assistant Org")

    created = _create_assistant(client, org)
    assert created["assistant_token"]
    assert created["assistant_url"].endswith("?assistant_token=" + created["assistant_token"])
    assert "<iframe" in created["embed_code"]
    assert created["assistant_url"] in created["embed_code"]
    assert created["assistant_token"] in created["embed_code"]

    assistant = get_assistant_for_organization(db_session, uuid.UUID(org.organization_id))
    assert assistant is not None
    assert assistant.token_hash != created["assistant_token"]
    assert len(assistant.token_hash) == 64

    status_response = client.get("/api/assistant", headers=org.headers)
    assert status_response.status_code == 200
    assert status_response.json()["configured"] is True
    assert "assistant_token" not in status_response.json()
    assert client.post("/api/assistant", headers=org.headers).status_code == 409


def test_regenerate_assistant_token_revokes_previous_token(client, embedding_provider, llm_provider):
    org = register_org(client, "assistant-rotate@example.com", "Assistant Org")
    created = _create_assistant(client, org)
    rotated = client.post("/api/assistant/regenerate", headers=org.headers)

    assert rotated.status_code == 200
    new_token = rotated.json()["assistant_token"]
    assert new_token != created["assistant_token"]
    assert _ask(client, created["assistant_token"], "What is the scholarship GPA?").status_code == 401
    assert _ask(client, new_token, "What is the scholarship GPA?").status_code == 200


def test_regenerate_requires_existing_assistant(client):
    org = register_org(client, "assistant-missing@example.com", "Assistant Org")
    response = client.post("/api/assistant/regenerate", headers=org.headers)
    assert response.status_code == 404


def test_public_ask_is_unauthenticated_but_returns_grounded_answer_and_citation(
    client, tmp_storage, embedding_provider, llm_provider
):
    org = register_org(client, "assistant-grounded@example.com", "Assistant Org")
    doc = upload_pdf(client, org, [GPA_TEXT], filename="Scholarship_Rules.pdf")
    assistant = _create_assistant(client, org)

    response = _ask(client, assistant["assistant_token"], "What GPA is required?")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] == llm_provider.answer
    assert body["citations"] == [
        {
            "document_id": doc["id"],
            "document_name": "Scholarship_Rules.pdf",
            "page_number": 1,
            "citation": "Scholarship_Rules.pdf — Page 1",
        }
    ]
    assert len(llm_provider.calls) == 1
    assert GPA_TEXT in llm_provider.calls[0][1]


def test_public_assistant_url_token_can_be_passed_as_query_parameter(
    client, tmp_storage, embedding_provider, llm_provider
):
    org = register_org(client, "assistant-url@example.com", "Assistant Org")
    upload_pdf(client, org, [GPA_TEXT])
    assistant = _create_assistant(client, org)

    response = _ask(client, assistant["assistant_token"], "What GPA is required?", query_token=True)
    assert response.status_code == 200


def test_assistant_url_opens_public_visitor_page_and_rejects_rotated_token(client):
    org = register_org(client, "assistant-visitor-page@example.com", "Assistant Org")
    created = _create_assistant(client, org)

    page = client.get(created["assistant_url"])
    assert page.status_code == 200
    assert "What would you like to know?" in page.text
    assert "citation" in page.text.lower()
    assert page.headers["referrer-policy"] == "no-referrer"

    rotated = client.post("/api/assistant/regenerate", headers=org.headers).json()
    expired_page = client.get(created["assistant_url"])
    assert "This assistant link is no longer available" in expired_page.text
    assert expired_page.status_code == 401
    assert client.get(rotated["assistant_url"]).status_code == 200


def test_insufficient_context_does_not_call_llm(client, tmp_storage, embedding_provider, llm_provider):
    org = register_org(client, "assistant-empty@example.com", "Assistant Org")
    upload_pdf(client, org, [GPA_TEXT])
    assistant = _create_assistant(client, org)

    response = _ask(client, assistant["assistant_token"], "Explain quantum chromodynamics lattice theory")

    assert response.status_code == 200
    assert "not available" in response.json()["answer"].lower()
    assert response.json()["citations"] == []
    assert llm_provider.calls == []


def test_assistant_token_is_tenant_boundary_for_retrieval(
    client, tmp_storage, embedding_provider, llm_provider
):
    org_a = register_org(client, "assistant-tenant-a@example.com", "Org A")
    org_b = register_org(client, "assistant-tenant-b@example.com", "Org B")
    doc_a = upload_pdf(client, org_a, [GPA_TEXT], filename="A_Private.pdf")
    doc_b = upload_pdf(client, org_b, [LEAVE_TEXT], filename="B_Private.pdf")
    assistant_a = _create_assistant(client, org_a)
    assistant_b = _create_assistant(client, org_b)

    response_a = _ask(client, assistant_a["assistant_token"], GPA_TEXT)
    # A caller-supplied org id must not redirect B's assistant to A's corpus.
    response_b = client.post(
        "/api/ask",
        json={
            "assistant_token": assistant_b["assistant_token"],
            "question": GPA_TEXT,
            "organization_id": org_a.organization_id,
        },
    )
    response_b_own = _ask(client, assistant_b["assistant_token"], LEAVE_TEXT)

    assert [item["document_id"] for item in response_a.json()["citations"]] == [doc_a["id"]]
    assert response_b.json()["citations"] == []
    assert "not available" in response_b.json()["answer"].lower()
    assert [item["document_id"] for item in response_b_own.json()["citations"]] == [doc_b["id"]]
    contexts = [context for _question, context in llm_provider.calls]
    assert any(GPA_TEXT in context for context in contexts)
    assert any(LEAVE_TEXT in context for context in contexts)
    assert all("A_Private.pdf" not in context for context in contexts if LEAVE_TEXT in context)
    assert "A_Private.pdf" not in response_b.text


def test_invalid_assistant_token_is_rejected(client, embedding_provider, llm_provider):
    response = _ask(client, "not-a-valid-assistant-token", "question")
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid assistant token"}


def test_invalid_questions_are_rejected(client, embedding_provider, llm_provider):
    org = register_org(client, "assistant-question@example.com", "Assistant Org")
    assistant = _create_assistant(client, org)

    assert _ask(client, assistant["assistant_token"], " \n\t ").status_code == 422
    assert _ask(client, assistant["assistant_token"], "x" * 2001).status_code == 422


def test_missing_providers_return_safe_503_without_external_calls(client):
    org = register_org(client, "assistant-provider-missing@example.com", "Assistant Org")
    assistant = _create_assistant(client, org)
    response = _ask(client, assistant["assistant_token"], "Anything?")
    assert response.status_code == 503
    assert response.json() == {"detail": "Embedding provider is not configured"}


def test_missing_llm_provider_returns_safe_503(client, embedding_provider):
    app.dependency_overrides[get_llm_provider] = lambda: None
    org = register_org(client, "assistant-llm-missing@example.com", "Assistant Org")
    assistant = _create_assistant(client, org)
    response = _ask(client, assistant["assistant_token"], "Anything?")
    assert response.status_code == 503
    assert response.json() == {"detail": "Language model provider is not configured"}


def test_usage_counts_are_authenticated_and_organization_scoped(
    client, tmp_storage, embedding_provider, llm_provider
):
    org_a = register_org(client, "assistant-usage-a@example.com", "Usage Org A")
    org_b = register_org(client, "assistant-usage-b@example.com", "Usage Org B")
    upload_pdf(client, org_a, [GPA_TEXT], filename="Usage_Rules.pdf")
    assistant_a = _create_assistant(client, org_a)
    assistant_b = _create_assistant(client, org_b)

    answered = _ask(client, assistant_a["assistant_token"], "What GPA is required?")
    unavailable = _ask(client, assistant_a["assistant_token"], "Explain quantum chromodynamics")
    foreign_public_access = client.get(
        "/api/assistant/usage", headers={"Authorization": f"Bearer {assistant_a['assistant_token']}"}
    )
    unauthenticated = client.get("/api/assistant/usage")

    assert answered.status_code == 200
    assert unavailable.status_code == 200
    assert unauthenticated.status_code == 401
    assert foreign_public_access.status_code == 401
    assert client.get("/api/assistant/usage", headers=org_a.headers).json() == {
        "questions_asked": 2,
        "questions_answered": 1,
        "questions_unavailable": 1,
    }
    assert client.get("/api/assistant/usage", headers=org_b.headers).json() == {
        "questions_asked": 0,
        "questions_answered": 0,
        "questions_unavailable": 0,
    }
    assert assistant_b["assistant_id"] != assistant_a["assistant_id"]
    assert client.get(
        "/api/documents", headers={"Authorization": f"Bearer {assistant_a['assistant_token']}"}
    ).status_code == 401
    assert client.get(
        "/api/assistant", headers={"Authorization": f"Bearer {assistant_a['assistant_token']}"}
    ).status_code == 401


def test_llm_provider_failure_returns_safe_503(client, embedding_provider):
    app.dependency_overrides[get_llm_provider] = lambda: MockLLMProvider(error=True)
    org = register_org(client, "assistant-llm-error@example.com", "Assistant Org")
    upload_pdf(client, org, [GPA_TEXT])
    assistant = _create_assistant(client, org)
    response = _ask(client, assistant["assistant_token"], "What GPA is required?")
    assert response.status_code == 503
    assert response.json() == {"detail": "Assistant is temporarily unavailable"}
    assert "simulated" not in response.text


def test_document_delete_is_authenticated_and_organization_scoped(
    client, tmp_storage, embedding_provider
):
    org_a = register_org(client, "assistant-delete-a@example.com", "Org A")
    org_b = register_org(client, "assistant-delete-b@example.com", "Org B")
    document = upload_pdf(client, org_a, [GPA_TEXT], filename="delete.pdf")

    unauthorized = client.delete(f"/api/documents/{document['id']}")
    cross_tenant = client.delete(
        f"/api/documents/{document['id']}", headers=org_b.headers
    )
    assert unauthorized.status_code == 401
    assert cross_tenant.status_code == 404

    deleted = client.delete(f"/api/documents/{document['id']}", headers=org_a.headers)
    assert deleted.status_code == 204
    assert client.get("/api/documents", headers=org_a.headers).json() == []
    assert list(tmp_storage.root_dir.rglob("*.pdf")) == []
