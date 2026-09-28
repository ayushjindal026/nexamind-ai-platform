"""
Retrieval tests: organization-scoped vector similarity search.

Ranking assertions rely on word overlap (the test embedder is a hashing
embedder, not a semantic model — see tests/embedding_fixtures.py). Tenant
isolation assertions are embedder-independent: they hold for any embedder.
"""

import uuid

from app.embeddings.provider import get_embedding_provider
from app.main import app
from app.models.document import Document, DocumentStatus
from app.retrieval.service import search_similar_chunks
from tests.embedding_fixtures import FailingEmbeddingProvider
from tests.rag_helpers import GPA_TEXT, HOSTEL_TEXT, LEAVE_TEXT, register_org, search, upload_pdf


# --- ranking & result shape ---


def test_search_ranks_the_relevant_chunk_first_with_page_citation(client, tmp_storage, embedding_provider):
    org = register_org(client, "a@acme-university.com", "Acme University")
    upload_pdf(client, org, [LEAVE_TEXT, GPA_TEXT, HOSTEL_TEXT], filename="Scholarship_Rules.pdf")

    response = search(client, org, "minimum GPA scholarship requirement")

    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 3
    top = results[0]
    assert top["page_number"] == 2  # the GPA page, not first in the document
    assert top["document_name"] == "Scholarship_Rules.pdf"
    assert top["citation"] == "Scholarship_Rules.pdf — Page 2"
    assert GPA_TEXT in top["text"]
    similarities = [r["similarity"] for r in results]
    assert similarities == sorted(similarities, reverse=True)
    assert top["similarity"] > results[1]["similarity"]


def test_identical_text_has_near_perfect_similarity(client, tmp_storage, embedding_provider):
    org = register_org(client, "b@acme-university.com", "Acme University")
    upload_pdf(client, org, [GPA_TEXT])

    results = search(client, org, GPA_TEXT).json()["results"]

    assert results[0]["similarity"] > 0.999


def test_top_k_limits_results(client, tmp_storage, embedding_provider):
    org = register_org(client, "c@acme-university.com", "Acme University")
    upload_pdf(client, org, [GPA_TEXT, LEAVE_TEXT, HOSTEL_TEXT])

    results = search(client, org, "policy", top_k=2).json()["results"]

    assert len(results) == 2


def test_min_similarity_threshold_excludes_irrelevant_chunks(client, tmp_storage, embedding_provider):
    """The relevance-threshold hook later phases use to say 'not in the knowledge base'."""
    org = register_org(client, "d@acme-university.com", "Acme University")
    upload_pdf(client, org, [GPA_TEXT, LEAVE_TEXT, HOSTEL_TEXT])

    unrelated = search(client, org, "quantum chromodynamics lattice", min_similarity=0.3)
    related = search(client, org, "minimum GPA scholarship", min_similarity=0.3)

    assert unrelated.json()["results"] == []
    assert len(related.json()["results"]) == 1
    assert related.json()["results"][0]["page_number"] == 1


def test_search_with_no_documents_returns_empty_list(client, tmp_storage, embedding_provider):
    org = register_org(client, "e@acme-university.com", "Acme University")

    response = search(client, org, "anything at all")

    assert response.status_code == 200
    assert response.json() == {"results": []}


# --- chunks that are not yet embedded / documents that aren't ready ---


def test_unembedded_chunks_are_invisible_to_search(client, tmp_storage, embedding_provider):
    org = register_org(client, "f@acme-university.com", "Acme University")
    upload_pdf(client, org, [GPA_TEXT], filename="embedded.pdf")

    # Second upload happens while the provider is "down": chunks stay pending.
    app.dependency_overrides[get_embedding_provider] = lambda: FailingEmbeddingProvider()
    pending = upload_pdf(client, org, [GPA_TEXT], filename="pending.pdf")
    assert pending["embedded_chunk_count"] == 0
    app.dependency_overrides[get_embedding_provider] = lambda: embedding_provider

    results = search(client, org, GPA_TEXT).json()["results"]

    assert [r["document_name"] for r in results] == ["embedded.pdf"]  # pending.pdf skipped, no error


def test_chunks_of_non_ready_documents_are_not_searchable(client, tmp_storage, embedding_provider, db_session):
    org = register_org(client, "g@acme-university.com", "Acme University")
    body = upload_pdf(client, org, [GPA_TEXT])
    document = db_session.get(Document, uuid.UUID(body["id"]))
    document.status = DocumentStatus.failed
    db_session.commit()

    assert search(client, org, GPA_TEXT).json()["results"] == []


# --- tenant isolation ---


def test_service_level_search_is_scoped_to_the_given_organization(client, tmp_storage, embedding_provider, db_session):
    org_a = register_org(client, "h@acme-university.com", "Acme University")
    org_b = register_org(client, "i@globaltech-institute.com", "Global Tech Institute")
    # Identical content in both tenants: only organization scoping can separate them.
    doc_a = upload_pdf(client, org_a, [GPA_TEXT], filename="acme.pdf")
    doc_b = upload_pdf(client, org_b, [GPA_TEXT], filename="globaltech.pdf")
    query_vector = embedding_provider.embed([GPA_TEXT])[0]

    results_a = search_similar_chunks(db_session, uuid.UUID(org_a.organization_id), query_vector)
    results_b = search_similar_chunks(db_session, uuid.UUID(org_b.organization_id), query_vector)

    assert [str(r.document_id) for r in results_a] == [doc_a["id"]]
    assert [str(r.document_id) for r in results_b] == [doc_b["id"]]


def test_service_search_for_an_organization_with_no_chunks_returns_nothing(client, tmp_storage, embedding_provider, db_session):
    org_a = register_org(client, "j@acme-university.com", "Acme University")
    upload_pdf(client, org_a, [GPA_TEXT])
    query_vector = embedding_provider.embed([GPA_TEXT])[0]

    results = search_similar_chunks(db_session, uuid.uuid4(), query_vector)  # unknown org

    assert results == []


def test_org_b_cannot_retrieve_org_a_knowledge_over_http(client, tmp_storage, embedding_provider):
    """The Phase 4 cross-tenant guarantee, at the HTTP level."""
    org_a = register_org(client, "k@acme-university.com", "Acme University")
    org_b = register_org(client, "l@globaltech-institute.com", "Global Tech Institute")
    acme_doc = upload_pdf(client, org_a, [GPA_TEXT], filename="Acme_Scholarship_Rules.pdf")
    globaltech_doc = upload_pdf(client, org_b, ["Internship policy requires a signed offer letter"], filename="Internship_Policy.pdf")

    # B asks about Acme's exact content: must get B's own (unrelated) chunk at most, never Acme's.
    response = search(client, org_b, GPA_TEXT)

    assert response.status_code == 200
    body = response.json()
    returned_documents = {r["document_id"] for r in body["results"]}
    assert acme_doc["id"] not in returned_documents
    assert returned_documents <= {globaltech_doc["id"]}
    assert "Acme_Scholarship_Rules.pdf" not in response.text
    assert GPA_TEXT not in response.text

    # And Acme still finds its own document.
    own = search(client, org_a, GPA_TEXT).json()["results"]
    assert [r["document_id"] for r in own] == [acme_doc["id"]]


def test_client_supplied_organization_id_is_ignored(client, tmp_storage, embedding_provider):
    org_a = register_org(client, "m@acme-university.com", "Acme University")
    org_b = register_org(client, "n@globaltech-institute.com", "Global Tech Institute")
    upload_pdf(client, org_a, [GPA_TEXT], filename="acme.pdf")

    # B tries to smuggle A's organization_id in the body (and query string).
    response = client.post(
        f"/api/retrieval/search?organization_id={org_a.organization_id}",
        headers=org_b.headers,
        json={"query": GPA_TEXT, "organization_id": org_a.organization_id},
    )

    assert response.status_code == 200
    assert response.json()["results"] == []


# --- auth & validation & failure handling ---


def test_search_requires_authentication(client, embedding_provider):
    response = client.post("/api/retrieval/search", json={"query": "anything"})
    assert response.status_code == 401


def test_blank_and_missing_queries_are_rejected(client, tmp_storage, embedding_provider):
    org = register_org(client, "o@acme-university.com", "Acme University")

    assert search(client, org, "").status_code == 422
    assert search(client, org, "   \n\t ").status_code == 422
    assert client.post("/api/retrieval/search", headers=org.headers, json={}).status_code == 422


def test_top_k_bounds_are_enforced(client, tmp_storage, embedding_provider):
    org = register_org(client, "p@acme-university.com", "Acme University")

    assert search(client, org, "policy", top_k=0).status_code == 422
    assert search(client, org, "policy", top_k=21).status_code == 422


def test_search_without_a_configured_provider_returns_503(client, tmp_storage):
    # No `embedding_provider` fixture -> conftest default (provider=None).
    org = register_org(client, "q@acme-university.com", "Acme University")

    response = search(client, org, "policy")

    assert response.status_code == 503
    assert response.json() == {"detail": "Embedding provider is not configured"}


def test_provider_failure_returns_generic_503_without_leaking_details(client, tmp_storage):
    app.dependency_overrides[get_embedding_provider] = lambda: FailingEmbeddingProvider()
    org = register_org(client, "r@acme-university.com", "Acme University")

    response = search(client, org, "policy")

    assert response.status_code == 503
    assert response.json() == {"detail": "Embedding service is temporarily unavailable"}
    assert "simulated" not in response.text  # raw provider error never reaches the client
