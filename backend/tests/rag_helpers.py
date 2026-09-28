"""Shared setup helpers for embedding/retrieval tests (HTTP-level, same as real clients)."""

import io
from dataclasses import dataclass

from tests.pdf_fixtures import make_pdf_bytes

GPA_TEXT = "Scholarship eligibility requires a minimum GPA of 3.0"
LEAVE_TEXT = "Staff leave policy allows twenty vacation days annually"
HOSTEL_TEXT = "Hostel curfew is ten pm for all residents"


@dataclass
class Org:
    token: str
    organization_id: str
    headers: dict


def register_org(client, email: str, org_name: str) -> Org:
    response = client.post(
        "/api/auth/register",
        json={"email": email, "password": "correct-horse-battery", "organization_name": org_name},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    return Org(
        token=body["access_token"],
        organization_id=body["organization"]["id"],
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )


def upload_pdf(client, org: Org, pages: list[str], filename: str = "policy.pdf") -> dict:
    response = client.post(
        "/api/documents",
        headers=org.headers,
        files={"file": (filename, io.BytesIO(make_pdf_bytes(pages)), "application/pdf")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def search(client, org: Org, query: str, **extra):
    return client.post("/api/retrieval/search", headers=org.headers, json={"query": query, **extra})
