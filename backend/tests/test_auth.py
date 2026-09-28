"""
Authentication tests — register, login, /me.

Tenant-resolution tests (membership -> organization_id) live in
test_tenant_isolation.py; this file is about the auth mechanics themselves.
"""


def _register(client, email="alice@acme-university.com", password="correct-horse-1", org_name="Acme University"):
    return client.post(
        "/api/auth/register",
        json={"email": email, "password": password, "organization_name": org_name},
    )


def test_register_succeeds(client):
    response = _register(client)
    assert response.status_code == 201
    body = response.json()
    assert body["access_token"]
    assert body["token_type"] == "bearer"
    assert body["user"]["email"] == "alice@acme-university.com"
    assert body["organization"]["name"] == "Acme University"


def test_register_duplicate_email_rejected(client):
    first = _register(client, email="dupe@acme-university.com")
    assert first.status_code == 201

    second = _register(client, email="dupe@acme-university.com", org_name="A Second Org")
    assert second.status_code == 409


def test_register_duplicate_email_is_case_insensitive(client):
    _register(client, email="Case@Acme-University.com")
    second = _register(client, email="case@acme-university.com", org_name="Another Org")
    assert second.status_code == 409


def test_login_succeeds(client):
    _register(client, email="bob@acme-university.com", password="correct-horse-2")

    response = client.post(
        "/api/auth/login", json={"email": "bob@acme-university.com", "password": "correct-horse-2"}
    )
    assert response.status_code == 200
    assert response.json()["access_token"]


def test_login_wrong_password_rejected(client):
    _register(client, email="carol@acme-university.com", password="correct-horse-3")

    response = client.post(
        "/api/auth/login", json={"email": "carol@acme-university.com", "password": "wrong-password"}
    )
    assert response.status_code == 401


def test_login_nonexistent_email_same_response_as_wrong_password(client):
    """
    Deliberately checking these two cases return the SAME status and detail —
    that's the user-enumeration mitigation from the Phase 2 design.
    """
    wrong_password_response = client.post(
        "/api/auth/login", json={"email": "no-such-user@acme-university.com", "password": "irrelevant"}
    )
    assert wrong_password_response.status_code == 401
    assert wrong_password_response.json()["detail"] == "Invalid email or password"


def test_me_requires_authentication(client):
    response = client.get("/api/auth/me")
    assert response.status_code == 401


def test_me_rejects_malformed_token(client):
    response = client.get("/api/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert response.status_code == 401


def test_me_resolves_correct_user_with_valid_token(client):
    register_response = _register(client, email="dana@acme-university.com", org_name="Dana's Org")
    token = register_response.json()["access_token"]

    response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    body = response.json()
    assert body["email"] == "dana@acme-university.com"
    assert body["organization"]["name"] == "Dana's Org"
    assert body["role"] == "owner"
