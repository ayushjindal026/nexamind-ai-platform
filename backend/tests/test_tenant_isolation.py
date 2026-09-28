"""
Tenant-resolution tests for Phase 2.

Phase 2 has no tenant-owned resources yet (Document arrives in Phase 3), so
there is nothing to attempt cross-tenant *access* to. What Phase 2 can and
must prove is the foundation that resource-level isolation will be built on:
that each user's server-derived organization_id is correct and never
crosses over. The actual "User A cannot fetch Org B's document by UUID"
test is written in Phase 3's test_tenant_isolation.py additions, using this
same registration pattern.
"""


def _register(client, email, org_name):
    response = client.post(
        "/api/auth/register",
        json={"email": email, "password": "correct-horse-battery", "organization_name": org_name},
    )
    assert response.status_code == 201
    return response.json()


def test_each_user_resolves_only_their_own_organization(client):
    org_a = _register(client, "usera@acme-university.com", "Acme University")
    org_b = _register(client, "userb@globaltech-institute.com", "Global Tech Institute")

    me_a = client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {org_a['access_token']}"}
    ).json()
    me_b = client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {org_b['access_token']}"}
    ).json()

    assert me_a["organization"]["id"] == org_a["organization"]["id"]
    assert me_b["organization"]["id"] == org_b["organization"]["id"]

    # The actual isolation claim: A's token never resolves to B's org, and vice versa.
    assert me_a["organization"]["id"] != me_b["organization"]["id"]
    assert me_a["organization"]["name"] == "Acme University"
    assert me_b["organization"]["name"] == "Global Tech Institute"


def test_organization_id_is_not_accepted_from_the_client(client):
    """
    /api/auth/me takes no input at all — this test documents, rather than
    merely asserts, that there is no field anywhere in this API for a client
    to supply an organization_id. If a future phase ever adds one, this test
    breaking is the intended signal to reconsider that change.
    """
    registered = _register(client, "eve@acme-university.com", "Eve's Org")
    token = registered["access_token"]

    # Sending an organization_id has no endpoint to land on for GET /me —
    # query params are simply ignored by FastAPI when the route declares none.
    response = client.get(
        "/api/auth/me?organization_id=00000000-0000-0000-0000-000000000000",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["organization"]["id"] == registered["organization"]["id"]


def test_membership_invariant_one_user_one_organization(client, db_session):
    """
    Confirms the schema-level guarantee (UNIQUE constraint on
    memberships.user_id) that get_current_membership() relies on: exactly one
    membership row exists per registered user.
    """
    from app.models.membership import Membership
    from app.models.user import User

    registered = _register(client, "frank@acme-university.com", "Frank's Org")

    user = db_session.query(User).filter(User.email == "frank@acme-university.com").one()
    memberships = db_session.query(Membership).filter(Membership.user_id == user.id).all()

    assert len(memberships) == 1
    assert str(memberships[0].organization_id) == registered["organization"]["id"]
    assert memberships[0].role.value == "owner"
