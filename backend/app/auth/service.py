"""
Auth business logic, kept separate from the router so the router stays a
thin HTTP-status translation layer (see auth/router.py).
"""

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.security import hash_password, verify_password
from app.models.membership import Membership, MembershipRole
from app.models.organization import Organization
from app.models.user import User


class EmailAlreadyRegisteredError(Exception):
    pass


class InvalidCredentialsError(Exception):
    pass


def register_user(
    db: Session, email: str, password: str, organization_name: str
) -> tuple[User, Organization]:
    """
    Creates User + Organization + Membership(role=owner) atomically.

    Every organization in this MVP is created this way — exactly one owner
    membership, at registration time. There is no separate "create org" or
    "invite member" flow (see membership.py's module docstring for why).
    """
    normalized_email = email.strip().lower()

    # Fast-path check for a friendly error before hitting the DB constraint.
    # This is a courtesy, not the actual guarantee — see the IntegrityError
    # handling below, which is the real backstop against a race between two
    # concurrent registrations for the same email.
    if db.query(User).filter(User.email == normalized_email).first() is not None:
        raise EmailAlreadyRegisteredError()

    user = User(email=normalized_email, password_hash=hash_password(password))
    organization = Organization(name=organization_name)
    membership = Membership(user=user, organization=organization, role=MembershipRole.owner)

    db.add_all([user, organization, membership])
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise EmailAlreadyRegisteredError() from None

    db.refresh(user)
    db.refresh(organization)
    return user, organization


def authenticate_user(db: Session, email: str, password: str) -> User:
    """
    Raises InvalidCredentialsError for both "no such user" and "wrong password" —
    deliberately the same exception and the same eventual HTTP response, so the
    API doesn't reveal which case occurred (see Phase 2 design, Section 8).
    """
    normalized_email = email.strip().lower()
    user = db.query(User).filter(User.email == normalized_email).first()
    if user is None or not verify_password(password, user.password_hash):
        raise InvalidCredentialsError()
    return user
