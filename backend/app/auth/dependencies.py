"""
The dependency chain every authenticated route relies on.

    Authorization: Bearer <JWT>
        -> get_current_user       decode JWT -> load User row
        -> get_current_membership load that user's Membership row
                                  -> organization_id, role now known,
                                     entirely server-derived

No function here accepts an organization_id parameter from anywhere. That's
not an oversight to fix later — it's the entire point.
"""

import uuid

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.security import decode_access_token
from app.db.session import get_db
from app.models.membership import Membership
from app.models.user import User

# auto_error=False so a missing header raises our own 401 with a consistent
# body, instead of FastAPI's default 403 for "no credentials provided."
_bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    subject = decode_access_token(credentials.credentials)
    if subject is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    try:
        user_id = uuid.UUID(subject)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    return user


def get_current_membership(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Membership:
    """
    Returns the current user's single Membership row (organization_id, role).

    The current schema enforces exactly one membership per user via a UNIQUE
    constraint on memberships.user_id (see models/membership.py), so `.first()`
    is deterministic by construction — not just an assumption made here. If
    that invariant is ever violated by a future schema change without updating
    this function, `membership is None` will never fire for a genuinely
    missing membership vs. an ambiguous one; this function only ever returns
    exactly one row or 403s, it never guesses among several.
    """
    membership = db.query(Membership).filter(Membership.user_id == user.id).first()
    if membership is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no organization membership",
        )
    return membership
