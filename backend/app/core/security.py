"""
Password hashing and JWT helpers.

Hashing: argon2-cffi used directly (not passlib — passlib is effectively
unmaintained and has known friction with newer bcrypt/argon2 releases; for a
single-algorithm choice that's already decided (Argon2id, per the approved
design), the extra abstraction layer buys nothing). PasswordHasher()'s
defaults are Argon2id with OWASP-reasonable cost parameters out of the box.

JWT: pyjwt, HS256, minimal claims. `sub` is the user's id — nothing else.
Organization is deliberately NOT embedded in the token; every request re-derives
it from the memberships table (see app/auth/dependencies.py). This costs one
indexed query per request and buys a single, auditable place where
"user -> organization" is decided.
"""

from datetime import datetime, timedelta, timezone

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from app.core.config import settings

_password_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    return _password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _password_hasher.verify(password_hash, password)
    except VerifyMismatchError:
        return False


def create_access_token(subject: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "iat": now,
        "exp": now + timedelta(minutes=settings.access_token_expire_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> str | None:
    """Returns the subject (user id, as a string) or None if invalid/expired/tampered."""
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError:
        return None
    return payload.get("sub")
