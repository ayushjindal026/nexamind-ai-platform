"""
User: dashboard login identity.

Not tenant-owned itself — a user's relationship to an organization exists
only through Membership (see membership.py). Email is normalized to
lowercase by application code (app/auth/service.py) before every insert
and lookup; the UNIQUE constraint here is the DB-level backstop for that
invariant, not a substitute for it.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # One-to-one for now: Membership.user_id carries a UNIQUE constraint enforcing
    # the current "one user -> one organization" invariant (see membership.py).
    membership: Mapped["Membership"] = relationship(  # noqa: F821
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
    )
