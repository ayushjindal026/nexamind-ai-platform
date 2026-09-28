"""
Membership: the only path from a User to an Organization.

This indirection — never a direct organization_id column on User — is what
makes "authenticated user -> membership lookup -> server-derived org_id"
possible as a real query instead of a convention.

Current MVP invariant: one user has exactly ONE membership (one user, one
organization). This is enforced at the schema level by the UNIQUE constraint
on `user_id` alone (not a composite (user_id, organization_id) constraint,
which would only prevent duplicate rows for the *same* org and would silently
allow multiple different orgs per user). If multi-organization membership is
ever implemented, the single schema change required is relaxing this to
UNIQUE(user_id, organization_id) — and app/auth/dependencies.py's
get_current_membership() would need to change from "the one membership" to
"which membership for this request," which is a real design decision, not
a trivial follow-up. Nothing in the current codebase assumes that future
shape; it's called out here so it isn't discovered by surprise.

`role` is stored but not yet enforced anywhere: every organization in this
MVP has exactly one member (created as `owner` at registration), so no
endpoint branches on role. See the Phase 2 design notes for the intended
(not-yet-implemented) meaning of admin/member.
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class MembershipRole(str, enum.Enum):
    owner = "owner"
    admin = "admin"
    member = "member"


class Membership(Base):
    __tablename__ = "memberships"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,  # enforces "exactly one membership per user" — see module docstring
    )
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[MembershipRole] = mapped_column(
        Enum(MembershipRole, name="membership_role", native_enum=True),
        nullable=False,
        default=MembershipRole.owner,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="membership")  # noqa: F821
    organization: Mapped["Organization"] = relationship(back_populates="memberships")  # noqa: F821
