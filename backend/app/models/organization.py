"""
Organization: the tenant boundary itself.

Deliberately no `slug` column — nothing in the current scope reads or writes
one. The assistant's opaque token (Phase 10) is the only "public identifier"
this assignment calls for; organizations are otherwise only referenced
internally, by UUID, via membership. Adding it later is a one-line migration
if a real use ever appears (e.g. a branded org page).
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Organization(Base):
    __tablename__ = "organizations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    memberships: Mapped[list["Membership"]] = relationship(  # noqa: F821
        back_populates="organization",
        cascade="all, delete-orphan",
    )
    documents: Mapped[list["Document"]] = relationship(  # noqa: F821
        back_populates="organization",
        cascade="all, delete-orphan",
    )
