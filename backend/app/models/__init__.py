"""
Model registration point.

Importing this package (directly, or transitively by importing any
app.models.<name> submodule -- Python always runs a package's __init__.py
before its submodules) registers every table on Base.metadata in a fixed,
dependency-safe order: Organization before the tables that reference it,
Document before DocumentPage/DocumentChunk.

This is also what Alembic's env.py imports (alongside Base) so that
--autogenerate can see the full schema. Every new model added in a later
phase (Assistant in Phase 10, etc.) gets one import line added here, and
nowhere else.
"""

from app.models.organization import Organization
from app.models.user import User
from app.models.membership import Membership, MembershipRole
from app.models.document import Document, DocumentStatus
from app.models.document_page import DocumentPage
from app.models.document_chunk import DocumentChunk

__all__ = [
    "Organization",
    "User",
    "Membership",
    "MembershipRole",
    "Document",
    "DocumentStatus",
    "DocumentPage",
    "DocumentChunk",
]
