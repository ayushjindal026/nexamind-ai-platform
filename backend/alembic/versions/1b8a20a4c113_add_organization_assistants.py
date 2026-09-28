"""add organization-scoped assistants

Revision ID: 1b8a20a4c113
Revises: 4c37d4c06710
Create Date: 2026-09-28

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "1b8a20a4c113"
down_revision: Union[str, None] = "4c37d4c06710"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "assistants",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("organization_id", sa.UUID(), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id"),
        sa.UniqueConstraint("token_hash"),
    )
    op.create_index("ix_assistants_token_hash", "assistants", ["token_hash"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_assistants_token_hash", table_name="assistants")
    op.drop_table("assistants")
