"""add assistant usage counts

Revision ID: 8c2f7a1d4b60
Revises: 1b8a20a4c113
Create Date: 2026-09-28

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "8c2f7a1d4b60"
down_revision: Union[str, None] = "1b8a20a4c113"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("assistants", sa.Column("questions_asked", sa.Integer(), server_default="0", nullable=False))
    op.add_column("assistants", sa.Column("questions_answered", sa.Integer(), server_default="0", nullable=False))
    op.add_column("assistants", sa.Column("questions_unavailable", sa.Integer(), server_default="0", nullable=False))


def downgrade() -> None:
    op.drop_column("assistants", "questions_unavailable")
    op.drop_column("assistants", "questions_answered")
    op.drop_column("assistants", "questions_asked")
