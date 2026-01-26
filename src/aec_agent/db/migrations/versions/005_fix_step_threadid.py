"""
Fix Step schema threadId.

Makes "threadId" column nullable in "Step" table to match Chainlit behavior.

Revision ID: 005_fix_step_threadid
Revises: 004_fix_step_schema
Create Date: 2024-01-26
"""

from alembic import op
import sqlalchemy as sa


# Revision identifiers
revision = "005_fix_step_threadid"
down_revision = "004_fix_step_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Make Step.threadId nullable."""
    op.execute('ALTER TABLE "Step" ALTER COLUMN "threadId" DROP NOT NULL;')


def downgrade() -> None:
    """Make Step.threadId required."""
    # Note: This might fail if there are rows with NULL threadIds
    op.execute('ALTER TABLE "Step" ALTER COLUMN "threadId" SET NOT NULL;')
