"""
Fix Step schema.

Makes "name" column nullable in "Step" table to match Chainlit behavior.

Revision ID: 004_fix_step_schema
Revises: 003_chainlit_schema
Create Date: 2024-01-26
"""

from alembic import op

# Revision identifiers
revision = "004_fix_step_schema"
down_revision = "003_chainlit_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Make Step.name nullable."""
    op.execute('ALTER TABLE "Step" ALTER COLUMN "name" DROP NOT NULL;')


def downgrade() -> None:
    """Make Step.name required."""
    # Note: This might fail if there are rows with NULL names
    op.execute('ALTER TABLE "Step" ALTER COLUMN "name" SET NOT NULL;')
