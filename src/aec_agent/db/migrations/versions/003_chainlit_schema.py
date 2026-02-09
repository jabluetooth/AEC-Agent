"""
Chainlit schema.

Creates tables required by Chainlit's default data layer:
- "User"
- "Thread"
- "Step"
- "Element"
- "Feedback"

Revision ID: 003_chainlit_schema
Revises: 002_mep_enhancement
Create Date: 2024-01-26
"""

from alembic import op

# Revision identifiers
revision = "003_chainlit_schema"
down_revision = "002_mep_enhancement"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create Chainlit schema."""

    # User table
    op.execute("""
        CREATE TABLE IF NOT EXISTS "User" (
            "id" TEXT PRIMARY KEY,
            "identifier" TEXT NOT NULL UNIQUE,
            "metadata" JSONB DEFAULT '{}',
            "createdAt" TIMESTAMPTZ DEFAULT NOW(),
            "updatedAt" TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    # Thread table
    op.execute("""
        CREATE TABLE IF NOT EXISTS "Thread" (
            "id" TEXT PRIMARY KEY,
            "createdAt" TIMESTAMPTZ DEFAULT NOW(),
            "name" TEXT,
            "userId" TEXT REFERENCES "User"("id") ON DELETE CASCADE,
            "userIdentifier" TEXT,
            "tags" TEXT[],
            "metadata" JSONB DEFAULT '{}',
            "firstName" TEXT,
            "lastName" TEXT,
            "updatedAt" TIMESTAMPTZ DEFAULT NOW(),
            "deletedAt" TIMESTAMPTZ
        );
    """)

    # Step table
    op.execute("""
        CREATE TABLE IF NOT EXISTS "Step" (
            "id" TEXT PRIMARY KEY,
            "name" TEXT NOT NULL,
            "type" TEXT NOT NULL,
            "threadId" TEXT NOT NULL REFERENCES "Thread"("id") ON DELETE CASCADE,
            "parentId" TEXT REFERENCES "Step"("id") ON DELETE CASCADE,
            "disableFeedback" BOOLEAN DEFAULT FALSE,
            "streaming" BOOLEAN DEFAULT FALSE,
            "waitForAnswer" BOOLEAN DEFAULT FALSE,
            "isError" BOOLEAN DEFAULT FALSE,
            "showInput" TEXT,
            "input" TEXT,
            "output" TEXT,
            "metadata" JSONB DEFAULT '{}',
            "start" TIMESTAMPTZ,
            "end" TIMESTAMPTZ,
            "createdAt" TIMESTAMPTZ DEFAULT NOW(),
            "startTime" TIMESTAMPTZ,
            "endTime" TIMESTAMPTZ,
            "generation" JSONB,
            "language" TEXT
        );
    """)

    # Element table
    op.execute("""
        CREATE TABLE IF NOT EXISTS "Element" (
            "id" TEXT PRIMARY KEY,
            "threadId" TEXT REFERENCES "Thread"("id") ON DELETE CASCADE,
            "stepId" TEXT REFERENCES "Step"("id") ON DELETE CASCADE,
            "metadata" JSONB DEFAULT '{}',
            "mime" TEXT,
            "name" TEXT,
            "objectKey" TEXT,
            "url" TEXT,
            "chainlitKey" TEXT,
            "display" TEXT,
            "size" TEXT,
            "language" TEXT,
            "page" INTEGER,
            "props" JSONB DEFAULT '{}'
        );
    """)

    # Feedback table
    op.execute("""
        CREATE TABLE IF NOT EXISTS "Feedback" (
            "id" TEXT PRIMARY KEY,
            "forId" TEXT,
            "stepId" TEXT REFERENCES "Step"("id") ON DELETE CASCADE,
            "value" FLOAT NOT NULL,
            "comment" TEXT,
            "name" TEXT,
            "strategy" TEXT DEFAULT 'BINARY'
        );
    """)

    # Indexes
    op.execute('CREATE INDEX IF NOT EXISTS "idx_user_identifier" ON "User"("identifier");')
    op.execute('CREATE INDEX IF NOT EXISTS "idx_thread_user" ON "Thread"("userId");')
    op.execute('CREATE INDEX IF NOT EXISTS "idx_step_thread" ON "Step"("threadId");')
    op.execute('CREATE INDEX IF NOT EXISTS "idx_element_thread" ON "Element"("threadId");')


def downgrade() -> None:
    """Drop Chainlit tables."""
    op.execute('DROP TABLE IF EXISTS "Feedback" CASCADE;')
    op.execute('DROP TABLE IF EXISTS "Element" CASCADE;')
    op.execute('DROP TABLE IF EXISTS "Step" CASCADE;')
    op.execute('DROP TABLE IF EXISTS "Thread" CASCADE;')
    op.execute('DROP TABLE IF EXISTS "User" CASCADE;')
