"""
Symbol library table for RAG-based symbol recognition.

Stores CAD symbol images with CLIP/DINOv2 embeddings for
visual similarity search during PDF vectorization.

Revision ID: 006_symbol_library
Revises: 005_fix_step_threadid
Create Date: 2026-02-26
"""

from alembic import op

# Revision identifiers
revision = "006_symbol_library"
down_revision = "005_fix_step_threadid"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create symbol library table with vector embeddings."""

    # Symbol library table - stores reference symbols for RAG lookup
    op.execute("""
        CREATE TABLE IF NOT EXISTS symbol_library (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

            -- Symbol identification
            block_name TEXT NOT NULL,
            display_name TEXT NOT NULL,
            description TEXT,

            -- Classification hierarchy
            domain TEXT NOT NULL,  -- electrical, mechanical, plumbing, fire, architectural
            category TEXT NOT NULL,  -- outlet, switch, diffuser, valve, detector, door, etc.
            subcategory TEXT,  -- duplex, gfci, supply_square, gate, smoke, etc.

            -- NCS/AIA layer mapping
            layer TEXT NOT NULL,  -- e.g., E-POWR-OUTL, M-DIFF, F-FIRE-ALARM

            -- Visual representation
            preview_image BYTEA,  -- PNG thumbnail (64x64 or 128x128)
            source_dwg_path TEXT,  -- Path to source DWG block definition

            -- Vector embedding for similarity search (CLIP ViT-B/32 = 512 dims)
            embedding vector(512),

            -- Symbol properties
            default_scale FLOAT DEFAULT 1.0,
            default_rotation FLOAT DEFAULT 0.0,
            attributes JSONB DEFAULT '{}',  -- Default attribute values

            -- Metadata
            standards TEXT[],  -- ['NCS', 'AIA', 'ASHRAE']
            jurisdiction TEXT,  -- 'LA', 'CA', 'National'
            code_references TEXT[],  -- ['NFPA 72', 'CEC 210.52']

            -- Status
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),

            UNIQUE(block_name, domain)
        );
    """)

    # HNSW index for fast similarity search
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_symbol_library_embedding
        ON symbol_library USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)

    # Indexes for filtering
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_symbol_library_domain
        ON symbol_library(domain);
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_symbol_library_category
        ON symbol_library(domain, category);
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_symbol_library_layer
        ON symbol_library(layer);
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_symbol_library_active
        ON symbol_library(domain, category)
        WHERE is_active = TRUE;
    """)

    # Symbol aliases table - alternative names for symbols
    op.execute("""
        CREATE TABLE IF NOT EXISTS symbol_aliases (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            symbol_id UUID NOT NULL REFERENCES symbol_library(id) ON DELETE CASCADE,
            alias TEXT NOT NULL,
            alias_type TEXT DEFAULT 'common',  -- common, manufacturer, legacy
            created_at TIMESTAMPTZ DEFAULT NOW(),

            UNIQUE(symbol_id, alias)
        );
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_symbol_aliases_alias
        ON symbol_aliases(LOWER(alias));
    """)

    # Symbol usage tracking - for learning which symbols are commonly used
    op.execute("""
        CREATE TABLE IF NOT EXISTS symbol_usage (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            symbol_id UUID NOT NULL REFERENCES symbol_library(id) ON DELETE CASCADE,
            project_id UUID REFERENCES projects(id) ON DELETE SET NULL,

            -- Recognition context
            confidence FLOAT NOT NULL,
            image_hash TEXT,  -- Hash of the input image patch

            -- Feedback
            was_correct BOOLEAN,  -- User feedback
            correct_symbol_id UUID REFERENCES symbol_library(id),  -- If incorrect, what was correct

            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_symbol_usage_symbol
        ON symbol_usage(symbol_id);
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_symbol_usage_project
        ON symbol_usage(project_id);
    """)


def downgrade() -> None:
    """Drop symbol library tables."""
    op.execute("DROP TABLE IF EXISTS symbol_usage CASCADE;")
    op.execute("DROP TABLE IF EXISTS symbol_aliases CASCADE;")
    op.execute("DROP TABLE IF EXISTS symbol_library CASCADE;")
