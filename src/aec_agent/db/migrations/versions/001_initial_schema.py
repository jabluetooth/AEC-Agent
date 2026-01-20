"""
Initial schema with PostGIS and pgvector.

Creates projects, elements, and element_relationships tables
with spatial and vector indexes.

Revision ID: 001_initial_schema
Revises: None
Create Date: 2024-01-20
"""

from alembic import op
import sqlalchemy as sa

# Revision identifiers
revision = "001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial schema."""

    # Enable extensions
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Projects table
    op.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name TEXT NOT NULL,
            source TEXT NOT NULL CHECK (source IN ('autocad', 'revit')),
            file_path TEXT,
            file_hash TEXT,
            extracted_at TIMESTAMPTZ,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    # Elements table
    op.execute("""
        CREATE TABLE IF NOT EXISTS elements (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            source_id TEXT NOT NULL,
            source TEXT NOT NULL CHECK (source IN ('autocad', 'revit')),

            -- Classification
            entity_type TEXT NOT NULL,
            layer TEXT,
            category TEXT,
            family TEXT,
            type_name TEXT,

            -- Geometry (PostGIS)
            geom GEOMETRY(GeometryZ, 0),
            centroid GEOMETRY(PointZ, 0),

            -- Properties
            properties JSONB DEFAULT '{}',

            -- Semantic search (pgvector)
            description TEXT,
            embedding vector(384),

            -- Lifecycle
            deleted_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),

            UNIQUE(project_id, source_id, source)
        );
    """)

    # Element relationships table
    op.execute("""
        CREATE TABLE IF NOT EXISTS element_relationships (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            from_element_id UUID NOT NULL REFERENCES elements(id) ON DELETE CASCADE,
            to_element_id UUID NOT NULL REFERENCES elements(id) ON DELETE CASCADE,

            relation_type TEXT NOT NULL,
            distance FLOAT,
            confidence FLOAT DEFAULT 1.0,
            source TEXT DEFAULT 'computed',
            metadata JSONB DEFAULT '{}',

            created_at TIMESTAMPTZ DEFAULT NOW(),

            UNIQUE(project_id, from_element_id, to_element_id, relation_type)
        );
    """)

    # Indexes for spatial queries (PostGIS GIST)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_elements_geom
        ON elements USING GIST (geom);
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_elements_centroid
        ON elements USING GIST (centroid);
    """)

    # Index for semantic search (pgvector HNSW)
    # HNSW is preferred over IVFFlat as it doesn't require training
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_elements_embedding
        ON elements USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)

    # Index for JSONB property queries
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_elements_properties
        ON elements USING GIN (properties);
    """)

    # Indexes for relationships
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_rel_from
        ON element_relationships(from_element_id);
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_rel_to
        ON element_relationships(to_element_id);
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_rel_type
        ON element_relationships(relation_type);
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_rel_project
        ON element_relationships(project_id);
    """)

    # Index for project lookups
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_projects_file_path
        ON projects(file_path);
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_projects_file_hash
        ON projects(file_hash);
    """)

    # Index for element queries
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_elements_project
        ON elements(project_id);
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_elements_entity_type
        ON elements(entity_type);
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_elements_layer
        ON elements(layer);
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_elements_category
        ON elements(category);
    """)

    # Partial index for non-deleted elements
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_elements_active
        ON elements(project_id, entity_type)
        WHERE deleted_at IS NULL;
    """)


def downgrade() -> None:
    """Drop all tables."""
    op.execute("DROP TABLE IF EXISTS element_relationships CASCADE;")
    op.execute("DROP TABLE IF EXISTS elements CASCADE;")
    op.execute("DROP TABLE IF EXISTS projects CASCADE;")
    # Note: We don't drop extensions as other databases might use them
