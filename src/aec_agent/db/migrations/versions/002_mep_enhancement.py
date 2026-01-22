"""
MEP Enhancement schema - domain rules, workflows, memory, preferences.

Adds tables for:
- Phase 2: domain_rules, system_priorities
- Phase 3: workflow_templates, workflow_executions
- Phase 4: project_facts, conversation_summaries
- Phase 5: user_preferences, user_shortcuts
- Phase 6: validation_results, design_suggestions

Revision ID: 002_mep_enhancement
Revises: 001_initial_schema
Create Date: 2024-01-22
"""

from alembic import op

# Revision identifiers
revision = "002_mep_enhancement"
down_revision = "001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create MEP enhancement schema."""

    # =========================================================================
    # Phase 2: Domain Knowledge Base
    # =========================================================================

    # Domain rules table
    op.execute("""
        CREATE TABLE IF NOT EXISTS domain_rules (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            domain TEXT NOT NULL DEFAULT 'mep',
            subdomain TEXT,
            rule_type TEXT NOT NULL,
            rule_name TEXT NOT NULL,
            condition JSONB NOT NULL DEFAULT '{}',
            action JSONB NOT NULL DEFAULT '{}',
            priority INTEGER DEFAULT 100,
            source TEXT DEFAULT 'best_practice',
            description TEXT,
            embedding vector(384),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(domain, subdomain, rule_name)
        );
    """)

    # System priorities table
    op.execute("""
        CREATE TABLE IF NOT EXISTS system_priorities (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
            system_type TEXT NOT NULL,
            priority_rank INTEGER NOT NULL,
            notes TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(project_id, system_type)
        );
    """)

    # Indexes for domain rules
    op.execute("CREATE INDEX IF NOT EXISTS idx_rules_domain ON domain_rules(domain, subdomain);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_rules_type ON domain_rules(rule_type);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_rules_embedding ON domain_rules USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);")

    # =========================================================================
    # Phase 3: Workflow Templates
    # =========================================================================

    # Workflow templates table
    op.execute("""
        CREATE TABLE IF NOT EXISTS workflow_templates (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name TEXT NOT NULL UNIQUE,
            domain TEXT NOT NULL DEFAULT 'mep',
            subdomain TEXT,
            description TEXT,
            steps JSONB NOT NULL DEFAULT '[]',
            required_context JSONB DEFAULT '[]',
            default_params JSONB DEFAULT '{}',
            estimated_tokens INTEGER,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    # Workflow executions table
    op.execute("""
        CREATE TABLE IF NOT EXISTS workflow_executions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            template_id UUID REFERENCES workflow_templates(id) ON DELETE SET NULL,
            project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
            user_id TEXT,
            user_session TEXT,
            status TEXT DEFAULT 'pending',
            current_step INTEGER DEFAULT 0,
            context JSONB DEFAULT '{}',
            results JSONB DEFAULT '[]',
            error_message TEXT,
            started_at TIMESTAMPTZ DEFAULT NOW(),
            completed_at TIMESTAMPTZ
        );
    """)

    # Indexes for workflows
    op.execute("CREATE INDEX IF NOT EXISTS idx_workflow_templates_domain ON workflow_templates(domain, subdomain);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_workflow_executions_project ON workflow_executions(project_id);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_workflow_executions_status ON workflow_executions(status);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_workflow_executions_user ON workflow_executions(user_id);")

    # =========================================================================
    # Phase 4: Project Memory
    # =========================================================================

    # Project facts table
    op.execute("""
        CREATE TABLE IF NOT EXISTS project_facts (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            fact_type TEXT NOT NULL,
            key TEXT NOT NULL,
            value JSONB NOT NULL,
            source TEXT DEFAULT 'user',
            confidence FLOAT DEFAULT 1.0,
            expires_at TIMESTAMPTZ,
            embedding vector(384),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(project_id, fact_type, key)
        );
    """)

    # Conversation summaries table
    op.execute("""
        CREATE TABLE IF NOT EXISTS conversation_summaries (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
            user_id TEXT,
            user_session TEXT,
            summary TEXT NOT NULL,
            key_decisions JSONB DEFAULT '[]',
            key_topics JSONB DEFAULT '[]',
            embedding vector(384),
            message_count INTEGER DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    # Indexes for project memory
    op.execute("CREATE INDEX IF NOT EXISTS idx_facts_project ON project_facts(project_id, fact_type);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_facts_key ON project_facts(key);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_facts_embedding ON project_facts USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_summaries_project ON conversation_summaries(project_id);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_summaries_user ON conversation_summaries(user_id);")

    # =========================================================================
    # Phase 5: User Preferences
    # =========================================================================

    # User preferences table
    op.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id TEXT NOT NULL,
            category TEXT NOT NULL,
            key TEXT NOT NULL,
            value JSONB NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(user_id, category, key)
        );
    """)

    # User shortcuts table
    op.execute("""
        CREATE TABLE IF NOT EXISTS user_shortcuts (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id TEXT NOT NULL,
            alias TEXT NOT NULL,
            expansion TEXT NOT NULL,
            description TEXT,
            usage_count INTEGER DEFAULT 0,
            last_used_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(user_id, alias)
        );
    """)

    # Indexes for user preferences
    op.execute("CREATE INDEX IF NOT EXISTS idx_prefs_user ON user_preferences(user_id);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_prefs_category ON user_preferences(user_id, category);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_shortcuts_user ON user_shortcuts(user_id);")

    # =========================================================================
    # Phase 6: Validation & Suggestions
    # =========================================================================

    # Validation results table
    op.execute("""
        CREATE TABLE IF NOT EXISTS validation_results (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            element_id UUID REFERENCES elements(id) ON DELETE CASCADE,
            rule_id UUID REFERENCES domain_rules(id) ON DELETE SET NULL,
            status TEXT NOT NULL,
            message TEXT,
            details JSONB DEFAULT '{}',
            computed_at TIMESTAMPTZ DEFAULT NOW(),
            expires_at TIMESTAMPTZ
        );
    """)

    # Design suggestions table
    op.execute("""
        CREATE TABLE IF NOT EXISTS design_suggestions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            context_element_id UUID REFERENCES elements(id) ON DELETE CASCADE,
            workflow_execution_id UUID REFERENCES workflow_executions(id) ON DELETE SET NULL,
            suggestion_type TEXT NOT NULL,
            suggestion TEXT NOT NULL,
            rationale TEXT,
            priority INTEGER DEFAULT 50,
            status TEXT DEFAULT 'pending',
            embedding vector(384),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            dismissed_at TIMESTAMPTZ,
            accepted_at TIMESTAMPTZ
        );
    """)

    # Indexes for validation
    op.execute("CREATE INDEX IF NOT EXISTS idx_validation_project ON validation_results(project_id);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_validation_element ON validation_results(element_id);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_validation_status ON validation_results(status);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_suggestions_project ON design_suggestions(project_id, status);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_suggestions_type ON design_suggestions(suggestion_type);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_suggestions_embedding ON design_suggestions USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);")


def downgrade() -> None:
    """Drop MEP enhancement tables."""
    # Phase 6
    op.execute("DROP TABLE IF EXISTS design_suggestions CASCADE;")
    op.execute("DROP TABLE IF EXISTS validation_results CASCADE;")

    # Phase 5
    op.execute("DROP TABLE IF EXISTS user_shortcuts CASCADE;")
    op.execute("DROP TABLE IF EXISTS user_preferences CASCADE;")

    # Phase 4
    op.execute("DROP TABLE IF EXISTS conversation_summaries CASCADE;")
    op.execute("DROP TABLE IF EXISTS project_facts CASCADE;")

    # Phase 3
    op.execute("DROP TABLE IF EXISTS workflow_executions CASCADE;")
    op.execute("DROP TABLE IF EXISTS workflow_templates CASCADE;")

    # Phase 2
    op.execute("DROP TABLE IF EXISTS system_priorities CASCADE;")
    op.execute("DROP TABLE IF EXISTS domain_rules CASCADE;")
