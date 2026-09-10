# MEP Enhancement Integration Plan

## Overview

All 6 phases of the MEP enhancement are **built** but not yet **integrated** into the main agent. This document tracks the remaining integration work.

> **2026-09-10 correction:** This plan assumed `frontend/agent.py` (the Chainlit UI
> process) could hold a `db_pool` directly and instantiate `ProjectMemory`/
> `RulesEngine`/etc. in-process. It cannot — only the separate FastMCP server
> process (`mcp/server.py`) ever calls `initialize_database_pool()`; the frontend
> only reaches the system via `MCPClient.call_tool(...)` over SSE. **Tasks 4, 5,
> 10, 12 are done** (see below) exactly as originally documented, since they're
> server-side and use the existing `get_database_pool()`/`get_embedding_service()`
> pattern from `mcp/server.py`. **Tasks 1, 2, 3, 6, 7, 8, 9, 11 are NOT done** and
> need redesigning around a new imperative MCP tool call from `agent.py`/`app.py`
> (e.g. a `get_session_context` tool) rather than direct object instantiation in
> the frontend process — not yet implemented.

### Done (2026-09-10)
- ✅ Task 4 — `src/aec_agent/mcp/tools/workflow_tools.py`: `list_workflows`, `start_workflow`
- ✅ Task 5 — `src/aec_agent/mcp/tools/validation_tools.py`: `validate_elements`, `get_suggestions`
- ✅ Task 12 — `src/aec_agent/mcp/tools/memory_tools.py`: `store_fact`, `recall_facts`
- ✅ Task 10 — `aec-agent seed-rules` CLI command (`cmd_seed_rules` in `cli.py`), using the actual seed function name `get_default_hvac_rules` (the doc's original `get_hvac_seed_rules` doesn't exist)
- All three new tool files pass `get_database_pool()`/`get_embedding_service()` from `mcp/server.py` into the factory functions, so semantic search (Task 9's intent) works for these tools without any frontend changes.

---

## Current State

### Completed Modules

| Module | Location | Status |
|--------|----------|--------|
| Intent Classifier | `src/aec_agent/intent/` | ✅ Built & Integrated |
| Domain Knowledge | `src/aec_agent/domain/` | ✅ Built, ❌ Not Integrated |
| Workflows | `src/aec_agent/workflows/` | ✅ Built, ❌ Not Integrated |
| Project Memory | `src/aec_agent/memory/` | ✅ Built, ❌ Not Integrated |
| User Preferences | `src/aec_agent/memory/user_preferences.py` | ✅ Built, ❌ Not Integrated |
| Rules Engine | `src/aec_agent/domain/rules_engine.py` | ✅ Built, ❌ Not Integrated |
| Validators | `src/aec_agent/domain/validators.py` | ✅ Built, ❌ Not Integrated |

### Database Tables (Migration Applied)

All tables exist in PostgreSQL:
- `domain_rules`, `system_priorities`
- `workflow_templates`, `workflow_executions`
- `project_facts`, `conversation_summaries`
- `user_preferences`, `user_shortcuts`
- `validation_results`, `design_suggestions`

---

## Integration Tasks

### Task 1: Initialize Services in Agent

**File:** `src/aec_agent/frontend/agent.py`

Add initialization in `AECAgent.__init__()` or `initialize()`:

```python
from aec_agent.domain import get_knowledge_base, get_rules_engine
from aec_agent.memory import get_project_memory, get_user_preferences
from aec_agent.workflows import get_workflow_executor

# In initialize() method:
self._knowledge_base = await get_knowledge_base(db_pool=self._db_pool)
self._project_memory = await get_project_memory(
    db_pool=self._db_pool,
    embedding_service=self._embedding_service
)
self._user_preferences = await get_user_preferences(
    db_pool=self._db_pool
)
self._workflow_executor = await get_workflow_executor(
    mcp_client=self._mcp_client,
    db_pool=self._db_pool
)
self._rules_engine = await get_rules_engine(
    knowledge_base=self._knowledge_base,
    db_pool=self._db_pool
)
```

---

### Task 2: Inject Memory Context into Prompts

**File:** `src/aec_agent/frontend/agent.py`

Before LLM call, inject project context:

```python
# In _build_messages() or similar:
if self._project_memory and project_id:
    context = await self._project_memory.get_context(
        project_id=project_id,
        user_id=self._user_preferences.user_id,
        query=user_input,
        token_budget=500
    )
    if context.facts or context.recent_summaries:
        system_prompt += "\n\n" + context.to_context_string()
```

**Token Impact:** -1800 tokens (200 token context vs 2000 history)

---

### Task 3: Load User Preferences at Session Start

**File:** `src/aec_agent/frontend/app.py`

In Chainlit `on_chat_start`:

```python
from aec_agent.memory import get_user_preferences, get_current_user

@cl.on_chat_start
async def on_chat_start():
    user_id = get_current_user()  # Windows username
    prefs = await get_user_preferences(db_pool=db_pool, user_id=user_id)

    # Store in session
    cl.user_session.set("user_preferences", prefs)
    cl.user_session.set("user_id", user_id)

    # Get MEP defaults for tool calls
    mep_defaults = await prefs.get_mep_defaults()
    cl.user_session.set("mep_defaults", mep_defaults)
```

---

### Task 4: Create Workflow MCP Tools

**File to Create:** `src/aec_agent/mcp/tools/workflow_tools.py`

```python
from aec_agent.workflows import get_workflow_executor

async def list_workflows(domain: str = "mep") -> dict:
    """List available workflow templates."""
    executor = await get_workflow_executor()
    templates = executor.list_templates(domain=domain)
    return {
        "success": True,
        "workflows": [
            {
                "name": t.name,
                "description": t.description,
                "required_inputs": t.required_context,
                "estimated_tokens_saved": t.estimated_tokens,
            }
            for t in templates
        ]
    }

async def start_workflow(
    workflow_name: str,
    context: dict,
    project_id: str = None
) -> dict:
    """Start and run a workflow to completion."""
    executor = await get_workflow_executor()
    execution = await executor.start_workflow(
        template_name=workflow_name,
        context=context,
        project_id=UUID(project_id) if project_id else None,
        user_id=get_current_user()
    )
    result = await executor.run_to_completion(execution.id)
    return result.to_dict()
```

**Register in MCP server** (`src/aec_agent/mcp/server.py`)

---

### Task 5: Create Validation MCP Tools

**File to Create:** `src/aec_agent/mcp/tools/validation_tools.py`

```python
from aec_agent.domain import get_rules_engine

async def validate_elements(
    project_id: str,
    element_ids: list[str] = None,
    rule_types: list[str] = None
) -> dict:
    """Validate elements against MEP rules."""
    engine = await get_rules_engine()
    # Fetch elements from DB, then validate
    results = await engine.validate_project(
        project_id=UUID(project_id),
        elements=elements,
        rule_types=rule_types
    )
    return results

async def get_suggestions(
    project_id: str,
    limit: int = 10
) -> dict:
    """Get pending design suggestions."""
    engine = await get_rules_engine()
    suggestions = await engine.get_pending_suggestions(
        project_id=UUID(project_id),
        limit=limit
    )
    return {
        "success": True,
        "suggestions": [s.to_dict() for s in suggestions]
    }
```

---

### Task 6: Expand Shortcuts in User Input

**File:** `src/aec_agent/frontend/agent.py`

Before processing user input:

```python
async def process_message(self, user_input: str) -> str:
    # Expand any user shortcuts
    if self._user_preferences:
        user_input = await self._user_preferences.expand_shortcut(user_input)

    # Continue with intent classification...
```

---

### Task 7: Store Conversation Summaries

**File:** `src/aec_agent/frontend/agent.py`

After N messages or on session end:

```python
async def _maybe_summarize_conversation(self):
    if len(self._messages) >= 20:  # Every 20 messages
        summary = await self._generate_summary()  # LLM call
        await self._project_memory.store_summary(
            project_id=self._project_id,
            user_id=self._user_preferences.user_id,
            user_session=self._session_id,
            summary=summary.text,
            key_decisions=summary.decisions,
            key_topics=summary.topics,
            message_count=len(self._messages)
        )
        # Trim old messages from context
        self._messages = self._messages[-5:]
```

---

### Task 8: Apply Rules During Element Operations

**File:** `src/aec_agent/mcp/tools/metadata.py` or new tool

After sync_metadata or find_elements, optionally validate:

```python
async def sync_metadata_with_validation(source: str, validate: bool = True):
    result = await sync_metadata(source)

    if validate and result["success"]:
        engine = await get_rules_engine()
        validation = await engine.validate_project(
            project_id=result["project_id"],
            elements=result["elements"]
        )
        result["validation"] = validation

    return result
```

---

## Verification Checklist

After integration, verify:

- [ ] `get_current_user()` returns Windows username correctly
- [ ] User preferences load at session start
- [ ] Shortcuts expand correctly (e.g., "rs" → "route supply duct")
- [ ] Project facts are injected into system prompt
- [ ] `list_workflows` MCP tool returns 6 HVAC workflows
- [ ] `start_workflow` executes multi-step workflow
- [ ] `validate_elements` returns clearance/sizing issues
- [ ] Conversation summaries are stored after 20 messages
- [ ] Token usage is reduced (measure before/after)

---

## Token Savings Summary

| Integration | Savings | Mechanism |
|-------------|---------|-----------|
| Intent → tool filter | -40-60% | Already integrated |
| Memory context | -1800 | Task 2 |
| User defaults | -20-50 | Task 3 |
| Workflows | -2000/workflow | Task 4 |
| Rules engine | -500-1000 | Task 5, 8 |

**Expected Total:** 50-70% reduction in token usage

---

## Quick Reference: Import Paths

```python
# Intent (already integrated)
from aec_agent.intent import IntentClassifier, IntentResult

# Domain Knowledge
from aec_agent.domain import (
    MEPKnowledgeBase, get_knowledge_base,
    RulesEngine, get_rules_engine,
    ClearanceValidator, SizingValidator, RoutingValidator, CoverageValidator,
    DomainRule, RuleType
)

# Workflows
from aec_agent.workflows import (
    WorkflowExecutor, get_workflow_executor,
    WorkflowTemplate, WorkflowStep, WorkflowExecution
)

# Memory
from aec_agent.memory import (
    ProjectMemory, get_project_memory,
    UserPreferences, get_user_preferences, get_current_user,
    ProjectFact, FactType, ConversationSummary
)
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/aec_agent/frontend/agent.py` | Tasks 1, 2, 6, 7 |
| `src/aec_agent/frontend/app.py` | Task 3 |
| `src/aec_agent/mcp/server.py` | Register new tools |
| `src/aec_agent/mcp/tools/workflow_tools.py` | Task 4 (create) |
| `src/aec_agent/mcp/tools/validation_tools.py` | Task 5 (create) |
| `src/aec_agent/mcp/tools/metadata.py` | Task 8 (optional) |

---

## Testing Commands

```bash
# Run all tests
pytest tests/ -v

# Test specific modules
pytest tests/unit/test_intent_classifier.py -v

# Test with coverage
pytest --cov=src/aec_agent --cov-report=html
```

---

## Additional Missing Pieces

### Task 9: Wire Up Embedding Service

**Problem:** Semantic search requires embeddings but the embedding service isn't passed to modules.

**File:** `src/aec_agent/frontend/agent.py`

```python
from aec_agent.semantic.embeddings import EmbeddingService

# In initialize():
self._embedding_service = EmbeddingService(
    model_name=settings.embedding_model  # "all-MiniLM-L6-v2"
)
await self._embedding_service.initialize()

# Pass to modules that need it:
self._project_memory = await get_project_memory(
    db_pool=self._db_pool,
    embedding_service=self._embedding_service  # <-- Required for semantic search
)
self._rules_engine = await get_rules_engine(
    knowledge_base=self._knowledge_base,
    db_pool=self._db_pool,
    embedding_service=self._embedding_service  # <-- Required for suggestions
)
```

---

### Task 10: Load HVAC Seed Data into Database

**Problem:** The `domain_rules` table is empty. Seed data exists in code but not in DB.

**Option A: Create a management command**

**File to Create:** `src/aec_agent/cli.py` (add command)

```python
@cli.command()
def seed_rules():
    """Load HVAC rules into database."""
    import asyncio
    from aec_agent.domain.seed_data import get_hvac_seed_rules
    from aec_agent.domain import get_knowledge_base

    async def run():
        kb = await get_knowledge_base(db_pool=get_db_pool())
        rules = get_hvac_seed_rules()
        for rule in rules:
            await kb.add_rule(rule)
        print(f"Loaded {len(rules)} HVAC rules")

    asyncio.run(run())
```

**Option B: Auto-load on first startup**

In `MEPKnowledgeBase.initialize()`, check if rules exist, if not, seed them.

---

### Task 11: Project ID Management

**Problem:** How does the agent know which project is active?

**Solution:** Track project in session

**File:** `src/aec_agent/frontend/app.py`

```python
@cl.on_chat_start
async def on_chat_start():
    # Create or get project from first sync
    cl.user_session.set("project_id", None)  # Set when first file synced

# When sync_metadata is called:
async def handle_sync_result(result):
    if result.get("project_id"):
        cl.user_session.set("project_id", result["project_id"])
```

**File:** `src/aec_agent/frontend/agent.py`

```python
@property
def project_id(self) -> Optional[UUID]:
    """Get active project ID from session or last sync."""
    return self._session.get("project_id")
```

---

### Task 12: Create Memory MCP Tools

**Problem:** No tools to store/recall project facts from chat.

**File to Create:** `src/aec_agent/mcp/tools/memory_tools.py`

```python
from aec_agent.memory import get_project_memory, FactType

async def store_fact(
    project_id: str,
    fact_type: str,  # decision, constraint, preference, note
    key: str,
    value: str
) -> dict:
    """Store a project fact."""
    memory = await get_project_memory()
    fact = await memory.store_fact(
        project_id=UUID(project_id),
        fact_type=FactType(fact_type),
        key=key,
        value=value
    )
    return {"success": True, "fact_id": str(fact.id)}

async def recall_facts(
    project_id: str,
    query: str = None,
    fact_type: str = None,
    limit: int = 10
) -> dict:
    """Recall project facts, optionally filtered by query."""
    memory = await get_project_memory()

    if query:
        facts = await memory.search_facts(UUID(project_id), query, limit)
    else:
        facts = await memory.get_facts(
            UUID(project_id),
            fact_type=FactType(fact_type) if fact_type else None,
            limit=limit
        )

    return {
        "success": True,
        "facts": [f.to_dict() for f in facts]
    }
```

---

### Task 13: Standards Extraction from CAD Templates (Optional)

**Problem:** Original plan included extracting standards from DWT/RTE files. Not yet built.

**File to Create:** `src/aec_agent/mcp/tools/standards_tools.py`

```python
async def extract_template_standards(template_path: str) -> dict:
    """
    Extract layer standards, blocks, properties from DWT/RTE.

    Calls sidecar to:
    1. Open template file
    2. Extract layer table (name, color, linetype)
    3. Extract block definitions
    4. Parse property defaults

    Returns standards that can be saved as domain_rules.
    """
    # Requires sidecar implementation
    pass
```

**Priority:** Low - can be added later when company standards need to be imported.

---

### Task 14: Create Unit Tests for New Modules

**Problem:** No tests for Phases 2-6 modules.

**Files to Create:**

```
tests/unit/
├── test_knowledge_base.py      # MEPKnowledgeBase tests
├── test_workflow_executor.py   # WorkflowExecutor tests
├── test_project_memory.py      # ProjectMemory tests
├── test_user_preferences.py    # UserPreferences tests
├── test_rules_engine.py        # RulesEngine tests
└── test_validators.py          # Validator tests
```

**Example test structure:**

```python
# tests/unit/test_user_preferences.py
import pytest
from aec_agent.memory import UserPreferences, get_current_user

class TestUserPreferences:
    def test_get_current_user_returns_string(self):
        user = get_current_user()
        assert isinstance(user, str)
        assert len(user) > 0

    @pytest.mark.asyncio
    async def test_default_preferences_loaded(self):
        prefs = UserPreferences()
        await prefs.initialize()
        units = await prefs.get("display", "units")
        assert units == "imperial"

    @pytest.mark.asyncio
    async def test_shortcut_expansion(self):
        prefs = UserPreferences()
        await prefs.initialize()
        await prefs.add_shortcut("rs", "route supply duct")
        result = await prefs.expand_shortcut("rs to diffusers")
        assert result == "route supply duct to diffusers"
```

---

## Complete Task Summary

| # | Task | Priority | Files |
|---|------|----------|-------|
| 1 | Initialize services in agent | High | agent.py |
| 2 | Inject memory context | High | agent.py |
| 3 | Load user preferences | High | app.py |
| 4 | Create workflow MCP tools | High | workflow_tools.py (new) |
| 5 | Create validation MCP tools | Medium | validation_tools.py (new) |
| 6 | Expand shortcuts | Medium | agent.py |
| 7 | Store conversation summaries | Medium | agent.py |
| 8 | Apply rules during operations | Low | metadata.py |
| 9 | Wire up embedding service | High | agent.py |
| 10 | Load HVAC seed data | High | cli.py or auto-load |
| 11 | Project ID management | High | app.py, agent.py |
| 12 | Create memory MCP tools | Medium | memory_tools.py (new) |
| 13 | Standards extraction | Low | standards_tools.py (new) |
| 14 | Unit tests for new modules | Medium | tests/unit/*.py |

---

## Recommended Order of Implementation

1. **Task 9** - Embedding service (required for semantic features)
2. **Task 10** - Seed HVAC rules (required for validation)
3. **Task 11** - Project ID management (required for memory)
4. **Task 1** - Initialize all services
5. **Task 3** - Load user preferences
6. **Task 4** - Workflow MCP tools
7. **Task 12** - Memory MCP tools
8. **Task 2** - Inject memory context
9. **Task 5** - Validation MCP tools
10. **Task 6, 7, 8** - Additional features
11. **Task 14** - Tests
12. **Task 13** - Standards extraction (future)
