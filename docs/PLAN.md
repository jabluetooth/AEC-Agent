# AEC Agent - Metadata Pipeline Architecture

## Overview

Build metadata extraction pipelines for AutoCAD and Revit to enable semantic CAD operations without manual coordinate input.

**Goal:** "Draw a line from this wall to that column" → System resolves coordinates automatically.

---

## Architecture Integration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              EXISTING STACK                                  │
├─────────────────┬───────────────────────────────────────────────────────────┤
│ Chainlit UI     │ Port 8000                                                 │
├─────────────────┼───────────────────────────────────────────────────────────┤
│ FastMCP Server  │ Port 54321 - NOW with DB-aware tools                      │
├─────────────────┼───────────────────────────────────────────────────────────┤
│ Sidecars        │ AutoCAD (.NET) / Revit (pyRevit)                          │
│                 │ + NEW: Extraction endpoints                               │
└────────┬────────┴───────────────────────────────────────────────────────────┘
         │
┌────────▼────────────────────────────────────────────────────────────────────┐
│                           NEW: DATA LAYER                                    │
├─────────────────┬───────────────────────────────────────────────────────────┤
│ PostgreSQL      │ + PostGIS (spatial) + pgvector (semantic)                 │
│                 │ Elements, relationships, embeddings                        │
├─────────────────┼───────────────────────────────────────────────────────────┤
│ SQLite (keep)   │ Local session cache only (fast, ephemeral)                │
└─────────────────┴───────────────────────────────────────────────────────────┘
```

---

## Stage 1: Data Extraction

### A) AutoCAD Extraction

**Two-tier approach:**

| Tool | Use Case | Capabilities |
|------|----------|--------------|
| **AutoLISP** | Quick queries, simple extraction | Read-only, limited to LISP-accessible data |
| **ObjectARX/.NET** | Full extraction, complex geometry | Full API access, events, extended data |

**Recommended: Extend existing .NET sidecar** (already have thread marshaling solved)

Extract per entity:
- Handle (stable ID)
- ObjectId (session ID)
- Entity type (DXF name)
- Layer, color, linetype
- Geometry (type-specific: endpoints, vertices, center/radius)
- Bounding box (for spatial indexing)
- Extended data (XData)
- Block attributes (if block reference)
- Custom properties

**Output:** Stream directly to PostgreSQL via sidecar HTTP endpoint (not files)

### B) Revit Extraction

**Extend existing pyRevit sidecar:**

Extract per element:
- ElementId (stable within document)
- UniqueId (GUID, stable across sessions)
- Category, Family, Type
- Level, Phase
- Location (point or curve)
- Bounding box
- Parameters (instance + type)
- Hosted elements (doors in walls)
- Connected elements (structural framing)

**Output:** Stream directly to PostgreSQL

---

## Stage 2: Database Schema

### Why PostgreSQL over SQLite?

| Requirement | SQLite | PostgreSQL |
|-------------|--------|------------|
| Spatial queries (find nearby) | Limited | PostGIS |
| Vector search (semantic) | None | pgvector |
| Concurrent pipeline writes | Locks | MVCC |
| Complex joins (relationships) | Slow | Optimized |
| JSON queries | Basic | JSONB + GIN |
| Scale (millions of entities) | Degrades | Handles well |

**Verdict:** PostgreSQL for metadata store, keep SQLite for session cache.

### Core Tables

```sql
-- Enable extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS vector;

-- Projects (drawings/models)
CREATE TABLE projects (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL,
    source          TEXT NOT NULL CHECK (source IN ('autocad', 'revit')),
    file_path       TEXT,
    file_hash       TEXT,                    -- For change detection
    extracted_at    TIMESTAMPTZ,
    metadata        JSONB DEFAULT '{}'
);

-- Elements (entities/elements)
CREATE TABLE elements (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id      UUID REFERENCES projects(id) ON DELETE CASCADE,
    source_id       TEXT NOT NULL,           -- Handle (AutoCAD) or ElementId (Revit)
    source          TEXT NOT NULL,

    -- Classification
    entity_type     TEXT NOT NULL,           -- LINE, WALL, DOOR, etc.
    layer           TEXT,                    -- AutoCAD layer
    category        TEXT,                    -- Revit category
    family          TEXT,                    -- Revit family
    type_name       TEXT,                    -- Revit type

    -- Geometry (PostGIS)
    geom            GEOMETRY(GeometryZ, 0),  -- Actual geometry (3D)
    bbox            BOX3D,                   -- Bounding box for fast queries
    centroid        GEOMETRY(PointZ, 0),     -- Center point

    -- Properties
    properties      JSONB DEFAULT '{}',      -- All parameters/xdata

    -- Semantic search (pgvector)
    embedding       vector(384),             -- sentence-transformers dimension
    description     TEXT,                    -- Human-readable for embedding

    -- Timestamps
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(project_id, source_id, source)
);

-- Spatial index
CREATE INDEX idx_elements_geom ON elements USING GIST (geom);
CREATE INDEX idx_elements_bbox ON elements USING GIST (bbox);

-- Vector index (for semantic search)
CREATE INDEX idx_elements_embedding ON elements USING ivfflat (embedding vector_cosine_ops);

-- JSONB index (for property queries)
CREATE INDEX idx_elements_properties ON elements USING GIN (properties);

-- Relationships (computed from geometry + Revit API)
CREATE TABLE element_relationships (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id      UUID REFERENCES projects(id) ON DELETE CASCADE,
    from_element_id UUID REFERENCES elements(id) ON DELETE CASCADE,
    to_element_id   UUID REFERENCES elements(id) ON DELETE CASCADE,

    relation_type   TEXT NOT NULL,           -- intersects, hosts, connected_to, near
    distance        FLOAT,                   -- Distance if applicable
    confidence      FLOAT DEFAULT 1.0,       -- For inferred relationships
    source          TEXT DEFAULT 'computed', -- computed, revit_api, user_defined

    metadata        JSONB DEFAULT '{}',

    UNIQUE(project_id, from_element_id, to_element_id, relation_type)
);

CREATE INDEX idx_relationships_from ON element_relationships(from_element_id);
CREATE INDEX idx_relationships_to ON element_relationships(to_element_id);
CREATE INDEX idx_relationships_type ON element_relationships(relation_type);
```

---

## Stage 3: Relationship Computation

### Use PostGIS, not Python

**Wrong (slow):**
```python
for e1 in elements:
    for e2 in elements:
        if bbox_intersects(e1, e2):  # O(n²)
            save_relationship(e1, e2)
```

**Right (fast):**
```sql
-- Find all intersecting elements (uses spatial index)
INSERT INTO element_relationships (project_id, from_element_id, to_element_id, relation_type)
SELECT DISTINCT
    e1.project_id,
    e1.id,
    e2.id,
    'intersects'
FROM elements e1
JOIN elements e2 ON ST_Intersects(e1.geom, e2.geom)
WHERE e1.project_id = $1
  AND e2.project_id = $1
  AND e1.id < e2.id;  -- Avoid duplicates

-- Find elements within distance
INSERT INTO element_relationships (project_id, from_element_id, to_element_id, relation_type, distance)
SELECT
    e1.project_id,
    e1.id,
    e2.id,
    'near',
    ST_3DDistance(e1.geom, e2.geom)
FROM elements e1
JOIN elements e2 ON ST_3DDWithin(e1.geom, e2.geom, 1.0)  -- Within 1 meter
WHERE e1.project_id = $1
  AND e2.project_id = $1
  AND e1.id != e2.id;
```

### Relationship Types

| Type | Source | Description |
|------|--------|-------------|
| `intersects` | PostGIS | Geometries touch/overlap |
| `near` | PostGIS | Within threshold distance |
| `hosts` | Revit API | Door hosted in wall |
| `connected_to` | Revit API | Structural connection |
| `on_layer` | AutoCAD | Same layer grouping |
| `same_block` | AutoCAD | Part of same block |

---

## Stage 4: Semantic Search with pgvector

### Generate Descriptions

```python
def generate_description(element: dict) -> str:
    """Create searchable text from element properties."""
    parts = []

    if element['source'] == 'autocad':
        parts.append(f"{element['entity_type']} on layer {element['layer']}")
        if element.get('properties', {}).get('length'):
            parts.append(f"length {element['properties']['length']:.1f}")

    elif element['source'] == 'revit':
        parts.append(f"{element['category']}: {element['family']} - {element['type_name']}")
        if element.get('properties', {}).get('Height'):
            parts.append(f"height {element['properties']['Height']:.1f}m")
        if element.get('properties', {}).get('Fire Rating'):
            parts.append(f"fire rating: {element['properties']['Fire Rating']}")

    return ', '.join(parts)

# Example outputs:
# "LINE on layer WALLS, length 10.5"
# "Wall: Basic Wall - Generic 200mm, height 3.0m, fire rating: 1HR"
```

### Query Flow

```sql
-- User: "Find fire-rated walls near the elevator"

-- 1. Semantic search for "fire-rated walls"
WITH query_embedding AS (
    SELECT $1::vector AS emb  -- From sentence-transformers
),
fire_walls AS (
    SELECT e.*, e.embedding <=> q.emb AS distance
    FROM elements e, query_embedding q
    WHERE e.category = 'Wall'
      AND e.properties->>'Fire Rating' IS NOT NULL
    ORDER BY e.embedding <=> q.emb
    LIMIT 20
),
-- 2. Find elevator
elevator AS (
    SELECT * FROM elements
    WHERE family ILIKE '%elevator%' OR type_name ILIKE '%elevator%'
    LIMIT 1
)
-- 3. Filter walls near elevator
SELECT fw.*
FROM fire_walls fw, elevator el
WHERE ST_3DDWithin(fw.geom, el.geom, 5.0);  -- Within 5 meters
```

---

## Stage 5: New MCP Tools

### Extend tool registry:

| Tool | Description |
|------|-------------|
| `find_elements` | Semantic + spatial search |
| `get_element_by_name` | Fuzzy match on description |
| `get_nearby_elements` | Spatial proximity query |
| `get_related_elements` | Traverse relationship graph |
| `resolve_coordinates` | Get centroid/endpoints from element reference |

### Example: Smart Line Drawing

```python
@mcp.tool()
async def draw_line_between(
    from_element: str,  # "the east wall" or element ID
    to_element: str,    # "column C3" or element ID
    layer: str = None
) -> dict:
    """Draw a line between two elements, resolving coordinates automatically."""

    # 1. Resolve 'from' element
    from_el = await resolve_element(from_element)  # Semantic search
    if not from_el:
        return error_result("Could not find: " + from_element)

    # 2. Resolve 'to' element
    to_el = await resolve_element(to_element)
    if not to_el:
        return error_result("Could not find: " + to_element)

    # 3. Get coordinates from database
    from_point = await get_centroid(from_el['id'])
    to_point = await get_centroid(to_el['id'])

    # 4. Call existing AutoCAD tool with resolved coordinates
    return await autocad_draw_line(
        start_x=from_point.x,
        start_y=from_point.y,
        end_x=to_point.x,
        end_y=to_point.y,
        layer=layer
    )
```

---

## Stage 6: Sync Strategy

### Initial Load
1. User opens drawing/model
2. Sidecar detects file open event
3. Full extraction triggered
4. Stream to PostgreSQL
5. Compute relationships (PostGIS)
6. Generate embeddings (async background)

### Incremental Updates
1. Sidecar hooks into modification events
2. Track changed entity IDs
3. On idle/save: sync only changed entities
4. Recompute affected relationships

### Change Detection
```python
# In sidecar
@acad_event('ObjectModified')
def on_object_modified(sender, args):
    entity_id = args.ObjectId
    pending_sync.add(entity_id)

@acad_event('DocumentSaved')
def on_save(sender, args):
    sync_pending_to_database(pending_sync)
    pending_sync.clear()
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [ ] Set up PostgreSQL with PostGIS + pgvector
- [ ] Create database schema and migrations
- [ ] Add `asyncpg` to Python dependencies
- [ ] Database connection pool in FastMCP server

### Phase 2: AutoCAD Pipeline (Week 3-4)
- [ ] Extend .NET sidecar with extraction endpoint
- [ ] Entity geometry serialization
- [ ] Streaming ingest to PostgreSQL
- [ ] Basic relationship computation

### Phase 3: Revit Pipeline (Week 5-6)
- [ ] Extend pyRevit sidecar with extraction endpoint
- [ ] Element + parameter extraction
- [ ] Host relationship extraction (native from API)
- [ ] Streaming ingest to PostgreSQL

### Phase 4: Semantic Layer (Week 7-8)
- [ ] Description generation
- [ ] Embedding computation (sentence-transformers)
- [ ] Semantic search MCP tools
- [ ] `find_elements`, `resolve_element` tools

### Phase 5: Smart Tools (Week 9-10)
- [ ] `draw_line_between` (element references)
- [ ] `move_element_to` (relative positioning)
- [ ] `copy_element_near` (spatial context)
- [ ] Integration tests

---

## Configuration (New Settings)

```bash
# .env additions
DATABASE_URL=postgresql://user:pass@localhost:5432/aec_agent
DATABASE_POOL_SIZE=5
DATABASE_POOL_MAX_OVERFLOW=10

# Embedding model
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# Sync settings
SYNC_ON_SAVE=true
SYNC_DEBOUNCE_MS=2000
RELATIONSHIP_DISTANCE_THRESHOLD=1.0
```

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Large drawing extraction slow | User waits | Background extraction, progress UI |
| PostgreSQL setup complexity | Deployment friction | Docker Compose, managed DB option |
| Embedding generation slow | Delays sync | Async queue, batch processing |
| Stale data after external edits | Wrong coordinates | File hash check, re-extract prompt |
| ObjectARX complexity | Dev time | Start with AutoLISP, add ObjectARX later |

---

## Decision Log

| Decision | Rationale |
|----------|-----------|
| PostgreSQL over SQLite | Spatial queries, vector search, concurrent writes |
| Keep SQLite for cache | Fast ephemeral session data |
| PostGIS over Python geometry | Index-backed spatial queries, orders of magnitude faster |
| pgvector over dedicated vector DB | Single database, simpler ops, good enough for scale |
| Extend sidecars vs new service | Reuse thread marshaling, single deployment |
| Stream to DB vs file export | No intermediate files, real-time sync possible |
