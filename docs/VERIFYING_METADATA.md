# Verifying Metadata Persistence

This guide explains how to verify that your AEC project metadata (elements, geometry, relationships) is effectively saved in the PostgreSQL database.

## Method 1: Using the Chat Interface (Easiest)

You can use the built-in AI agent tools to check the status of the database and the active project.

**Ask the agent:**
> "Check cache status"

**Expected Release:**
The agent will call `get_cache_status` and return:
- **Database Configured**: `True`
- **Active Project ID**: The UUID of the current project (if one is loaded)
- **Cache Elements**: Number of elements currently in the database
- **Cache Fresh**: `True` (if recently synced)

**Ask the agent:**
> "What is in this file?" or "Get file context"

**Expected Result:**
The agent will call `get_file_context` and show a summary of:
- Total elements
- Count by category/layer
- Relationship counts

## Method 2: Direct Database Inspection (SQL)

If you have a database tool (like pgAdmin, DBeaver, or command line `psql`), you can query the tables directly.

**1. Check Projects**
See which projects have been synchronized.
```sql
SELECT id, name, source, file_path, created_at, updated_at 
FROM projects 
ORDER BY updated_at DESC;
```

**2. Check Elements**
See the elements for a specific project (replace `YOUR_PROJECT_ID` with the UUID from step 1).
```sql
SELECT entity_type, COUNT(*) 
FROM elements 
WHERE project_id = 'YOUR_PROJECT_ID' 
GROUP BY entity_type;
```

**3. Check Relationships**
Verify that relationships between elements have been computed.
```sql
SELECT relation_type, COUNT(*) 
FROM element_relationships 
WHERE project_id = 'YOUR_PROJECT_ID' 
GROUP BY relation_type;
```

**4. Check Semantic Embeddings**
Verify that vector embeddings have been generated for semantic search.
```sql
SELECT COUNT(*) 
FROM elements 
WHERE project_id = 'YOUR_PROJECT_ID' 
  AND embedding IS NOT NULL;
```

## Method 3: Checking Logs

When the agent is running, check the terminal output for sync confirmations.

**Look for:**
- `Synced {number} elements`
- `Batch upserted elements`
- `Created project`

These logs indicate that the `sync_metadata` tool has successfully executed and written to the database.
