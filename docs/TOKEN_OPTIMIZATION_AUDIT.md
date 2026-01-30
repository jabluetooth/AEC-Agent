# Token Optimization Audit Report

## Current State Analysis

### Input Token Sources

| Source | Avg Tokens | % of Total | Optimization Potential |
|--------|------------|------------|----------------------|
| System Prompt | ~100 | 5% | Low (already minimal) |
| Tool Schemas (20 tools) | ~2000-4000 | 40-60% | **High** |
| Conversation History | ~500-2000 | 20-30% | Medium |
| Tool Results | ~200-1000 | 10-20% | Medium |

### Output Token Sources

| Source | Avg Tokens | Optimization Potential |
|--------|------------|----------------------|
| LLM Response | ~100-500 | Low (user facing) |
| Tool Call Args | ~50-200 | Low (necessary) |

---

## Existing Optimizations (Well Implemented)

### 1. Tool Schema Compression
**File**: `src/aec_agent/frontend/tool_optimization.py`

```python
# Compression modes available
"full"     -> No compression (development)
"standard" -> Remove examples, first sentence only
"minimal"  -> Types only, pre-defined descriptions (~50 tokens)
"ultra"    -> Required params only (~20 tokens)
```

**Token Savings**: 50-80% per tool schema

### 2. Smart Tool Routing
**File**: `src/aec_agent/frontend/agent.py:914-999`

- Filters tools by app context (AutoCAD vs Revit)
- Uses MEP intent classification
- Loads only relevant tool tiers

**Token Savings**: 40-60% by loading 8-12 tools instead of 20+

### 3. Conversation History Trimming
**File**: `src/aec_agent/frontend/agent.py:812-824`

```python
def _trim_history(self) -> None:
    if len(self.messages) <= self.max_history_messages + 1:
        return
    system_msg = self.messages[0]
    recent_msgs = self.messages[-(self.max_history_messages):]
    self.messages = [system_msg] + recent_msgs
```

**Token Savings**: Prevents unbounded growth

### 4. Tool Result Truncation
**File**: `src/aec_agent/frontend/mcp_client.py:162-232`

- Smart truncation preserving JSON structure
- Configurable `max_tool_result_chars` (default 2000)
- Lists truncated with item counts

---

## Optimization Opportunities

### HIGH PRIORITY

#### 1. Dynamic Compression Based on Context Length

**Problem**: Static compression mode regardless of conversation length
**Solution**: Auto-escalate compression as context grows

```python
# Recommended implementation
def get_dynamic_compression_mode(current_tokens: int, max_tokens: int = 8000) -> str:
    ratio = current_tokens / max_tokens
    if ratio < 0.3:
        return "standard"
    elif ratio < 0.6:
        return "minimal"
    else:
        return "ultra"
```

**Files to modify**:
- `src/aec_agent/frontend/agent.py` - Add token tracking
- `src/aec_agent/config/settings.py` - Add `MAX_CONTEXT_TOKENS`

**Estimated Savings**: 20-30% in long conversations

---

#### 2. Tool Schema Caching with Fingerprinting

**Problem**: Tool schemas regenerated on every request
**Solution**: Cache compressed schemas with version fingerprint

```python
# Recommended implementation
_schema_cache: dict[str, dict] = {}

def get_cached_schema(tool_name: str, mode: str) -> dict:
    key = f"{tool_name}:{mode}"
    if key not in _schema_cache:
        _schema_cache[key] = generate_compressed_schema(tool_name, mode)
    return _schema_cache[key]
```

**Estimated Savings**: ~50ms latency per request (not tokens, but performance)

---

#### 3. MEP Domain-Specific Tool Loading

**Problem**: All MEP tools loaded when any MEP query detected
**Solution**: Load only domain-specific tools

```python
# Current (loads all MEP tools)
if intent.is_mep_specific:
    include_metadata = True

# Recommended (load domain-specific)
MEP_DOMAIN_TOOLS = {
    "hvac": ["find_elements", "get_nearby_elements", "route_duct"],
    "electrical": ["find_elements", "circuit_analysis"],
    "plumbing": ["find_elements", "size_pipe"],
}

def get_domain_tools(domain: MEPDomain) -> list[str]:
    return MEP_DOMAIN_TOOLS.get(domain.value, [])
```

**Estimated Savings**: 15-25% when MEP domain is clear

---

### MEDIUM PRIORITY

#### 4. Minimal System Prompt

**Current** (~100 tokens):
```python
SYSTEM_PROMPT = """You are an AEC AI assistant for AutoCAD and Revit automation.

Tools available: layers, drawing (lines/circles/rectangles), levels, walls, rooms, document queries.

Guidelines:
- Explain actions briefly before executing
- Confirm destructive operations (delete)
- Report errors clearly with alternatives
- Units: meters for Revit, drawing units for AutoCAD
- Operations are single-threaded; one at a time"""
```

**Optimized** (~60 tokens):
```python
SYSTEM_PROMPT = """AEC assistant for AutoCAD/Revit. Tools: layers, drawing, levels, walls, rooms.
Rules: Explain briefly. Confirm deletes. Report errors. Units: meters (Revit), drawing units (AutoCAD). One operation at a time."""
```

**Estimated Savings**: 40 tokens per request

---

#### 5. Conversation Summarization

**Problem**: Old messages consume tokens even when trimmed
**Solution**: Summarize older context into compact form

```python
# Recommended implementation
async def summarize_old_context(messages: list[Message], keep_recent: int = 5) -> str:
    if len(messages) <= keep_recent + 1:
        return None

    old_messages = messages[1:-keep_recent]  # Exclude system + recent
    summary_prompt = "Summarize this conversation context in 2 sentences:"

    # Use cheaper model for summarization
    summary = await cheap_llm.generate(summary_prompt + str(old_messages))
    return summary
```

**Estimated Savings**: 30-50% in long conversations

---

#### 6. Tool Result Field Filtering

**Problem**: Full entity data returned when only some fields needed
**Solution**: Return only fields relevant to the query

```python
# Current (returns all fields)
return {"elements": [full_element_data...]}

# Recommended (configurable fields)
FIELD_PRESETS = {
    "minimal": ["id", "name"],
    "standard": ["id", "name", "type", "layer"],
    "full": None  # All fields
}

def filter_result_fields(data: dict, preset: str = "standard") -> dict:
    fields = FIELD_PRESETS.get(preset)
    if fields is None:
        return data
    return {k: v for k, v in data.items() if k in fields}
```

**Estimated Savings**: 20-40% on large result sets

---

### LOW PRIORITY

#### 7. Prompt Caching (Provider-Specific)

**Problem**: System prompt and tool schemas sent every request
**Solution**: Use provider-specific caching

- **Anthropic**: Use prompt caching beta
- **OpenAI**: Use fine-tuning for system behavior

---

#### 8. Streaming Compression

**Problem**: Full responses buffered before processing
**Solution**: Compress/filter during streaming

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)
- [ ] Implement dynamic compression mode
- [ ] Reduce system prompt to minimal
- [ ] Add tool schema caching

### Phase 2: Core Improvements (3-5 days)
- [ ] MEP domain-specific tool loading
- [ ] Tool result field filtering
- [ ] Add token tracking/logging

### Phase 3: Advanced (1-2 weeks)
- [ ] Conversation summarization
- [ ] Provider-specific caching
- [ ] Streaming compression

---

## Monitoring Recommendations

Add token usage tracking:

```python
# Add to settings.py
enable_token_logging: bool = Field(default=True)

# Add to agent.py
async def log_token_usage(self, input_tokens: int, output_tokens: int):
    logger.info(
        "Token usage",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total=input_tokens + output_tokens,
        history_length=len(self.messages),
        tools_loaded=len(self._current_tools),
    )
```

---

## Estimated Total Savings

| Optimization | Input Savings | Output Savings |
|--------------|---------------|----------------|
| Dynamic compression | 20-30% | - |
| Domain-specific tools | 15-25% | - |
| Minimal system prompt | 5% | - |
| Conversation summary | 30-50%* | - |
| Result field filtering | - | 20-40% |

*In long conversations

**Combined Potential**: 40-60% input token reduction without quality loss

---

*Audit completed: 2026-01-23*
