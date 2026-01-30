# AEC Agent Implementation Plan

## Overview

This plan provides a comprehensive roadmap for optimizing the AEC Agent system, focusing on token efficiency, MEP workflow enhancement, and Low Voltage support. Each phase is designed to be independently deployable.

---

## Phase 1: Quick Wins (1-2 Days)

### 1.1 Dynamic Compression Mode

**Goal**: Automatically escalate compression as context grows

**Files to Modify**:
- `src/aec_agent/config/settings.py`
- `src/aec_agent/frontend/agent.py`

**Implementation Steps**:

```
Step 1: Add settings
─────────────────────────────────────────────────────────
File: src/aec_agent/config/settings.py

Add:
  max_context_tokens: int = Field(
      default=8000,
      ge=2000,
      le=128000,
      description="Maximum context tokens before aggressive compression"
  )

  enable_dynamic_compression: bool = Field(
      default=True,
      description="Auto-escalate compression based on context size"
  )
```

```
Step 2: Add token estimation utility
─────────────────────────────────────────────────────────
File: src/aec_agent/frontend/agent.py

Add method to AECAgent class:

  def _estimate_context_tokens(self) -> int:
      """Estimate current context token count."""
      total = 0
      for msg in self.messages:
          total += len(msg.content) // 4
          if msg.tool_calls:
              total += len(json.dumps(msg.tool_calls)) // 4
      return total

  def _get_dynamic_compression_mode(self) -> str:
      """Get compression mode based on context size."""
      settings = get_settings()
      if not settings.enable_dynamic_compression:
          return settings.tool_compression_mode

      tokens = self._estimate_context_tokens()
      ratio = tokens / settings.max_context_tokens

      if ratio < 0.3:
          return "standard"
      elif ratio < 0.6:
          return "minimal"
      else:
          return "ultra"
```

```
Step 3: Integrate into _get_tools()
─────────────────────────────────────────────────────────
File: src/aec_agent/frontend/agent.py

Modify _get_tools() method:

  # Replace static compression_mode with dynamic
  compression_mode = self._get_dynamic_compression_mode()
```

**Testing**:
```bash
pytest tests/unit/test_tool_optimization.py -v
# Add new test for dynamic compression
```

---

### 1.2 Minimal System Prompt

**Goal**: Reduce system prompt from ~100 to ~60 tokens

**File to Modify**:
- `src/aec_agent/frontend/agent.py`

**Implementation**:

```
Current (Line 23-32):
─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an AEC AI assistant for AutoCAD and Revit automation.

Tools available: layers, drawing (lines/circles/rectangles), levels, walls, rooms, document queries.

Guidelines:
- Explain actions briefly before executing
- Confirm destructive operations (delete)
- Report errors clearly with alternatives
- Units: meters for Revit, drawing units for AutoCAD
- Operations are single-threaded; one at a time"""
```

```
Optimized:
─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """AEC assistant for AutoCAD/Revit automation.
Tools: layers, drawing, levels, walls, rooms, queries.
Rules: Brief explanations. Confirm deletes. Report errors with alternatives.
Units: meters (Revit), drawing units (AutoCAD). One operation at a time."""
```

**Token Savings**: ~40 tokens per request

---

### 1.3 Tool Schema Caching

**Goal**: Cache compressed schemas to avoid regeneration

**File to Modify**:
- `src/aec_agent/frontend/tool_optimization.py`

**Implementation**:

```
Add at module level:
─────────────────────────────────────────────────────────
# Schema cache: {tool_name:mode -> compressed_schema}
_schema_cache: dict[str, dict] = {}
_description_cache: dict[str, str] = {}

def clear_schema_cache() -> None:
    """Clear cached schemas (call when tools change)."""
    global _schema_cache, _description_cache
    _schema_cache.clear()
    _description_cache.clear()
```

```
Modify get_optimized_description():
─────────────────────────────────────────────────────────
def get_optimized_description(tool_name: str, full_desc: str, mode: str = "standard") -> str:
    cache_key = f"{tool_name}:{mode}"
    if cache_key in _description_cache:
        return _description_cache[cache_key]

    # ... existing logic ...

    _description_cache[cache_key] = result
    return result
```

```
Modify get_optimized_schema():
─────────────────────────────────────────────────────────
def get_optimized_schema(schema: dict[str, Any], mode: str = "standard") -> dict[str, Any]:
    # Create cache key from schema hash + mode
    schema_hash = hash(json.dumps(schema, sort_keys=True))
    cache_key = f"{schema_hash}:{mode}"

    if cache_key in _schema_cache:
        return _schema_cache[cache_key]

    # ... existing logic ...

    _schema_cache[cache_key] = result
    return result
```

---

### 1.4 Token Usage Logging

**Goal**: Track token usage for monitoring and debugging

**Files to Modify**:
- `src/aec_agent/config/settings.py`
- `src/aec_agent/frontend/agent.py`

**Implementation**:

```
Step 1: Add setting
─────────────────────────────────────────────────────────
File: src/aec_agent/config/settings.py

Add:
  enable_token_logging: bool = Field(
      default=True,
      description="Log token usage per request"
  )
```

```
Step 2: Add logging to agent
─────────────────────────────────────────────────────────
File: src/aec_agent/frontend/agent.py

Add after line 1032 (response = await self.backend.generate...):

  if get_settings().enable_token_logging:
      logger.info(
          "Token usage",
          estimated_input=self._estimate_context_tokens(),
          tools_loaded=len(tools),
          compression_mode=self._get_dynamic_compression_mode(),
          history_messages=len(self.messages),
      )
```

---

## Phase 2: MEP Domain Optimization (3-5 Days)

### 2.1 Domain-Specific Tool Loading

**Goal**: Load only tools relevant to detected MEP domain

**Files to Modify**:
- `src/aec_agent/frontend/tool_optimization.py`
- `src/aec_agent/frontend/agent.py`

**Implementation**:

```
Step 1: Define domain tool mappings
─────────────────────────────────────────────────────────
File: src/aec_agent/frontend/tool_optimization.py

Add after MEP_TOOL_TIERS:

# Domain-specific tool priorities
DOMAIN_PRIORITY_TOOLS = {
    "hvac": {
        "must_have": ["find_elements", "get_nearby_elements", "sync_metadata"],
        "useful": ["get_related_elements", "draw_line_between", "get_distance_between"],
        "exclude": []  # Tools to never load for HVAC
    },
    "electrical": {
        "must_have": ["find_elements", "get_nearby_elements", "sync_metadata"],
        "useful": ["get_related_elements"],
        "exclude": []
    },
    "plumbing": {
        "must_have": ["find_elements", "get_nearby_elements", "sync_metadata"],
        "useful": ["get_related_elements", "get_distance_between"],
        "exclude": []
    },
    "fire_protection": {
        "must_have": ["find_elements", "get_nearby_elements"],
        "useful": ["sync_metadata"],
        "exclude": []
    },
    "low_voltage": {
        "must_have": ["find_elements", "get_nearby_elements"],
        "useful": ["get_related_elements"],
        "exclude": []
    },
}

def get_domain_priority_tools(domain: str, include_useful: bool = True) -> set[str]:
    """Get prioritized tools for an MEP domain."""
    domain_lower = domain.lower()
    if domain_lower not in DOMAIN_PRIORITY_TOOLS:
        return set()

    config = DOMAIN_PRIORITY_TOOLS[domain_lower]
    tools = set(config["must_have"])
    if include_useful:
        tools.update(config["useful"])
    return tools
```

```
Step 2: Integrate into agent._get_tools()
─────────────────────────────────────────────────────────
File: src/aec_agent/frontend/agent.py

In _get_tools() method, after intent classification:

  # Domain-specific tool filtering
  domain_tools = None
  if intent and intent.is_mep_specific and intent.confidence > 0.5:
      from aec_agent.frontend.tool_optimization import get_domain_priority_tools
      domain_tools = get_domain_priority_tools(
          intent.domain.value,
          include_useful=(intent.confidence > 0.7)
      )
      logger.debug(
          "Domain-specific tool filtering",
          domain=intent.domain.value,
          priority_tools=list(domain_tools),
      )
```

---

### 2.2 Add Low Voltage Domain

**Goal**: Dedicated support for low voltage systems (security, fire alarm, BMS, AV, data/telecom)

**Files to Modify**:
- `src/aec_agent/intent/models.py`
- `src/aec_agent/intent/patterns.py`
- `src/aec_agent/frontend/tool_optimization.py`

**Implementation**:

```
Step 1: Add domain enum
─────────────────────────────────────────────────────────
File: src/aec_agent/intent/models.py

Modify MEPDomain enum:

class MEPDomain(str, Enum):
    HVAC = "hvac"
    ELECTRICAL = "electrical"
    PLUMBING = "plumbing"
    FIRE_PROTECTION = "fire_protection"
    LOW_VOLTAGE = "low_voltage"      # NEW
    GENERAL = "general"
```

```
Step 2: Add patterns
─────────────────────────────────────────────────────────
File: src/aec_agent/intent/patterns.py

Add to MEP_PATTERNS:

MEPDomain.LOW_VOLTAGE: {
    # Security
    "access control": (1.0, "security"),
    "card reader": (1.0, "security"),
    "cctv": (1.0, "security"),
    "camera": (0.8, "security"),
    "security panel": (1.0, "security"),
    "door contact": (0.9, "security"),
    "motion sensor": (0.8, "security"),
    "intrusion": (0.9, "security"),

    # Fire Alarm (different from fire protection - detection vs suppression)
    "fire alarm panel": (1.0, "fire_alarm"),
    "smoke detector": (1.0, "fire_alarm"),
    "heat detector": (1.0, "fire_alarm"),
    "pull station": (1.0, "fire_alarm"),
    "horn strobe": (1.0, "fire_alarm"),
    "notification appliance": (0.9, "fire_alarm"),
    "facp": (1.0, "fire_alarm"),

    # BMS/BAS
    "bms": (1.0, "bms"),
    "bas": (1.0, "bms"),
    "building automation": (1.0, "bms"),
    "ddc": (0.9, "bms"),
    "controller": (0.6, "bms"),
    "sensor": (0.5, "bms"),
    "thermostat": (0.8, "bms"),

    # AV
    "audio visual": (1.0, "av"),
    "av": (0.8, "av"),
    "speaker": (0.7, "av"),
    "projector": (0.9, "av"),
    "display": (0.6, "av"),
    "microphone": (0.8, "av"),

    # Data/Telecom
    "data outlet": (1.0, "data"),
    "network": (0.7, "data"),
    "ethernet": (0.9, "data"),
    "patch panel": (1.0, "data"),
    "idf": (1.0, "data"),
    "mdf": (1.0, "data"),
    "telecom": (0.9, "data"),
    "cat6": (1.0, "data"),
    "fiber": (0.8, "data"),
    "wifi": (0.8, "data"),
    "access point": (0.9, "data"),
    "wap": (0.9, "data"),
},
```

```
Step 3: Add category filters
─────────────────────────────────────────────────────────
File: src/aec_agent/frontend/tool_optimization.py

Add to MEP_CATEGORY_FILTERS:

"low_voltage": {
    "revit_categories": [
        "Communication Devices",
        "Data Devices",
        "Fire Alarm Devices",
        "Security Devices",
        "Telephone Devices",
        "Nurse Call Devices",
    ],
    "autocad_layers": [
        "LV-DATA", "LV-TELE", "LV-SEC", "LV-FA", "LV-AV",
        "LV-BMS", "LV-CCTV", "LV-ACC", "D-", "T-",
        "COMM", "TELECOM", "DATA", "SECURITY", "CCTV",
    ],
    "entity_types": [
        "data outlet", "camera", "card reader", "smoke detector",
        "speaker", "access point", "patch panel", "controller",
    ],
},
```

---

### 2.3 Result Field Filtering

**Goal**: Return only relevant fields based on query type

**Files to Modify**:
- `src/aec_agent/config/settings.py`
- `src/aec_agent/frontend/mcp_client.py`

**Implementation**:

```
Step 1: Add settings
─────────────────────────────────────────────────────────
File: src/aec_agent/config/settings.py

Add:
  result_field_preset: str = Field(
      default="standard",
      description="Result field filtering: minimal, standard, full"
  )
```

```
Step 2: Add field filtering
─────────────────────────────────────────────────────────
File: src/aec_agent/frontend/mcp_client.py

Add after ToolResult class:

RESULT_FIELD_PRESETS = {
    "minimal": {"id", "name", "type"},
    "standard": {"id", "name", "type", "layer", "category", "level", "location"},
    "full": None,  # All fields
}

def _filter_result_fields(data: Any, preset: str = "standard") -> Any:
    """Filter result data to include only specified fields."""
    allowed = RESULT_FIELD_PRESETS.get(preset)
    if allowed is None:
        return data

    if isinstance(data, dict):
        # Filter dict keys
        if "elements" in data and isinstance(data["elements"], list):
            data["elements"] = [
                _filter_result_fields(el, preset) for el in data["elements"]
            ]
        return {k: v for k, v in data.items() if k in allowed or k in {"elements", "count", "message"}}

    if isinstance(data, list):
        return [_filter_result_fields(item, preset) for item in data]

    return data
```

```
Step 3: Integrate into ToolResult.to_message_content()
─────────────────────────────────────────────────────────
Modify to_message_content():

  def to_message_content(self, max_length: Optional[int] = None) -> str:
      settings = get_settings()
      if max_length is None:
          max_length = settings.max_tool_result_chars

      if self.success and self.data:
          # Filter fields before serialization
          filtered_data = _filter_result_fields(self.data, settings.result_field_preset)
          content = json.dumps(filtered_data, separators=(',', ':'))
      # ... rest of method
```

---

## Phase 3: Advanced Optimizations (1-2 Weeks)

### 3.1 Conversation Summarization

**Goal**: Summarize old context to reduce tokens while preserving knowledge

**Files to Create/Modify**:
- `src/aec_agent/memory/summarizer.py` (NEW)
- `src/aec_agent/frontend/agent.py`

**Implementation**:

```
Step 1: Create summarizer module
─────────────────────────────────────────────────────────
File: src/aec_agent/memory/summarizer.py

"""
Conversation summarization for token optimization.
"""

from typing import Optional
import structlog

logger = structlog.get_logger(__name__)


class ConversationSummarizer:
    """Summarizes conversation history to reduce token usage."""

    def __init__(self, backend=None, max_summary_tokens: int = 200):
        self._backend = backend
        self._max_summary_tokens = max_summary_tokens
        self._cached_summary: Optional[str] = None
        self._summarized_until: int = 0

    async def get_summary(
        self,
        messages: list,
        keep_recent: int = 5,
    ) -> Optional[str]:
        """
        Get a summary of older messages.

        Args:
            messages: Full message list (excluding system)
            keep_recent: Number of recent messages to keep verbatim

        Returns:
            Summary string or None if not enough old messages
        """
        if len(messages) <= keep_recent:
            return None

        old_messages = messages[:-keep_recent]

        # Check if we need to regenerate
        if self._summarized_until >= len(old_messages):
            return self._cached_summary

        # Generate summary
        summary = await self._summarize_messages(old_messages)
        self._cached_summary = summary
        self._summarized_until = len(old_messages)

        return summary

    async def _summarize_messages(self, messages: list) -> str:
        """Generate summary of messages."""
        if self._backend is None:
            # Fallback: extract key actions without LLM
            return self._extract_key_points(messages)

        # Use LLM for smart summarization
        content = "\n".join([
            f"{m.role}: {m.content[:200]}" for m in messages
        ])

        prompt = f"""Summarize this conversation context in 2-3 sentences.
Focus on: what was requested, what was done, current state.

{content}

Summary:"""

        # Use cheaper/faster model if available
        response = await self._backend.generate(
            messages=[{"role": "user", "content": prompt}],
            tools=[],
        )
        return response.content[:self._max_summary_tokens * 4]

    def _extract_key_points(self, messages: list) -> str:
        """Extract key points without LLM."""
        actions = []
        for msg in messages:
            if msg.role == "assistant" and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc.get("function", {}).get("name", "unknown")
                    actions.append(tool_name)

        if actions:
            unique_actions = list(dict.fromkeys(actions))  # Preserve order, remove dupes
            return f"Previous actions: {', '.join(unique_actions[:10])}"
        return ""

    def clear(self) -> None:
        """Clear cached summary."""
        self._cached_summary = None
        self._summarized_until = 0
```

```
Step 2: Integrate into agent
─────────────────────────────────────────────────────────
File: src/aec_agent/frontend/agent.py

Add to AECAgent.__init__():

  self._summarizer = ConversationSummarizer()

Modify _trim_history() or create new method:

  async def _optimize_context(self) -> None:
      """Optimize context using summarization."""
      settings = get_settings()

      if not settings.enable_conversation_summarization:
          self._trim_history()
          return

      # Get summary of old messages
      non_system = self.messages[1:]  # Exclude system
      summary = await self._summarizer.get_summary(
          non_system,
          keep_recent=settings.max_history_messages // 2,
      )

      if summary:
          # Rebuild messages: system + summary + recent
          system_msg = self.messages[0]
          recent = self.messages[-(settings.max_history_messages // 2):]

          summary_msg = Message(
              role="system",
              content=f"[Previous context: {summary}]"
          )

          self.messages = [system_msg, summary_msg] + recent
```

---

### 3.2 MEP-Specific Tools

**Goal**: Add specialized tools for MEP workflows

**Files to Create**:
- `src/aec_agent/mcp/tools/mep_tools.py` (NEW)

**Implementation**:

```
File: src/aec_agent/mcp/tools/mep_tools.py
─────────────────────────────────────────────────────────

"""
MEP-specific MCP tools for HVAC, electrical, plumbing, and low voltage.
"""

from typing import Optional
from aec_agent.mcp.server import mcp
from aec_agent.mcp.tools.base import success_result, error_result, safe_tool

import structlog

logger = structlog.get_logger(__name__)


@mcp.tool()
@safe_tool
async def check_clearances(
    element_id: str,
    clearance_type: str = "maintenance",
    min_distance: float = 0.6,
) -> dict:
    """
    Check clearance requirements around an MEP element.

    Args:
        element_id: Element to check clearances for
        clearance_type: Type of clearance (maintenance, code, access)
        min_distance: Minimum required clearance in meters

    Returns:
        Clearance analysis with violations if any
    """
    # Implementation would use get_nearby_elements and compare distances
    pass


@mcp.tool()
@safe_tool
async def trace_system(
    start_element_id: str,
    system_type: str,
    max_depth: int = 50,
) -> dict:
    """
    Trace an MEP system from a starting element.

    Args:
        start_element_id: Starting element (e.g., diffuser, outlet)
        system_type: Type of system (duct, pipe, conduit, cable_tray)
        max_depth: Maximum elements to trace

    Returns:
        Connected elements in the system path
    """
    pass


@mcp.tool()
@safe_tool
async def find_clashes(
    system1: Optional[str] = None,
    system2: Optional[str] = None,
    tolerance: float = 0.01,
) -> dict:
    """
    Find clashes between MEP systems.

    Args:
        system1: First system type (hvac, electrical, plumbing) or None for all
        system2: Second system type or None for all
        tolerance: Clash tolerance in meters

    Returns:
        List of clashing elements with locations
    """
    pass


@mcp.tool()
@safe_tool
async def validate_mep_spacing(
    domain: str,
    level: Optional[str] = None,
) -> dict:
    """
    Validate MEP element spacing against standards.

    Args:
        domain: MEP domain (hvac, electrical, plumbing, low_voltage)
        level: Specific level to check or None for all

    Returns:
        Validation results with violations
    """
    pass
```

---

### 3.3 Provider-Specific Caching

**Goal**: Leverage provider caching capabilities

**Files to Modify**:
- `src/aec_agent/frontend/agent.py`

**Implementation** (for Anthropic):

```
Modify AnthropicBackend.generate():
─────────────────────────────────────────────────────────

# Add cache control for system prompt
if system_content:
    kwargs["system"] = [
        {
            "type": "text",
            "text": system_content,
            "cache_control": {"type": "ephemeral"}
        }
    ]

# Add cache control for tools
if tools:
    kwargs["tools"] = [
        {**tool, "cache_control": {"type": "ephemeral"}}
        for tool in tools
    ]
```

---

## Testing Plan

### Unit Tests

```bash
# Phase 1
pytest tests/unit/test_tool_optimization.py -v -k "dynamic_compression"
pytest tests/unit/test_tool_optimization.py -v -k "schema_cache"

# Phase 2
pytest tests/unit/test_intent_classifier.py -v -k "low_voltage"
pytest tests/unit/test_tool_optimization.py -v -k "domain_priority"

# Phase 3
pytest tests/unit/test_summarizer.py -v
pytest tests/unit/test_mep_tools.py -v
```

### Integration Tests

```bash
# Full system test with token logging
LOG_LEVEL=DEBUG pytest tests/test_mcp_server.py -v -s
```

### Manual Testing Checklist

- [ ] Verify compression escalates as conversation grows
- [ ] Confirm domain-specific tools load for MEP queries
- [ ] Test low voltage pattern matching
- [ ] Verify result field filtering works
- [ ] Check summarization preserves key context

---

## Rollout Strategy

### Phase 1: Conservative Defaults
```bash
# .env settings for initial rollout
ENABLE_DYNAMIC_COMPRESSION=true
TOOL_COMPRESSION_MODE=standard
ENABLE_TOKEN_LOGGING=true
```

### Phase 2: Enable New Features
```bash
MEP_DOMAIN_PRIORITY=all
RESULT_FIELD_PRESET=standard
```

### Phase 3: Full Optimization
```bash
ENABLE_CONVERSATION_SUMMARIZATION=true
RESULT_FIELD_PRESET=minimal  # For high-volume usage
```

---

## Success Metrics

| Metric | Baseline | Phase 1 Target | Phase 2 Target | Phase 3 Target |
|--------|----------|----------------|----------------|----------------|
| Avg Input Tokens | 4000 | 3200 (-20%) | 2400 (-40%) | 2000 (-50%) |
| Tools Loaded | 20+ | 15 | 10 | 8 |
| Response Latency | Baseline | -10% | -15% | -20% |
| MEP Query Accuracy | Baseline | Baseline | +10% | +15% |

---

## Dependencies & Prerequisites

### Phase 1
- No new dependencies
- Backward compatible

### Phase 2
- Update `intent/patterns.py` with new patterns
- Migration: None required

### Phase 3
- Optional: Anthropic prompt caching beta access
- PostgreSQL for relationship tracing

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Over-compression loses context | Token logging + fallback to standard |
| Wrong domain classification | Confidence threshold (>0.5 required) |
| Summary loses critical info | Keep recent messages verbatim |
| Breaking changes | Feature flags for all new behavior |

---

*Plan created: 2026-01-23*
*Estimated total effort: 2-3 weeks*
