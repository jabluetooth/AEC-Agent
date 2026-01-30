# MEP Workflow Enhancement Plan

## Overview

This document tracks the implementation of advanced MEP (Mechanical, Electrical, Plumbing) workflow support for the AEC Agent, with a focus on minimizing token usage while capturing how MEP designers think.

**Key Goals:**
- Reduce token usage by 50-70% for typical MEP workflows
- Capture MEP design patterns (routing priorities, sizing logic, coordination rules)
- Enable workflow automation without replacing designers
- Leverage existing PostgreSQL + pgvector infrastructure

---

## User Requirements

| Requirement | Choice |
|-------------|--------|
| **MEP Priority** | HVAC first (most complex discipline) |
| **Starting Phase** | Phase 1: Intent Classification |
| **User Identity** | Windows username (`os.getlogin()`) |
| **Standards Source** | CAD templates (DWT/RTE files) |

---

## Implementation Status

| Phase | Status | Token Impact |
|-------|--------|--------------|
| Phase 1: Intent Classification | ✅ Complete | -40-60% tools |
| Phase 2: MEP Knowledge Base | ✅ Complete | -500-1000 |
| Phase 3: Workflow Templates | ✅ Complete | -2000/workflow |
| Phase 4: Project Memory | ✅ Complete | -1800 |
| Phase 5: User Preferences | ✅ Complete | -20-50 |
| Phase 6: Rules Engine | ✅ Complete | -500-1000 |

---

## Phase 1: Intent Classification ✅

**Objective:** Replace basic keyword matching with MEP-aware intent classification.

### Files Created
```
src/aec_agent/intent/
├── __init__.py          # Module exports
├── models.py            # IntentResult, MEPDomain, MEPAction, AppContext
├── patterns.py          # 150+ HVAC-focused keywords
└── classifier.py        # IntentClassifier with regex matching

tests/unit/
└── test_intent_classifier.py  # 42 unit tests
```

### Files Modified
- `src/aec_agent/frontend/agent.py` - Integrated IntentClassifier
- `src/aec_agent/frontend/tool_optimization.py` - Added MEP_TOOL_TIERS
- `src/aec_agent/config/settings.py` - Added intent settings

### Key Components

```python
@dataclass
class IntentResult:
    domain: MEPDomain      # hvac, electrical, plumbing, fire_protection, general
    subdomain: str         # equipment, distribution, terminals, etc.
    action: MEPAction      # query, create, modify, route, analyze, etc.
    app_context: AppContext  # autocad, revit, both
    confidence: float
    suggested_tools: list[str]
```

### HVAC Patterns (Priority)
- **Equipment**: AHU, VAV, FCU, RTU, chiller, boiler
- **Distribution**: ductwork, supply/return air, main trunk, branch
- **Terminals**: diffuser, grille, register, air terminal
- **Controls**: thermostat, damper, control valve, sensor
- **Parameters**: CFM, airflow, static pressure, velocity, ACH
- **Sizing**: duct sizing, pressure drop, friction loss

### Settings Added
```python
enable_intent_classification: bool = True
use_intent_embeddings: bool = False
intent_min_confidence: float = 0.3
mep_domain_priority: str = "hvac"
```

---

## Phase 2: MEP Knowledge Base ✅

**Objective:** Store MEP design rules in PostgreSQL for on-demand retrieval.

### Files Created
```
src/aec_agent/domain/
├── __init__.py          # Module exports
├── models.py            # DomainRule, SystemPriority, ValidationStatus
├── knowledge.py         # MEPKnowledgeBase class with rule management
└── seed_data.py         # 20+ pre-populated HVAC rules
```

### Database Tables (in `002_mep_enhancement.py`)
- `domain_rules` - Stores clearance, routing, sizing, priority rules
- `system_priorities` - Coordination priority rankings by project

### Pre-populated Rules (HVAC)
| Rule Type | Example |
|-----------|---------|
| Clearance | Duct min 6" from structural beam |
| Clearance | Maintain 2" insulation clearance for supply ducts |
| Routing | Gravity drain has priority over pressurized |
| Routing | Main trunk parallel to structure, branches perpendicular |
| Sizing | Max duct velocity 4000 FPM for low noise |
| Sizing | Return air velocity can be 20% higher than supply |
| Access | Dampers need 18" clear for service access |

### Key Class: MEPKnowledgeBase
```python
class MEPKnowledgeBase:
    def get_clearance_rules(self, element_type, near_type)
    def get_sizing_rules(self, system_type)
    def get_routing_rules(self, subdomain)
    def get_applicable_rules(self, context)
    async def search_rules_semantic(self, query, limit)
```

---

## Phase 3: Workflow Templates ✅

**Objective:** Pre-defined multi-step workflows that execute without per-step LLM calls.

### Files Created
```
src/aec_agent/workflows/
├── __init__.py          # Module exports
├── models.py            # WorkflowStep, WorkflowTemplate, WorkflowExecution, StepResult
├── executor.py          # WorkflowExecutor class
└── templates.py         # 6 pre-defined HVAC workflows
```

### Database Tables (in `002_mep_enhancement.py`)
- `workflow_templates` - Stores reusable workflow definitions
- `workflow_executions` - Tracks running/completed workflow instances

### Key Class: WorkflowExecutor
```python
class WorkflowExecutor:
    async def start_workflow(self, template_name, context, project_id, user_id)
    async def execute_step(self, execution_id)
    async def run_to_completion(self, execution_id, max_steps)
    def list_templates(self, domain)
```

### HVAC Workflows Implemented

1. **hvac_duct_routing_analysis**
   - sync_metadata → find_elements (AHU, VAV) → get_nearby_elements → check clearances
   - Inputs: source, start_equipment, end_terminals
   - Outputs: route path, clearance warnings

2. **hvac_equipment_schedule**
   - sync_metadata → find_elements (equipment, AHUs, VAVs, FCUs)
   - Inputs: source
   - Outputs: formatted equipment schedule

3. **hvac_duct_sizing_check**
   - sync_metadata → find_elements (supply, return, exhaust ducts)
   - Inputs: source, max_velocity_fpm
   - Outputs: sizing report with velocity analysis

4. **hvac_diffuser_coverage**
   - sync_metadata → find_elements (diffusers, rooms) → check_spacing
   - Inputs: source, room_filter, max_spacing_m
   - Outputs: coverage analysis

5. **mep_clash_detection**
   - sync_metadata → find_elements (ducts, pipes) → check_intersections
   - Inputs: source
   - Outputs: clash report

6. **hvac_system_balance**
   - sync_metadata → find_elements (supply, return, exhaust terminals)
   - Inputs: source
   - Outputs: air balance analysis

---

## Phase 4: Project Memory ✅

**Objective:** Persist project context and decisions across sessions.

### Files Created
```
src/aec_agent/memory/
├── __init__.py          # Module exports
├── models.py            # ProjectFact, FactType, ConversationSummary, MemoryContext
└── project_memory.py    # ProjectMemory class
```

### Database Tables (in `002_mep_enhancement.py`)
- `project_facts` - Stores decisions, constraints, preferences, notes
- `conversation_summaries` - Compressed conversation history

### Key Class: ProjectMemory
```python
class ProjectMemory:
    async def store_fact(self, project_id, fact_type, key, value, source, confidence)
    async def get_facts(self, project_id, fact_type, include_expired, min_confidence)
    async def search_facts(self, project_id, query, limit)  # Semantic search
    async def store_summary(self, project_id, user_id, summary, key_decisions)
    async def get_context(self, project_id, user_id, query, token_budget)
```

### Fact Types
- `decision`: "HVAC system will be VAV with central AHU"
- `constraint`: "Max ceiling height 3.0m on Level 2"
- `preference`: "Use rectangular duct over spiral"
- `note`: "Client wants extra receptacles in conference rooms"
- `standard`: "ASHRAE 62.1 ventilation requirements"
- `assumption`: "Assume 10 CFM/person for offices"

### Context Generation
```python
# Get compact context for LLM injection
context = await memory.get_context(project_id, user_id, query, token_budget=500)
prompt += context.to_context_string()  # ~200 tokens vs ~2000 for full history
```

---

## Phase 5: User Preferences ✅

**Objective:** Store user-specific settings using Windows username.

### Files Created
```
src/aec_agent/memory/
└── user_preferences.py  # UserPreferences class with shortcuts
```

### Database Tables (in `002_mep_enhancement.py`)
- `user_preferences` - Category-organized user settings
- `user_shortcuts` - Command aliases with usage tracking

### User Identification
```python
from aec_agent.memory import get_current_user
user = get_current_user()  # Returns 'frelatorre' on Windows
```

### Key Class: UserPreferences
```python
class UserPreferences:
    async def get(self, category, key, default)
    async def set(self, category, key, value)
    async def get_category(self, category)
    async def add_shortcut(self, alias, expansion, description)
    async def expand_shortcut(self, text)
    async def get_mep_defaults(self)
    async def apply_unit_conversion(self, value, from_unit, to_preference)
```

### Default Preferences (HVAC Focus)
```python
DEFAULT_PREFERENCES = {
    "display": {
        "units": "imperial",
        "pressure_units": "inWG",
        "velocity_units": "fpm",
    },
    "workflow": {
        "preferred_duct_shape": "rectangular",
        "routing_method": "trunk_and_branch",
        "auto_size_ducts": True,
    },
    "defaults": {
        "max_supply_velocity": 2000,  # fpm
        "max_return_velocity": 2400,
        "friction_loss_target": 0.08,  # inWG/100ft
    },
    "mep": {
        "primary_discipline": "hvac",
        "coordination_priority": ["plumbing", "hvac", "electrical"],
    },
}
```

### Shortcuts Example
```python
prefs = await get_user_preferences()
await prefs.add_shortcut("rs", "route supply duct from AHU", "Start supply routing")
text = await prefs.expand_shortcut("rs to diffusers")
# Returns: "route supply duct from AHU to diffusers"
```

---

## Phase 6: Rules Engine ✅

**Objective:** Proactive validation and suggestions during design operations.

### Files Created
```
src/aec_agent/domain/
├── rules_engine.py      # RulesEngine class with validation and suggestions
└── validators.py        # ClearanceValidator, SizingValidator, RoutingValidator, CoverageValidator
```

### Database Tables (in `002_mep_enhancement.py`)
- `validation_results` - Cached validation checks per element
- `design_suggestions` - AI-generated design recommendations

### Key Class: RulesEngine
```python
class RulesEngine:
    async def validate_element(self, project_id, element, rule_types)
    async def validate_project(self, project_id, elements, rule_types)
    async def generate_suggestions(self, project_id, element, context)
    async def get_pending_suggestions(self, project_id, limit)
    async def dismiss_suggestion(self, suggestion_id)
    async def accept_suggestion(self, suggestion_id)
```

### Built-in Validators

**1. ClearanceValidator**
```python
validator = ClearanceValidator()
issue = validator.check_clearance(duct_element, beam_element)
# Returns ValidationIssue if clearance < 6" (duct-to-beam minimum)
```

**2. SizingValidator**
```python
validator = SizingValidator()
issue = validator.check_duct_velocity(element, space_type="office")
# Checks velocity limits: 2000 fpm max for supply branch in office
issue = validator.check_pipe_velocity(element)
# Checks: cold water 8 fps max, hot water 5 fps max
```

**3. RoutingValidator**
```python
validator = RoutingValidator()
issue = validator.check_slope(condensate_drain)
# Checks: minimum 1% slope for condensate
issue = validator.check_offset(duct_element, max_offset_angle=45)
```

**4. CoverageValidator**
```python
validator = CoverageValidator()
issue = validator.check_diffuser_coverage(room_area=800, diffuser_count=4)
# Checks: max 200 sq ft per square diffuser
issue = validator.check_sprinkler_coverage(area=1000, count=8, hazard="ordinary")
# Checks: NFPA 13 - max 130 sq ft per head for ordinary hazard
```

### Validation Results
```python
@dataclass
class ValidationResult:
    element_id: UUID
    rule_name: str
    status: ValidationStatus  # PASS, WARNING, ERROR
    message: str
    details: dict  # actual vs required values
```

---

## Implementation Dependencies

```
Phase 1 (Intent) ────────────────────────────────────┐
    │                                                │
    v                                                │
Phase 2 (Knowledge) ────> Phase 3 (Workflows)        │
    │                         │                      │
    v                         v                      │
Phase 4 (Memory) ────────> Phase 5 (Preferences)     │
    │                         │                      │
    └───────────┬─────────────┘                      │
                v                                    │
        Phase 6 (Rules Engine) <─────────────────────┘
```

**Critical Path:** Phase 1 → Phase 2 → Phase 4 → Phase 6

---

## Token Impact Summary

| Phase | Per-Query Savings | Mechanism |
|-------|-------------------|-----------|
| Phase 1 | -40-60% tools | Load only relevant tools |
| Phase 2 | -500-1000 | Rules in DB, not prompt |
| Phase 3 | -2000/workflow | Single summary call |
| Phase 4 | -1800 | Context injection vs history |
| Phase 5 | -20-50 | Pre-applied defaults |
| Phase 6 | -500-1000 | Prevents error cycles |

**Total Estimated Savings:** 50-70% reduction in token usage

---

## Critical Files Reference

| File | Role |
|------|------|
| `src/aec_agent/frontend/agent.py` | Core agent loop, intent integration |
| `src/aec_agent/frontend/tool_optimization.py` | Tool compression, MEP tiers |
| `src/aec_agent/db/repository.py` | Database queries |
| `src/aec_agent/semantic/search.py` | Semantic search |
| `src/aec_agent/config/settings.py` | Configuration |

---

## Testing

```bash
# Run intent classification tests
pytest tests/unit/test_intent_classifier.py -v

# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=src/aec_agent --cov-report=html
```
