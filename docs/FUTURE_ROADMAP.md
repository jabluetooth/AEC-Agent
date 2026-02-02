# AEC Agent - Future Development Roadmap

## Executive Summary

This document outlines the development roadmap for the AEC Agent, focusing on achieving the full vision: **an AI that can autonomously design MEP, Low Voltage, and Fire Alarm systems** using AutoCAD/Revit, with intelligent memory and semantic search.

**Key Insight**: Python-side OpenCV handles vectorization (lines-first detection, FastLineDetector, circle pixel validation). The AI focuses on understanding, designing, and decision-making.

---

## Vision Statement

```
┌─────────────────────────────────────────────────────────────┐
│                    WHAT THE AI WILL DO                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  "Design the HVAC system for Floor 3"                       │
│       ↓                                                      │
│  AI reads cached building geometry                           │
│       ↓                                                      │
│  AI calculates CFM per room (ASHRAE 62.1)                   │
│       ↓                                                      │
│  AI places diffusers, sizes ducts, routes system            │
│       ↓                                                      │
│  AI validates against codes, checks clashes                  │
│       ↓                                                      │
│  Complete HVAC design in AutoCAD/Revit                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Current State Assessment

### What's Working

| Component | Status | Quality |
|-----------|--------|---------|
| Multi-layer architecture | Deployed | Excellent |
| AutoCAD/Revit sidecar integration | Working | Good |
| MCP tool framework | Working | Good |
| PostgreSQL + PostGIS + pgvector | Ready | Good |
| Multi-LLM provider support | Working | Excellent |
| Token optimization | Implemented | Excellent |
| MEP intent classification | Working | Good |
| Conversation summarization | Implemented | Good |
| Raster Design / PDF vectorization | Working | Excellent |
| OpenCV vectorizer (lines-first, FLD, circle validation) | Working | Excellent |

### What's Missing

| Component | Priority | Notes |
|-----------|----------|-------|
| ~~Raster Design integration~~ | ~~**Critical**~~ | ~~Core PDF workflow~~ **DONE** |
| Auto-cache on file open | **Critical** | Memory foundation |
| Design knowledge base | **Critical** | Codes, standards, formulas |
| Autonomous design tools | **High** | Equipment placement, routing |
| Model routing by task | Medium | Cost optimization |
| Embedding-based tool selection | Medium | Smarter tool matching |

---

## Tool Gap Analysis

### Current Tool Inventory

**Total MCP Tools: 50**

| Category | Count | Tools |
|----------|-------|-------|
| Common | 4 | `ping`, `get_server_status`, `check_sidecar`, `sync_cache` |
| AutoCAD | 8 | Layers (3), Drawing (3), Query (2) |
| Revit | 7 | Levels (2), Walls (2), Rooms (1), Document (2) |
| Metadata | 5 | `find_elements`, `get_nearby_elements`, `get_related_elements`, `resolve_coordinates`, `sync_metadata` |
| MEP | 5 | `check_clearances`, `validate_mep_spacing`, `trace_system`, `find_clashes`, `get_mep_summary` |
| Smart Drawing | 4 | `draw_line_between`, `draw_circle_at`, `draw_rectangle_around`, `get_distance_between` |
| Raster Design | 17 | `raster_convert_pdf`, `raster_import_pdf`, `raster_attach_image`, `raster_cleanup`, `raster_vectorize`, `raster_auto_vectorize`, `raster_process_image`, `raster_create_primitive`, `raster_select_entities`, `raster_follower`, `raster_recognize_text`, `raster_ocr_extract`, `raster_get_status`, `raster_get_entity_count`, `raster_fade_image`, `raster_store_vectorized`, `raster_pdf_to_vector_pipeline` |

### Tool Readiness by Category

```
┌─────────────────────────────────────────────────────────────┐
│                    TOOL READINESS SCORE                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Query/Search:           ████████████████████ 100%          │
│  Basic Drawing:          ████████████████░░░░  80%          │
│  MEP Analysis:           ████████████░░░░░░░░  60%          │
│  Element Placement:      ░░░░░░░░░░░░░░░░░░░░   0%          │
│  Engineering Calcs:      ░░░░░░░░░░░░░░░░░░░░   0%          │
│  Routing/Pathfinding:    ░░░░░░░░░░░░░░░░░░░░   0%          │
│  Code Validation:        ░░░░░░░░░░░░░░░░░░░░   0%          │
│  Knowledge Query:        ░░░░░░░░░░░░░░░░░░░░   0%          │
│                                                              │
│  Raster/Vectorization:     ████████████████████ 100%          │
│                                                              │
│  OVERALL: 40% ready for autonomous design                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Gap Analysis by Phase

#### Phase 4: HVAC Autonomous Design

| Required Capability | Current Tool | Status |
|---------------------|--------------|--------|
| Read room geometry/area | `revit_list_rooms` | ✅ Has area_sqm |
| Calculate CFM per room | None | 🔴 **MISSING** |
| Select diffuser by CFM | None | 🔴 **MISSING** |
| Place diffuser at location | None | 🔴 **MISSING** |
| Size duct by CFM | None | 🔴 **MISSING** |
| Route duct (pathfinding) | None | 🔴 **MISSING** |
| Place VAV/AHU equipment | None | 🔴 **MISSING** |
| Detect clashes | `find_clashes` | ✅ Works |
| Validate against code | None | 🔴 **MISSING** |

**HVAC Readiness: 2/9 (22%)**

#### Phase 5: Fire Alarm Autonomous Design

| Required Capability | Current Tool | Status |
|---------------------|--------------|--------|
| Get room occupancy type | None | 🔴 **MISSING** |
| Calculate detector spacing | None | 🔴 **MISSING** |
| Place smoke detectors | None | 🔴 **MISSING** |
| Place heat detectors | None | 🔴 **MISSING** |
| Place horn/strobes | None | 🔴 **MISSING** |
| Design circuits/wiring | None | 🔴 **MISSING** |
| Place FACP | None | 🔴 **MISSING** |
| Validate NFPA 72 | None | 🔴 **MISSING** |

**Fire Alarm Readiness: 0/8 (0%)**

#### Phase 6: Low Voltage Autonomous Design

| Required Capability | Current Tool | Status |
|---------------------|--------------|--------|
| Place data outlets | None | 🔴 **MISSING** |
| Calculate WAP coverage | None | 🔴 **MISSING** |
| Place WAPs | None | 🔴 **MISSING** |
| Place cameras | None | 🔴 **MISSING** |
| Place access control | None | 🔴 **MISSING** |
| Route pathways | None | 🔴 **MISSING** |
| Size telecom room | None | 🔴 **MISSING** |

**Low Voltage Readiness: 0/7 (0%)**

#### Phase 7: Electrical Autonomous Design

| Required Capability | Current Tool | Status |
|---------------------|--------------|--------|
| Calculate room loads | None | 🔴 **MISSING** |
| Place receptacles | None | 🔴 **MISSING** |
| Calculate lighting | None | 🔴 **MISSING** |
| Place light fixtures | None | 🔴 **MISSING** |
| Size panels | None | 🔴 **MISSING** |
| Design circuits | None | 🔴 **MISSING** |
| Route conduit | None | 🔴 **MISSING** |
| Validate CEC | None | 🔴 **MISSING** |

**Electrical Readiness: 0/8 (0%)**

#### Phase 8: Plumbing Autonomous Design

| Required Capability | Current Tool | Status |
|---------------------|--------------|--------|
| Calculate fixture units | None | 🔴 **MISSING** |
| Size supply piping | None | 🔴 **MISSING** |
| Size DWV piping | None | 🔴 **MISSING** |
| Place plumbing fixtures | None | 🔴 **MISSING** |
| Place floor drains | None | 🔴 **MISSING** |
| Route piping (supply/waste) | None | 🔴 **MISSING** |
| Size water heater | None | 🔴 **MISSING** |
| Validate CPC | None | 🔴 **MISSING** |

**Plumbing Readiness: 0/8 (0%)**

---

### Critical Missing Tool Categories

#### 1. Family/Block Placement (BLOCKING)

```
Current State:
- Can draw primitives (lines, circles, rectangles)
- Cannot place Revit families or AutoCAD blocks

Impact:
- AI knows "place a 24x24 diffuser at (10, 15, 3)"
- No tool exists to actually place it
- Blocks ALL autonomous design phases
```

#### 2. Engineering Calculations (BLOCKING)

```
Missing Calculations:
- CFM calculation (room volume × ACH)
- Duct sizing (CFM → diameter using velocity limits)
- Electrical load (VA per sq ft × area)
- Lighting calculation (lumens/sq ft → fixture count)
- Detector spacing (NFPA 72 tables by ceiling height)
- Pipe sizing (fixture units → pipe diameter)
```

#### 3. Knowledge Base Query (BLOCKING)

```
Current State:
- Semantic search works on CAD elements only
- Cannot query codes/standards/equipment catalogs

Impact:
- AI asks: "What size duct for 600 CFM?"
- No tool to query duct sizing tables
- AI cannot access design rules at runtime
```

#### 4. Routing/Pathfinding (HIGH)

```
Current State:
- Can trace existing systems (trace_system)
- Cannot create new routes

Impact:
- AI knows: "Route duct from AHU to diffuser"
- No tool to find optimal path avoiding obstacles
- No tool to create the actual duct run
```

#### 5. Code Validation (HIGH)

```
Missing Validations (Los Angeles):
- CMC: California Mechanical Code (ventilation, HVAC)
- CPC: California Plumbing Code (fixture units, pipe sizing)
- CEC: California Electrical Code (receptacles, circuits, Title 24 lighting)
- CFC: California Fire Code (detector coverage, sprinklers)
- LAMC: Los Angeles Municipal Code (local amendments)
- ASHRAE 62.1: Ventilation rates (ref by CMC)
- NFPA 72/13: Fire alarm & sprinkler (ref by CFC)
```

---

### Priority Tools to Build

#### Tier 1: Unlock All Placement (Critical)

| Tool | Target | Description |
|------|--------|-------------|
| `place_revit_family` | Revit | Place any family at XYZ with rotation |
| `place_autocad_block` | AutoCAD | Insert block at XYZ with scale/rotation |
| `query_knowledge_base` | Both | Search codes/standards/equipment |

#### Tier 2: Enable Calculations (Critical)

| Tool | Domain | Description |
|------|--------|-------------|
| `calculate_ventilation` | Mechanical | CFM by room type/occupancy (ASHRAE 62.1) |
| `calculate_duct_size` | Mechanical | CFM → duct dimensions |
| `calculate_fixture_units` | Plumbing | Fixture units per CPC Table 7-3 |
| `calculate_pipe_size` | Plumbing | Fixture units → pipe diameter |
| `calculate_electrical_load` | Electrical | VA by room type/area (CEC 220) |
| `calculate_detector_spacing` | Fire | NFPA 72 spacing tables |
| `select_equipment` | All | Match requirements to catalog |

#### Tier 3: Enable Routing (High)

| Tool | Domain | Description |
|------|--------|-------------|
| `find_route` | All | A* pathfinding avoiding obstacles |
| `create_duct_run` | Mechanical | Create duct between points |
| `create_pipe_run` | Plumbing | Create supply/DWV pipe runs |
| `create_conduit_run` | Electrical | Create conduit between points |
| `create_cable_tray_run` | Low Voltage | Create cable tray routes |

#### Tier 4: Enable Validation (High)

| Tool | Code | Description |
|------|------|-------------|
| `validate_cmc` | CMC + ASHRAE | Check mechanical/HVAC compliance |
| `validate_cpc` | CPC | Check plumbing compliance |
| `validate_cec` | CEC + Title 24 | Check electrical compliance |
| `validate_cfc` | CFC + NFPA 72/13 | Check fire protection compliance |
| `validate_lamc` | LAMC | Check LA-specific amendments |
| `generate_compliance_report` | All | Summary of violations by code

---

### Tool Development Sequence

```
Phase 3 (Knowledge Base):
├── query_knowledge_base        ← Unlocks design rules access
├── select_equipment            ← Unlocks equipment selection
└── calculate_* (all)           ← Unlocks engineering math

Phase 4 (HVAC):
├── place_revit_family          ← Unlocks ALL family placement
├── place_autocad_block         ← Unlocks ALL block insertion
├── find_route                  ← Unlocks pathfinding
├── create_duct_run             ← Unlocks duct creation
└── validate_ashrae             ← Unlocks code checking

Phase 5-7 (Fire/LV/Elec):
├── create_conduit_run          ← Reuse routing logic
├── create_cable_tray_run       ← Reuse routing logic
├── validate_nfpa72             ← Fire alarm validation
└── validate_nec                ← Electrical validation
```

---

### Summary: The AI Can Think But Has No Hands

```
┌─────────────────────────────────────────────────────────────┐
│                    CURRENT CAPABILITY                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ✅ WORKING (Analysis & Understanding)                       │
│  ├── Find elements by natural language                       │
│  ├── Query spatial relationships                             │
│  ├── Trace existing MEP systems                              │
│  ├── Detect clashes between systems                          │
│  ├── Check clearances around elements                        │
│  └── Draw primitive shapes (lines, circles)                  │
│                                                              │
│  🔴 MISSING (Design & Creation)                              │
│  ├── Place equipment (diffusers, panels, detectors)         │
│  ├── Calculate engineering values (CFM, loads, spacing)     │
│  ├── Query design rules (codes, standards)                   │
│  ├── Route systems (ducts, conduits, pipes)                  │
│  └── Validate against codes                                  │
│                                                              │
│  CONCLUSION:                                                 │
│  AI can understand and analyze existing designs              │
│  AI cannot create new MEP designs                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Free Tier Model Strategy

### Recommended Stack (No Vision Required)

```
┌─────────────────────────────────────────────────────────────┐
│                    MODEL ALLOCATION                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ORCHESTRATOR / ROUTING                                      │
│  └─► Groq Llama 3.3 70B (fast, free)                        │
│                                                              │
│  DESIGN & ENGINEERING DECISIONS                              │
│  └─► Groq Llama 3.3 70B (primary)                           │
│  └─► Gemini 1.5 Flash (fallback)                            │
│                                                              │
│  TOOL EXECUTION                                              │
│  └─► Groq Llama 3.3 70B (fastest)                           │
│                                                              │
│  EMBEDDINGS                                                  │
│  └─► all-MiniLM-L6-v2 (local, free)                         │
│                                                              │
│  OFFLINE FALLBACK                                            │
│  └─► Ollama + Llama 3.2 (local, unlimited)                  │
│                                                              │
│  VISION (Optional - Edge Cases Only)                         │
│  └─► Gemini 1.5 Flash (when needed)                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Future Paid Upgrades

| When | Upgrade | Why |
|------|---------|-----|
| Production use | Groq → Claude Sonnet | Better reasoning |
| Enterprise | Claude Opus | Complex multi-system design |
| High accuracy needed | GPT-4o | Validated performance |

---

## Knowledge Base Architecture

### What is the Knowledge Base?

The knowledge base stores **engineering rules, codes, standards, and formulas** that the AI uses to make design decisions. It's not hardcoded - you can update it anytime.

### Format Options

| Format | Pros | Cons | Recommended For |
|--------|------|------|-----------------|
| **Markdown files** | Easy to edit, version control | No structure validation | General guidelines |
| **JSON/YAML** | Structured, parseable | Harder to read | Formulas, equipment data |
| **PostgreSQL tables** | Queryable, relational | Needs DB access to edit | Large datasets |
| **Embeddings + RAG** | Semantic search | More complex | Large code books |

### Jurisdiction: Los Angeles, California

All codes are based on **California state codes** with **Los Angeles local amendments**.

```
Code Hierarchy:
┌─────────────────────────────────────────────────────────────┐
│  Federal / National Standards                               │
│  (ASHRAE, NFPA, SMACNA, ICC model codes)                   │
│                         ↓                                   │
│  California State Codes (Title 24)                          │
│  (CBC, CEC, CMC, CPC, CFC, Energy Code)                    │
│                         ↓                                   │
│  Los Angeles Municipal Code (LAMC)                          │
│  (Local amendments, stricter requirements)                  │
└─────────────────────────────────────────────────────────────┘
```

### Recommended Hybrid Approach

```
knowledge_base/
├── codes/
│   │
│   │  # === MECHANICAL (HVAC) ===
│   ├── cmc_2022.md               # California Mechanical Code (based on UMC)
│   ├── ashrae_62_1.md            # Ventilation for Acceptable IAQ
│   ├── ashrae_90_1.md            # Energy Standard (ref by Title 24)
│   ├── smacna_duct.md            # Duct construction standards
│   ├── title_24_part6.md         # California Energy Code
│   │
│   │  # === ELECTRICAL ===
│   ├── cec_2022.md               # California Electrical Code (based on NEC)
│   ├── lamc_electrical.md        # LA amendments to CEC
│   ├── title_24_lighting.md      # CA lighting power density
│   │
│   │  # === PLUMBING ===
│   ├── cpc_2022.md               # California Plumbing Code (based on UPC)
│   ├── lamc_plumbing.md          # LA amendments to CPC
│   ├── iapmo_guidelines.md       # IAPMO installation standards
│   │
│   │  # === FIRE PROTECTION ===
│   ├── cfc_2022.md               # California Fire Code
│   ├── nfpa_72.md                # Fire alarm systems
│   ├── nfpa_13.md                # Sprinkler systems
│   ├── lafd_requirements.md      # LA Fire Dept specific rules
│   │
│   │  # === BUILDING / GENERAL ===
│   ├── cbc_2022.md               # California Building Code (based on IBC)
│   ├── lamc_building.md          # LA amendments to CBC
│   └── ada_accessibility.md      # ADA/CBC Chapter 11B
│
├── standards/
│   │
│   │  # === MECHANICAL ===
│   ├── duct_sizing.yaml          # Duct sizing (equal friction method)
│   ├── diffuser_selection.yaml   # Diffuser throw/NC ratings
│   ├── vav_sizing.yaml           # VAV box selection
│   ├── refrigerant_piping.yaml   # Refrigerant line sizing
│   │
│   │  # === PLUMBING ===
│   ├── pipe_sizing.yaml          # DWV and water pipe sizing
│   ├── fixture_units.yaml        # Fixture unit values (CPC Table 7-3)
│   ├── water_heater_sizing.yaml  # Water heater capacity
│   ├── gas_pipe_sizing.yaml      # Natural gas piping
│   │
│   │  # === ELECTRICAL ===
│   ├── electrical_load.yaml      # Load calculations per CEC
│   ├── wire_sizing.yaml          # Conductor sizing tables
│   ├── conduit_fill.yaml         # Conduit fill calculations
│   ├── panel_sizing.yaml         # Panel/breaker selection
│   │
│   │  # === FIRE / LOW VOLTAGE ===
│   ├── fire_alarm_spacing.yaml   # Detector spacing (NFPA 72)
│   ├── sprinkler_coverage.yaml   # Sprinkler layout (NFPA 13)
│   └── low_voltage_standards.yaml
│
├── equipment/
│   ├── diffusers.json            # Diffuser catalog
│   ├── vav_boxes.json            # VAV equipment
│   ├── air_handlers.json         # AHU specifications
│   ├── pumps.json                # Plumbing pumps
│   ├── water_heaters.json        # Water heater catalog
│   ├── plumbing_fixtures.json    # Fixture specifications
│   ├── fire_alarm_devices.json   # FA device catalog
│   ├── electrical_panels.json    # Panel specifications
│   └── lighting_fixtures.json    # Light fixture catalog
│
├── formulas/
│   ├── hvac_formulas.yaml        # CFM, static pressure, psychrometrics
│   ├── plumbing_formulas.yaml    # GPM, head loss, fixture units
│   ├── electrical_formulas.yaml  # VA, voltage drop, fault current
│   └── fire_formulas.yaml        # Sprinkler hydraulics, detector coverage
│
└── preferences/
    ├── company_standards.yaml    # TBD - will be provided later
    └── project_defaults.yaml     # Project-specific overrides
```

### California Code Reference Table

| Domain | California Code | Based On | Current Edition | LA Amendments |
|--------|----------------|----------|-----------------|---------------|
| **Building** | CBC (Title 24 Part 2) | IBC | 2022 | LAMC Chapter IX |
| **Mechanical** | CMC (Title 24 Part 4) | UMC | 2022 | LAMC §91.100+ |
| **Plumbing** | CPC (Title 24 Part 5) | UPC | 2022 | LAMC §94.1000+ |
| **Electrical** | CEC (Title 24 Part 3) | NEC | 2022 | LAMC §93.0100+ |
| **Fire** | CFC (Title 24 Part 9) | IFC | 2022 | LAMC Chapter 57 |
| **Energy** | Title 24 Part 6 | ASHRAE 90.1 | 2022 | — |
| **Green** | CALGreen (Title 24 Part 11) | IgCC | 2022 | LA Green Building |

### Key LA-Specific Requirements

```yaml
los_angeles_amendments:
  mechanical:
    - "Seismic bracing required for all ductwork (LAMC)"
    - "Kitchen exhaust: Type I hood for all commercial cooking"
    - "Smoke control required for high-rise (75ft+)"

  plumbing:
    - "Water conservation: max 1.28 GPF toilets, 1.8 GPM faucets"
    - "Graywater systems encouraged (CA Water Code)"
    - "Backflow prevention on all potable connections"

  electrical:
    - "AFCI protection required in more locations than NEC"
    - "EV charging rough-in required (CALGreen)"
    - "Solar-ready requirements for new construction"

  fire:
    - "LAFD Plan Check for all commercial projects"
    - "High-rise: emergency voice/alarm required"
    - "Automatic sprinklers required at lower thresholds"
```

### Example: Duct Sizing Rules (YAML)

```yaml
# knowledge_base/standards/duct_sizing.yaml
duct_sizing:
  method: equal_friction

  velocity_limits:
    main_duct:
      max_fpm: 2000
      recommended_fpm: 1500
    branch_duct:
      max_fpm: 1200
      recommended_fpm: 800
    flex_duct:
      max_fpm: 800
      max_length_ft: 6

  friction_rate:
    low_pressure:
      max_in_wg_per_100ft: 0.08
    medium_pressure:
      max_in_wg_per_100ft: 0.15

  aspect_ratio:
    max: 4:1
    preferred: 2:1
```

### Example: Fire Alarm Spacing (YAML)

```yaml
# knowledge_base/standards/fire_alarm_spacing.yaml
nfpa_72:
  smoke_detectors:
    spot_type:
      max_spacing_ft: 30
      max_from_wall_ft: 15
      ceiling_height_adjustment:
        - ceiling_ft: 10
          spacing_ft: 30
        - ceiling_ft: 20
          spacing_ft: 28
        - ceiling_ft: 30
          spacing_ft: 26

    beam_detector:
      max_spacing_ft: 60
      max_width_ft: 60

  heat_detectors:
    max_spacing_ft: 50
    max_from_wall_ft: 25

  notification_appliances:
    horn_strobe:
      max_room_size_sqft: 3600
      candela_by_room_size:
        - max_sqft: 400
          candela: 15
        - max_sqft: 900
          candela: 30
        - max_sqft: 1600
          candela: 60
```

### Example: Equipment Catalog (JSON)

```json
{
  "diffusers": [
    {
      "model": "24x24-4way",
      "manufacturer": "Generic",
      "type": "4-way ceiling diffuser",
      "size_in": [24, 24],
      "cfm_range": [200, 800],
      "neck_sizes": [8, 10, 12, 14],
      "throw_ft": {
        "200cfm": 4,
        "400cfm": 7,
        "600cfm": 10,
        "800cfm": 12
      },
      "nc_rating": 25
    }
  ]
}
```

### How to Update/Overwrite

| Method | How | When to Use |
|--------|-----|-------------|
| **Edit files directly** | Modify .md/.yaml/.json files | Small updates |
| **Admin UI (future)** | Web interface for editing | Non-technical users |
| **API endpoint** | POST to update endpoint | Automated updates |
| **Git version control** | Commit changes to repo | Track history |
| **Project overrides** | Project-specific file | Per-project customization |

### Override Priority

```
Base Knowledge (codes, standards)
        ↓
Company Standards (your preferences)
        ↓
Project Overrides (this project only)
        ↓
User Preferences (this user only)
        ↓
Conversation Context (this session only)
```

Higher levels override lower levels. You can always go back to defaults.

### Loading into the System

```
Startup:
1. Load base knowledge files
2. Generate embeddings for semantic search
3. Store in PostgreSQL for fast retrieval

Runtime:
1. User asks: "Size the duct for 600 CFM"
2. AI searches knowledge base for duct sizing rules
3. AI applies formula with user's parameters
4. AI returns: "Use 12" round duct (velocity: 764 FPM)"
```

---

## Development Phases (Revised)

### Phase 1: Foundation -- COMPLETE

**Goal**: File caching + Groq/Gemini integration

| Task | Description | Status |
|------|-------------|--------|
| 1.1 | Add Groq as primary LLM provider | Done |
| 1.2 | Add Gemini as fallback provider | Done |
| 1.3 | Implement auto-cache on file open event | Done |
| 1.4 | Extract file metadata to PostgreSQL | Done |
| 1.5 | Load cached context instead of re-reading files | Done |
| 1.6 | LLM provider fallback chain (Groq -> Gemini -> OpenAI) | Done |
| 1.7 | REST endpoint for sidecar file-open notifications | Done |
| 1.8 | Register all metadata + MEP tools in MCP server | Done |

**Deliverables**:
- PostgreSQL 18.1 with PostGIS 3.6.1 + pgvector 0.8.1 (15 tables)
- Files cached automatically when opened (REST + MCP tool)
- 90% token reduction via `get_file_context` tool
- Groq + Gemini + fallback chain working
- 179 tests passing

---

### Phase 2: Raster Design Integration -- COMPLETE

**Goal**: PDF to DWG pipeline via Python-side OpenCV vectorization

| Task | Description | Status |
|------|-------------|--------|
| 2.1 | Add Raster Design commands to AutoCAD sidecar (14 commands) | Done |
| 2.2 | PDF import and raster image handling (PyMuPDF → bitonal TIFF) | Done |
| 2.3 | Despeckle and cleanup automation | Done |
| 2.4 | OpenCV auto-vectorization (replaced interactive VTools) | Done |
| 2.5 | OCR text extraction (`irectext`) | Done |
| 2.6 | Store vectorized entities in PostgreSQL with embeddings | Done |
| 2.7 | Lines-first detection order (fix false circle detection) | Done |
| 2.8 | FastLineDetector (FLD) integration with HoughLinesP fallback | Done |
| 2.9 | Circle pixel validation (circumference ink sampling) | Done |
| 2.10 | Topology cleanup by default (NetworkX graph-based) | Done |
| 2.11 | Tool prefix fix (`raster_` included in AutoCAD context) | Done |

**Deliverables**:
- Upload PDF → vectorized DWG (17 MCP tools, 14 sidecar commands)
- Lines-first detection: Lines 85→1,973 (+2,221%), Circles 205→6 (-97%) on test PDF
- FastLineDetector (opencv-contrib) as primary detector with HoughLinesP fallback
- Circle pixel validation rejects false circles at line intersections
- Topology cleanup merges fragmented lines, snaps dangling endpoints
- Text extracted and stored
- Entities searchable in database
- 179 tests passing

---

### Phase 3: Knowledge Base (3-4 weeks)

**Goal**: Design knowledge that AI can query and you can update

**Jurisdiction**: Los Angeles, California (California codes + LA amendments)

| Task | Description |
|------|-------------|
| 3.1 | Create knowledge base file structure |
| 3.2 | Add **Mechanical** codes (CMC, ASHRAE, SMACNA) |
| 3.3 | Add **Plumbing** codes (CPC, IAPMO, fixture units) |
| 3.4 | Add **Electrical** codes (CEC, Title 24 lighting) |
| 3.5 | Add **Fire Protection** codes (CFC, NFPA 72/13, LAFD) |
| 3.6 | Add **Low Voltage** standards (TIA/EIA, BICSI) |
| 3.7 | Add **LA amendments** (LAMC Chapter IX, 93, 94, 57) |
| 3.8 | Knowledge base loader + embeddings |
| 3.9 | Semantic search over knowledge |
| 3.10 | Override/update mechanism |

**Deliverables**:
- Knowledge base files in place (all MEP disciplines)
- California/LA code compliance built-in
- AI can query codes/standards
- Easy to update/override
- Company standards slot ready (TBD)

---

### Phase 4: Autonomous Design - HVAC (4-6 weeks)

**Goal**: AI designs HVAC systems from requirements

| Task | Description |
|------|-------------|
| 4.1 | Room-by-room CFM calculation |
| 4.2 | Diffuser selection and placement |
| 4.3 | Duct sizing calculations |
| 4.4 | Duct routing (pathfinding) |
| 4.5 | Equipment placement (AHU, VAV) |
| 4.6 | Clash detection during design |
| 4.7 | Design validation against codes |

**Deliverables**:
- "Design HVAC for Floor 3" → Complete system
- Proper sizing, routing, placement
- Code-compliant output

---

### Phase 5: Autonomous Design - Fire Alarm (3-4 weeks)

**Goal**: AI designs fire alarm systems per NFPA 72

| Task | Description |
|------|-------------|
| 5.1 | Room classification (occupancy type) |
| 5.2 | Detector spacing calculations |
| 5.3 | Detector placement by coverage |
| 5.4 | Notification appliance layout |
| 5.5 | Device wiring/circuit design |
| 5.6 | FACP location logic |
| 5.7 | Validation against NFPA 72 |

**Deliverables**:
- "Design fire alarm for Building A" → Complete system
- Proper coverage and spacing
- NFPA 72 compliant

---

### Phase 6: Autonomous Design - Low Voltage (3-4 weeks)

**Goal**: AI designs low voltage systems

| Task | Description |
|------|-------------|
| 6.1 | Data outlet placement per room type |
| 6.2 | WAP placement for coverage |
| 6.3 | Security camera placement |
| 6.4 | Access control device layout |
| 6.5 | Pathway/raceway routing |
| 6.6 | Telecom room sizing |
| 6.7 | Structured cabling design |

**Deliverables**:
- "Design data network for Floor 2" → Complete design
- Proper coverage calculations
- Standards-compliant layout

---

### Phase 7: Autonomous Design - Electrical (4-6 weeks)

**Goal**: AI designs electrical systems per CEC (California Electrical Code)

| Task | Description |
|------|-------------|
| 7.1 | Load calculations per room (CEC Article 220) |
| 7.2 | Receptacle placement per code (CEC 210.52) |
| 7.3 | Lighting layout calculations (Title 24 Part 6) |
| 7.4 | Panel sizing and placement |
| 7.5 | Circuit design and homerun |
| 7.6 | Conduit routing |
| 7.7 | CEC validation |

**Deliverables**:
- "Design electrical for Office Area" → Complete system
- Load calcs, panel schedules
- CEC compliant (California Electrical Code)

---

### Phase 8: Autonomous Design - Plumbing (3-4 weeks)

**Goal**: AI designs plumbing systems per CPC (California Plumbing Code)

| Task | Description |
|------|-------------|
| 8.1 | Fixture unit calculations (CPC Table 7-3) |
| 8.2 | Water supply pipe sizing |
| 8.3 | DWV (drain/waste/vent) pipe sizing |
| 8.4 | Fixture placement by code |
| 8.5 | Floor drain placement |
| 8.6 | Water heater sizing and location |
| 8.7 | Gas piping (if applicable) |
| 8.8 | Pipe routing (supply, waste, vent) |
| 8.9 | CPC validation |

**Deliverables**:
- "Design plumbing for Restroom Core" → Complete system
- Proper pipe sizing, fixture units
- CPC compliant (California Plumbing Code)

---

### Phase 9: Multi-System Coordination (2-3 weeks)

**Goal**: AI coordinates across all MEP systems

| Task | Description |
|------|-------------|
| 9.1 | Cross-system clash detection (M/E/P/FP) |
| 9.2 | Elevation/priority rules (gravity vs pressure) |
| 9.3 | Automated clash resolution |
| 9.4 | Coordination report generation |
| 9.5 | Iterative design refinement |

**Deliverables**:
- AI detects clashes across all systems
- Suggests or auto-resolves conflicts
- Coordination documentation

---

### Phase 10: Intelligence & Learning (Ongoing)

**Goal**: AI gets smarter over time

| Task | Description |
|------|-------------|
| 10.1 | Track design decisions |
| 10.2 | Learn from user corrections |
| 10.3 | Project-specific memory |
| 10.4 | Cross-project learning |
| 10.5 | Preference prediction |

**Deliverables**:
- AI remembers past decisions
- Gets better with use
- Personalized to your workflow

---

## Database Schema (Updated)

```
┌─────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE BASE TABLES                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  knowledge_entries                                           │
│  ├── id (UUID)                                               │
│  ├── category (ENUM: code, standard, formula, equipment)    │
│  ├── domain (ENUM: hvac, electrical, plumbing, fire, lv)    │
│  ├── source (TEXT) -- "ASHRAE 62.1", "NFPA 72"              │
│  ├── title (TEXT)                                            │
│  ├── content (TEXT)                                          │
│  ├── content_embedding (VECTOR)                              │
│  ├── metadata (JSONB) -- formulas, values, etc.             │
│  ├── version (TEXT)                                          │
│  ├── is_override (BOOL)                                      │
│  ├── override_source (TEXT) -- "company", "project", "user" │
│  ├── created_at (TIMESTAMP)                                  │
│  └── updated_at (TIMESTAMP)                                  │
│                                                              │
│  equipment_catalog                                           │
│  ├── id (UUID)                                               │
│  ├── category (TEXT) -- "diffuser", "vav", "panel"          │
│  ├── manufacturer (TEXT)                                     │
│  ├── model (TEXT)                                            │
│  ├── specifications (JSONB)                                  │
│  ├── description_embedding (VECTOR)                          │
│  └── is_preferred (BOOL)                                     │
│                                                              │
│  design_decisions                                            │
│  ├── id (UUID)                                               │
│  ├── project_id (UUID)                                       │
│  ├── system_type (TEXT)                                      │
│  ├── decision (TEXT)                                         │
│  ├── rationale (TEXT) -- why AI made this choice            │
│  ├── knowledge_refs (UUID[]) -- which rules were used       │
│  ├── user_approved (BOOL)                                    │
│  ├── user_modified (BOOL)                                    │
│  └── created_at (TIMESTAMP)                                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Timeline Summary (Revised)

| Phase | Focus | Duration | Cumulative |
|-------|-------|----------|------------|
| 1 | Foundation (cache, providers) | **COMPLETE** | - |
| 2 | Raster Design integration | **COMPLETE** | - |
| 3 | Knowledge base (LA codes) | 3-4 weeks | 10 weeks |
| 4 | **Mechanical** (HVAC) autonomous design | 4-6 weeks | 16 weeks |
| 5 | **Fire Protection** autonomous design | 3-4 weeks | 20 weeks |
| 6 | **Low Voltage** autonomous design | 3-4 weeks | 24 weeks |
| 7 | **Electrical** autonomous design | 4-6 weeks | 30 weeks |
| 8 | **Plumbing** autonomous design | 3-4 weeks | 34 weeks |
| 9 | Multi-system coordination | 2-3 weeks | 37 weeks |
| 10 | Intelligence & learning | Ongoing | - |

**Total: ~9 months** to full autonomous MEP design capability

### MEP Coverage Summary

| Discipline | Phase | California Code |
|------------|-------|-----------------|
| **M** - Mechanical (HVAC) | 4 | CMC + ASHRAE |
| **E** - Electrical | 7 | CEC + Title 24 |
| **P** - Plumbing | 8 | CPC |
| Fire Protection | 5 | CFC + NFPA |
| Low Voltage | 6 | TIA/EIA + BICSI |

---

## Knowledge Base Maintenance

### How to Add New Knowledge

```
1. Create/edit file in knowledge_base/
2. Restart system (or call reload API)
3. System generates embeddings
4. Knowledge available immediately
```

### How to Override

```yaml
# knowledge_base/preferences/company_standards.yaml

overrides:
  - source: "ASHRAE 62.1"
    rule: "office_cfm_per_person"
    original_value: 5
    override_value: 7
    reason: "Company standard for better air quality"

  - source: "duct_sizing"
    rule: "max_velocity_fpm"
    original_value: 2000
    override_value: 1800
    reason: "Noise reduction preference"
```

### How to Add Project-Specific Rules

```yaml
# knowledge_base/projects/building_a.yaml

project: "Building A"
overrides:
  - rule: "diffuser_type"
    value: "linear_slot"
    reason: "Architect requirement"

  - rule: "ceiling_height_ft"
    value: 12
    reason: "Actual field condition"
```

---

## Success Metrics

### Phase 1-2 Success
- [x] Files cached automatically
- [x] PDF → DWG vectorization pipeline working (17 tools)
- [x] 90% token reduction for file queries
- [x] Lines-first detection order eliminates false circles
- [x] FastLineDetector + HoughLinesP dual-detector pipeline
- [x] Circle pixel validation (35% ink threshold)
- [x] Topology cleanup (merge degree-2 nodes, snap dangling endpoints)

### Phase 3 Success
- [ ] Knowledge base searchable
- [ ] Overrides working correctly
- [ ] AI references correct codes

### Phase 4-7 Success
- [ ] Autonomous design accuracy > 80%
- [ ] Code compliance validation working
- [ ] User corrections < 20% of designs

### Phase 8-9 Success
- [ ] Cross-system clashes detected > 90%
- [ ] AI remembers past decisions
- [ ] Design time reduced by 50%

---

## Next Steps (Immediate)

### Phase 2 Wrap-Up
1. ~~**Add Groq provider**~~ - Done (primary free LLM)
2. ~~**Implement file open caching**~~ - Done (Foundation for memory)
3. **End-to-end integration test** with real PDF + running AutoCAD sidecar

### Critical Tools (Phase 3-4)
4. **Build `query_knowledge_base` tool** - Unlock design rules access
5. **Build `place_revit_family` tool** - Unlock ALL element placement
6. **Build `place_autocad_block` tool** - Unlock ALL block insertion
7. **Build calculation tools** - `calculate_ventilation`, `calculate_duct_size`

### Supporting Tools (Phase 4+)
8. **Build `find_route` tool** - A* pathfinding for routing
9. **Build `create_duct_run` tool** - Duct creation
10. **Build validation tools** - Code compliance checking

### Order of Priority
```
query_knowledge_base  →  place_revit_family  →  calculate_*  →  find_route
        ↓                       ↓                    ↓              ↓
   Access rules          Place equipment      Size systems    Route systems
```

---

*Document created: 2025-01-23*
*Last updated: 2026-02-02*
*Jurisdiction: Los Angeles, California*
*Status: Phase 1-2 COMPLETE, Phase 3 (Knowledge Base) next*
