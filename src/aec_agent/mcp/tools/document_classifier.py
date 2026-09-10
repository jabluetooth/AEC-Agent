"""
Document classification for AEC drawings.

Uses LLM to classify drawing type and configure the vectorization pipeline
appropriately. This is Layer 1 of the Semantic Intelligence Pipeline.

Drawing types determine:
- Which symbols to look for
- How to interpret text annotations
- What geometric patterns to expect
- Which layers to assign to elements
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class DrawingType(str, Enum):
    """Type of engineering drawing."""
    FLOOR_PLAN = "floor_plan"           # Architectural floor plan
    MEP_PLAN = "mep_plan"               # Mechanical/Electrical/Plumbing overlay
    ELECTRICAL_PLAN = "electrical"       # Power, lighting, panels
    HVAC_PLAN = "hvac"                  # Ductwork, diffusers, equipment
    PLUMBING_PLAN = "plumbing"          # Piping, fixtures, drains
    FIRE_ALARM = "fire_alarm"           # FA devices, wiring, FACP
    FIRE_PROTECTION = "fire_protection"  # Sprinklers, standpipes
    SINGLE_LINE = "single_line"         # Electrical single-line diagram
    RISER_DIAGRAM = "riser"             # Vertical system diagram
    DETAIL = "detail"                   # Construction detail
    SCHEDULE = "schedule"               # Equipment/room schedule
    LEGEND = "legend"                   # Symbol legend
    SITE_PLAN = "site_plan"             # Site/civil plan
    STRUCTURAL = "structural"           # Structural framing
    UNKNOWN = "unknown"


@dataclass
class DrawingClassification:
    """Result of drawing classification."""
    drawing_type: DrawingType
    confidence: float  # 0.0 - 1.0
    discipline: str  # "architectural", "mechanical", "electrical", etc.
    subdiscipline: str | None = None  # More specific category
    sheet_number: str | None = None  # Extracted from title block
    sheet_name: str | None = None  # Extracted from title block
    scale: str | None = None  # Drawing scale if detected
    reasoning: str = ""  # LLM's explanation for classification
    metadata: dict[str, Any] = field(default_factory=dict)


# Mapping of drawing types to disciplines and expected content
DRAWING_TYPE_CONFIG: dict[DrawingType, dict[str, Any]] = {
    DrawingType.FLOOR_PLAN: {
        "discipline": "architectural",
        "symbol_categories": ["doors", "windows", "stairs", "elevators"],
        "text_focus": ["room_names", "room_numbers", "areas"],
        "geometry_patterns": ["walls", "partitions", "doors"],
        "layer_prefix": "A-",
    },
    DrawingType.MEP_PLAN: {
        "discipline": "mep",
        "symbol_categories": ["mechanical", "electrical", "plumbing", "fire"],
        "text_focus": ["equipment_tags", "sizes", "flow_rates"],
        "geometry_patterns": ["ducts", "pipes", "conduits"],
        "layer_prefix": "M-,E-,P-",
    },
    DrawingType.ELECTRICAL_PLAN: {
        "discipline": "electrical",
        "symbol_categories": ["electrical"],
        "text_focus": ["circuit_ids", "panel_names", "wattages"],
        "geometry_patterns": ["conduits", "wiring_runs"],
        "layer_prefix": "E-",
    },
    DrawingType.HVAC_PLAN: {
        "discipline": "mechanical",
        "symbol_categories": ["mechanical"],
        "text_focus": ["cfm", "equipment_tags", "duct_sizes"],
        "geometry_patterns": ["ducts", "rectangular_ducts", "round_ducts"],
        "layer_prefix": "M-",
    },
    DrawingType.PLUMBING_PLAN: {
        "discipline": "plumbing",
        "symbol_categories": ["plumbing"],
        "text_focus": ["pipe_sizes", "fixture_counts", "flow_rates"],
        "geometry_patterns": ["pipes", "fixtures"],
        "layer_prefix": "P-",
    },
    DrawingType.FIRE_ALARM: {
        "discipline": "fire_alarm",
        "symbol_categories": ["fire"],
        "text_focus": ["device_tags", "zone_ids", "circuit_ids"],
        "geometry_patterns": ["wiring_runs", "conduits"],
        "layer_prefix": "FA-",
    },
    DrawingType.FIRE_PROTECTION: {
        "discipline": "fire_protection",
        "symbol_categories": ["fire"],
        "text_focus": ["pipe_sizes", "head_types", "coverage_areas"],
        "geometry_patterns": ["pipes", "branch_lines", "mains"],
        "layer_prefix": "FP-",
    },
    DrawingType.SINGLE_LINE: {
        "discipline": "electrical",
        "symbol_categories": ["electrical"],
        "text_focus": ["equipment_names", "ratings", "connections"],
        "geometry_patterns": ["connection_lines", "bus_bars"],
        "layer_prefix": "E-DIAG-",
    },
    DrawingType.RISER_DIAGRAM: {
        "discipline": "mep",
        "symbol_categories": ["mechanical", "electrical", "plumbing"],
        "text_focus": ["floor_labels", "equipment_tags", "sizes"],
        "geometry_patterns": ["vertical_lines", "floor_markers"],
        "layer_prefix": "DIAG-",
    },
    DrawingType.DETAIL: {
        "discipline": "general",
        "symbol_categories": [],
        "text_focus": ["dimensions", "notes", "materials"],
        "geometry_patterns": ["all"],
        "layer_prefix": "DET-",
    },
    DrawingType.SCHEDULE: {
        "discipline": "general",
        "symbol_categories": [],
        "text_focus": ["table_data", "specifications"],
        "geometry_patterns": ["table_lines", "grid"],
        "layer_prefix": "SCHED-",
    },
    DrawingType.LEGEND: {
        "discipline": "general",
        "symbol_categories": ["all"],
        "text_focus": ["symbol_descriptions", "abbreviations"],
        "geometry_patterns": ["symbols", "lines"],
        "layer_prefix": "LEG-",
    },
    DrawingType.SITE_PLAN: {
        "discipline": "civil",
        "symbol_categories": [],
        "text_focus": ["dimensions", "elevations", "bearings"],
        "geometry_patterns": ["property_lines", "contours"],
        "layer_prefix": "C-",
    },
    DrawingType.STRUCTURAL: {
        "discipline": "structural",
        "symbol_categories": [],
        "text_focus": ["grid_lines", "beam_sizes", "column_marks"],
        "geometry_patterns": ["grid", "beams", "columns"],
        "layer_prefix": "S-",
    },
    DrawingType.UNKNOWN: {
        "discipline": "unknown",
        "symbol_categories": ["all"],
        "text_focus": ["all"],
        "geometry_patterns": ["all"],
        "layer_prefix": "",
    },
}


# Keywords that indicate drawing types (used for fast rule-based classification)
DRAWING_TYPE_KEYWORDS: dict[DrawingType, list[str]] = {
    DrawingType.FLOOR_PLAN: [
        "floor plan", "floorplan", "architectural", "arch plan",
        "a-", "a1", "a2", "a3", "a100", "a101", "a200",
    ],
    DrawingType.ELECTRICAL_PLAN: [
        "electrical", "power plan", "lighting plan", "elec",
        "e-", "e1", "e2", "e100", "e101", "e200",
        "panel", "circuit", "outlet", "receptacle",
    ],
    DrawingType.HVAC_PLAN: [
        "hvac", "mechanical", "mech", "ductwork", "duct plan",
        "m-", "m1", "m2", "m100", "m101", "m200",
        "diffuser", "vav", "ahu", "cfm",
    ],
    DrawingType.PLUMBING_PLAN: [
        "plumbing", "plumb", "piping",
        "p-", "p1", "p2", "p100", "p101", "p200",
        "fixture", "drain", "waste", "vent", "dwv",
    ],
    DrawingType.FIRE_ALARM: [
        "fire alarm", "fa plan", "fire detection",
        "fa-", "fa1", "fa2",
        "smoke detector", "pull station", "facp", "horn strobe",
    ],
    DrawingType.FIRE_PROTECTION: [
        "fire protection", "sprinkler", "fp plan",
        "fp-", "fp1", "fp2",
        "standpipe", "fire pump", "sprinkler head",
    ],
    DrawingType.SINGLE_LINE: [
        "single line", "single-line", "one line", "one-line",
        "riser diagram", "power riser",
    ],
    DrawingType.RISER_DIAGRAM: [
        "riser", "vertical diagram", "stack diagram",
    ],
    DrawingType.MEP_PLAN: [
        "mep", "m/e/p", "combined",
    ],
    DrawingType.DETAIL: [
        "detail", "section", "elevation", "typ",
    ],
    DrawingType.SCHEDULE: [
        "schedule", "equipment list", "fixture list",
        "door schedule", "room schedule", "panel schedule",
    ],
    DrawingType.LEGEND: [
        "legend", "symbols", "abbreviations", "notes",
    ],
    DrawingType.SITE_PLAN: [
        "site plan", "civil", "grading", "survey",
        "c-", "c1", "c100",
    ],
    DrawingType.STRUCTURAL: [
        "structural", "framing", "foundation",
        "s-", "s1", "s100", "s200",
    ],
}


def classify_by_keywords(text: str) -> tuple[DrawingType, float]:
    """
    Fast rule-based classification using keyword matching.

    Args:
        text: Text from title block or drawing content.

    Returns:
        Tuple of (DrawingType, confidence)
    """
    text_lower = text.lower()
    scores: dict[DrawingType, int] = {}

    for dtype, keywords in DRAWING_TYPE_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in text_lower:
                # Longer keywords get higher weight
                score += len(keyword)
        if score > 0:
            scores[dtype] = score

    if not scores:
        return (DrawingType.UNKNOWN, 0.0)

    # Find the highest scoring type
    best_type = max(scores, key=lambda k: scores[k])
    best_score = scores[best_type]

    # Normalize confidence (higher score = higher confidence, max ~0.8 for keywords)
    max_possible = 50  # Approximate max keyword score
    confidence = min(0.8, best_score / max_possible)

    return (best_type, confidence)


async def classify_document(
    image_path: str | None = None,
    title_block_text: str | None = None,
    sample_annotations: list[str] | None = None,
    use_llm: bool = True,
    llm_provider: str = "groq",
) -> DrawingClassification:
    """
    Classify the type of engineering drawing.

    Uses a multi-stage approach:
    1. Keyword matching on title block text (fast, no LLM cost)
    2. LLM classification if keywords are ambiguous (optional)
    3. Visual analysis of drawing content (future, with Vision LLM)

    Args:
        image_path: Path to the drawing image (for future Vision LLM)
        title_block_text: Text extracted from the title block
        sample_annotations: Sample text annotations from the drawing
        use_llm: Whether to use LLM for ambiguous cases
        llm_provider: LLM provider to use ("groq", "gemini", "openai")

    Returns:
        DrawingClassification with type, confidence, and metadata
    """
    # Combine available text for keyword analysis
    all_text = ""
    if title_block_text:
        all_text += title_block_text + " "
    if sample_annotations:
        all_text += " ".join(sample_annotations)

    # Stage 1: Keyword-based classification
    keyword_type, keyword_confidence = classify_by_keywords(all_text)

    logger.info(
        "Keyword classification result",
        drawing_type=keyword_type.value,
        confidence=keyword_confidence,
    )

    # If high confidence from keywords, use that result
    if keyword_confidence >= 0.6:
        config = DRAWING_TYPE_CONFIG.get(keyword_type, {})
        return DrawingClassification(
            drawing_type=keyword_type,
            confidence=keyword_confidence,
            discipline=config.get("discipline", "unknown"),
            reasoning=f"Classified by keyword matching (confidence: {keyword_confidence:.2f})",
        )

    # Stage 2: LLM classification for ambiguous cases
    if use_llm and all_text.strip():
        try:
            llm_result = await _classify_with_llm(all_text, llm_provider)
            if llm_result.confidence > keyword_confidence:
                return llm_result
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")

    # Fallback to keyword result or UNKNOWN
    if keyword_type != DrawingType.UNKNOWN:
        config = DRAWING_TYPE_CONFIG.get(keyword_type, {})
        return DrawingClassification(
            drawing_type=keyword_type,
            confidence=keyword_confidence,
            discipline=config.get("discipline", "unknown"),
            reasoning="Classified by keyword matching (low confidence)",
        )

    return DrawingClassification(
        drawing_type=DrawingType.UNKNOWN,
        confidence=0.0,
        discipline="unknown",
        reasoning="Could not determine drawing type",
    )


async def _classify_with_llm(
    text: str,
    provider: str = "groq",
) -> DrawingClassification:
    """
    Use LLM to classify drawing type from text content.

    Args:
        text: Combined text from title block and annotations
        provider: LLM provider to use

    Returns:
        DrawingClassification from LLM analysis
    """
    import json

    # Build the classification prompt
    drawing_types_list = ", ".join([dt.value for dt in DrawingType])

    prompt = f"""Analyze this text from an engineering drawing and classify it.

Text from drawing:
---
{text[:2000]}  # Limit text length
---

Classify this as ONE of these drawing types:
{drawing_types_list}

Return a JSON object with these fields:
- drawing_type: one of the types listed above
- confidence: 0.0 to 1.0
- discipline: "architectural", "mechanical", "electrical", "plumbing", "fire_alarm", "fire_protection", "civil", "structural", or "general"
- sheet_number: extracted sheet number if visible (e.g., "E1.01", "M-101"), or null
- sheet_name: extracted sheet name if visible, or null
- reasoning: brief explanation of why you chose this classification

Return ONLY valid JSON, no markdown or explanation."""

    try:
        # Try to use the configured LLM provider
        response_text = await _call_llm(prompt, provider)

        # Parse JSON response
        # Handle potential markdown code blocks
        if "```" in response_text:
            # Extract JSON from code block
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                response_text = response_text[start:end]

        result = json.loads(response_text)

        # Map to DrawingType enum
        dtype_str = result.get("drawing_type", "unknown")
        try:
            dtype = DrawingType(dtype_str)
        except ValueError:
            dtype = DrawingType.UNKNOWN

        config = DRAWING_TYPE_CONFIG.get(dtype, {})

        return DrawingClassification(
            drawing_type=dtype,
            confidence=float(result.get("confidence", 0.5)),
            discipline=result.get("discipline", config.get("discipline", "unknown")),
            sheet_number=result.get("sheet_number"),
            sheet_name=result.get("sheet_name"),
            reasoning=result.get("reasoning", ""),
        )

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM response as JSON: {e}")
        raise
    except Exception as e:
        logger.warning(f"LLM classification error: {e}")
        raise


async def _call_llm(prompt: str, provider: str = "groq") -> str:
    """
    Call LLM provider with the given prompt.

    Supports: groq, gemini, openai

    Args:
        prompt: The prompt to send
        provider: LLM provider name

    Returns:
        LLM response text
    """
    if provider == "groq":
        return await _call_groq(prompt)
    elif provider == "gemini":
        return await _call_gemini(prompt)
    elif provider == "openai":
        return await _call_openai(prompt)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


async def _call_groq(prompt: str) -> str:
    """Call Groq API."""
    import httpx

    from aec_agent.config.settings import get_settings

    settings = get_settings()
    api_key = settings.groq_api_key
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.groq_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 500,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


async def _call_gemini(prompt: str) -> str:
    """Call Google Gemini API."""
    import os

    import httpx

    from aec_agent.config.settings import get_settings

    settings = get_settings()
    # Preserve original fallback order: GOOGLE_API_KEY env var takes precedence
    # over the settings-backed GEMINI_API_KEY.
    api_key = os.environ.get("GOOGLE_API_KEY") or settings.gemini_api_key
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{settings.gemini_model}:generateContent?key={api_key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 500},
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


async def _call_openai(prompt: str) -> str:
    """Call OpenAI API."""
    import httpx

    from aec_agent.config.settings import get_settings

    settings = get_settings()
    api_key = settings.openai_api_key
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 500,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


def get_pipeline_config(classification: DrawingClassification) -> dict[str, Any]:
    """
    Get vectorization pipeline configuration for a classified drawing.

    Returns configuration parameters that should be used for the
    vectorization pipeline based on the drawing type.

    Args:
        classification: The drawing classification result

    Returns:
        Dictionary of pipeline configuration parameters
    """
    config = DRAWING_TYPE_CONFIG.get(classification.drawing_type, {})

    return {
        # Symbol detection configuration
        "symbol_categories": config.get("symbol_categories", ["all"]),
        "symbol_detection": bool(config.get("symbol_categories")),

        # OCR configuration
        "ocr_focus": config.get("text_focus", ["all"]),
        "ocr_masking": True,

        # Geometry configuration
        "geometry_patterns": config.get("geometry_patterns", ["all"]),
        "aec_heuristics": True,

        # Layer assignment
        "layer_prefix": config.get("layer_prefix", ""),

        # Drawing metadata
        "discipline": classification.discipline,
        "drawing_type": classification.drawing_type.value,
    }
