"""
Tests for symbol classification prompts.

Phase C: Symbol Intelligence - Tests for Vision LLM prompt generation.
"""

import pytest

from aec_agent.mcp.tools.document_classifier import DrawingType
from aec_agent.prompts.symbol_classification import (
    SYMBOL_SUBTYPES,
    get_symbol_classification_prompt,
    get_unknown_symbol_prompt,
    get_batch_symbol_prompt,
    get_valve_classification_prompt,
    get_diffuser_classification_prompt,
    get_outlet_classification_prompt,
    get_detector_classification_prompt,
)


class TestSymbolSubtypes:
    """Tests for SYMBOL_SUBTYPES dictionary."""

    def test_has_all_categories(self):
        """Should have all MEP categories."""
        expected = {"mechanical", "electrical", "fire", "plumbing", "low_voltage"}
        assert set(SYMBOL_SUBTYPES.keys()) == expected

    def test_mechanical_has_valve_subtypes(self):
        """Mechanical category should have valve subtypes."""
        assert "valve" in SYMBOL_SUBTYPES["mechanical"]
        valves = SYMBOL_SUBTYPES["mechanical"]["valve"]
        assert "gate_valve" in valves
        assert "ball_valve" in valves
        assert "butterfly_valve" in valves
        assert len(valves) >= 8

    def test_mechanical_has_diffuser_subtypes(self):
        """Mechanical category should have diffuser subtypes."""
        assert "diffuser" in SYMBOL_SUBTYPES["mechanical"]
        diffusers = SYMBOL_SUBTYPES["mechanical"]["diffuser"]
        assert "supply_air_diffuser" in diffusers
        assert "return_air_grille" in diffusers
        assert len(diffusers) >= 6

    def test_electrical_has_outlet_subtypes(self):
        """Electrical category should have outlet subtypes."""
        assert "outlet" in SYMBOL_SUBTYPES["electrical"]
        outlets = SYMBOL_SUBTYPES["electrical"]["outlet"]
        assert "duplex_outlet" in outlets
        assert "gfci_outlet" in outlets
        assert len(outlets) >= 6

    def test_fire_has_detection_subtypes(self):
        """Fire category should have detection subtypes."""
        assert "detection" in SYMBOL_SUBTYPES["fire"]
        detectors = SYMBOL_SUBTYPES["fire"]["detection"]
        assert "smoke_detector" in detectors
        assert "heat_detector" in detectors


class TestSymbolClassificationPrompt:
    """Tests for get_symbol_classification_prompt."""

    def test_basic_prompt_generation(self):
        """Should generate a valid prompt with basic inputs."""
        prompt = get_symbol_classification_prompt(
            yolo_class="valve",
            yolo_category="mechanical",
            drawing_type=DrawingType.HVAC_PLAN,
        )
        assert "valve" in prompt.lower()
        assert "mechanical" in prompt.lower()
        assert "hvac" in prompt.lower()
        assert "JSON" in prompt

    def test_includes_nearby_text(self):
        """Should include nearby text in prompt."""
        prompt = get_symbol_classification_prompt(
            yolo_class="valve",
            yolo_category="mechanical",
            drawing_type=DrawingType.PLUMBING_PLAN,
            nearby_text=["3/4\" GV", "DCW"],
        )
        assert "3/4\" GV" in prompt
        assert "DCW" in prompt

    def test_includes_confidence(self):
        """Should include YOLO confidence in prompt."""
        prompt = get_symbol_classification_prompt(
            yolo_class="outlet",
            yolo_category="electrical",
            drawing_type=DrawingType.ELECTRICAL_PLAN,
            yolo_confidence=0.75,
        )
        assert "0.75" in prompt

    def test_includes_subtypes_hint(self):
        """Should include possible subtypes when available."""
        prompt = get_symbol_classification_prompt(
            yolo_class="valve",
            yolo_category="mechanical",
            drawing_type=DrawingType.PLUMBING_PLAN,
        )
        assert "gate_valve" in prompt or "ball_valve" in prompt

    def test_includes_discipline_hints(self):
        """Should include discipline-specific hints."""
        hvac_prompt = get_symbol_classification_prompt(
            yolo_class="diffuser",
            yolo_category="mechanical",
            drawing_type=DrawingType.HVAC_PLAN,
        )
        assert "CFM" in hvac_prompt or "SA" in hvac_prompt

        elec_prompt = get_symbol_classification_prompt(
            yolo_class="outlet",
            yolo_category="electrical",
            drawing_type=DrawingType.ELECTRICAL_PLAN,
        )
        assert "Circuit" in elec_prompt or "GFI" in elec_prompt

    def test_requests_json_output(self):
        """Should request JSON output format."""
        prompt = get_symbol_classification_prompt(
            yolo_class="valve",
            yolo_category="mechanical",
            drawing_type=DrawingType.MEP_PLAN,
        )
        assert "JSON" in prompt
        assert "category" in prompt
        assert "subtype" in prompt
        assert "confidence" in prompt


class TestUnknownSymbolPrompt:
    """Tests for get_unknown_symbol_prompt."""

    def test_basic_unknown_prompt(self):
        """Should generate prompt for unknown symbol."""
        prompt = get_unknown_symbol_prompt(drawing_type=DrawingType.HVAC_PLAN)
        assert "Identify" in prompt or "identify" in prompt
        assert "JSON" in prompt
        assert "category" in prompt

    def test_includes_nearby_text(self):
        """Should include nearby text context."""
        prompt = get_unknown_symbol_prompt(
            drawing_type=DrawingType.ELECTRICAL_PLAN,
            nearby_text=["20A", "GFCI"],
        )
        assert "20A" in prompt
        assert "GFCI" in prompt

    def test_includes_discipline_context(self):
        """Should include discipline-specific context."""
        prompt = get_unknown_symbol_prompt(drawing_type=DrawingType.FIRE_ALARM)
        assert "fire" in prompt.lower() or "alarm" in prompt.lower()


class TestBatchSymbolPrompt:
    """Tests for get_batch_symbol_prompt."""

    def test_batch_prompt_generation(self):
        """Should generate batch prompt for multiple symbols."""
        symbols = [
            {"yolo_class": "valve", "yolo_category": "mechanical", "nearby_text": []},
            {"yolo_class": "outlet", "yolo_category": "electrical", "nearby_text": ["GFCI"]},
        ]
        prompt = get_batch_symbol_prompt(symbols, DrawingType.MEP_PLAN)
        assert "1." in prompt
        assert "2." in prompt
        assert "valve" in prompt.lower()
        assert "outlet" in prompt.lower()
        assert "JSON array" in prompt

    def test_limits_to_10_symbols(self):
        """Should limit batch to 10 symbols."""
        symbols = [
            {"yolo_class": f"symbol{i}", "yolo_category": "unknown", "nearby_text": []}
            for i in range(15)
        ]
        prompt = get_batch_symbol_prompt(symbols, DrawingType.MEP_PLAN)
        assert "10." in prompt
        assert "11." not in prompt

    def test_includes_nearby_text_per_symbol(self):
        """Should include nearby text for each symbol."""
        symbols = [
            {"yolo_class": "valve", "yolo_category": "mechanical", "nearby_text": ["GV", "3/4\""]},
        ]
        prompt = get_batch_symbol_prompt(symbols, DrawingType.PLUMBING_PLAN)
        assert "GV" in prompt


class TestValveClassificationPrompt:
    """Tests for get_valve_classification_prompt."""

    def test_valve_prompt_has_valve_types(self):
        """Should list all common valve types."""
        prompt = get_valve_classification_prompt("bowtie shape", [])
        assert "Gate Valve" in prompt
        assert "Ball Valve" in prompt
        assert "Butterfly Valve" in prompt
        assert "Check Valve" in prompt

    def test_valve_prompt_includes_abbreviations(self):
        """Should include common valve abbreviations."""
        prompt = get_valve_classification_prompt("", ["GV"])
        assert "GV" in prompt  # Gate Valve
        assert "BV" in prompt  # Ball Valve
        assert "CV" in prompt  # Check Valve

    def test_valve_prompt_requests_json(self):
        """Should request JSON with valve-specific fields."""
        prompt = get_valve_classification_prompt("", [])
        assert "subtype" in prompt
        assert "size_inches" in prompt
        assert "JSON" in prompt


class TestDiffuserClassificationPrompt:
    """Tests for get_diffuser_classification_prompt."""

    def test_diffuser_prompt_has_types(self):
        """Should list diffuser types."""
        prompt = get_diffuser_classification_prompt([])
        assert "Square" in prompt or "Rectangular" in prompt
        assert "Linear" in prompt
        assert "Return Air" in prompt

    def test_diffuser_prompt_mentions_cfm(self):
        """Should mention CFM annotations."""
        prompt = get_diffuser_classification_prompt(["24x24", "200 CFM"])
        assert "CFM" in prompt
        assert "24x24" in prompt

    def test_diffuser_prompt_requests_json(self):
        """Should request JSON with diffuser-specific fields."""
        prompt = get_diffuser_classification_prompt([])
        assert "cfm" in prompt
        assert "air_type" in prompt


class TestOutletClassificationPrompt:
    """Tests for get_outlet_classification_prompt."""

    def test_outlet_prompt_has_types(self):
        """Should list outlet types."""
        prompt = get_outlet_classification_prompt([])
        assert "Duplex" in prompt
        assert "GFCI" in prompt
        assert "Quad" in prompt
        assert "Floor" in prompt

    def test_outlet_prompt_requests_json(self):
        """Should request JSON with outlet-specific fields."""
        prompt = get_outlet_classification_prompt([])
        assert "voltage" in prompt
        assert "amperage" in prompt


class TestDetectorClassificationPrompt:
    """Tests for get_detector_classification_prompt."""

    def test_detector_prompt_has_types(self):
        """Should list detector types."""
        prompt = get_detector_classification_prompt([])
        assert "Smoke Detector" in prompt
        assert "Heat Detector" in prompt
        assert "Horn/Strobe" in prompt or "Horn Strobe" in prompt.replace("/", " ")

    def test_detector_prompt_includes_categories(self):
        """Should include detection, notification, control categories."""
        prompt = get_detector_classification_prompt([])
        # Check for category keywords
        prompt_lower = prompt.lower()
        assert "detection" in prompt_lower or "detector" in prompt_lower
        assert "notification" in prompt_lower or "strobe" in prompt_lower

    def test_detector_prompt_requests_json(self):
        """Should request JSON with detector-specific fields."""
        prompt = get_detector_classification_prompt([])
        assert "zone" in prompt
        assert "mounting" in prompt


class TestPromptQuality:
    """Tests for prompt quality and consistency."""

    @pytest.mark.parametrize("drawing_type", [
        DrawingType.HVAC_PLAN,
        DrawingType.ELECTRICAL_PLAN,
        DrawingType.PLUMBING_PLAN,
        DrawingType.FIRE_ALARM,
        DrawingType.FLOOR_PLAN,
    ])
    def test_all_drawing_types_have_context(self, drawing_type):
        """All drawing types should generate valid prompts with context."""
        prompt = get_symbol_classification_prompt(
            yolo_class="symbol",
            yolo_category="unknown",
            drawing_type=drawing_type,
        )
        assert len(prompt) > 200
        assert drawing_type.value in prompt

    def test_all_prompts_request_json(self):
        """All prompts should request JSON output."""
        prompts = [
            get_symbol_classification_prompt("valve", "mechanical", DrawingType.MEP_PLAN),
            get_unknown_symbol_prompt(DrawingType.MEP_PLAN),
            get_batch_symbol_prompt([{"yolo_class": "x", "yolo_category": "y", "nearby_text": []}], DrawingType.MEP_PLAN),
            get_valve_classification_prompt("", []),
            get_diffuser_classification_prompt([]),
            get_outlet_classification_prompt([]),
            get_detector_classification_prompt([]),
        ]
        for prompt in prompts:
            assert "JSON" in prompt

    def test_all_prompts_have_reasonable_length(self):
        """All prompts should be reasonably sized."""
        prompts = [
            get_symbol_classification_prompt("valve", "mechanical", DrawingType.MEP_PLAN),
            get_unknown_symbol_prompt(DrawingType.MEP_PLAN),
            get_valve_classification_prompt("", []),
            get_diffuser_classification_prompt([]),
            get_outlet_classification_prompt([]),
            get_detector_classification_prompt([]),
        ]
        for prompt in prompts:
            # Should be substantial but not excessive
            assert 300 < len(prompt) < 5000
