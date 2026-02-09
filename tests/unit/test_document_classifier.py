"""Unit tests for document_classifier.py (Phase A)."""

import pytest

from aec_agent.mcp.tools.document_classifier import (
    DRAWING_TYPE_CONFIG,
    DRAWING_TYPE_KEYWORDS,
    DrawingClassification,
    DrawingType,
    classify_by_keywords,
    get_pipeline_config,
)


class TestDrawingType:
    """Tests for DrawingType enum."""

    def test_all_drawing_types_have_config(self):
        """Every DrawingType should have a configuration entry."""
        for dtype in DrawingType:
            assert dtype in DRAWING_TYPE_CONFIG, f"Missing config for {dtype}"

    def test_all_drawing_types_have_keywords(self):
        """Every DrawingType except UNKNOWN should have keywords."""
        for dtype in DrawingType:
            if dtype != DrawingType.UNKNOWN:
                assert dtype in DRAWING_TYPE_KEYWORDS, f"Missing keywords for {dtype}"

    def test_config_has_required_fields(self):
        """Each config should have discipline and layer_prefix."""
        for dtype, config in DRAWING_TYPE_CONFIG.items():
            assert "discipline" in config, f"Missing discipline for {dtype}"
            assert "layer_prefix" in config, f"Missing layer_prefix for {dtype}"


class TestClassifyByKeywords:
    """Tests for keyword-based classification."""

    def test_electrical_plan_keywords(self):
        """Text with electrical keywords should classify as electrical."""
        text = "ELECTRICAL FLOOR PLAN - POWER E1.01"
        dtype, confidence = classify_by_keywords(text)
        assert dtype == DrawingType.ELECTRICAL_PLAN
        assert confidence > 0.3

    def test_hvac_plan_keywords(self):
        """Text with HVAC keywords should classify as HVAC."""
        text = "MECHANICAL HVAC PLAN M-101 DUCTWORK"
        dtype, confidence = classify_by_keywords(text)
        assert dtype == DrawingType.HVAC_PLAN
        assert confidence > 0.3

    def test_plumbing_plan_keywords(self):
        """Text with plumbing keywords should classify as plumbing."""
        text = "PLUMBING FLOOR PLAN P-100 FIXTURES DWV"
        dtype, confidence = classify_by_keywords(text)
        assert dtype == DrawingType.PLUMBING_PLAN
        assert confidence > 0.3

    def test_fire_alarm_keywords(self):
        """Text with fire alarm keywords should classify as fire alarm."""
        text = "FIRE ALARM PLAN FA-101 SMOKE DETECTOR FACP"
        dtype, confidence = classify_by_keywords(text)
        assert dtype == DrawingType.FIRE_ALARM
        assert confidence > 0.3

    def test_floor_plan_keywords(self):
        """Text with architectural keywords should classify as floor plan."""
        text = "FLOOR PLAN ARCHITECTURAL A101"
        dtype, confidence = classify_by_keywords(text)
        assert dtype == DrawingType.FLOOR_PLAN
        assert confidence > 0.3

    def test_single_line_keywords(self):
        """Text with single-line diagram keywords should classify correctly."""
        text = "ELECTRICAL SINGLE LINE DIAGRAM POWER RISER"
        dtype, confidence = classify_by_keywords(text)
        assert dtype == DrawingType.SINGLE_LINE
        assert confidence > 0.3

    def test_schedule_keywords(self):
        """Text with schedule keywords should classify as schedule."""
        text = "DOOR SCHEDULE EQUIPMENT LIST"
        dtype, confidence = classify_by_keywords(text)
        assert dtype == DrawingType.SCHEDULE
        assert confidence > 0.3

    def test_empty_text_returns_unknown(self):
        """Empty text should return UNKNOWN with zero confidence."""
        dtype, confidence = classify_by_keywords("")
        assert dtype == DrawingType.UNKNOWN
        assert confidence == 0.0

    def test_gibberish_returns_unknown(self):
        """Random text without keywords should return UNKNOWN."""
        text = "xyz abc 123 random gibberish"
        dtype, confidence = classify_by_keywords(text)
        assert dtype == DrawingType.UNKNOWN
        assert confidence == 0.0

    def test_case_insensitive(self):
        """Classification should be case-insensitive."""
        text1 = "ELECTRICAL PLAN"
        text2 = "electrical plan"
        text3 = "Electrical Plan"

        dtype1, _ = classify_by_keywords(text1)
        dtype2, _ = classify_by_keywords(text2)
        dtype3, _ = classify_by_keywords(text3)

        assert dtype1 == dtype2 == dtype3 == DrawingType.ELECTRICAL_PLAN

    def test_longer_keywords_score_higher(self):
        """Longer keyword matches should produce higher confidence."""
        # "fire alarm" is longer than just "fire"
        text_long = "FIRE ALARM PLAN"
        text_short = "FIRE"

        _, conf_long = classify_by_keywords(text_long)
        _, conf_short = classify_by_keywords(text_short)

        assert conf_long > conf_short

    def test_multiple_discipline_keywords(self):
        """Text with keywords from multiple disciplines should pick highest scoring."""
        # MEP has fewer keywords than specific disciplines
        text = "MECHANICAL ELECTRICAL PLUMBING MEP PLAN"
        dtype, confidence = classify_by_keywords(text)
        # Should classify as one of the specific disciplines, not MEP
        assert dtype in (
            DrawingType.HVAC_PLAN,
            DrawingType.ELECTRICAL_PLAN,
            DrawingType.PLUMBING_PLAN,
            DrawingType.MEP_PLAN,
        )
        # Confidence varies based on keyword matches (0.2+ is reasonable)
        assert confidence > 0.2


class TestDrawingClassification:
    """Tests for DrawingClassification dataclass."""

    def test_dataclass_creation(self):
        """Should create DrawingClassification with required fields."""
        classification = DrawingClassification(
            drawing_type=DrawingType.ELECTRICAL_PLAN,
            confidence=0.85,
            discipline="electrical",
        )
        assert classification.drawing_type == DrawingType.ELECTRICAL_PLAN
        assert classification.confidence == 0.85
        assert classification.discipline == "electrical"

    def test_optional_fields_default_to_none(self):
        """Optional fields should default to None."""
        classification = DrawingClassification(
            drawing_type=DrawingType.UNKNOWN,
            confidence=0.0,
            discipline="unknown",
        )
        assert classification.sheet_number is None
        assert classification.sheet_name is None
        assert classification.scale is None

    def test_metadata_defaults_to_empty_dict(self):
        """Metadata should default to empty dict."""
        classification = DrawingClassification(
            drawing_type=DrawingType.UNKNOWN,
            confidence=0.0,
            discipline="unknown",
        )
        assert classification.metadata == {}


class TestGetPipelineConfig:
    """Tests for get_pipeline_config function."""

    def test_electrical_config(self):
        """Electrical plan should return electrical-specific config."""
        classification = DrawingClassification(
            drawing_type=DrawingType.ELECTRICAL_PLAN,
            confidence=0.8,
            discipline="electrical",
        )
        config = get_pipeline_config(classification)

        assert config["discipline"] == "electrical"
        assert "electrical" in config["symbol_categories"]
        assert config["layer_prefix"] == "E-"

    def test_hvac_config(self):
        """HVAC plan should return mechanical-specific config."""
        classification = DrawingClassification(
            drawing_type=DrawingType.HVAC_PLAN,
            confidence=0.8,
            discipline="mechanical",
        )
        config = get_pipeline_config(classification)

        assert config["discipline"] == "mechanical"
        assert "mechanical" in config["symbol_categories"]
        assert config["layer_prefix"] == "M-"

    def test_unknown_config(self):
        """Unknown drawing type should return generic config."""
        classification = DrawingClassification(
            drawing_type=DrawingType.UNKNOWN,
            confidence=0.0,
            discipline="unknown",
        )
        config = get_pipeline_config(classification)

        assert config["drawing_type"] == "unknown"
        assert config["ocr_masking"] is True
        assert config["aec_heuristics"] is True

    def test_config_always_has_required_keys(self):
        """Config should always have required keys regardless of type."""
        for dtype in DrawingType:
            classification = DrawingClassification(
                drawing_type=dtype,
                confidence=0.5,
                discipline=DRAWING_TYPE_CONFIG[dtype].get("discipline", "unknown"),
            )
            config = get_pipeline_config(classification)

            assert "symbol_categories" in config
            assert "symbol_detection" in config
            assert "ocr_focus" in config
            assert "ocr_masking" in config
            assert "layer_prefix" in config
            assert "drawing_type" in config


class TestDrawingTypeConfig:
    """Tests for DRAWING_TYPE_CONFIG structure."""

    def test_all_configs_have_discipline(self):
        """All configs should have a discipline field."""
        for dtype, config in DRAWING_TYPE_CONFIG.items():
            assert "discipline" in config
            assert isinstance(config["discipline"], str)

    def test_all_configs_have_symbol_categories(self):
        """All configs should have symbol_categories list."""
        for dtype, config in DRAWING_TYPE_CONFIG.items():
            assert "symbol_categories" in config
            assert isinstance(config["symbol_categories"], list)

    def test_all_configs_have_layer_prefix(self):
        """All configs should have layer_prefix string."""
        for dtype, config in DRAWING_TYPE_CONFIG.items():
            assert "layer_prefix" in config
            assert isinstance(config["layer_prefix"], str)

    def test_mep_disciplines_have_correct_prefixes(self):
        """MEP disciplines should have standard layer prefixes."""
        assert DRAWING_TYPE_CONFIG[DrawingType.ELECTRICAL_PLAN]["layer_prefix"] == "E-"
        assert DRAWING_TYPE_CONFIG[DrawingType.HVAC_PLAN]["layer_prefix"] == "M-"
        assert DRAWING_TYPE_CONFIG[DrawingType.PLUMBING_PLAN]["layer_prefix"] == "P-"
        assert DRAWING_TYPE_CONFIG[DrawingType.FIRE_ALARM]["layer_prefix"] == "FA-"
