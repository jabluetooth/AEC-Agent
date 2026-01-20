"""
Unit tests for tool optimization module.
"""

import pytest
from aec_agent.frontend.tool_optimization import (
    compress_description_aggressive,
    get_optimized_description,
    get_optimized_schema,
    create_minimal_schema,
    get_tools_for_tier,
    estimate_token_count,
    MINIMAL_DESCRIPTIONS,
    TOOL_TIERS,
)


class TestCompressDescriptionAggressive:
    """Tests for aggressive description compression."""

    def test_removes_examples(self):
        desc = """Draw a line in AutoCAD.

        Args:
            start_x: X coordinate

        Example:
            draw_line(0, 0, 100, 100)

        Returns:
            Created line
        """
        result = compress_description_aggressive(desc)
        assert "Example" not in result
        assert "draw_line" not in result

    def test_keeps_first_sentence(self):
        desc = "Draw a line in AutoCAD. This is additional detail."
        result = compress_description_aggressive(desc)
        assert "Draw a line in AutoCAD" in result

    def test_handles_empty_description(self):
        assert compress_description_aggressive("") == ""
        assert compress_description_aggressive(None) == ""


class TestGetOptimizedDescription:
    """Tests for optimized description retrieval."""

    def test_full_mode_returns_original(self):
        original = "This is the full description."
        result = get_optimized_description("test_tool", original, "full")
        assert result == original

    def test_minimal_mode_uses_predefined(self):
        result = get_optimized_description(
            "autocad_draw_line",
            "Very long original description...",
            "minimal"
        )
        assert result == MINIMAL_DESCRIPTIONS["autocad_draw_line"]

    def test_ultra_mode_returns_name(self):
        result = get_optimized_description(
            "autocad_draw_line",
            "Very long description",
            "ultra"
        )
        assert result == "Autocad Draw Line"


class TestGetOptimizedSchema:
    """Tests for schema optimization."""

    def test_full_mode_returns_original(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The name"}
            }
        }
        result = get_optimized_schema(schema, "full")
        assert result == schema

    def test_minimal_mode_removes_descriptions(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Very long description that adds tokens"
                }
            }
        }
        result = get_optimized_schema(schema, "minimal")
        assert "description" not in result["properties"]["name"]
        assert result["properties"]["name"]["type"] == "string"

    def test_ultra_mode_keeps_only_required(self):
        schema = {
            "type": "object",
            "properties": {
                "required_param": {"type": "string"},
                "optional_param": {"type": "number", "default": 10}
            },
            "required": ["required_param"]
        }
        result = get_optimized_schema(schema, "ultra")
        assert "required_param" in result.get("properties", {})
        # Optional params may be excluded in ultra mode


class TestCreateMinimalSchema:
    """Tests for minimal schema creation."""

    def test_preserves_type(self):
        schema = {"type": "object"}
        result = create_minimal_schema(schema)
        assert result["type"] == "object"

    def test_preserves_enum_values(self):
        schema = {
            "type": "object",
            "properties": {
                "color": {"type": "string", "enum": ["red", "blue"]}
            }
        }
        result = create_minimal_schema(schema)
        assert result["properties"]["color"]["enum"] == ["red", "blue"]

    def test_preserves_defaults(self):
        schema = {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 100}
            }
        }
        result = create_minimal_schema(schema)
        assert result["properties"]["limit"]["default"] == 100


class TestGetToolsForTier:
    """Tests for tool tier retrieval."""

    def test_essential_tier(self):
        tools = get_tools_for_tier("essential")
        assert "ping" in tools
        assert "autocad_draw_line" in tools
        # Should NOT include standard tier tools
        assert "autocad_create_layer" not in tools

    def test_standard_tier_includes_essential(self):
        tools = get_tools_for_tier("standard")
        # Essential tools
        assert "ping" in tools
        assert "autocad_draw_line" in tools
        # Standard tools
        assert "autocad_create_layer" in tools
        # Should NOT include advanced
        assert "find_elements" not in tools

    def test_advanced_tier_includes_all(self):
        tools = get_tools_for_tier("advanced")
        # Essential
        assert "ping" in tools
        # Standard
        assert "autocad_create_layer" in tools
        # Advanced
        assert "find_elements" in tools
        assert "draw_line_between" in tools


class TestEstimateTokenCount:
    """Tests for token estimation."""

    def test_basic_estimation(self):
        # GPT-style: ~4 chars per token
        text = "Hello world!"  # 12 chars = ~3 tokens
        tokens = estimate_token_count(text)
        assert tokens == 3

    def test_empty_string(self):
        assert estimate_token_count("") == 0


class TestTokenReduction:
    """Integration tests for token reduction."""

    def test_compression_reduces_tokens(self):
        """Verify that compression actually reduces token count."""
        full_desc = """
        Draw a line in AutoCAD between two points.

        This function creates a line entity in the current AutoCAD drawing
        from the specified start point to the specified end point.

        Args:
            start_x: The X coordinate of the starting point
            start_y: The Y coordinate of the starting point
            end_x: The X coordinate of the ending point
            end_y: The Y coordinate of the ending point
            layer: Optional layer name to draw on

        Returns:
            A dictionary containing the handle of the created line entity

        Example:
            autocad_draw_line(0, 0, 100, 100, "Construction")
        """

        full_tokens = estimate_token_count(full_desc)
        minimal_desc = MINIMAL_DESCRIPTIONS.get("autocad_draw_line", "")
        minimal_tokens = estimate_token_count(minimal_desc)

        # Minimal should be significantly smaller
        assert minimal_tokens < full_tokens * 0.5

    def test_all_tools_have_minimal_descriptions(self):
        """Verify all tiered tools have minimal descriptions."""
        all_tiered_tools = set()
        for tier_tools in TOOL_TIERS.values():
            all_tiered_tools.update(tier_tools)

        for tool_name in all_tiered_tools:
            assert tool_name in MINIMAL_DESCRIPTIONS, (
                f"Missing minimal description for {tool_name}"
            )
