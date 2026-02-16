"""
Unit tests for Phase 5: AutoCAD Entity Creation.

Tests the gemini_first.autocad_creation module which creates AutoCAD entities
from the ExtractionResult of Phase 4.
"""

from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

import pytest

# Import module under test
from aec_agent.mcp.tools.gemini_first.autocad_creation import (
    EntityCreationResult,
    LayerCreationResult,
    CreationStatistics,
    AutoCADCreationResult,
    create_entities_in_autocad,
    create_entities_batch,
    create_single_entity,
    create_layer_if_needed,
    create_line_entity,
    create_arc_entity,
    create_circle_entity,
    create_text_entity,
    create_block_entity,
    create_polyline_entity,
    get_entity_type_stats,
    get_failed_by_type,
    get_color_for_layer,
    LAYER_PREFIX_COLORS,
    ENTITY_CREATORS,
)
from aec_agent.mcp.tools.gemini_first.adaptive_extraction import (
    EntityToCreate,
    EntityType,
    ExtractionSource,
    ExtractionResult,
)


class TestEntityCreationResult:
    """Tests for EntityCreationResult dataclass."""

    def test_success_result(self):
        """Test creating a successful result."""
        result = EntityCreationResult(
            entity_type="line",
            layer="A-WALL",
            success=True,
            handle="1A3",
        )

        assert result.success is True
        assert result.handle == "1A3"
        assert result.error is None

    def test_failure_result(self):
        """Test creating a failure result."""
        result = EntityCreationResult(
            entity_type="block",
            layer="M-EQPM",
            success=False,
            error="Block not defined",
        )

        assert result.success is False
        assert result.error == "Block not defined"
        assert result.handle is None

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = EntityCreationResult(
            entity_type="circle",
            layer="E-POWR",
            success=True,
            handle="2B4",
            source_element="circle_column",
        )

        d = result.to_dict()
        assert d["entity_type"] == "circle"
        assert d["layer"] == "E-POWR"
        assert d["success"] is True
        assert d["handle"] == "2B4"
        assert d["source_element"] == "circle_column"


class TestLayerCreationResult:
    """Tests for LayerCreationResult dataclass."""

    def test_new_layer_created(self):
        """Test result when layer is newly created."""
        result = LayerCreationResult(
            name="M-DUCT",
            success=True,
            already_existed=False,
        )

        assert result.success is True
        assert result.already_existed is False

    def test_layer_already_existed(self):
        """Test result when layer already existed."""
        result = LayerCreationResult(
            name="A-WALL",
            success=True,
            already_existed=True,
        )

        assert result.success is True
        assert result.already_existed is True

    def test_layer_creation_failed(self):
        """Test result when layer creation failed."""
        result = LayerCreationResult(
            name="INVALID**LAYER",
            success=False,
            error="Invalid layer name",
        )

        assert result.success is False
        assert "Invalid" in result.error


class TestCreationStatistics:
    """Tests for CreationStatistics dataclass."""

    def test_default_values(self):
        """Test default statistics are zero."""
        stats = CreationStatistics()

        assert stats.total_entities == 0
        assert stats.success_count == 0
        assert stats.failure_count == 0
        assert stats.lines_created == 0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        stats = CreationStatistics(
            total_entities=100,
            success_count=95,
            failure_count=5,
            lines_created=50,
            circles_created=20,
            texts_created=15,
            blocks_created=10,
            layers_created=5,
        )

        d = stats.to_dict()
        assert d["total_entities"] == 100
        assert d["success_count"] == 95
        assert d["by_type"]["lines"] == 50
        assert d["by_type"]["circles"] == 20
        assert d["layers"]["created"] == 5


class TestAutoCADCreationResult:
    """Tests for AutoCADCreationResult dataclass."""

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        result = AutoCADCreationResult(success=True)
        result.statistics.total_entities = 100
        result.statistics.success_count = 80
        result.statistics.failure_count = 20

        assert result.success_rate == 0.8
        assert result.success_count == 80
        assert result.failure_count == 20

    def test_success_rate_empty(self):
        """Test success rate when no entities."""
        result = AutoCADCreationResult(success=True)

        assert result.success_rate == 1.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = AutoCADCreationResult(
            success=True,
            drawing_type="floor_plan",
            extraction_source="direct",
        )
        result.statistics.total_entities = 50
        result.statistics.success_count = 50
        result.created_handles = ["1A", "1B", "1C"]

        d = result.to_dict()
        assert d["success"] is True
        assert d["metadata"]["drawing_type"] == "floor_plan"
        assert len(d["created_handles"]) == 3


class TestGetColorForLayer:
    """Tests for get_color_for_layer function."""

    def test_architectural_layer(self):
        """Test color for architectural layer."""
        assert get_color_for_layer("A-WALL") == 7  # White

    def test_mechanical_layer(self):
        """Test color for mechanical layer."""
        assert get_color_for_layer("M-DUCT") == 4  # Cyan

    def test_electrical_layer(self):
        """Test color for electrical layer."""
        assert get_color_for_layer("E-POWR-OUTL") == 1  # Red

    def test_plumbing_layer(self):
        """Test color for plumbing layer."""
        assert get_color_for_layer("P-DOMW-PIPE") == 5  # Blue

    def test_fire_layer(self):
        """Test color for fire protection layer."""
        assert get_color_for_layer("F-ALRM-DETC") == 1  # Red

    def test_telecom_layer(self):
        """Test color for telecom/low voltage layer."""
        assert get_color_for_layer("T-DATA-OUTL") == 3  # Green

    def test_unknown_layer(self):
        """Test color for unknown layer prefix."""
        assert get_color_for_layer("X-UNKNOWN") == 7  # Default white

    def test_layer_zero(self):
        """Test color for layer 0."""
        assert get_color_for_layer("0") == 7  # Default white


class TestCreateLayerIfNeeded:
    """Tests for create_layer_if_needed function."""

    @pytest.mark.asyncio
    async def test_skip_layer_zero(self):
        """Test that layer '0' is skipped."""
        mock_command = AsyncMock()

        result = await create_layer_if_needed("0", mock_command)

        assert result.success is True
        assert result.already_existed is True
        mock_command.assert_not_called()

    @pytest.mark.asyncio
    async def test_layer_in_existing_set(self):
        """Test that known existing layers are skipped."""
        mock_command = AsyncMock()
        existing = {"A-WALL", "M-DUCT"}

        result = await create_layer_if_needed("A-WALL", mock_command, existing)

        assert result.success is True
        assert result.already_existed is True
        mock_command.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_new_layer(self):
        """Test creating a new layer."""
        mock_command = AsyncMock(return_value={"success": True})

        result = await create_layer_if_needed("M-DIFF", mock_command)

        assert result.success is True
        assert result.already_existed is False
        mock_command.assert_called_once()
        call_args = mock_command.call_args
        assert call_args[0][0] == "create_layer"
        assert call_args[0][1]["name"] == "M-DIFF"
        assert call_args[0][1]["color"] == 4  # Cyan for M-

    @pytest.mark.asyncio
    async def test_layer_already_exists_error(self):
        """Test handling 'already exists' error."""
        mock_command = AsyncMock(return_value={
            "success": False,
            "error": {"message": "Layer already exists"}
        })

        result = await create_layer_if_needed("A-WALL", mock_command)

        assert result.success is True
        assert result.already_existed is True

    @pytest.mark.asyncio
    async def test_layer_creation_exception(self):
        """Test handling exception during layer creation."""
        mock_command = AsyncMock(side_effect=Exception("Connection error"))

        result = await create_layer_if_needed("E-POWR", mock_command)

        assert result.success is False
        assert "Connection error" in result.error


class TestCreateLineEntity:
    """Tests for create_line_entity function."""

    @pytest.mark.asyncio
    async def test_create_line_success(self):
        """Test successful line creation."""
        entity = EntityToCreate(
            entity_type=EntityType.LINE,
            layer="A-WALL",
            properties={
                "start": (0.0, 0.0),
                "end": (100.0, 0.0),
                "linetype": "Continuous",
            },
        )
        mock_command = AsyncMock(return_value={
            "success": True,
            "data": {"handle": "1A3"}
        })

        result = await create_line_entity(entity, mock_command)

        assert result.success is True
        assert result.handle == "1A3"
        assert result.entity_type == "line"

    @pytest.mark.asyncio
    async def test_create_line_with_list_coords(self):
        """Test line creation with list coordinates."""
        entity = EntityToCreate(
            entity_type=EntityType.LINE,
            layer="M-DUCT",
            properties={
                "start": [10.0, 20.0],
                "end": [110.0, 20.0],
            },
        )
        mock_command = AsyncMock(return_value={"success": True, "data": {}})

        result = await create_line_entity(entity, mock_command)

        assert result.success is True
        call_params = mock_command.call_args[0][1]
        assert call_params["start"] == [10.0, 20.0]
        assert call_params["end"] == [110.0, 20.0]

    @pytest.mark.asyncio
    async def test_create_line_invalid_start(self):
        """Test line creation with invalid start point."""
        entity = EntityToCreate(
            entity_type=EntityType.LINE,
            layer="A-WALL",
            properties={
                "start": "invalid",
                "end": (100.0, 0.0),
            },
        )
        mock_command = AsyncMock()

        result = await create_line_entity(entity, mock_command)

        assert result.success is False
        assert "Invalid start point" in result.error
        mock_command.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_line_failure(self):
        """Test line creation failure from AutoCAD."""
        entity = EntityToCreate(
            entity_type=EntityType.LINE,
            layer="A-WALL",
            properties={"start": (0, 0), "end": (100, 0)},
        )
        mock_command = AsyncMock(return_value={
            "success": False,
            "error": {"message": "Layer locked"}
        })

        result = await create_line_entity(entity, mock_command)

        assert result.success is False
        assert "Layer locked" in result.error


class TestCreateArcEntity:
    """Tests for create_arc_entity function."""

    @pytest.mark.asyncio
    async def test_create_arc_success(self):
        """Test successful arc creation."""
        entity = EntityToCreate(
            entity_type=EntityType.ARC,
            layer="A-DOOR",
            properties={
                "center": (50.0, 50.0),
                "radius": 25.0,
                "start_angle": 0.0,
                "end_angle": 90.0,
            },
        )
        mock_command = AsyncMock(return_value={
            "success": True,
            "data": {"handle": "2B4"}
        })

        result = await create_arc_entity(entity, mock_command)

        assert result.success is True
        assert result.handle == "2B4"

    @pytest.mark.asyncio
    async def test_create_arc_invalid_center(self):
        """Test arc creation with invalid center."""
        entity = EntityToCreate(
            entity_type=EntityType.ARC,
            layer="A-DOOR",
            properties={
                "center": 50.0,  # Should be tuple/list
                "radius": 25.0,
                "start_angle": 0.0,
                "end_angle": 90.0,
            },
        )
        mock_command = AsyncMock()

        result = await create_arc_entity(entity, mock_command)

        assert result.success is False
        assert "Invalid center point" in result.error


class TestCreateCircleEntity:
    """Tests for create_circle_entity function."""

    @pytest.mark.asyncio
    async def test_create_circle_success(self):
        """Test successful circle creation."""
        entity = EntityToCreate(
            entity_type=EntityType.CIRCLE,
            layer="M-EQPM",
            properties={
                "center": (100.0, 100.0),
                "radius": 5.0,
            },
        )
        mock_command = AsyncMock(return_value={
            "success": True,
            "data": {"handle": "3C5"}
        })

        result = await create_circle_entity(entity, mock_command)

        assert result.success is True
        assert result.handle == "3C5"

    @pytest.mark.asyncio
    async def test_create_circle_zero_radius(self):
        """Test circle creation with zero radius."""
        entity = EntityToCreate(
            entity_type=EntityType.CIRCLE,
            layer="M-EQPM",
            properties={
                "center": (100.0, 100.0),
                "radius": 0.0,
            },
        )
        mock_command = AsyncMock()

        result = await create_circle_entity(entity, mock_command)

        assert result.success is False
        assert "Invalid radius" in result.error


class TestCreateTextEntity:
    """Tests for create_text_entity function."""

    @pytest.mark.asyncio
    async def test_create_text_success(self):
        """Test successful text creation."""
        entity = EntityToCreate(
            entity_type=EntityType.MTEXT,
            layer="G-ANNO-TEXT",
            properties={
                "content": "Room 101",
                "position": (50.0, 50.0),
                "height": 0.125,
            },
        )
        mock_command = AsyncMock(return_value={
            "success": True,
            "data": {"handle": "4D6"}
        })

        result = await create_text_entity(entity, mock_command)

        assert result.success is True
        assert result.handle == "4D6"

    @pytest.mark.asyncio
    async def test_create_text_empty_content(self):
        """Test text creation with empty content."""
        entity = EntityToCreate(
            entity_type=EntityType.MTEXT,
            layer="G-ANNO-TEXT",
            properties={
                "content": "",
                "position": (50.0, 50.0),
                "height": 0.125,
            },
        )
        mock_command = AsyncMock()

        result = await create_text_entity(entity, mock_command)

        assert result.success is False
        assert "Empty text" in result.error

    @pytest.mark.asyncio
    async def test_create_text_minimum_height(self):
        """Test text creation enforces minimum height."""
        entity = EntityToCreate(
            entity_type=EntityType.MTEXT,
            layer="G-ANNO-TEXT",
            properties={
                "content": "Test",
                "position": (0.0, 0.0),
                "height": 0.01,  # Very small
            },
        )
        mock_command = AsyncMock(return_value={"success": True, "data": {}})

        await create_text_entity(entity, mock_command)

        call_params = mock_command.call_args[0][1]
        assert call_params["height"] >= 0.0625  # Minimum 1/16"


class TestCreateBlockEntity:
    """Tests for create_block_entity function."""

    @pytest.mark.asyncio
    async def test_create_block_success(self):
        """Test successful block insertion."""
        entity = EntityToCreate(
            entity_type=EntityType.BLOCK,
            layer="M-DIFF",
            properties={
                "block_name": "M-DIFF-SQ",
                "position": (150.0, 200.0),
                "rotation": 45.0,
                "scale": 1.0,
                "attributes": {"SIZE": "24x24"},
            },
        )
        mock_command = AsyncMock(return_value={
            "success": True,
            "data": {"handle": "5E7"}
        })

        result = await create_block_entity(entity, mock_command)

        assert result.success is True
        assert result.handle == "5E7"

    @pytest.mark.asyncio
    async def test_create_block_missing_name(self):
        """Test block creation with missing name."""
        entity = EntityToCreate(
            entity_type=EntityType.BLOCK,
            layer="M-DIFF",
            properties={
                "position": (150.0, 200.0),
            },
        )
        mock_command = AsyncMock()

        result = await create_block_entity(entity, mock_command)

        assert result.success is False
        assert "Missing block name" in result.error

    @pytest.mark.asyncio
    async def test_create_block_not_defined(self):
        """Test block creation when block not defined."""
        entity = EntityToCreate(
            entity_type=EntityType.BLOCK,
            layer="M-DIFF",
            properties={
                "block_name": "UNKNOWN-BLOCK",
                "position": (150.0, 200.0),
            },
        )
        mock_command = AsyncMock(return_value={
            "success": False,
            "error": {"message": "Block not found"}
        })

        result = await create_block_entity(entity, mock_command)

        assert result.success is False
        assert "not defined" in result.error


class TestCreatePolylineEntity:
    """Tests for create_polyline_entity function."""

    @pytest.mark.asyncio
    async def test_create_polyline_success(self):
        """Test successful polyline creation."""
        entity = EntityToCreate(
            entity_type=EntityType.POLYLINE,
            layer="A-WALL",
            properties={
                "points": [(0, 0), (100, 0), (100, 50), (0, 50)],
                "closed": True,
            },
        )
        mock_command = AsyncMock(return_value={
            "success": True,
            "data": {"handle": "6F8"}
        })

        result = await create_polyline_entity(entity, mock_command)

        assert result.success is True
        assert result.handle == "6F8"

    @pytest.mark.asyncio
    async def test_create_polyline_too_few_points(self):
        """Test polyline creation with insufficient points."""
        entity = EntityToCreate(
            entity_type=EntityType.POLYLINE,
            layer="A-WALL",
            properties={
                "points": [(0, 0)],  # Only 1 point
            },
        )
        mock_command = AsyncMock()

        result = await create_polyline_entity(entity, mock_command)

        assert result.success is False
        assert "at least 2 points" in result.error


class TestCreateSingleEntity:
    """Tests for create_single_entity dispatch function."""

    @pytest.mark.asyncio
    async def test_dispatch_line(self):
        """Test dispatching to line creator."""
        entity = EntityToCreate(
            entity_type=EntityType.LINE,
            layer="A-WALL",
            properties={"start": (0, 0), "end": (100, 0)},
        )
        mock_command = AsyncMock(return_value={"success": True, "data": {}})

        result = await create_single_entity(entity, mock_command)

        assert result.entity_type == "line"

    @pytest.mark.asyncio
    async def test_dispatch_string_type(self):
        """Test dispatching with string entity type."""
        entity = EntityToCreate(
            entity_type="circle",  # String type
            layer="M-EQPM",
            properties={"center": (50, 50), "radius": 10},
        )
        mock_command = AsyncMock(return_value={"success": True, "data": {}})

        result = await create_single_entity(entity, mock_command)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_dispatch_unsupported_type(self):
        """Test dispatching with unsupported entity type."""
        entity = EntityToCreate(
            entity_type="hatch",  # Unsupported
            layer="A-PATT",
            properties={},
        )
        mock_command = AsyncMock()

        result = await create_single_entity(entity, mock_command)

        assert result.success is False
        assert "Unsupported entity type" in result.error


class TestCreateEntitiesInAutocad:
    """Tests for create_entities_in_autocad main function."""

    @pytest.mark.asyncio
    async def test_empty_extraction(self):
        """Test with empty extraction result."""
        extraction = ExtractionResult()
        mock_command = AsyncMock()

        result = await create_entities_in_autocad(
            extraction, call_command=mock_command
        )

        assert result.success is True
        assert result.statistics.total_entities == 0

    @pytest.mark.asyncio
    async def test_create_layers_enabled(self):
        """Test that layers are created when enabled."""
        entity = EntityToCreate(
            entity_type=EntityType.LINE,
            layer="M-DUCT",
            properties={"start": (0, 0), "end": (100, 0)},
        )
        extraction = ExtractionResult(entities=[entity])

        mock_command = AsyncMock(return_value={"success": True, "data": {}})

        result = await create_entities_in_autocad(
            extraction,
            call_command=mock_command,
            create_layers=True,
        )

        # Should have calls for layer creation and entity creation
        assert mock_command.call_count >= 1

    @pytest.mark.asyncio
    async def test_create_multiple_entities(self):
        """Test creating multiple entities."""
        entities = [
            EntityToCreate(
                entity_type=EntityType.LINE,
                layer="A-WALL",
                properties={"start": (0, 0), "end": (100, 0)},
            ),
            EntityToCreate(
                entity_type=EntityType.CIRCLE,
                layer="M-EQPM",
                properties={"center": (50, 50), "radius": 10},
            ),
            EntityToCreate(
                entity_type=EntityType.MTEXT,
                layer="G-ANNO-TEXT",
                properties={"content": "Test", "position": (0, 0), "height": 0.125},
            ),
        ]
        extraction = ExtractionResult(entities=entities)

        mock_command = AsyncMock(return_value={
            "success": True,
            "data": {"handle": "XXX"}
        })

        result = await create_entities_in_autocad(
            extraction,
            call_command=mock_command,
            create_layers=False,  # Skip layer creation for simplicity
        )

        assert result.statistics.total_entities == 3
        assert result.statistics.success_count == 3
        assert result.statistics.lines_created == 1
        assert result.statistics.circles_created == 1
        assert result.statistics.texts_created == 1

    @pytest.mark.asyncio
    async def test_partial_failure(self):
        """Test when some entities fail to create."""
        entities = [
            EntityToCreate(
                entity_type=EntityType.LINE,
                layer="A-WALL",
                properties={"start": (0, 0), "end": (100, 0)},
            ),
            EntityToCreate(
                entity_type=EntityType.BLOCK,
                layer="M-DIFF",
                properties={"block_name": "MISSING"},  # Will fail
            ),
        ]
        extraction = ExtractionResult(entities=entities)

        def command_response(cmd, params):
            if cmd == "draw_line":
                return {"success": True, "data": {"handle": "1A"}}
            elif cmd == "insert_block":
                return {"success": False, "error": {"message": "Block not found"}}
            return {"success": True, "data": {}}

        mock_command = AsyncMock(side_effect=command_response)

        result = await create_entities_in_autocad(
            extraction,
            call_command=mock_command,
            create_layers=False,
        )

        assert result.statistics.success_count == 1
        assert result.statistics.failure_count == 1
        assert len(result.failed_entities) == 1

    @pytest.mark.asyncio
    async def test_success_determination(self):
        """Test success determination based on success rate."""
        # Create 10 entities, 6 succeed = 60% > 50% threshold
        entities = [
            EntityToCreate(
                entity_type=EntityType.LINE,
                layer="A-WALL",
                properties={"start": (i, 0), "end": (i+10, 0)},
            )
            for i in range(10)
        ]
        extraction = ExtractionResult(entities=entities)

        call_count = 0
        def command_response(cmd, params):
            nonlocal call_count
            call_count += 1
            if cmd == "draw_line":
                if call_count <= 6:
                    return {"success": True, "data": {}}
                else:
                    return {"success": False, "error": {"message": "Failed"}}
            return {"success": True, "data": {}}

        mock_command = AsyncMock(side_effect=command_response)

        result = await create_entities_in_autocad(
            extraction,
            call_command=mock_command,
            create_layers=False,
        )

        assert result.success is True  # >50% success rate


class TestCreateEntitiesBatch:
    """Tests for create_entities_batch convenience function."""

    @pytest.mark.asyncio
    async def test_batch_creation(self):
        """Test batch creation of entities."""
        entities = [
            EntityToCreate(
                entity_type=EntityType.LINE,
                layer="A-WALL",
                properties={"start": (0, 0), "end": (100, 0)},
            ),
        ]
        mock_command = AsyncMock(return_value={"success": True, "data": {}})

        result = await create_entities_batch(
            entities,
            call_command=mock_command,
            create_layers=False,
        )

        assert result.statistics.total_entities == 1


class TestGetEntityTypeStats:
    """Tests for get_entity_type_stats helper function."""

    def test_get_stats(self):
        """Test getting entity type statistics."""
        result = AutoCADCreationResult(success=True)
        result.statistics.lines_created = 10
        result.statistics.circles_created = 5
        result.statistics.texts_created = 3

        stats = get_entity_type_stats(result)

        assert stats["lines"] == 10
        assert stats["circles"] == 5
        assert stats["texts"] == 3


class TestGetFailedByType:
    """Tests for get_failed_by_type helper function."""

    def test_group_failures(self):
        """Test grouping failed entities by type."""
        result = AutoCADCreationResult(success=True)
        result.failed_entities = [
            {"entity_type": "block", "error": "Not defined"},
            {"entity_type": "block", "error": "Another error"},
            {"entity_type": "line", "error": "Layer locked"},
        ]

        grouped = get_failed_by_type(result)

        assert len(grouped["block"]) == 2
        assert len(grouped["line"]) == 1


class TestEntityCreators:
    """Tests for ENTITY_CREATORS mapping."""

    def test_all_types_have_creators(self):
        """Test that all EntityType enums have creators."""
        supported_types = [
            EntityType.LINE,
            EntityType.ARC,
            EntityType.CIRCLE,
            EntityType.MTEXT,
            EntityType.TEXT,
            EntityType.BLOCK,
            EntityType.POLYLINE,
        ]

        for entity_type in supported_types:
            assert entity_type in ENTITY_CREATORS, f"Missing creator for {entity_type}"

    def test_string_aliases(self):
        """Test that string aliases work."""
        string_types = ["line", "arc", "circle", "mtext", "text", "block", "polyline"]

        for type_str in string_types:
            assert type_str in ENTITY_CREATORS, f"Missing creator for '{type_str}'"


class TestLayerPrefixColors:
    """Tests for LAYER_PREFIX_COLORS constant."""

    def test_all_disciplines_have_colors(self):
        """Test that all discipline prefixes have colors."""
        expected_prefixes = ["A-", "M-", "E-", "P-", "F-", "T-", "G-", "S-", "C-"]

        for prefix in expected_prefixes:
            assert prefix in LAYER_PREFIX_COLORS, f"Missing color for {prefix}"

    def test_colors_are_valid(self):
        """Test that all colors are valid AutoCAD color indices."""
        for prefix, color in LAYER_PREFIX_COLORS.items():
            assert 1 <= color <= 255, f"Invalid color {color} for {prefix}"
