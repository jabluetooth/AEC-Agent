"""Unit tests for relationship_builder.py (Phase E)."""

import pytest

from aec_agent.mcp.tools.geometry_classifier import (
    ClassifiedLine,
    GeometryClassification,
    GeometrySystem,
    GeometryType,
)
from aec_agent.mcp.tools.topology_analyzer import (
    EdgeType,
    NodeType,
    TopologyEdge,
    TopologyGraph,
    TopologyNode,
)
from aec_agent.mcp.tools.relationship_builder import (
    InferredRelationship,
    RelationshipBuilder,
    RelationshipBuilderResult,
    RelationType,
    RoomBoundary,
    build_relationships,
    find_connected_chain,
    find_elements_in_room,
)


class TestRelationType:
    """Tests for RelationType enum."""

    def test_containment_types(self):
        """Containment types should exist."""
        assert RelationType.CONTAINS.value == "contains"
        assert RelationType.CONTAINED_BY.value == "contained_by"

    def test_connectivity_types(self):
        """Connectivity types should exist."""
        assert RelationType.CONNECTED_TO.value == "connected_to"
        assert RelationType.BRANCHES_FROM.value == "branches_from"
        assert RelationType.MERGES_INTO.value == "merges_into"

    def test_flow_types(self):
        """Flow types should exist."""
        assert RelationType.FEEDS.value == "feeds"
        assert RelationType.FED_BY.value == "fed_by"
        assert RelationType.SUPPLIES.value == "supplies"
        assert RelationType.RETURNS_TO.value == "returns_to"

    def test_spatial_types(self):
        """Spatial types should exist."""
        assert RelationType.SERVES.value == "serves"
        assert RelationType.ADJACENT_TO.value == "adjacent_to"
        assert RelationType.NEAR.value == "near"

    def test_annotation_types(self):
        """Annotation types should exist."""
        assert RelationType.LABELS.value == "labels"
        assert RelationType.REFERENCES.value == "references"


class TestInferredRelationship:
    """Tests for InferredRelationship dataclass."""

    def test_default_values(self):
        """Default values should be set correctly."""
        rel = InferredRelationship(
            source_id="src_1",
            target_id="tgt_1",
            relation_type=RelationType.CONTAINS,
        )
        assert rel.source_id == "src_1"
        assert rel.target_id == "tgt_1"
        assert rel.relation_type == RelationType.CONTAINS
        assert rel.confidence == 1.0
        assert rel.distance is None
        assert rel.bidirectional is False

    def test_custom_values(self):
        """Custom values should be preserved."""
        rel = InferredRelationship(
            source_id="pipe_1",
            target_id="valve_1",
            relation_type=RelationType.CONNECTED_TO,
            confidence=0.85,
            distance=5.5,
            system=GeometrySystem.DOMESTIC_COLD,
            bidirectional=True,
            reasoning="Endpoint proximity",
        )
        assert rel.confidence == 0.85
        assert rel.distance == 5.5
        assert rel.system == GeometrySystem.DOMESTIC_COLD
        assert rel.bidirectional is True

    def test_has_uuid(self):
        """Relationship should have auto-generated UUID."""
        rel = InferredRelationship(
            source_id="a",
            target_id="b",
            relation_type=RelationType.NEAR,
        )
        assert rel.id is not None


class TestRoomBoundary:
    """Tests for RoomBoundary dataclass."""

    def test_default_values(self):
        """Default values should be set correctly."""
        room = RoomBoundary(id="room_1")
        assert room.id == "room_1"
        assert room.name is None
        assert room.boundary == []
        assert room.area == 0.0

    def test_with_boundary(self):
        """Room should accept boundary polygon."""
        boundary = [(0, 0), (100, 0), (100, 100), (0, 100)]
        room = RoomBoundary(
            id="room_101",
            name="Conference Room",
            boundary=boundary,
            center=(50, 50),
            area=10000.0,
        )
        assert room.name == "Conference Room"
        assert len(room.boundary) == 4
        assert room.center == (50, 50)


class TestRelationshipBuilder:
    """Tests for RelationshipBuilder class."""

    @pytest.fixture
    def builder(self):
        """Create a builder instance."""
        return RelationshipBuilder()

    @pytest.fixture
    def square_room(self):
        """Create a simple square room."""
        return RoomBoundary(
            id="room_1",
            name="Room 101",
            boundary=[(0, 0), (100, 0), (100, 100), (0, 100)],
            center=(50, 50),
        )

    @pytest.fixture
    def adjacent_rooms(self):
        """Create two adjacent rooms sharing a wall."""
        room1 = RoomBoundary(
            id="room_1",
            name="Room 101",
            boundary=[(0, 0), (100, 0), (100, 100), (0, 100)],
        )
        room2 = RoomBoundary(
            id="room_2",
            name="Room 102",
            boundary=[(100, 0), (200, 0), (200, 100), (100, 100)],
        )
        return [room1, room2]

    @pytest.fixture
    def mock_symbol_inside(self):
        """Create a mock symbol inside the square room."""
        class MockSymbol:
            id = "diff_1"
            type = "diffuser"
            position = (50, 50)
            category = "mechanical"
        return MockSymbol()

    @pytest.fixture
    def mock_symbol_outside(self):
        """Create a mock symbol outside the square room."""
        class MockSymbol:
            id = "valve_1"
            type = "valve"
            position = (150, 50)
            category = "plumbing"
        return MockSymbol()

    def test_initialization(self, builder):
        """Builder should initialize with defaults."""
        assert builder.proximity_tolerance == 15.0
        assert builder.containment_margin == 5.0
        assert builder.adjacency_threshold == 50.0

    def test_custom_initialization(self):
        """Builder should accept custom parameters."""
        builder = RelationshipBuilder(
            proximity_tolerance=20.0,
            adjacency_threshold=100.0,
        )
        assert builder.proximity_tolerance == 20.0
        assert builder.adjacency_threshold == 100.0


class TestPointInPolygon:
    """Tests for point-in-polygon algorithm."""

    @pytest.fixture
    def builder(self):
        return RelationshipBuilder()

    @pytest.fixture
    def square(self):
        """Simple square polygon."""
        return [(0, 0), (100, 0), (100, 100), (0, 100)]

    @pytest.fixture
    def triangle(self):
        """Simple triangle polygon."""
        return [(0, 0), (100, 0), (50, 100)]

    @pytest.fixture
    def l_shaped(self):
        """L-shaped polygon."""
        return [(0, 0), (100, 0), (100, 50), (50, 50), (50, 100), (0, 100)]

    def test_point_inside_square(self, builder, square):
        """Point inside square should return True."""
        assert builder._point_in_polygon((50, 50), square) is True
        assert builder._point_in_polygon((10, 10), square) is True
        assert builder._point_in_polygon((90, 90), square) is True

    def test_point_outside_square(self, builder, square):
        """Point outside square should return False."""
        assert builder._point_in_polygon((150, 50), square) is False
        assert builder._point_in_polygon((-10, 50), square) is False
        assert builder._point_in_polygon((50, 150), square) is False

    def test_point_inside_triangle(self, builder, triangle):
        """Point inside triangle should return True."""
        assert builder._point_in_polygon((50, 30), triangle) is True

    def test_point_outside_triangle(self, builder, triangle):
        """Point outside triangle should return False."""
        assert builder._point_in_polygon((10, 90), triangle) is False

    def test_point_inside_l_shaped(self, builder, l_shaped):
        """Point inside L-shape should return True."""
        assert builder._point_in_polygon((25, 25), l_shaped) is True
        assert builder._point_in_polygon((25, 75), l_shaped) is True

    def test_point_outside_l_shaped(self, builder, l_shaped):
        """Point in the L-shape cutout should return False."""
        assert builder._point_in_polygon((75, 75), l_shaped) is False


class TestContainmentRelationships:
    """Tests for containment relationship detection."""

    @pytest.fixture
    def builder(self):
        return RelationshipBuilder()

    @pytest.fixture
    def room_with_equipment(self):
        """Room and equipment for containment test."""
        room = RoomBoundary(
            id="room_1",
            name="Room 101",
            boundary=[(0, 0), (100, 0), (100, 100), (0, 100)],
        )

        class MockDiffuser:
            id = "diff_1"
            type = "diffuser"
            position = (50, 50)

        class MockOutlet:
            id = "outlet_1"
            type = "outlet"
            position = (30, 30)

        return room, [MockDiffuser(), MockOutlet()]

    @pytest.mark.asyncio
    async def test_detect_containment(self, builder, room_with_equipment):
        """Should detect equipment inside room."""
        room, symbols = room_with_equipment

        relationships = await builder._build_containment_relationships([room], symbols)

        # Should have 2 CONTAINS and 2 SERVES relationships
        contains = [r for r in relationships if r.relation_type == RelationType.CONTAINS]
        serves = [r for r in relationships if r.relation_type == RelationType.SERVES]

        assert len(contains) == 2
        assert len(serves) == 2  # diffuser and outlet are terminals

    @pytest.mark.asyncio
    async def test_equipment_outside_room(self, builder):
        """Should not create relationship for equipment outside room."""
        room = RoomBoundary(
            id="room_1",
            boundary=[(0, 0), (100, 0), (100, 100), (0, 100)],
        )

        class MockSymbol:
            id = "sym_1"
            type = "valve"
            position = (150, 50)  # Outside

        relationships = await builder._build_containment_relationships([room], [MockSymbol()])
        assert len(relationships) == 0


class TestConnectivityRelationships:
    """Tests for connectivity relationship detection."""

    @pytest.fixture
    def builder(self):
        return RelationshipBuilder()

    @pytest.fixture
    def pipe_with_valves(self):
        """Pipe connecting two valves."""
        class MockValve1:
            id = "valve_1"
            type = "valve"
            position = (0, 0)

        class MockValve2:
            id = "valve_2"
            type = "valve"
            position = (100, 0)

        pipe = ClassifiedLine(
            start=(0, 0),
            end=(100, 0),
            linetype="CONTINUOUS",
            classification=GeometryClassification(
                geometry_type=GeometryType.PIPE,
                system=GeometrySystem.DOMESTIC_COLD,
            ),
        )

        return [MockValve1(), MockValve2()], [pipe]

    @pytest.mark.asyncio
    async def test_detect_connectivity(self, builder, pipe_with_valves):
        """Should detect symbols connected by geometry."""
        symbols, geometry = pipe_with_valves

        relationships = await builder._build_connectivity_relationships(symbols, geometry)

        # Should detect connections
        assert len(relationships) >= 2  # Line to valves + valve to valve

        connected = [r for r in relationships if r.relation_type == RelationType.CONNECTED_TO]
        assert len(connected) >= 2

    @pytest.mark.asyncio
    async def test_connectivity_with_tolerance(self, builder):
        """Should connect symbols within tolerance."""
        class MockSymbol:
            id = "sym_1"
            type = "valve"
            position = (5, 0)  # 5 units from line endpoint

        pipe = ClassifiedLine(
            start=(0, 0),
            end=(100, 0),
            classification=GeometryClassification(GeometryType.PIPE),
        )

        relationships = await builder._build_connectivity_relationships([MockSymbol()], [pipe])

        # Should still connect (within default 15 unit tolerance)
        connected = [r for r in relationships if r.relation_type == RelationType.CONNECTED_TO]
        assert len(connected) >= 1


class TestSpatialRelationships:
    """Tests for spatial relationship detection."""

    @pytest.fixture
    def builder(self):
        return RelationshipBuilder()

    @pytest.mark.asyncio
    async def test_room_adjacency(self, builder):
        """Should detect adjacent rooms."""
        room1 = RoomBoundary(
            id="room_1",
            boundary=[(0, 0), (100, 0), (100, 100), (0, 100)],
        )
        room2 = RoomBoundary(
            id="room_2",
            boundary=[(100, 0), (200, 0), (200, 100), (100, 100)],
        )

        relationships = await builder._build_spatial_relationships([room1, room2], [])

        adjacent = [r for r in relationships if r.relation_type == RelationType.ADJACENT_TO]
        assert len(adjacent) == 1
        assert adjacent[0].bidirectional is True

    @pytest.mark.asyncio
    async def test_symbol_proximity(self, builder):
        """Should detect symbols near each other."""
        class MockSymbol1:
            id = "sym_1"
            type = "valve"
            position = (0, 0)

        class MockSymbol2:
            id = "sym_2"
            type = "fitting"
            position = (20, 0)  # 20 units away (within 50 threshold)

        relationships = await builder._build_spatial_relationships(
            [], [MockSymbol1(), MockSymbol2()]
        )

        near = [r for r in relationships if r.relation_type == RelationType.NEAR]
        assert len(near) == 1
        assert near[0].distance == pytest.approx(20.0, abs=0.1)


class TestFlowRelationships:
    """Tests for flow relationship detection from topology."""

    @pytest.fixture
    def builder(self):
        return RelationshipBuilder()

    @pytest.fixture
    def hvac_topology(self):
        """Simple HVAC topology: AHU -> TEE -> Diffuser1, Diffuser2"""
        graph = TopologyGraph()

        ahu = TopologyNode(
            id="ahu_1",
            node_type=NodeType.SOURCE,
            position=(0, 0),
            system=GeometrySystem.SUPPLY_AIR,
            symbol_type="ahu",
        )
        tee = TopologyNode(
            id="tee_1",
            node_type=NodeType.TEE,
            position=(100, 0),
            system=GeometrySystem.SUPPLY_AIR,
        )
        diff1 = TopologyNode(
            id="diff_1",
            node_type=NodeType.TERMINAL,
            position=(150, 50),
            system=GeometrySystem.SUPPLY_AIR,
            symbol_type="diffuser",
        )
        diff2 = TopologyNode(
            id="diff_2",
            node_type=NodeType.TERMINAL,
            position=(150, -50),
            system=GeometrySystem.SUPPLY_AIR,
            symbol_type="diffuser",
        )

        graph.add_node(ahu)
        graph.add_node(tee)
        graph.add_node(diff1)
        graph.add_node(diff2)

        graph.add_edge(TopologyEdge(
            id="e1", source_id="ahu_1", target_id="tee_1",
            edge_type=EdgeType.DUCT_RUN, length=100,
        ))
        graph.add_edge(TopologyEdge(
            id="e2", source_id="tee_1", target_id="diff_1",
            edge_type=EdgeType.DUCT_RUN, length=70,
        ))
        graph.add_edge(TopologyEdge(
            id="e3", source_id="tee_1", target_id="diff_2",
            edge_type=EdgeType.DUCT_RUN, length=70,
        ))

        return graph

    @pytest.mark.asyncio
    async def test_detect_feeds_relationships(self, builder, hvac_topology):
        """Should detect FEEDS relationships from source to terminals."""
        relationships = await builder._build_flow_relationships(hvac_topology, [])

        feeds = [r for r in relationships if r.relation_type == RelationType.FEEDS]
        assert len(feeds) == 2  # AHU feeds both diffusers

        # Verify both diffusers are targets
        target_ids = {r.target_id for r in feeds}
        assert "diff_1" in target_ids
        assert "diff_2" in target_ids


class TestBuildRelationshipsFunction:
    """Tests for the build_relationships convenience function."""

    @pytest.mark.asyncio
    async def test_basic_call(self):
        """Should work with minimal arguments."""
        result = await build_relationships()
        assert isinstance(result, RelationshipBuilderResult)
        assert len(result.relationships) == 0

    @pytest.mark.asyncio
    async def test_with_rooms_and_symbols(self):
        """Should detect relationships with rooms and symbols."""
        room = RoomBoundary(
            id="room_1",
            boundary=[(0, 0), (100, 0), (100, 100), (0, 100)],
        )

        class MockSymbol:
            id = "diff_1"
            type = "diffuser"
            position = (50, 50)

        result = await build_relationships(
            rooms=[room],
            symbols=[MockSymbol()],
        )

        assert len(result.containment_relationships) >= 1
        assert result.statistics["total_relationships"] >= 1


class TestFindElementsInRoom:
    """Tests for find_elements_in_room function."""

    @pytest.mark.asyncio
    async def test_find_elements(self):
        """Should find symbols inside room."""
        room = RoomBoundary(
            id="room_1",
            boundary=[(0, 0), (100, 0), (100, 100), (0, 100)],
        )

        class MockInside:
            position = (50, 50)

        class MockOutside:
            position = (150, 50)

        symbols = [MockInside(), MockOutside()]
        contained = await find_elements_in_room(room, symbols)

        assert len(contained) == 1


class TestFindConnectedChain:
    """Tests for find_connected_chain function."""

    @pytest.mark.asyncio
    async def test_find_chain(self):
        """Should find chain of connected symbols."""
        class MockSymbol:
            def __init__(self, sym_id, x, y):
                self.id = sym_id
                self.position = (x, y)

        # Create a chain: sym1 -- line1 -- sym2 -- line2 -- sym3
        sym1 = MockSymbol("sym_1", 0, 0)
        sym2 = MockSymbol("sym_2", 100, 0)
        sym3 = MockSymbol("sym_3", 200, 0)

        line1 = ClassifiedLine(
            start=(0, 0), end=(100, 0),
            classification=GeometryClassification(GeometryType.PIPE),
        )
        line2 = ClassifiedLine(
            start=(100, 0), end=(200, 0),
            classification=GeometryClassification(GeometryType.PIPE),
        )

        chain = await find_connected_chain(
            sym1, [sym1, sym2, sym3], [line1, line2], max_depth=10
        )

        assert "sym_1" in chain
        assert "sym_2" in chain
        assert "sym_3" in chain


class TestEdgesOverlap:
    """Tests for edge overlap detection (room adjacency)."""

    @pytest.fixture
    def builder(self):
        return RelationshipBuilder()

    def test_horizontal_overlap(self, builder):
        """Horizontal edges that overlap should be detected."""
        # Edge 1: (50, 0) to (150, 0)
        # Edge 2: (100, 0) to (200, 0)
        p1, p2 = (50, 0), (150, 0)
        p3, p4 = (100, 0), (200, 0)

        assert builder._edges_overlap(p1, p2, p3, p4) is True

    def test_vertical_overlap(self, builder):
        """Vertical edges that overlap should be detected."""
        p1, p2 = (0, 50), (0, 150)
        p3, p4 = (0, 100), (0, 200)

        assert builder._edges_overlap(p1, p2, p3, p4) is True

    def test_no_overlap(self, builder):
        """Non-overlapping parallel edges should not match."""
        p1, p2 = (0, 0), (100, 0)
        p3, p4 = (200, 0), (300, 0)

        assert builder._edges_overlap(p1, p2, p3, p4) is False

    def test_non_parallel(self, builder):
        """Non-parallel edges should not match."""
        p1, p2 = (0, 0), (100, 0)
        p3, p4 = (50, 0), (50, 100)

        assert builder._edges_overlap(p1, p2, p3, p4) is False


class TestStatistics:
    """Tests for relationship statistics computation."""

    @pytest.mark.asyncio
    async def test_statistics_computed(self):
        """Statistics should be computed correctly."""
        room = RoomBoundary(
            id="room_1",
            boundary=[(0, 0), (100, 0), (100, 100), (0, 100)],
        )

        class MockDiffuser:
            id = "diff_1"
            type = "diffuser"
            position = (50, 50)

        result = await build_relationships(
            rooms=[room],
            symbols=[MockDiffuser()],
        )

        assert "total_relationships" in result.statistics
        assert "containment_count" in result.statistics
        assert "type_counts" in result.statistics
        assert "average_confidence" in result.statistics


class TestSymbolPositionExtraction:
    """Tests for extracting positions from various symbol formats."""

    @pytest.fixture
    def builder(self):
        return RelationshipBuilder()

    def test_position_as_tuple(self, builder):
        """Should extract position from tuple attribute."""
        class MockSymbol:
            position = (10.5, 20.5)

        pos = builder._get_symbol_position(MockSymbol())
        assert pos == (10.5, 20.5)

    def test_position_as_point_object(self, builder):
        """Should extract position from point object with x,y."""
        class Point:
            x = 15.0
            y = 25.0

        class MockSymbol:
            position = Point()

        pos = builder._get_symbol_position(MockSymbol())
        assert pos == (15.0, 25.0)

    def test_center_fallback(self, builder):
        """Should use center if position not available."""
        class MockSymbol:
            center = (30.0, 40.0)

        pos = builder._get_symbol_position(MockSymbol())
        assert pos == (30.0, 40.0)

    def test_xy_attributes(self, builder):
        """Should use x,y attributes directly."""
        class MockSymbol:
            x = 45.0
            y = 55.0

        pos = builder._get_symbol_position(MockSymbol())
        assert pos == (45.0, 55.0)

    def test_no_position(self, builder):
        """Should return None if no position found."""
        class MockSymbol:
            name = "no position"

        pos = builder._get_symbol_position(MockSymbol())
        assert pos is None
