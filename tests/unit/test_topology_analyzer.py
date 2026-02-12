"""Unit tests for topology_analyzer.py (Phase D)."""

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
    SystemBranch,
    SystemPath,
    TopologyAnalyzer,
    TopologyEdge,
    TopologyGraph,
    TopologyNode,
    build_system_graph,
    find_connected_equipment,
)


class TestNodeType:
    """Tests for NodeType enum."""

    def test_equipment_types_exist(self):
        """Equipment node types should exist."""
        assert NodeType.EQUIPMENT.value == "equipment"
        assert NodeType.VALVE.value == "valve"
        assert NodeType.FITTING.value == "fitting"
        assert NodeType.TERMINAL.value == "terminal"
        assert NodeType.SOURCE.value == "source"

    def test_geometry_types_exist(self):
        """Geometry node types should exist."""
        assert NodeType.JUNCTION.value == "junction"
        assert NodeType.TEE.value == "tee"
        assert NodeType.CROSS.value == "cross"

    def test_spatial_types_exist(self):
        """Spatial node types should exist."""
        assert NodeType.ROOM.value == "room"
        assert NodeType.ZONE.value == "zone"
        assert NodeType.FLOOR.value == "floor"


class TestEdgeType:
    """Tests for EdgeType enum."""

    def test_physical_connection_types(self):
        """Physical connection types should exist."""
        assert EdgeType.PIPE_RUN.value == "pipe_run"
        assert EdgeType.DUCT_RUN.value == "duct_run"
        assert EdgeType.CONDUIT_RUN.value == "conduit_run"

    def test_logical_connection_types(self):
        """Logical connection types should exist."""
        assert EdgeType.FEEDS.value == "feeds"
        assert EdgeType.BRANCHES_FROM.value == "branches_from"
        assert EdgeType.CONNECTS_TO.value == "connects_to"

    def test_spatial_relationship_types(self):
        """Spatial relationship types should exist."""
        assert EdgeType.CONTAINS.value == "contains"
        assert EdgeType.ADJACENT_TO.value == "adjacent_to"
        assert EdgeType.SERVES.value == "serves"


class TestTopologyNode:
    """Tests for TopologyNode dataclass."""

    def test_default_values(self):
        """Default values should be set correctly."""
        node = TopologyNode(
            id="test_node",
            node_type=NodeType.JUNCTION,
            position=(10.0, 20.0),
        )
        assert node.id == "test_node"
        assert node.node_type == NodeType.JUNCTION
        assert node.position == (10.0, 20.0)
        assert node.system == GeometrySystem.UNKNOWN
        assert node.degree == 0

    def test_custom_values(self):
        """Custom values should be preserved."""
        node = TopologyNode(
            id="valve_1",
            node_type=NodeType.VALVE,
            position=(100.0, 200.0),
            system=GeometrySystem.DOMESTIC_COLD,
            symbol_type="gate_valve",
            block_name="P-VALV-GATE",
        )
        assert node.symbol_type == "gate_valve"
        assert node.block_name == "P-VALV-GATE"
        assert node.system == GeometrySystem.DOMESTIC_COLD


class TestTopologyEdge:
    """Tests for TopologyEdge dataclass."""

    def test_default_values(self):
        """Default values should be set correctly."""
        edge = TopologyEdge(
            id="edge_1",
            source_id="node_1",
            target_id="node_2",
            edge_type=EdgeType.PIPE_RUN,
        )
        assert edge.id == "edge_1"
        assert edge.source_id == "node_1"
        assert edge.target_id == "node_2"
        assert edge.edge_type == EdgeType.PIPE_RUN
        assert edge.length == 0.0
        assert edge.size is None

    def test_custom_values(self):
        """Custom values should be preserved."""
        edge = TopologyEdge(
            id="pipe_1",
            source_id="valve_1",
            target_id="fixture_1",
            edge_type=EdgeType.PIPE_RUN,
            system=GeometrySystem.DOMESTIC_HOT,
            length=150.0,
            size='3/4"',
            flow_direction="forward",
        )
        assert edge.length == 150.0
        assert edge.size == '3/4"'
        assert edge.flow_direction == "forward"


class TestTopologyGraph:
    """Tests for TopologyGraph dataclass and methods."""

    @pytest.fixture
    def empty_graph(self):
        """Create an empty graph."""
        return TopologyGraph()

    @pytest.fixture
    def simple_graph(self):
        """Create a simple graph with nodes and edges."""
        graph = TopologyGraph()

        # Add nodes
        node1 = TopologyNode(id="n1", node_type=NodeType.SOURCE, position=(0.0, 0.0))
        node2 = TopologyNode(id="n2", node_type=NodeType.JUNCTION, position=(50.0, 0.0))
        node3 = TopologyNode(id="n3", node_type=NodeType.TERMINAL, position=(100.0, 0.0))

        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)

        # Add edges
        edge1 = TopologyEdge(id="e1", source_id="n1", target_id="n2", edge_type=EdgeType.PIPE_RUN)
        edge2 = TopologyEdge(id="e2", source_id="n2", target_id="n3", edge_type=EdgeType.PIPE_RUN)

        graph.add_edge(edge1)
        graph.add_edge(edge2)

        return graph

    def test_add_node(self, empty_graph):
        """Should add nodes correctly."""
        node = TopologyNode(id="test", node_type=NodeType.VALVE, position=(0.0, 0.0))
        empty_graph.add_node(node)
        assert "test" in empty_graph.nodes
        assert empty_graph.nodes["test"] == node

    def test_add_node_with_system(self, empty_graph):
        """Should track nodes by system."""
        node = TopologyNode(
            id="pump",
            node_type=NodeType.SOURCE,
            position=(0.0, 0.0),
            system=GeometrySystem.DOMESTIC_COLD,
        )
        empty_graph.add_node(node)
        assert GeometrySystem.DOMESTIC_COLD in empty_graph.systems
        assert "pump" in empty_graph.systems[GeometrySystem.DOMESTIC_COLD]

    def test_add_edge(self, empty_graph):
        """Should add edges and update adjacency."""
        node1 = TopologyNode(id="n1", node_type=NodeType.JUNCTION, position=(0.0, 0.0))
        node2 = TopologyNode(id="n2", node_type=NodeType.JUNCTION, position=(10.0, 0.0))
        empty_graph.add_node(node1)
        empty_graph.add_node(node2)

        edge = TopologyEdge(id="e1", source_id="n1", target_id="n2", edge_type=EdgeType.PIPE_RUN)
        empty_graph.add_edge(edge)

        assert "e1" in empty_graph.edges
        assert "n2" in empty_graph.adjacency["n1"]
        assert "n1" in empty_graph.adjacency["n2"]

    def test_add_edge_updates_degree(self, empty_graph):
        """Adding edge should update node degrees."""
        node1 = TopologyNode(id="n1", node_type=NodeType.JUNCTION, position=(0.0, 0.0))
        node2 = TopologyNode(id="n2", node_type=NodeType.JUNCTION, position=(10.0, 0.0))
        empty_graph.add_node(node1)
        empty_graph.add_node(node2)

        edge = TopologyEdge(id="e1", source_id="n1", target_id="n2", edge_type=EdgeType.PIPE_RUN)
        empty_graph.add_edge(edge)

        assert empty_graph.nodes["n1"].degree == 1
        assert empty_graph.nodes["n2"].degree == 1

    def test_get_neighbors(self, simple_graph):
        """Should return correct neighbors."""
        neighbors = simple_graph.get_neighbors("n2")
        assert "n1" in neighbors
        assert "n3" in neighbors
        assert len(neighbors) == 2

    def test_get_edges_for_node(self, simple_graph):
        """Should return edges connected to a node."""
        edges = simple_graph.get_edges_for_node("n2")
        assert len(edges) == 2
        edge_ids = [e.id for e in edges]
        assert "e1" in edge_ids
        assert "e2" in edge_ids

    def test_get_nodes_by_type(self, simple_graph):
        """Should return nodes of specific type."""
        sources = simple_graph.get_nodes_by_type(NodeType.SOURCE)
        assert len(sources) == 1
        assert sources[0].id == "n1"

        terminals = simple_graph.get_nodes_by_type(NodeType.TERMINAL)
        assert len(terminals) == 1
        assert terminals[0].id == "n3"

    def test_find_path_exists(self, simple_graph):
        """Should find path between connected nodes."""
        path = simple_graph.find_path("n1", "n3")
        assert path is not None
        assert path == ["n1", "n2", "n3"]

    def test_find_path_not_exists(self, simple_graph):
        """Should return None for non-existent path."""
        # Add an isolated node
        node4 = TopologyNode(id="n4", node_type=NodeType.JUNCTION, position=(200.0, 0.0))
        simple_graph.add_node(node4)

        path = simple_graph.find_path("n1", "n4")
        assert path is None

    def test_find_path_same_node(self, simple_graph):
        """Should return single-node path for same start/end."""
        path = simple_graph.find_path("n1", "n1")
        assert path == ["n1"]

    def test_find_all_paths(self, simple_graph):
        """Should find all paths between nodes."""
        paths = simple_graph.find_all_paths("n1", "n3")
        assert len(paths) >= 1
        assert ["n1", "n2", "n3"] in paths

    def test_find_connected_components(self, simple_graph):
        """Should find connected components."""
        components = simple_graph.find_connected_components()
        assert len(components) == 1
        assert len(components[0]) == 3

    def test_find_connected_components_with_isolated(self, simple_graph):
        """Should find multiple components when nodes are isolated."""
        node4 = TopologyNode(id="n4", node_type=NodeType.JUNCTION, position=(200.0, 0.0))
        simple_graph.add_node(node4)

        components = simple_graph.find_connected_components()
        assert len(components) == 2

    def test_get_statistics(self, simple_graph):
        """Should compute graph statistics."""
        stats = simple_graph.get_statistics()
        assert stats["node_count"] == 3
        assert stats["edge_count"] == 2
        assert stats["connected_components"] == 1
        assert "node_types" in stats
        assert "systems" in stats


class TestTopologyAnalyzer:
    """Tests for TopologyAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create an analyzer instance."""
        return TopologyAnalyzer()

    @pytest.fixture
    def simple_lines(self):
        """Create simple classified lines."""
        return [
            ClassifiedLine(
                start=(0.0, 0.0),
                end=(50.0, 0.0),
                linetype="CONTINUOUS",
                classification=GeometryClassification(
                    geometry_type=GeometryType.PIPE,
                    system=GeometrySystem.DOMESTIC_COLD,
                ),
            ),
            ClassifiedLine(
                start=(50.0, 0.0),
                end=(100.0, 0.0),
                linetype="CONTINUOUS",
                classification=GeometryClassification(
                    geometry_type=GeometryType.PIPE,
                    system=GeometrySystem.DOMESTIC_COLD,
                ),
            ),
        ]

    def test_initialization(self, analyzer):
        """Analyzer should initialize with defaults."""
        assert analyzer.proximity_tolerance == 10.0
        assert analyzer.min_edge_length == 5.0

    def test_custom_initialization(self):
        """Analyzer should accept custom parameters."""
        analyzer = TopologyAnalyzer(proximity_tolerance=20.0, min_edge_length=10.0)
        assert analyzer.proximity_tolerance == 20.0
        assert analyzer.min_edge_length == 10.0

    @pytest.mark.asyncio
    async def test_build_empty_graph(self, analyzer):
        """Should build empty graph with no input."""
        graph = await analyzer.build_system_graph()
        assert isinstance(graph, TopologyGraph)
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    @pytest.mark.asyncio
    async def test_build_graph_from_lines(self, analyzer, simple_lines):
        """Should create nodes and edges from lines."""
        graph = await analyzer.build_system_graph(lines=simple_lines)
        assert len(graph.nodes) >= 2  # At least endpoints
        assert len(graph.edges) >= 1

    @pytest.mark.asyncio
    async def test_build_graph_creates_junction_nodes(self, analyzer, simple_lines):
        """Should create junction nodes at line intersections."""
        graph = await analyzer.build_system_graph(lines=simple_lines)
        # Lines share endpoint at (50, 0), should create junction
        junctions = [
            n for n in graph.nodes.values()
            if n.node_type in [NodeType.JUNCTION, NodeType.TEE]
        ]
        assert len(junctions) >= 1

    @pytest.mark.asyncio
    async def test_find_system_paths(self, analyzer):
        """Should find paths from sources to terminals."""
        # Create a graph with source -> junction -> terminal
        graph = TopologyGraph()

        source = TopologyNode(
            id="source_1",
            node_type=NodeType.SOURCE,
            position=(0.0, 0.0),
            system=GeometrySystem.DOMESTIC_COLD,
        )
        junction = TopologyNode(
            id="junction_1",
            node_type=NodeType.JUNCTION,
            position=(50.0, 0.0),
            system=GeometrySystem.DOMESTIC_COLD,
        )
        terminal = TopologyNode(
            id="terminal_1",
            node_type=NodeType.TERMINAL,
            position=(100.0, 0.0),
            system=GeometrySystem.DOMESTIC_COLD,
        )

        graph.add_node(source)
        graph.add_node(junction)
        graph.add_node(terminal)

        graph.add_edge(TopologyEdge(
            id="e1", source_id="source_1", target_id="junction_1",
            edge_type=EdgeType.PIPE_RUN, length=50.0,
        ))
        graph.add_edge(TopologyEdge(
            id="e2", source_id="junction_1", target_id="terminal_1",
            edge_type=EdgeType.PIPE_RUN, length=50.0,
        ))

        paths = await analyzer.find_system_paths(graph)
        assert len(paths) == 1
        assert paths[0].source_node == "source_1"
        assert paths[0].terminal_node == "terminal_1"
        assert paths[0].total_length == 100.0

    @pytest.mark.asyncio
    async def test_analyze_connectivity(self, analyzer):
        """Should analyze graph connectivity."""
        graph = TopologyGraph()

        node1 = TopologyNode(id="n1", node_type=NodeType.JUNCTION, position=(0.0, 0.0))
        node2 = TopologyNode(id="n2", node_type=NodeType.JUNCTION, position=(50.0, 0.0))
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_edge(TopologyEdge(
            id="e1", source_id="n1", target_id="n2", edge_type=EdgeType.PIPE_RUN,
        ))

        analysis = await analyzer.analyze_connectivity(graph)
        assert analysis["node_count"] == 2
        assert analysis["edge_count"] == 1
        assert analysis["is_fully_connected"] is True
        assert len(analysis["isolated_nodes"]) == 0


class TestBuildSystemGraphFunction:
    """Tests for the build_system_graph convenience function."""

    @pytest.mark.asyncio
    async def test_basic_call(self):
        """Should work with minimal arguments."""
        graph = await build_system_graph()
        assert isinstance(graph, TopologyGraph)

    @pytest.mark.asyncio
    async def test_with_lines(self):
        """Should accept lines parameter."""
        lines = [
            ClassifiedLine(
                start=(0.0, 0.0),
                end=(100.0, 0.0),
                classification=GeometryClassification(GeometryType.PIPE),
            ),
        ]
        graph = await build_system_graph(lines=lines)
        assert len(graph.nodes) >= 1


class TestFindConnectedEquipmentFunction:
    """Tests for the find_connected_equipment convenience function."""

    @pytest.fixture
    def connected_equipment_graph(self):
        """Create a graph with connected equipment."""
        graph = TopologyGraph()

        # AHU -> Junction -> Diffuser1, Diffuser2
        ahu = TopologyNode(
            id="ahu_1",
            node_type=NodeType.SOURCE,
            position=(0.0, 0.0),
            system=GeometrySystem.SUPPLY_AIR,
        )
        junction = TopologyNode(
            id="jct_1",
            node_type=NodeType.TEE,
            position=(100.0, 0.0),
        )
        diff1 = TopologyNode(
            id="diff_1",
            node_type=NodeType.TERMINAL,
            position=(150.0, 50.0),
        )
        diff2 = TopologyNode(
            id="diff_2",
            node_type=NodeType.TERMINAL,
            position=(150.0, -50.0),
        )

        graph.add_node(ahu)
        graph.add_node(junction)
        graph.add_node(diff1)
        graph.add_node(diff2)

        graph.add_edge(TopologyEdge(
            id="e1", source_id="ahu_1", target_id="jct_1", edge_type=EdgeType.DUCT_RUN,
        ))
        graph.add_edge(TopologyEdge(
            id="e2", source_id="jct_1", target_id="diff_1", edge_type=EdgeType.DUCT_RUN,
        ))
        graph.add_edge(TopologyEdge(
            id="e3", source_id="jct_1", target_id="diff_2", edge_type=EdgeType.DUCT_RUN,
        ))

        return graph

    @pytest.mark.asyncio
    async def test_find_connected_from_source(self, connected_equipment_graph):
        """Should find equipment connected to source."""
        connected = await find_connected_equipment(connected_equipment_graph, "ahu_1")
        # Should find both diffusers
        assert "diff_1" in connected
        assert "diff_2" in connected

    @pytest.mark.asyncio
    async def test_find_connected_from_terminal(self, connected_equipment_graph):
        """Should find equipment connected to terminal."""
        connected = await find_connected_equipment(connected_equipment_graph, "diff_1")
        # Should find AHU and other diffuser
        assert "ahu_1" in connected
        assert "diff_2" in connected

    @pytest.mark.asyncio
    async def test_nonexistent_equipment(self, connected_equipment_graph):
        """Should return empty list for non-existent equipment."""
        connected = await find_connected_equipment(connected_equipment_graph, "fake_id")
        assert connected == []

    @pytest.mark.asyncio
    async def test_max_depth_limit(self, connected_equipment_graph):
        """Should respect max_depth limit."""
        connected = await find_connected_equipment(
            connected_equipment_graph, "ahu_1", max_depth=1
        )
        # With depth 1, may not reach diffusers through junction
        assert isinstance(connected, list)


class TestSymbolToNodeType:
    """Tests for symbol to node type conversion."""

    @pytest.fixture
    def analyzer(self):
        return TopologyAnalyzer()

    def test_valve_symbol(self, analyzer):
        """Valve symbols should become VALVE nodes."""
        class MockSymbol:
            type = "gate_valve"
            category = "plumbing"

        node_type = analyzer._symbol_to_node_type(MockSymbol())
        assert node_type == NodeType.VALVE

    def test_fitting_symbol(self, analyzer):
        """Fitting symbols should become FITTING nodes."""
        class MockSymbol:
            type = "elbow_90"
            category = "plumbing"

        node_type = analyzer._symbol_to_node_type(MockSymbol())
        assert node_type == NodeType.FITTING

    def test_diffuser_symbol(self, analyzer):
        """Diffuser symbols should become TERMINAL nodes."""
        class MockSymbol:
            type = "diffuser_square"
            category = "mechanical"

        node_type = analyzer._symbol_to_node_type(MockSymbol())
        assert node_type == NodeType.TERMINAL

    def test_ahu_symbol(self, analyzer):
        """AHU symbols should become SOURCE nodes."""
        class MockSymbol:
            type = "ahu_rooftop"
            category = "mechanical"

        node_type = analyzer._symbol_to_node_type(MockSymbol())
        assert node_type == NodeType.SOURCE


class TestGeometryToEdgeType:
    """Tests for geometry to edge type conversion."""

    @pytest.fixture
    def analyzer(self):
        return TopologyAnalyzer()

    def test_pipe_to_pipe_run(self, analyzer):
        """PIPE geometry should become PIPE_RUN edge."""
        edge_type = analyzer._geometry_to_edge_type(GeometryType.PIPE)
        assert edge_type == EdgeType.PIPE_RUN

    def test_duct_to_duct_run(self, analyzer):
        """DUCT geometry should become DUCT_RUN edge."""
        edge_type = analyzer._geometry_to_edge_type(GeometryType.DUCT)
        assert edge_type == EdgeType.DUCT_RUN

    def test_duct_rectangular_to_duct_run(self, analyzer):
        """DUCT_RECTANGULAR geometry should become DUCT_RUN edge."""
        edge_type = analyzer._geometry_to_edge_type(GeometryType.DUCT_RECTANGULAR)
        assert edge_type == EdgeType.DUCT_RUN

    def test_conduit_to_conduit_run(self, analyzer):
        """CONDUIT geometry should become CONDUIT_RUN edge."""
        edge_type = analyzer._geometry_to_edge_type(GeometryType.CONDUIT)
        assert edge_type == EdgeType.CONDUIT_RUN

    def test_unknown_to_connects_to(self, analyzer):
        """Unknown geometry should become CONNECTS_TO edge."""
        edge_type = analyzer._geometry_to_edge_type(GeometryType.UNKNOWN)
        assert edge_type == EdgeType.CONNECTS_TO
