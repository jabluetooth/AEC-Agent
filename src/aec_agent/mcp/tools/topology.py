"""
Topology cleanup for vectorized geometry using NetworkX.

Converts detected lines and polylines into a graph, merges artificial
breaks (degree-2 nodes), snaps dangling endpoints, and converts the
cleaned graph back to vectorization primitives.

This module is used as an optional post-processing step in
``vectorize_bitonal_image()`` when ``topology_cleanup=True``.
"""

import math
from typing import Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


def build_segment_graph(
    result: "VectorizationResult",
    snap_tolerance: float = 5.0,
) -> "nx.Graph":
    """
    Convert detected lines and polylines into a NetworkX graph.

    Nodes are unique endpoint coordinates (snapped within tolerance).
    Edges carry the original geometry as attributes.

    Args:
        result: VectorizationResult from image_vectorizer.
        snap_tolerance: Maximum distance in drawing units to consider
                        two endpoints as the same node.

    Returns:
        NetworkX Graph with nodes as (x, y) tuples and edges carrying
        ``geometry_type`` and ``points`` attributes.
    """
    import networkx as nx

    graph = nx.Graph()
    # Pool of known node positions for snapping
    node_pool: List[Tuple[float, float]] = []

    def _snap_point(pt: Tuple[float, float]) -> Tuple[float, float]:
        """Snap a point to the nearest existing node within tolerance."""
        best_dist = snap_tolerance
        best_node: Optional[Tuple[float, float]] = None
        for existing in node_pool:
            d = math.sqrt((pt[0] - existing[0]) ** 2 + (pt[1] - existing[1]) ** 2)
            if d < best_dist:
                best_dist = d
                best_node = existing
        if best_node is not None:
            return best_node
        # New node — add to pool
        node_pool.append(pt)
        return pt

    # Add lines as single-segment edges
    for line in result.lines:
        n1 = _snap_point(line.start)
        n2 = _snap_point(line.end)
        if n1 == n2:
            continue  # degenerate
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_edge(n1, n2, geometry_type="line", points=[n1, n2])

    # Add polylines as chains of edges
    for pline in result.polylines:
        if len(pline.points) < 2:
            continue
        snapped = [_snap_point(p) for p in pline.points]
        for i in range(len(snapped) - 1):
            n1 = snapped[i]
            n2 = snapped[i + 1]
            if n1 == n2:
                continue
            graph.add_node(n1)
            graph.add_node(n2)
            graph.add_edge(n1, n2, geometry_type="polyline_seg", points=[n1, n2])
        # Close the polyline if flagged
        if pline.closed and len(snapped) > 2 and snapped[0] != snapped[-1]:
            graph.add_edge(
                snapped[-1], snapped[0],
                geometry_type="polyline_seg", points=[snapped[-1], snapped[0]],
            )

    logger.info(
        "Segment graph built",
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
    )
    return graph


def merge_degree2_nodes(graph: "nx.Graph") -> "nx.Graph":
    """
    Merge degree-2 nodes (artificial breaks in straight lines).

    A node with exactly 2 edges is a "knee" — an artificial break where
    a single continuous line was split into two segments.  This function
    removes the node and merges the two edges into one, preserving the
    chain of intermediate points.

    Only merges if the two edges are roughly collinear (angle < 20 deg).

    Args:
        graph: NetworkX graph from ``build_segment_graph``.

    Returns:
        The same graph with degree-2 nodes merged.
    """
    merged_count = 0
    changed = True

    while changed:
        changed = False
        for node in list(graph.nodes()):
            if graph.degree(node) != 2:
                continue

            neighbors = list(graph.neighbors(node))
            if len(neighbors) != 2:
                continue

            n1, n2 = neighbors[0], neighbors[1]

            # Check collinearity: angle between the two segments
            v1 = (n1[0] - node[0], n1[1] - node[1])
            v2 = (n2[0] - node[0], n2[1] - node[1])
            len1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
            len2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
            if len1 < 1e-9 or len2 < 1e-9:
                continue

            # Angle between vectors (they point AWAY from the node, so
            # collinear segments have ~180° between them)
            cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2)
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle = math.degrees(math.acos(cos_angle))

            # Nearly 180° = collinear (tolerance: 160-180°)
            if angle < 160.0:
                continue

            # Merge: remove node, connect n1 to n2
            e1_data = graph.edges[node, n1]
            e2_data = graph.edges[node, n2]
            pts1 = list(e1_data.get("points", [n1, node]))
            pts2 = list(e2_data.get("points", [node, n2]))

            # Build merged point chain: n1 → ... → node → ... → n2
            # Ensure pts1 ends at node and pts2 starts at node
            if pts1 and pts1[-1] != node:
                pts1 = pts1[::-1]
            if pts2 and pts2[0] != node:
                pts2 = pts2[::-1]

            merged_pts = pts1 + pts2[1:]  # skip duplicate node

            graph.remove_node(node)
            if n1 != n2 and not graph.has_edge(n1, n2):
                graph.add_edge(n1, n2, geometry_type="line", points=merged_pts)
            merged_count += 1
            changed = True
            break  # restart iteration after mutation

    logger.info("Degree-2 merge complete", nodes_merged=merged_count)
    return graph


def snap_dangling_endpoints(
    graph: "nx.Graph",
    tolerance: float = 5.0,
) -> "nx.Graph":
    """
    Snap degree-1 nodes (dangling endpoints) to nearby nodes or edges.

    A dangling endpoint is a line that "almost" connects to another
    feature but falls short by a few pixels.  This function finds the
    nearest existing node within tolerance and creates a new edge to
    close the gap.

    Args:
        graph: NetworkX graph (post degree-2 merge).
        tolerance: Maximum distance in drawing units to snap.

    Returns:
        The same graph with dangling endpoints snapped.
    """
    snapped_count = 0

    for node in list(graph.nodes()):
        if graph.degree(node) != 1:
            continue

        best_dist = tolerance
        best_target: Optional[Tuple[float, float]] = None

        for other in graph.nodes():
            if other == node:
                continue
            # Don't snap to the node's own neighbor
            if graph.has_edge(node, other):
                continue
            d = math.sqrt((node[0] - other[0]) ** 2 + (node[1] - other[1]) ** 2)
            if d < best_dist:
                best_dist = d
                best_target = other

        if best_target is not None:
            graph.add_edge(
                node, best_target,
                geometry_type="snap", points=[node, best_target],
            )
            snapped_count += 1

    logger.info("Endpoint snapping complete", endpoints_snapped=snapped_count)
    return graph


def graph_to_vectorization_result(
    graph: "nx.Graph",
) -> "VectorizationResult":
    """
    Convert a cleaned NetworkX graph back to lines and polylines.

    Simple edges (2-point) become DetectedLine.
    Multi-point edges become DetectedPolyline.

    Circles, arcs, and ellipses are NOT in the graph and must be
    preserved separately by the caller.

    Args:
        graph: Cleaned NetworkX graph.

    Returns:
        VectorizationResult with lines and polylines populated.
    """
    from aec_agent.mcp.tools.image_vectorizer import (
        DetectedLine,
        DetectedPolyline,
        VectorizationResult,
    )

    result = VectorizationResult()

    for u, v, data in graph.edges(data=True):
        pts = data.get("points", [u, v])

        if len(pts) <= 2:
            result.lines.append(DetectedLine(start=pts[0], end=pts[-1]))
        else:
            result.polylines.append(DetectedPolyline(
                points=pts,
                closed=False,
                bulges=[0.0] * len(pts),
            ))

    logger.info(
        "Graph converted to vectorization result",
        lines=len(result.lines),
        polylines=len(result.polylines),
    )
    return result
