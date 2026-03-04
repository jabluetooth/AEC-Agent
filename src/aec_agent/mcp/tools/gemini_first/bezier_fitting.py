"""
Bezier Curve Fitting Module.

Implements Schneider's algorithm for fitting cubic Bezier curves to point
sequences. This enables accurate vectorization of curved elements like
mechanical parts, contours, and decorative elements.

Algorithm Reference:
- "An Algorithm for Automatically Fitting Digitized Curves"
  Philip J. Schneider, Graphics Gems (1990)

Features:
- Cubic Bezier fitting with configurable error tolerance
- Automatic corner detection for curve splitting
- G1 continuity at join points
- Curve detection in binary images
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import math

import numpy as np
from scipy import ndimage
from scipy.interpolate import splprep, splev
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class BezierCurve:
    """A cubic Bezier curve defined by 4 control points."""
    # Control points: P0 (start), P1, P2, P3 (end)
    control_points: List[Tuple[float, float]]
    start: Tuple[float, float] = field(init=False)
    end: Tuple[float, float] = field(init=False)
    fit_error: float = 0.0  # Maximum deviation from original points

    def __post_init__(self):
        if len(self.control_points) != 4:
            raise ValueError("Cubic Bezier requires exactly 4 control points")
        self.start = self.control_points[0]
        self.end = self.control_points[3]

    def evaluate(self, t: float) -> Tuple[float, float]:
        """
        Evaluate the Bezier curve at parameter t (0 to 1).

        Uses De Casteljau's algorithm for numerical stability.
        """
        p = self.control_points
        # De Casteljau's algorithm
        q0 = self._lerp(p[0], p[1], t)
        q1 = self._lerp(p[1], p[2], t)
        q2 = self._lerp(p[2], p[3], t)

        r0 = self._lerp(q0, q1, t)
        r1 = self._lerp(q1, q2, t)

        return self._lerp(r0, r1, t)

    def evaluate_derivative(self, t: float) -> Tuple[float, float]:
        """Evaluate the first derivative (tangent) at parameter t."""
        p = self.control_points
        # Derivative of cubic Bezier
        t2 = t * t
        mt = 1 - t
        mt2 = mt * mt

        dx = 3 * mt2 * (p[1][0] - p[0][0]) + \
             6 * mt * t * (p[2][0] - p[1][0]) + \
             3 * t2 * (p[3][0] - p[2][0])

        dy = 3 * mt2 * (p[1][1] - p[0][1]) + \
             6 * mt * t * (p[2][1] - p[1][1]) + \
             3 * t2 * (p[3][1] - p[2][1])

        return (dx, dy)

    @staticmethod
    def _lerp(p1: Tuple[float, float], p2: Tuple[float, float], t: float) -> Tuple[float, float]:
        """Linear interpolation between two points."""
        return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))

    def to_polyline(self, num_segments: int = 20) -> List[Tuple[float, float]]:
        """Convert to polyline by sampling the curve."""
        points = []
        for i in range(num_segments + 1):
            t = i / num_segments
            points.append(self.evaluate(t))
        return points

    def to_dict(self) -> dict:
        return {
            "control_points": self.control_points,
            "start": self.start,
            "end": self.end,
            "fit_error": self.fit_error,
        }


@dataclass
class CurveSegment:
    """A detected curve segment from an image."""
    points: np.ndarray  # Nx2 array of (x, y) points
    is_closed: bool = False
    start_tangent: Optional[Tuple[float, float]] = None
    end_tangent: Optional[Tuple[float, float]] = None


@dataclass
class BezierFitConfig:
    """Configuration for Bezier curve fitting."""
    max_error: float = 2.0  # Maximum allowed fitting error in pixels
    max_iterations: int = 4  # Maximum iterations for fitting refinement
    corner_threshold: float = 0.5  # Curvature threshold for corner detection (radians)
    min_segment_length: int = 4  # Minimum points in a segment
    subdivision_threshold: float = 4.0  # Error threshold for subdivision


def _compute_chord_length_params(points: np.ndarray) -> np.ndarray:
    """
    Compute parameterization based on chord length.

    Args:
        points: Nx2 array of points

    Returns:
        Array of parameter values (0 to 1)
    """
    n = len(points)
    if n < 2:
        return np.array([0.0])

    # Compute cumulative chord lengths
    diffs = np.diff(points, axis=0)
    chord_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    cumulative = np.zeros(n)
    cumulative[1:] = np.cumsum(chord_lengths)

    # Normalize to [0, 1]
    total_length = cumulative[-1]
    if total_length > 0:
        return cumulative / total_length
    return np.linspace(0, 1, n)


def _compute_left_tangent(points: np.ndarray) -> Tuple[float, float]:
    """Compute unit tangent at the start of a curve."""
    if len(points) < 2:
        return (1.0, 0.0)

    dx = points[1][0] - points[0][0]
    dy = points[1][1] - points[0][1]
    length = math.sqrt(dx * dx + dy * dy)

    if length > 0:
        return (dx / length, dy / length)
    return (1.0, 0.0)


def _compute_right_tangent(points: np.ndarray) -> Tuple[float, float]:
    """Compute unit tangent at the end of a curve."""
    if len(points) < 2:
        return (1.0, 0.0)

    dx = points[-1][0] - points[-2][0]
    dy = points[-1][1] - points[-2][1]
    length = math.sqrt(dx * dx + dy * dy)

    if length > 0:
        return (dx / length, dy / length)
    return (1.0, 0.0)


def _compute_center_tangent(points: np.ndarray, index: int) -> Tuple[float, float]:
    """Compute unit tangent at an interior point."""
    if index <= 0 or index >= len(points) - 1:
        return (1.0, 0.0)

    dx = points[index + 1][0] - points[index - 1][0]
    dy = points[index + 1][1] - points[index - 1][1]
    length = math.sqrt(dx * dx + dy * dy)

    if length > 0:
        return (dx / length, dy / length)
    return (1.0, 0.0)


def _generate_bezier(
    points: np.ndarray,
    params: np.ndarray,
    left_tangent: Tuple[float, float],
    right_tangent: Tuple[float, float],
) -> BezierCurve:
    """
    Generate a cubic Bezier curve using the method from Schneider's algorithm.

    This uses least-squares fitting with fixed endpoint tangents.
    """
    n = len(points)
    if n < 2:
        p = (float(points[0][0]), float(points[0][1]))
        return BezierCurve(control_points=[p, p, p, p])

    # First and last control points are the endpoints
    p0 = (float(points[0][0]), float(points[0][1]))
    p3 = (float(points[-1][0]), float(points[-1][1]))

    # Compute A matrices for least squares
    A = np.zeros((n, 2, 2))
    for i, t in enumerate(params):
        t2 = t * t
        t3 = t2 * t
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt

        # Bernstein polynomials for B1 and B2
        b1 = 3 * mt2 * t
        b2 = 3 * mt * t2

        A[i, 0, 0] = left_tangent[0] * b1
        A[i, 0, 1] = right_tangent[0] * b2
        A[i, 1, 0] = left_tangent[1] * b1
        A[i, 1, 1] = right_tangent[1] * b2

    # Compute C and X for least squares
    C = np.zeros((2, 2))
    X = np.zeros(2)

    for i in range(n):
        t = params[i]
        t2 = t * t
        t3 = t2 * t
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt

        C[0, 0] += np.dot(A[i, :, 0], A[i, :, 0])
        C[0, 1] += np.dot(A[i, :, 0], A[i, :, 1])
        C[1, 0] = C[0, 1]
        C[1, 1] += np.dot(A[i, :, 1], A[i, :, 1])

        # RHS
        tmp = np.array([
            points[i][0] - (mt3 * p0[0] + t3 * p3[0]),
            points[i][1] - (mt3 * p0[1] + t3 * p3[1])
        ])

        X[0] += np.dot(A[i, :, 0], tmp)
        X[1] += np.dot(A[i, :, 1], tmp)

    # Solve for alpha values
    det = C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0]

    if abs(det) < 1e-12:
        # Fallback: use simple heuristic for control points
        dist = math.sqrt((p3[0] - p0[0]) ** 2 + (p3[1] - p0[1]) ** 2) / 3.0
        alpha_l = dist
        alpha_r = dist
    else:
        alpha_l = (C[1, 1] * X[0] - C[0, 1] * X[1]) / det
        alpha_r = (C[0, 0] * X[1] - C[1, 0] * X[0]) / det

    # Ensure positive alphas (tangent direction is correct)
    seg_length = math.sqrt((p3[0] - p0[0]) ** 2 + (p3[1] - p0[1]) ** 2)
    epsilon = 1e-6 * seg_length

    if alpha_l < epsilon or alpha_r < epsilon:
        alpha_l = alpha_r = seg_length / 3.0

    # Compute control points
    p1 = (p0[0] + alpha_l * left_tangent[0], p0[1] + alpha_l * left_tangent[1])
    p2 = (p3[0] - alpha_r * right_tangent[0], p3[1] - alpha_r * right_tangent[1])

    return BezierCurve(control_points=[p0, p1, p2, p3])


def _compute_max_error(
    curve: BezierCurve,
    points: np.ndarray,
    params: np.ndarray,
) -> Tuple[float, int]:
    """
    Compute maximum fitting error and the index of the worst point.

    Returns:
        Tuple of (max_error, split_index)
    """
    max_error = 0.0
    split_index = len(points) // 2

    for i, (point, t) in enumerate(zip(points, params)):
        fitted = curve.evaluate(t)
        error = math.sqrt((point[0] - fitted[0]) ** 2 + (point[1] - fitted[1]) ** 2)

        if error > max_error:
            max_error = error
            split_index = i

    return max_error, split_index


def _reparameterize(
    curve: BezierCurve,
    points: np.ndarray,
    params: np.ndarray,
) -> np.ndarray:
    """
    Newton-Raphson refinement of parameterization.

    Finds better parameter values that minimize the distance to the curve.
    """
    new_params = np.zeros_like(params)

    for i, (point, t) in enumerate(zip(points, params)):
        # Newton-Raphson iteration
        for _ in range(3):  # Few iterations usually sufficient
            p = curve.evaluate(t)
            d = curve.evaluate_derivative(t)

            # Numerator: (p - point) . d
            numerator = (p[0] - point[0]) * d[0] + (p[1] - point[1]) * d[1]

            # Denominator: |d|^2 + (p - point) . d'
            d_len_sq = d[0] ** 2 + d[1] ** 2
            denominator = d_len_sq

            if abs(denominator) > 1e-12:
                t = t - numerator / denominator
                t = max(0.0, min(1.0, t))

        new_params[i] = t

    return new_params


def fit_bezier_to_points(
    points: Union[List[Tuple[float, float]], np.ndarray],
    max_error: float = 2.0,
    left_tangent: Optional[Tuple[float, float]] = None,
    right_tangent: Optional[Tuple[float, float]] = None,
    config: Optional[BezierFitConfig] = None,
) -> List[BezierCurve]:
    """
    Fit cubic Bezier curves to a sequence of points using Schneider's algorithm.

    This is the main entry point for Bezier fitting. It automatically splits
    the curve at corners and ensures G1 continuity.

    Args:
        points: List or array of (x, y) points
        max_error: Maximum allowed fitting error in pixels
        left_tangent: Optional fixed tangent at start
        right_tangent: Optional fixed tangent at end
        config: Optional configuration

    Returns:
        List of BezierCurve objects that approximate the input points

    Example:
        >>> points = [(0, 0), (10, 5), (20, 0), (30, -5), (40, 0)]
        >>> curves = fit_bezier_to_points(points, max_error=1.0)
        >>> for curve in curves:
        ...     print(f"Curve: {curve.control_points}")
    """
    if config is None:
        config = BezierFitConfig(max_error=max_error)
    else:
        config.max_error = max_error

    # Convert to numpy array
    if isinstance(points, list):
        points = np.array(points, dtype=np.float64)
    else:
        points = np.asarray(points, dtype=np.float64)

    n = len(points)
    if n < 2:
        return []

    if n == 2:
        # Line segment - create a degenerate Bezier
        p0 = (float(points[0][0]), float(points[0][1]))
        p1 = (float(points[1][0]), float(points[1][1]))
        mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
        return [BezierCurve(control_points=[p0, mid, mid, p1], fit_error=0.0)]

    # Compute tangents if not provided
    if left_tangent is None:
        left_tangent = _compute_left_tangent(points)
    if right_tangent is None:
        right_tangent = _compute_right_tangent(points)

    # Initial parameterization
    params = _compute_chord_length_params(points)

    # Iterative fitting with reparameterization
    for iteration in range(config.max_iterations):
        curve = _generate_bezier(points, params, left_tangent, right_tangent)
        max_err, split_idx = _compute_max_error(curve, points, params)

        if max_err <= max_error:
            curve.fit_error = max_err
            return [curve]

        # Try reparameterization
        if iteration < config.max_iterations - 1:
            params = _reparameterize(curve, points, params)
            new_curve = _generate_bezier(points, params, left_tangent, right_tangent)
            new_err, _ = _compute_max_error(new_curve, points, params)

            if new_err <= max_error:
                new_curve.fit_error = new_err
                return [new_curve]

    # Subdivision required
    if split_idx <= 1:
        split_idx = 2
    elif split_idx >= n - 2:
        split_idx = n - 3

    # Split and recurse
    center_tangent = _compute_center_tangent(points, split_idx)
    neg_tangent = (-center_tangent[0], -center_tangent[1])

    left_curves = fit_bezier_to_points(
        points[:split_idx + 1],
        max_error=max_error,
        left_tangent=left_tangent,
        right_tangent=center_tangent,
        config=config,
    )

    right_curves = fit_bezier_to_points(
        points[split_idx:],
        max_error=max_error,
        left_tangent=neg_tangent,
        right_tangent=right_tangent,
        config=config,
    )

    return left_curves + right_curves


def detect_corners(
    points: np.ndarray,
    threshold: float = 0.5,
) -> List[int]:
    """
    Detect corner points in a point sequence based on curvature.

    Args:
        points: Nx2 array of points
        threshold: Curvature threshold in radians

    Returns:
        List of indices where corners are detected
    """
    n = len(points)
    if n < 3:
        return []

    corners = []

    for i in range(1, n - 1):
        # Vectors to neighboring points
        v1 = points[i] - points[i - 1]
        v2 = points[i + 1] - points[i]

        # Compute angle between vectors
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)

        if len1 > 0 and len2 > 0:
            cos_angle = np.dot(v1, v2) / (len1 * len2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = math.acos(cos_angle)

            # Large angle deviation = corner
            if abs(math.pi - angle) > threshold:
                corners.append(i)

    return corners


def detect_curves(
    binary_image: np.ndarray,
    min_curvature: float = 0.1,
    min_length: int = 10,
) -> List[CurveSegment]:
    """
    Detect curved segments in a binary image.

    This extracts contours and identifies segments that have significant
    curvature, suitable for Bezier fitting.

    Args:
        binary_image: Binary image (0 = background, 255 = foreground)
        min_curvature: Minimum average curvature to consider a segment curved
        min_length: Minimum segment length in pixels

    Returns:
        List of CurveSegment objects
    """
    import cv2

    # Find contours
    contours, _ = cv2.findContours(
        binary_image.astype(np.uint8),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )

    segments = []

    for contour in contours:
        if len(contour) < min_length:
            continue

        # Reshape to Nx2
        points = contour.reshape(-1, 2).astype(np.float64)

        # Check if closed
        is_closed = np.linalg.norm(points[0] - points[-1]) < 5

        # Compute curvature along the contour
        avg_curvature = _compute_average_curvature(points)

        # Only include if curved enough (exclude straight lines)
        if avg_curvature >= min_curvature:
            segments.append(CurveSegment(
                points=points,
                is_closed=is_closed,
            ))

    return segments


def _compute_average_curvature(points: np.ndarray) -> float:
    """Compute average curvature of a point sequence."""
    n = len(points)
    if n < 3:
        return 0.0

    curvatures = []

    for i in range(1, n - 1):
        # Use three consecutive points to estimate curvature
        p1 = points[i - 1]
        p2 = points[i]
        p3 = points[i + 1]

        # Vectors
        v1 = p2 - p1
        v2 = p3 - p2

        # Cross product (z-component of 3D cross product)
        cross = v1[0] * v2[1] - v1[1] * v2[0]

        # Lengths
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        len3 = np.linalg.norm(p3 - p1)

        if len1 > 0 and len2 > 0 and len3 > 0:
            # Curvature approximation
            curvature = 2 * abs(cross) / (len1 * len2 * len3)
            curvatures.append(curvature)

    return np.mean(curvatures) if curvatures else 0.0


def curves_to_spline_data(
    curves: List[BezierCurve],
) -> dict:
    """
    Convert Bezier curves to data suitable for AutoCAD SPLINE creation.

    Args:
        curves: List of fitted BezierCurve objects

    Returns:
        Dictionary with spline data for AutoCAD:
        - fit_points: Points the spline passes through
        - control_points: Bezier control points
        - degree: Spline degree (3 for cubic)
    """
    if not curves:
        return {"fit_points": [], "control_points": [], "degree": 3}

    # Collect all control points
    control_points = []
    for curve in curves:
        if not control_points:
            control_points.extend(curve.control_points)
        else:
            # Skip first point (same as previous end)
            control_points.extend(curve.control_points[1:])

    # Generate fit points by sampling
    fit_points = []
    for curve in curves:
        for t in np.linspace(0, 1, 10):
            fit_points.append(curve.evaluate(t))

    return {
        "fit_points": fit_points,
        "control_points": control_points,
        "degree": 3,
    }
