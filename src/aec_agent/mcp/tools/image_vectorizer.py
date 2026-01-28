"""
Python-side image vectorization using OpenCV.

Detects geometric features (lines, circles, arcs, ellipses, contours/polylines)
from bitonal TIFF images and returns them as coordinate lists ready to be sent
to AutoCAD via draw_line / draw_polyline / draw_circle / draw_arc / draw_ellipse
sidecar commands.

This replaces AutoCAD Raster Design VTools (vline, vpline, varc, vcircle)
which are interactive and cannot be automated via SendStringToExecute.

Coordinate conversion:
    pixel (x, y) → drawing units: x_dwg = px_x * scale, y_dwg = (height - px_y) * scale
    The Y-axis is flipped because images have origin at top-left,
    AutoCAD has origin at bottom-left.

    The ``scale`` parameter must match the scale used when attaching the raster
    image in AutoCAD.  With ``scale=1.0`` (default), 1 pixel = 1 drawing unit,
    which corresponds to ``raster_attach_image`` at scale 1.0.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DetectedLine:
    """A line segment detected from raster data."""
    start: Tuple[float, float]
    end: Tuple[float, float]


@dataclass
class DetectedCircle:
    """A circle detected from raster data."""
    center: Tuple[float, float]
    radius: float


@dataclass
class DetectedArc:
    """A circular arc detected from raster data."""
    center: Tuple[float, float]
    radius: float
    start_angle: float  # degrees, 0 = +X axis, CCW positive
    end_angle: float    # degrees


@dataclass
class DetectedEllipse:
    """An ellipse detected from raster data."""
    center: Tuple[float, float]
    major_axis_endpoint: Tuple[float, float]  # endpoint of major axis relative to center
    axis_ratio: float  # minor/major ratio (0..1]
    start_angle: float  # degrees, 0 = full ellipse
    end_angle: float    # degrees, 360 = full ellipse


@dataclass
class DetectedPolyline:
    """A polyline (contour) detected from raster data."""
    points: List[Tuple[float, float]]
    closed: bool = False
    bulges: List[float] = field(default_factory=list)  # bulge per vertex (0=straight, nonzero=arc)


@dataclass
class VectorizationResult:
    """Results from image vectorization."""
    lines: List[DetectedLine] = field(default_factory=list)
    circles: List[DetectedCircle] = field(default_factory=list)
    arcs: List[DetectedArc] = field(default_factory=list)
    ellipses: List[DetectedEllipse] = field(default_factory=list)
    polylines: List[DetectedPolyline] = field(default_factory=list)
    image_width_px: int = 0
    image_height_px: int = 0
    dpi: int = 300


def vectorize_bitonal_image(
    image_path: str,
    dpi: int = 300,
    scale: float = 1.0,
    min_line_length: int = 100,
    max_line_gap: int = 10,
    hough_threshold: int = 150,
    min_circle_radius: int = 20,
    max_circle_radius: int = 500,
    contour_epsilon_factor: float = 0.01,
    min_contour_points: int = 5,
    min_contour_area: float = 2000.0,
    ellipse_fit_threshold: float = 0.85,
    arc_coverage_min: float = 30.0,
    arc_coverage_max: float = 350.0,
    line_merge_angle_tol: float = 5.0,
    line_merge_dist_tol: float = 15.0,
    circle_merge_center_tol: float = 30.0,
    circle_merge_radius_tol: float = 20.0,
) -> VectorizationResult:
    """
    Detect geometric features from a bitonal image using OpenCV.

    The image is pre-processed with morphological operations to remove noise
    from gradient fills, text, and scan artifacts before detection.  After
    detection, near-duplicate lines and circles are merged.

    Args:
        image_path: Path to the bitonal TIFF image.
        dpi: Image DPI (used for metadata, not coordinate conversion).
        scale: Coordinate scale factor. Pixel coords are multiplied by this value.
               Must match the scale used for raster_attach_image in AutoCAD.
               Default 1.0 means 1 pixel = 1 drawing unit.
        min_line_length: Minimum line length in pixels for HoughLinesP (default 100).
        max_line_gap: Maximum gap between line segments to merge (default 10).
        hough_threshold: Accumulator threshold for HoughLinesP (default 150).
        min_circle_radius: Minimum circle radius in pixels (default 20).
        max_circle_radius: Maximum circle radius in pixels, 0=unlimited (default 500).
        contour_epsilon_factor: Polyline approximation tolerance as fraction
                                of contour perimeter (default 0.01).
        min_contour_points: Minimum points for a contour to be kept (default 5).
        min_contour_area: Minimum contour area in pixels to filter noise (default 2000).
        ellipse_fit_threshold: Goodness-of-fit threshold (0-1) for ellipse/arc
                               detection. Higher = stricter matching (default 0.85).
        arc_coverage_min: Minimum arc coverage in degrees to accept as arc (default 30).
        arc_coverage_max: Maximum arc coverage in degrees before treating as
                          full ellipse/circle (default 350).
        line_merge_angle_tol: Max angle difference (degrees) to merge two lines (default 5).
        line_merge_dist_tol: Max perpendicular distance (pixels) to merge lines (default 15).
        circle_merge_center_tol: Max center distance (pixels) to merge circles (default 30).
        circle_merge_radius_tol: Max radius difference (pixels) to merge circles (default 20).

    Returns:
        VectorizationResult with detected lines, circles, arcs, ellipses,
        and polylines in drawing-unit coordinates.

    Raises:
        FileNotFoundError: If the image file does not exist.
        RuntimeError: If vectorization fails.
    """
    import cv2
    import numpy as np
    import math

    image_path = os.path.abspath(image_path)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    logger.info("Starting image vectorization", image_path=image_path, dpi=dpi, scale=scale)

    try:
        # Load image as grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to load image: {image_path}")

        height, width = img.shape[:2]
        result = VectorizationResult(
            image_width_px=width,
            image_height_px=height,
            dpi=dpi,
        )

        # =================================================================
        # PRE-PROCESSING: Clean the image to isolate line work from fills,
        # text, gradients, and scan noise.
        # =================================================================

        # 1. Adaptive threshold handles uneven lighting and gradients better
        #    than a simple global threshold.  It binarizes based on local
        #    pixel neighbourhood, so gradient fills (which change slowly)
        #    become white while thin dark lines survive.
        binary = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=25, C=10,
        )

        # 2. Morphological close: fill tiny gaps in lines so they connect
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        # 3. Morphological open: remove small blobs (text characters, speckles,
        #    gradient dithering artifacts).  A 3x3 open removes features
        #    thinner than ~3 px while keeping real geometry lines.
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=2)

        # 4. Remove small connected components (text, dots, annotations).
        #    This is the most effective filter for removing dimension text
        #    and arrowheads while preserving large geometry.
        min_component_area = max(min_contour_area, width * height * 0.001)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for lbl in range(1, n_labels):
            if stats[lbl, cv2.CC_STAT_AREA] < min_component_area:
                binary[labels == lbl] = 0

        logger.info(
            "Pre-processing complete",
            components_kept=sum(
                1 for lbl in range(1, n_labels)
                if stats[lbl, cv2.CC_STAT_AREA] >= min_component_area
            ),
            components_removed=sum(
                1 for lbl in range(1, n_labels)
                if stats[lbl, cv2.CC_STAT_AREA] < min_component_area
            ),
        )

        # Helper: convert pixel coords to drawing units.
        def px_to_dwg(px_x: float, px_y: float) -> Tuple[float, float]:
            return (px_x * scale, (height - px_y) * scale)

        def px_dist_to_dwg(px_dist: float) -> float:
            return px_dist * scale

        # =================================================================
        # LINE DETECTION (HoughLinesP) + Deduplication
        # =================================================================
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        raw_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=hough_threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )

        # Deduplicate lines: merge nearly-parallel, closely-spaced segments
        deduped_lines = _deduplicate_lines(
            raw_lines, line_merge_angle_tol, line_merge_dist_tol,
        )
        for (x1, y1, x2, y2) in deduped_lines:
            start = px_to_dwg(float(x1), float(y1))
            end = px_to_dwg(float(x2), float(y2))
            result.lines.append(DetectedLine(start=start, end=end))

        logger.info(
            f"Detected {len(result.lines)} lines "
            f"(raw: {len(raw_lines) if raw_lines is not None else 0}, "
            f"after dedup: {len(deduped_lines)})"
        )

        # =================================================================
        # CIRCLE DETECTION (HoughCircles) + Deduplication
        # =================================================================
        blurred = cv2.medianBlur(img, 7)
        raw_circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.5,
            minDist=max(50, min_circle_radius * 3),
            param1=120,
            param2=60,
            minRadius=min_circle_radius,
            maxRadius=max_circle_radius if max_circle_radius > 0 else 0,
        )

        deduped_circles = _deduplicate_circles(
            raw_circles, circle_merge_center_tol, circle_merge_radius_tol,
        )
        for (cx, cy, r) in deduped_circles:
            center = px_to_dwg(float(cx), float(cy))
            radius = px_dist_to_dwg(float(r))
            result.circles.append(DetectedCircle(center=center, radius=radius))

        logger.info(
            f"Detected {len(result.circles)} circles "
            f"(raw: {len(raw_circles[0]) if raw_circles is not None else 0}, "
            f"after dedup: {len(deduped_circles)})"
        )

        # =================================================================
        # CONTOUR DETECTION → arcs, ellipses, polylines
        # =================================================================
        # Use RETR_EXTERNAL to get only outermost contours (skip nested
        # contours from gradient fill boundaries, hatches, etc.)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        # Track detected circle centers to avoid double-detection
        circle_centers_px: List[Tuple[float, float, float]] = []
        for (cx, cy, r) in deduped_circles:
            circle_centers_px.append((cx, cy, r))

        def _is_near_detected_circle(cx: float, cy: float, tol: float = 30.0) -> bool:
            for (px, py, _r) in circle_centers_px:
                if abs(cx - px) < tol and abs(cy - py) < tol:
                    return True
            return False

        def _ellipse_fit_error(contour, ellipse) -> float:
            (cx, cy), (ma, MA), angle = ellipse
            if ma < 1 or MA < 1:
                return float('inf')
            a = MA / 2.0
            b = ma / 2.0
            angle_rad = math.radians(angle)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            total_err = 0.0
            for pt in contour:
                dx = float(pt[0][0]) - cx
                dy = float(pt[0][1]) - cy
                lx = dx * cos_a + dy * sin_a
                ly = -dx * sin_a + dy * cos_a
                if a > 0 and b > 0:
                    d = math.sqrt((lx / a) ** 2 + (ly / b) ** 2)
                    total_err += abs(d - 1.0)
                else:
                    total_err += 1.0
            return total_err / max(len(contour), 1)

        def _contour_angular_coverage(contour, cx: float, cy: float) -> float:
            angles = []
            for pt in contour:
                dx = float(pt[0][0]) - cx
                dy = float(pt[0][1]) - cy
                angles.append(math.degrees(math.atan2(dy, dx)) % 360)
            if not angles:
                return 0.0
            angles.sort()
            max_gap = 0.0
            for i in range(len(angles)):
                gap = angles[(i + 1) % len(angles)] - angles[i]
                if gap < 0:
                    gap += 360
                max_gap = max(max_gap, gap)
            return 360.0 - max_gap

        def _compute_arc_angles(contour, cx: float, cy: float) -> Tuple[float, float]:
            angles = []
            for pt in contour:
                dx = float(pt[0][0]) - cx
                dy = float(pt[0][1]) - cy
                angles.append(math.degrees(math.atan2(dy, dx)) % 360)
            if not angles:
                return (0.0, 0.0)
            angles.sort()
            max_gap = 0.0
            max_gap_idx = 0
            for i in range(len(angles)):
                gap = angles[(i + 1) % len(angles)] - angles[i]
                if gap < 0:
                    gap += 360
                if gap > max_gap:
                    max_gap = gap
                    max_gap_idx = i
            start_angle = angles[(max_gap_idx + 1) % len(angles)]
            end_angle = angles[max_gap_idx]
            return (start_angle, end_angle)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_contour_area:
                continue

            n_pts = len(contour)

            # Try ellipse/arc fitting if we have enough points
            if n_pts >= 5:
                ellipse = cv2.fitEllipse(contour)
                (cx, cy), (minor_axis, major_axis), angle = ellipse

                if _is_near_detected_circle(cx, cy):
                    continue

                fit_error = _ellipse_fit_error(contour, ellipse)
                good_fit = fit_error < (1.0 - ellipse_fit_threshold)

                if good_fit:
                    coverage = _contour_angular_coverage(contour, cx, cy)
                    is_circular = abs(major_axis - minor_axis) / max(major_axis, 1) < 0.15

                    if coverage >= arc_coverage_max:
                        if is_circular:
                            continue  # Already detected by HoughCircles
                        else:
                            semi_major = (major_axis / 2.0)
                            angle_rad = math.radians(angle)
                            maj_dx = semi_major * math.cos(angle_rad)
                            maj_dy = semi_major * math.sin(angle_rad)
                            center_dwg = px_to_dwg(cx, cy)
                            maj_end_dwg = (
                                px_dist_to_dwg(maj_dx),
                                -px_dist_to_dwg(maj_dy),
                            )
                            ratio = minor_axis / max(major_axis, 1e-6)
                            result.ellipses.append(DetectedEllipse(
                                center=center_dwg,
                                major_axis_endpoint=maj_end_dwg,
                                axis_ratio=min(ratio, 1.0),
                                start_angle=0.0,
                                end_angle=360.0,
                            ))
                            continue

                    elif coverage >= arc_coverage_min:
                        start_deg, end_deg = _compute_arc_angles(contour, cx, cy)
                        start_acad = (360.0 - end_deg) % 360.0
                        end_acad = (360.0 - start_deg) % 360.0

                        if is_circular:
                            center_dwg = px_to_dwg(cx, cy)
                            radius_dwg = px_dist_to_dwg(major_axis / 2.0)
                            result.arcs.append(DetectedArc(
                                center=center_dwg,
                                radius=radius_dwg,
                                start_angle=start_acad,
                                end_angle=end_acad,
                            ))
                        else:
                            semi_major = (major_axis / 2.0)
                            angle_rad = math.radians(angle)
                            maj_dx = semi_major * math.cos(angle_rad)
                            maj_dy = semi_major * math.sin(angle_rad)
                            center_dwg = px_to_dwg(cx, cy)
                            maj_end_dwg = (
                                px_dist_to_dwg(maj_dx),
                                -px_dist_to_dwg(maj_dy),
                            )
                            ratio = minor_axis / max(major_axis, 1e-6)
                            result.ellipses.append(DetectedEllipse(
                                center=center_dwg,
                                major_axis_endpoint=maj_end_dwg,
                                axis_ratio=min(ratio, 1.0),
                                start_angle=start_acad,
                                end_angle=end_acad,
                            ))
                        continue

            # --- Fallback: polyline / contour ---
            perimeter = cv2.arcLength(contour, True)
            epsilon = contour_epsilon_factor * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) < min_contour_points:
                continue

            points = []
            for pt in approx:
                dwg_pt = px_to_dwg(float(pt[0][0]), float(pt[0][1]))
                points.append(dwg_pt)

            is_closed = cv2.isContourConvex(approx) or (
                len(approx) > 2 and
                np.linalg.norm(approx[0][0] - approx[-1][0]) < max_line_gap
            )

            bulges = _compute_bulges(contour, approx, is_closed)

            result.polylines.append(DetectedPolyline(
                points=points,
                closed=is_closed,
                bulges=bulges,
            ))

        logger.info(f"Detected {len(result.arcs)} arcs")
        logger.info(f"Detected {len(result.ellipses)} ellipses")
        logger.info(f"Detected {len(result.polylines)} polylines/contours")

        total = (
            len(result.lines) + len(result.circles) + len(result.arcs)
            + len(result.ellipses) + len(result.polylines)
        )
        logger.info(
            "Vectorization complete",
            total_features=total,
            lines=len(result.lines),
            circles=len(result.circles),
            arcs=len(result.arcs),
            ellipses=len(result.ellipses),
            polylines=len(result.polylines),
        )

        return result

    except (FileNotFoundError, RuntimeError):
        raise
    except Exception as e:
        raise RuntimeError(f"Image vectorization failed: {e}") from e


# =========================================================================
# Deduplication helpers
# =========================================================================

def _deduplicate_lines(
    raw_lines,
    angle_tol: float = 5.0,
    dist_tol: float = 15.0,
) -> List[Tuple[float, float, float, float]]:
    """
    Merge near-duplicate lines detected by HoughLinesP.

    Two lines are considered duplicates if:
    - Their angles differ by less than ``angle_tol`` degrees, AND
    - Their midpoints are within ``dist_tol`` pixels perpendicular distance.

    When duplicates are found, the longest line is kept.
    """
    import numpy as np
    import math

    if raw_lines is None or len(raw_lines) == 0:
        return []

    # Extract line data: (x1, y1, x2, y2, angle, length, midpoint)
    lines_data = []
    for line in raw_lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180  # 0-180
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        lines_data.append((x1, y1, x2, y2, angle, length, mx, my))

    # Sort by length descending — keep longer lines first
    lines_data.sort(key=lambda x: -x[5])

    kept = []
    used = [False] * len(lines_data)

    for i, (x1, y1, x2, y2, ang, length, mx, my) in enumerate(lines_data):
        if used[i]:
            continue
        kept.append((x1, y1, x2, y2))
        used[i] = True

        # Mark near-duplicates as used
        for j in range(i + 1, len(lines_data)):
            if used[j]:
                continue
            _, _, _, _, ang_j, _, mx_j, my_j = lines_data[j]

            # Check angle similarity
            angle_diff = abs(ang - ang_j)
            if angle_diff > 90:
                angle_diff = 180 - angle_diff
            if angle_diff > angle_tol:
                continue

            # Check perpendicular distance between midpoints
            dx = mx_j - mx
            dy = my_j - my
            # Perpendicular distance from midpoint j to line i
            line_dir = np.array([x2 - x1, y2 - y1], dtype=float)
            line_len = np.linalg.norm(line_dir)
            if line_len < 1:
                continue
            line_unit = line_dir / line_len
            perp = np.array([-line_unit[1], line_unit[0]])
            perp_dist = abs(dx * perp[0] + dy * perp[1])

            if perp_dist < dist_tol:
                used[j] = True  # Mark as duplicate

    return kept


def _deduplicate_circles(
    raw_circles,
    center_tol: float = 30.0,
    radius_tol: float = 20.0,
) -> List[Tuple[float, float, float]]:
    """
    Merge near-duplicate circles detected by HoughCircles.

    Two circles are duplicates if their centers are within ``center_tol``
    pixels and their radii differ by less than ``radius_tol`` pixels.
    When duplicates are found, the one with higher accumulator support
    (first in the list) is kept.
    """
    import math

    if raw_circles is None:
        return []

    circles_list = []
    for c in raw_circles[0]:
        circles_list.append((float(c[0]), float(c[1]), float(c[2])))

    kept = []
    used = [False] * len(circles_list)

    for i, (cx, cy, r) in enumerate(circles_list):
        if used[i]:
            continue
        kept.append((cx, cy, r))
        used[i] = True

        for j in range(i + 1, len(circles_list)):
            if used[j]:
                continue
            cx_j, cy_j, r_j = circles_list[j]

            center_dist = math.sqrt((cx - cx_j) ** 2 + (cy - cy_j) ** 2)
            if center_dist < center_tol and abs(r - r_j) < radius_tol:
                used[j] = True

    return kept


def _compute_bulges(
    original_contour,
    simplified: "np.ndarray",
    is_closed: bool,
) -> List[float]:
    """
    Compute bulge values for a simplified polyline by comparing to the
    original contour.  A bulge of 0 means a straight segment; non-zero
    means an arc segment (bulge = tan(included_angle / 4)).

    For each segment of the simplified polyline, we find the corresponding
    points on the original contour and measure the maximum deviation.  If
    the deviation is significant relative to the segment length, we
    compute a bulge value.
    """
    import numpy as np
    import math

    n = len(simplified)
    bulges: List[float] = [0.0] * n
    if n < 2:
        return bulges

    # Flatten original contour to Nx2
    orig_pts = original_contour.reshape(-1, 2).astype(float)

    def _find_nearest_idx(arr: "np.ndarray", point: "np.ndarray") -> int:
        dists = np.sum((arr - point) ** 2, axis=1)
        return int(np.argmin(dists))

    seg_count = n if is_closed else n - 1
    for i in range(seg_count):
        j = (i + 1) % n
        p1 = simplified[i][0].astype(float)
        p2 = simplified[j][0].astype(float)

        seg_vec = p2 - p1
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 1.0:
            continue

        # Find the original-contour indices closest to p1 and p2
        idx1 = _find_nearest_idx(orig_pts, p1)
        idx2 = _find_nearest_idx(orig_pts, p2)

        # Extract the sub-contour between these indices
        if idx2 >= idx1:
            sub = orig_pts[idx1:idx2 + 1]
        else:
            sub = np.vstack([orig_pts[idx1:], orig_pts[:idx2 + 1]])

        if len(sub) < 3:
            continue

        # Compute perpendicular distance of each sub-point from the segment line
        seg_unit = seg_vec / seg_len
        perp = np.array([-seg_unit[1], seg_unit[0]])

        deviations = (sub - p1) @ perp
        max_dev = np.max(np.abs(deviations))

        # Sagitta-based bulge: if max deviation is significant relative to chord
        sagitta_ratio = max_dev / seg_len
        if sagitta_ratio > 0.02:  # at least 2% deviation → curved
            # Determine sign: positive = CCW bulge
            mid_dev = np.median(deviations)
            sign = 1.0 if mid_dev >= 0 else -1.0

            # Approximate: sagitta s = max_dev, chord c = seg_len
            # bulge = tan(theta/4) where theta = 4 * atan(2*s/c)
            theta = 4.0 * math.atan2(2.0 * max_dev, seg_len)
            bulge_val = math.tan(theta / 4.0)
            bulges[i] = sign * abs(bulge_val)

    return bulges
