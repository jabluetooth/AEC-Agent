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
    min_line_length: int = 80,
    max_line_gap: int = 10,
    hough_threshold: int = 150,
    min_circle_radius: int = 20,
    max_circle_radius: int = 500,
    hough_circles_dp: float = 1.2,
    hough_circles_param1: float = 200.0,
    hough_circles_param2: float = 100.0,
    hough_circles_min_dist: int = 100,
    contour_epsilon_factor: float = 0.01,
    min_contour_points: int = 5,
    min_contour_area: float = 3000.0,
    ellipse_fit_threshold: float = 0.85,
    arc_coverage_min: float = 30.0,
    arc_coverage_max: float = 350.0,
    line_merge_angle_tol: float = 5.0,
    line_merge_dist_tol: float = 15.0,
    circle_merge_center_tol: float = 30.0,
    circle_merge_radius_tol: float = 20.0,
    # --- Signal Restoration parameters ---
    signal_restore: bool = True,
    signal_close_kernel_length: int = 15,
    signal_close_angle_step: int = 15,
    signal_close_iterations: int = 1,
    signal_smooth_ksize: int = 3,
    signal_adaptive_block: int = 15,
    signal_adaptive_c: int = 2,
    # --- Iterative Masking parameters ---
    mask_detected_circles: bool = True,
    mask_detected_lines: bool = False,
    mask_thickness: int = 5,
    # --- Skeletonization & Topology parameters ---
    skeletonize: bool = False,
    topology_cleanup: bool = False,
    snap_tolerance: float = 5.0,
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
        min_line_length: Minimum line length in pixels for HoughLinesP (default 80).
        max_line_gap: Maximum gap between line segments to merge (default 10).
        hough_threshold: Accumulator threshold for HoughLinesP (default 150).
        min_circle_radius: Minimum circle radius in pixels (default 20).
        max_circle_radius: Maximum circle radius in pixels, 0=unlimited (default 500).
        hough_circles_dp: Inverse ratio of accumulator resolution to image
                          resolution. Lower = finer detection (default 1.2).
        hough_circles_param1: Canny high threshold used internally by
                              HoughCircles. Higher = fewer edges = fewer false
                              circles (default 200).
        hough_circles_param2: Accumulator threshold for circle centers.
                              Higher = fewer but more confident circles.
                              This is the most important param for reducing
                              false positives (default 100).
        hough_circles_min_dist: Minimum distance in pixels between detected
                                circle centers (default 100).
        contour_epsilon_factor: Polyline approximation tolerance as fraction
                                of contour perimeter (default 0.01).
        min_contour_points: Minimum points for a contour to be kept (default 5).
        min_contour_area: Minimum contour area in pixels to filter noise (default 3000).
        ellipse_fit_threshold: Goodness-of-fit threshold (0-1) for ellipse/arc
                               detection. Higher = stricter matching (default 0.85).
        arc_coverage_min: Minimum arc coverage in degrees to accept as arc (default 30).
        arc_coverage_max: Maximum arc coverage in degrees before treating as
                          full ellipse/circle (default 350).
        line_merge_angle_tol: Max angle difference (degrees) to merge two lines (default 5).
        line_merge_dist_tol: Max perpendicular distance (pixels) to merge lines (default 15).
        circle_merge_center_tol: Max center distance (pixels) to merge circles (default 30).
        circle_merge_radius_tol: Max radius difference (pixels) to merge circles (default 20).
        signal_restore: Enable Signal Restoration phase to bridge gaps in
                        dashed/broken lines before detection (default True).
        signal_close_kernel_length: Length in pixels of the directional line
                                    kernel used for morphological closing.
                                    Larger values bridge wider gaps but risk
                                    merging nearby parallel lines (default 15).
        signal_close_angle_step: Angular step in degrees between directional
                                 closing passes.  Smaller = more directions
                                 tested = better isotropy but slower (default 15).
        signal_close_iterations: Morphological close iterations per direction.
                                 More iterations = more aggressive bridging
                                 (default 1).
        signal_smooth_ksize: Gaussian blur kernel size for post-close smoothing.
                             Must be odd.  Removes jagged staircase edges left
                             by the directional close (default 3).
        signal_adaptive_block: Block size for the adaptive threshold that
                               re-binarizes after smoothing (default 15).
        signal_adaptive_c: Constant subtracted from the adaptive threshold
                           mean (default 2).
        mask_detected_circles: After detecting circles, erase their pixels
                               from the binary image so the contour pass does
                               not re-trace them as polylines (default True).
        mask_detected_lines: After detecting lines, erase their pixels from
                             the binary image.  Off by default — can remove
                             too much on dense drawings (default False).
        mask_thickness: Pixel thickness of the mask painted over detected
                        features.  Larger values erase more aggressively
                        (default 5).
        skeletonize: Run morphological skeletonization (scikit-image) after
                     signal restoration to reduce thick lines to 1px
                     centerlines before Hough detection (default False).
                     Requires ``scikit-image`` to be installed.
        topology_cleanup: After all detection, build a NetworkX graph from
                          detected lines/polylines, merge degree-2 nodes
                          (artificial breaks), and snap dangling endpoints
                          (default False).  Requires ``networkx``.
        snap_tolerance: Maximum distance in drawing units to snap dangling
                        endpoints during topology cleanup (default 5.0).

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
        # PRE-PROCESSING: Recover design intent from scanned geometry.
        #
        # A scanned drawing is a degraded copy of precise geometry.
        # The goal is to isolate the ink/line work so that Hough
        # transforms can recover the INTENDED lines, circles, and arcs
        # — not reproduce scan artifacts.  Every step must preserve
        # the structural geometry (≥2 px wide at 300 DPI).
        # =================================================================

        # 0. Auto-detect image polarity.
        #    Scanned blueprints are dark-on-light (ink on paper).
        #    CAD screenshots and inverted scans are light-on-dark.
        #    The adaptive threshold with BINARY_INV expects dark-on-light.
        #    If the image is predominantly dark, invert it first.
        mean_val = float(np.mean(img))
        if mean_val < 128:
            img = cv2.bitwise_not(img)
            logger.info("Auto-inverted light-on-dark image", mean_value=mean_val)

        # 1. Gaussian blur: smooth scan grain and pixel noise BEFORE
        #    thresholding.  A 5x5 kernel at σ=0 (auto) removes
        #    high-frequency scan artifacts without blurring real geometry.
        blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

        # 2. Adaptive threshold: binarize based on local pixel
        #    neighbourhood.  blockSize=51 tolerates uneven scan lighting.
        #    C=12 balances noise rejection vs line preservation — lower
        #    than C=15 to keep faint scan lines that represent real
        #    geometry.
        binary = cv2.adaptiveThreshold(
            blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=51, C=12,
        )

        # 3. Morphological close: bridge tiny gaps in lines (1-2 px)
        #    caused by scan artifacts or threshold edge effects.
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        # 4. Morphological open: remove single-pixel noise (speckles,
        #    scan dithering).  ONE iteration only — two iterations
        #    destroy 2px-wide ink lines which are common at 300 DPI.
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)

        # =================================================================
        # SIGNAL RESTORATION: Bridge gaps in dashed / broken lines.
        #
        # Standard vectorizers trace exactly what they see: gaps in the
        # raster pixels produce dashed output even when the design intent
        # is a continuous line.  This phase applies directional
        # morphological closing at multiple orientations to bridge those
        # gaps, then re-smooths and re-binarizes so downstream Hough
        # transforms see continuous geometry.
        # =================================================================
        if signal_restore:
            # 5a. Directional morphological close.
            #     A single isotropic (square) kernel cannot reliably bridge
            #     gaps in lines at arbitrary angles without also merging
            #     nearby parallel features.  Instead, we sweep a thin LINE
            #     kernel across multiple orientations (0°, 15°, 30°, …, 165°)
            #     and OR the results.  Each kernel bridges gaps only along
            #     its direction, preserving perpendicular separation.
            restored = np.zeros_like(binary)
            for angle_deg in range(0, 180, signal_close_angle_step):
                # Build a rotated line kernel of the requested length
                k_len = signal_close_kernel_length
                kern = np.zeros((k_len, k_len), dtype=np.uint8)
                center = k_len // 2
                angle_rad = math.radians(angle_deg)
                dx = math.cos(angle_rad)
                dy = math.sin(angle_rad)
                for t in range(-center, center + 1):
                    x = int(round(center + t * dx))
                    y = int(round(center + t * dy))
                    if 0 <= x < k_len and 0 <= y < k_len:
                        kern[y, x] = 1

                closed_dir = cv2.morphologyEx(
                    binary, cv2.MORPH_CLOSE, kern,
                    iterations=signal_close_iterations,
                )
                restored = cv2.bitwise_or(restored, closed_dir)

            # 5b. Gaussian blur + adaptive threshold to smooth jagged
            #     staircase edges introduced by the directional close and
            #     to re-binarize the result cleanly.
            sk = signal_smooth_ksize if signal_smooth_ksize % 2 == 1 else signal_smooth_ksize + 1
            smoothed = cv2.GaussianBlur(restored, (sk, sk), 0)
            binary = cv2.adaptiveThreshold(
                smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, blockSize=signal_adaptive_block, C=signal_adaptive_c,
            )

            logger.info(
                "Signal restoration complete",
                kernel_length=signal_close_kernel_length,
                angle_step=signal_close_angle_step,
                directions=180 // signal_close_angle_step,
            )

        # 4. Remove small connected components (text, dots, annotations,
        #    scan artifacts).  Uses BOTH area AND bounding-box extent:
        #    - Small area AND small bbox → noise (dots, speckles) → remove
        #    - Small area BUT large bbox → thin line work → KEEP
        #    This preserves long thin contour lines that have small pixel
        #    area but span a significant distance across the drawing.
        min_component_area = max(min_contour_area, width * height * 0.001)
        min_component_dim = max(50, min(width, height) * 0.03)  # 3% of shorter side
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for lbl in range(1, n_labels):
            comp_area = stats[lbl, cv2.CC_STAT_AREA]
            comp_w = stats[lbl, cv2.CC_STAT_WIDTH]
            comp_h = stats[lbl, cv2.CC_STAT_HEIGHT]
            # Keep if large area OR large bounding box (thin line work)
            if comp_area < min_component_area and max(comp_w, comp_h) < min_component_dim:
                binary[labels == lbl] = 0

        def _component_kept(lbl: int) -> bool:
            return (stats[lbl, cv2.CC_STAT_AREA] >= min_component_area or
                    max(stats[lbl, cv2.CC_STAT_WIDTH],
                        stats[lbl, cv2.CC_STAT_HEIGHT]) >= min_component_dim)

        logger.info(
            "Pre-processing complete",
            components_kept=sum(1 for lbl in range(1, n_labels) if _component_kept(lbl)),
            components_removed=sum(1 for lbl in range(1, n_labels) if not _component_kept(lbl)),
        )

        # =================================================================
        # OPTIONAL: Skeletonization (reduce thick lines to 1px centers)
        # =================================================================
        if skeletonize:
            try:
                from skimage.morphology import skeletonize as _skeletonize
                # skimage expects bool array (True = foreground)
                skel = _skeletonize(binary > 0)
                binary = (skel.astype(np.uint8)) * 255
                logger.info("Skeletonization complete (lines reduced to 1px centerlines)")
            except ImportError:
                logger.warning(
                    "scikit-image not installed — skipping skeletonization. "
                    "Install with: pip install scikit-image>=0.21.0"
                )

        # Helper: convert pixel coords to drawing units.
        def px_to_dwg(px_x: float, px_y: float) -> Tuple[float, float]:
            return (px_x * scale, (height - px_y) * scale)

        def px_dist_to_dwg(px_dist: float) -> float:
            return px_dist * scale

        # =================================================================
        # CIRCLE DETECTION (HoughCircles) + Deduplication
        # =================================================================
        # Circles are detected FIRST because they are the most "semantic"
        # primitive and the most prone to fragmentation by contour tracers.
        # After detection, their pixels can be masked out of the binary
        # image so subsequent line and contour passes don't re-trace them.
        circle_input = cv2.medianBlur(blurred_img, 7)
        raw_circles = cv2.HoughCircles(
            circle_input,
            cv2.HOUGH_GRADIENT,
            dp=hough_circles_dp,
            minDist=max(hough_circles_min_dist, min_circle_radius * 4),
            param1=hough_circles_param1,
            param2=hough_circles_param2,
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

        # --- Mask detected circles from binary image ---
        if mask_detected_circles and deduped_circles:
            circles_masked = 0
            for (cx, cy, r) in deduped_circles:
                cv2.circle(
                    binary, (int(round(cx)), int(round(cy))),
                    int(round(r)) + mask_thickness, 0, mask_thickness * 2,
                )
                circles_masked += 1
            logger.info(
                "Masked detected circles from binary",
                circles_masked=circles_masked,
                mask_thickness=mask_thickness,
            )

        # =================================================================
        # LINE DETECTION (HoughLinesP) + Deduplication
        # =================================================================
        # Run on the (potentially circle-masked) binary image so line
        # detection doesn't pick up circle edges as straight segments.
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

        # --- Mask detected lines from binary image ---
        if mask_detected_lines and deduped_lines:
            lines_masked = 0
            for (x1, y1, x2, y2) in deduped_lines:
                cv2.line(
                    binary,
                    (int(round(x1)), int(round(y1))),
                    (int(round(x2)), int(round(y2))),
                    0, mask_thickness * 2,
                )
                lines_masked += 1
            logger.info(
                "Masked detected lines from binary",
                lines_masked=lines_masked,
                mask_thickness=mask_thickness,
            )

        # =================================================================
        # CONTOUR DETECTION → arcs, ellipses, polylines
        # =================================================================
        # Use RETR_CCOMP (two-level hierarchy) to capture both outer
        # contours and inner features (holes, slots).  The hierarchy
        # array lets us distinguish outer vs inner if needed later.
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE,
        )

        # Track detected circle centers to avoid double-detection.
        # Only suppress a contour if its fitted ellipse closely matches an
        # already-detected circle (center within the circle radius AND
        # contour size similar to circle size).  This prevents large outer
        # contours from being killed by small interior circles.
        circle_centers_px: List[Tuple[float, float, float]] = []
        for (cx, cy, r) in deduped_circles:
            circle_centers_px.append((cx, cy, r))

        def _is_duplicate_of_detected_circle(
            cx: float, cy: float, contour_area: float,
        ) -> bool:
            for (px, py, r) in circle_centers_px:
                center_dist = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                circle_area = math.pi * r * r
                # Center must be within half the circle radius AND
                # contour area must be within 2x of the circle area
                if (center_dist < r * 0.5 and
                        circle_area > 0 and
                        0.3 < contour_area / circle_area < 3.0):
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

                if _is_duplicate_of_detected_circle(cx, cy, area):
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

        # =================================================================
        # POST-PROCESSING: Remove lines/circles that overlap with contours
        # =================================================================
        # HoughLinesP and HoughCircles detect features along contour edges,
        # producing redundant geometry (lines on top of polylines, circles
        # on curved contour sections).  Filter them out by checking whether
        # sample points lie close to a detected contour edge.
        #
        # cv2.pointPolygonTest(contour, pt, measureDist=True) returns the
        # signed distance from a point to the nearest contour edge.  If
        # abs(distance) < tolerance, the point sits ON the contour.

        overlap_tol = max(5.0, min_line_length * 0.05)  # pixels

        # Collect significant contours for overlap testing (area > threshold)
        significant_contours = [
            c for c in contours
            if cv2.contourArea(c) >= min_contour_area
        ]

        if significant_contours:
            # --- Filter redundant lines ---
            lines_before = len(result.lines)
            kept_lines: List[DetectedLine] = []
            for line in result.lines:
                # Convert DWG coords back to pixel coords for testing
                sx = line.start[0] / scale if scale else 0
                sy = height - (line.start[1] / scale if scale else 0)
                ex = line.end[0] / scale if scale else 0
                ey = height - (line.end[1] / scale if scale else 0)
                mx, my = (sx + ex) / 2.0, (sy + ey) / 2.0

                on_contour = False
                for contour in significant_contours:
                    d_start = abs(cv2.pointPolygonTest(
                        contour, (float(sx), float(sy)), True,
                    ))
                    d_end = abs(cv2.pointPolygonTest(
                        contour, (float(ex), float(ey)), True,
                    ))
                    d_mid = abs(cv2.pointPolygonTest(
                        contour, (float(mx), float(my)), True,
                    ))
                    # All three sample points near the contour edge → redundant
                    if d_start < overlap_tol and d_end < overlap_tol and d_mid < overlap_tol:
                        on_contour = True
                        break
                if not on_contour:
                    kept_lines.append(line)

            result.lines = kept_lines
            logger.info(
                f"Line overlap cleanup: {lines_before} → {len(result.lines)} "
                f"(removed {lines_before - len(result.lines)} redundant lines)"
            )

            # --- Filter redundant circles ---
            circles_before = len(result.circles)
            kept_circles: List[DetectedCircle] = []
            for circle in result.circles:
                cx_px = circle.center[0] / scale if scale else 0
                cy_px = height - (circle.center[1] / scale if scale else 0)
                r_px = circle.radius / scale if scale else 0

                on_contour = False
                for contour in significant_contours:
                    # Sample 8 points around the circle perimeter
                    perimeter_on_contour = 0
                    for angle_i in range(8):
                        a = angle_i * (2 * math.pi / 8)
                        px = cx_px + r_px * math.cos(a)
                        py = cy_px + r_px * math.sin(a)
                        d = abs(cv2.pointPolygonTest(
                            contour, (float(px), float(py)), True,
                        ))
                        if d < overlap_tol:
                            perimeter_on_contour += 1
                    # If most of the perimeter lies on a contour → redundant
                    if perimeter_on_contour >= 5:
                        on_contour = True
                        break
                if not on_contour:
                    kept_circles.append(circle)

            result.circles = kept_circles
            logger.info(
                f"Circle overlap cleanup: {circles_before} → {len(result.circles)} "
                f"(removed {circles_before - len(result.circles)} redundant circles)"
            )

        # =================================================================
        # OPTIONAL: Topology cleanup (merge degree-2, snap endpoints)
        # =================================================================
        if topology_cleanup:
            try:
                from aec_agent.mcp.tools.topology import (
                    build_segment_graph,
                    merge_degree2_nodes,
                    snap_dangling_endpoints,
                    graph_to_vectorization_result,
                )
                graph = build_segment_graph(result, snap_tolerance)
                graph = merge_degree2_nodes(graph)
                graph = snap_dangling_endpoints(graph, snap_tolerance)
                cleaned = graph_to_vectorization_result(graph)
                # Replace lines and polylines with cleaned versions;
                # preserve circles, arcs, and ellipses (graph doesn't touch those).
                result.lines = cleaned.lines
                result.polylines = cleaned.polylines
                logger.info(
                    "Topology cleanup complete",
                    lines_after=len(result.lines),
                    polylines_after=len(result.polylines),
                )
            except ImportError:
                logger.warning(
                    "networkx not installed — skipping topology cleanup. "
                    "Install with: pip install networkx>=3.0"
                )

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
