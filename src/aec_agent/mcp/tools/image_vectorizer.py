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

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DetectedLine:
    """A line segment detected from raster data."""
    start: tuple[float, float]
    end: tuple[float, float]
    linetype: str = "CONTINUOUS"  # Phase 2.5: "CONTINUOUS", "DASHED", "HIDDEN", etc.


@dataclass
class DetectedCircle:
    """A circle detected from raster data."""
    center: tuple[float, float]
    radius: float


@dataclass
class DetectedArc:
    """A circular arc detected from raster data."""
    center: tuple[float, float]
    radius: float
    start_angle: float  # degrees, 0 = +X axis, CCW positive
    end_angle: float    # degrees


@dataclass
class DetectedEllipse:
    """An ellipse detected from raster data."""
    center: tuple[float, float]
    major_axis_endpoint: tuple[float, float]  # endpoint of major axis relative to center
    axis_ratio: float  # minor/major ratio (0..1]
    start_angle: float  # degrees, 0 = full ellipse
    end_angle: float    # degrees, 360 = full ellipse


@dataclass
class DetectedPolyline:
    """A polyline (contour) detected from raster data."""
    points: list[tuple[float, float]]
    closed: bool = False
    bulges: list[float] = field(default_factory=list)  # bulge per vertex (0=straight, nonzero=arc)


@dataclass
class VectorizationResult:
    """Results from image vectorization."""
    lines: list[DetectedLine] = field(default_factory=list)
    circles: list[DetectedCircle] = field(default_factory=list)
    arcs: list[DetectedArc] = field(default_factory=list)
    ellipses: list[DetectedEllipse] = field(default_factory=list)
    polylines: list[DetectedPolyline] = field(default_factory=list)
    # Phase 2.5: Semantic pipeline outputs
    texts: list = field(default_factory=list)  # List[DetectedText] from ocr_masking
    blocks: list = field(default_factory=list)  # List[DetectedBlock] from symbol_detection
    image_width_px: int = 0
    image_height_px: int = 0
    dpi: int = 300
    # Phase A: Document classification and region segmentation
    drawing_type: str | None = None  # DrawingType value
    drawing_discipline: str | None = None  # e.g., "mechanical", "electrical"
    classification_confidence: float = 0.0
    regions: list = field(default_factory=list)  # List[DetectedRegion]
    title_block_text: dict = field(default_factory=dict)  # Extracted title block info
    # Phase B: Semantic text parsing and association
    parsed_annotations: list = field(default_factory=list)  # List[ParsedAnnotation]
    text_associations: list = field(default_factory=list)  # List[TextAssociation]
    enriched_elements: list = field(default_factory=list)  # Elements with associated text
    # Phase C: Symbol Intelligence
    smart_symbols: list = field(default_factory=list)  # List[SmartSymbol] with Vision LLM classification
    vision_llm_used: bool = False  # Whether Vision LLM was used for classification


def vectorize_bitonal_image(
    image_path: str,
    dpi: int = 300,
    scale: float = 1.0,
    min_line_length: int = 50,
    max_line_gap: int = 15,
    hough_threshold: int = 80,
    min_circle_radius: int = 20,
    max_circle_radius: int = 500,
    hough_circles_dp: float = 1.2,
    hough_circles_param1: float = 200.0,
    hough_circles_param2: float = 200.0,
    hough_circles_min_dist: int = 100,
    contour_epsilon_factor: float = 0.01,
    min_contour_points: int = 3,  # Lowered to catch small curves
    min_contour_area: float = 100.0,  # Lowered: 3000 was too high for thin curves
    ellipse_fit_threshold: float = 0.85,
    arc_coverage_min: float = 30.0,
    arc_coverage_max: float = 350.0,
    line_merge_angle_tol: float = 5.0,
    line_merge_dist_tol: float = 15.0,
    circle_merge_center_tol: float = 30.0,
    circle_merge_radius_tol: float = 20.0,
    # --- Morphological parameters ---
    morph_open: bool = False,  # Disabled: erodes thin lines
    # --- Signal Restoration parameters ---
    signal_restore: bool = False,  # Disabled: destroys parallel lines
    signal_close_kernel_length: int = 15,
    signal_close_angle_step: int = 15,
    signal_close_iterations: int = 1,
    signal_smooth_ksize: int = 3,
    signal_adaptive_block: int = 15,
    signal_adaptive_c: int = 2,
    # --- Iterative Masking parameters ---
    mask_detected_circles: bool = True,
    mask_detected_lines: bool = True,
    mask_thickness: int = 5,
    # --- Circle validation parameters ---
    circle_pixel_validation: bool = True,
    circle_min_ink_ratio: float = 0.35,
    # --- FastLineDetector parameters ---
    use_fast_line_detector: bool = True,
    fld_length_threshold: int = 30,
    fld_distance_threshold: float = 1.414,
    fld_canny_aperture: int = 3,
    fld_canny_th1: float = 50.0,
    fld_canny_th2: float = 50.0,
    # --- Skeletonization & Topology parameters ---
    skeletonize: bool = True,
    topology_cleanup: bool = True,
    snap_tolerance: float = 5.0,
    # --- Position Refinement parameters ---
    refine_positions: bool = True,
    refine_search_radius: int = 10,
    # --- Debug output ---
    debug_output_dir: str | None = None,
    # --- Phase A: Document Classification & Region Segmentation ---
    document_classification: bool = True,  # Classify drawing type
    region_segmentation: bool = True,  # Detect title block, legend, etc.
    use_llm_classification: bool = False,  # Use LLM for ambiguous cases
    llm_provider: str = "groq",  # LLM provider for classification
    process_drawing_area_only: bool = False,  # Only vectorize main drawing area
    # --- Phase B: Semantic OCR & Text Association ---
    semantic_ocr: bool = True,  # Parse OCR text into structured annotations
    text_association: bool = True,  # Associate text with symbols/geometry
    semantic_llm_fallback: bool = False,  # Use LLM for unrecognized patterns
    # --- Phase 2.5: Semantic Pipeline parameters ---
    # OCR Text Masking
    ocr_masking: bool = True,  # Enabled: auto-detects pytesseract availability
    ocr_min_confidence: int = 60,
    ocr_min_text_height: int = 8,
    ocr_lang: str = "eng",
    # Symbol Detection
    symbol_detection: bool = True,  # Enabled: auto-detects template availability
    symbol_backend: str = "auto",  # "auto", "yolo", "template"
    symbol_threshold: float = 0.8,
    symbol_nms_distance: float = 20.0,
    # Phase 2.5.1: YOLO Symbol Detection
    yolo_model_path: str | None = None,  # Custom YOLO model path
    yolo_confidence: float = 0.5,  # YOLO confidence threshold
    yolo_iou_threshold: float = 0.45,  # YOLO IoU for NMS
    # Phase C: Symbol Intelligence (Vision LLM)
    vision_llm_classification: bool = True,  # Use Vision LLM for detailed subtyping
    vision_llm_provider: str = "auto",  # Vision provider: "auto", "gemini", "openai", "anthropic"
    vision_llm_confidence_threshold: float = 0.85,  # Use Vision LLM if YOLO confidence below this
    # AEC Geometric Heuristics
    aec_heuristics: bool = True,  # Enabled by default
    orthogonal_snap: bool = True,
    orthogonal_angle_tolerance: float = 2.0,
    collinear_merge: bool = True,
    collinear_angle_tolerance: float = 2.0,
    collinear_distance_tolerance: float = 5.0,
    collinear_gap_tolerance: float = 20.0,
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
        min_line_length: Minimum line length in pixels for HoughLinesP (default 50).
        max_line_gap: Maximum gap between line segments to merge (default 15).
        hough_threshold: Accumulator threshold for HoughLinesP (default 80).
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
                              false positives (default 200).
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
                             the binary image so circle detection doesn't
                             misinterpret line intersections as circles
                             (default True).
        mask_thickness: Pixel thickness of the mask painted over detected
                        features.  Larger values erase more aggressively
                        (default 5).
        circle_pixel_validation: After detecting circles, verify that the
                                circumference has actual ink pixels in the
                                binary image.  Rejects false positives from
                                line intersections and noise (default True).
        circle_min_ink_ratio: Minimum fraction of circumference sample points
                              that must have ink pixels to accept a circle.
                              Higher = stricter.  0.35 means at least 35% of
                              sampled points must have ink (default 0.35).
        use_fast_line_detector: Use OpenCV FastLineDetector (FLD) as the
                                primary line detector instead of HoughLinesP.
                                FLD is superior for engineering drawings as it
                                better preserves corners and straightness.
                                HoughLinesP results are merged in as a
                                supplement (default True).
        fld_length_threshold: Minimum segment length for FLD (default 30).
        fld_distance_threshold: Max distance between original line and fitted
                                line for FLD (default 1.414).
        fld_canny_aperture: Canny aperture size for FLD (default 3).
        fld_canny_th1: First Canny threshold for FLD (default 50.0).
        fld_canny_th2: Second Canny threshold for FLD (default 50.0).
        skeletonize: Run morphological skeletonization after signal
                     restoration to reduce thick lines to 1px centerlines
                     before detection (default True).  Uses scikit-image
                     if available, falls back to OpenCV ximgproc thinning.
                     Essential for drawings with thick lines to prevent
                     outline artifacts.
        topology_cleanup: After all detection, build a NetworkX graph from
                          detected lines/polylines, merge degree-2 nodes
                          (artificial breaks), and snap dangling endpoints
                          (default True).  Requires ``networkx``.
        snap_tolerance: Maximum distance in drawing units to snap dangling
                        endpoints during topology cleanup (default 5.0).
        refine_positions: After detection, refine line/circle positions
                          against the original clean binary (before signal
                          restoration and skeletonization).  This corrects
                          positional drift introduced by morphological
                          pre-processing (default True).
        refine_search_radius: Pixel radius to search for ink pixels when
                              refining positions (default 10).

    Returns:
        VectorizationResult with detected lines, circles, arcs, ellipses,
        and polylines in drawing-unit coordinates.

    Raises:
        FileNotFoundError: If the image file does not exist.
        RuntimeError: If vectorization fails.
    """
    import math

    import cv2
    import numpy as np

    # --- Debug checkpoint helper ---
    def _save_debug(name: str, image: "np.ndarray", annotations: list = None):
        """Save a debug checkpoint image if debug_output_dir is set."""
        if not debug_output_dir:
            return
        os.makedirs(debug_output_dir, exist_ok=True)
        filename = f"{name}.png"
        filepath = os.path.join(debug_output_dir, filename)

        # Convert to BGR for color annotations if needed
        if annotations and len(image.shape) == 2:
            out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            out = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Draw annotations (lines, circles, etc.)
        if annotations:
            for ann in annotations:
                if ann["type"] == "line":
                    cv2.line(out, ann["p1"], ann["p2"], ann.get("color", (0, 255, 0)), ann.get("thickness", 2))
                elif ann["type"] == "circle":
                    cv2.circle(out, ann["center"], ann["radius"], ann.get("color", (255, 0, 0)), ann.get("thickness", 2))
                elif ann["type"] == "text":
                    cv2.putText(out, ann["text"], ann["pos"], cv2.FONT_HERSHEY_SIMPLEX,
                                ann.get("scale", 0.5), ann.get("color", (255, 255, 255)), 1)

        cv2.imwrite(filepath, out)
        print(f"[debug] Saved checkpoint: {filepath}")

    image_path = os.path.abspath(image_path)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    logger.info("Starting image vectorization", image_path=image_path, dpi=dpi, scale=scale)
    print(f"[vectorize] Starting: {image_path}  (dpi={dpi}, scale={scale})")

    try:
        # Load image as grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to load image: {image_path}")

        height, width = img.shape[:2]
        print(f"[vectorize] Image loaded: {width}x{height}px, mean={float(np.mean(img)):.1f}")
        _save_debug("01_original_grayscale", img)
        result = VectorizationResult(
            image_width_px=width,
            image_height_px=height,
            dpi=dpi,
        )

        # =================================================================
        # PHASE A STAGE 1: DOCUMENT CLASSIFICATION
        #
        # Classify the drawing type to configure pipeline appropriately.
        # Uses keyword matching first, with optional LLM for ambiguous cases.
        # =================================================================
        if document_classification:
            try:
                from aec_agent.mcp.tools.document_classifier import (
                    classify_by_keywords,
                    get_pipeline_config,
                )

                # First, try to extract some text for classification
                # We'll use a quick OCR on the bottom-right (likely title block)
                title_text = ""
                try:
                    import pytesseract
                    # Sample bottom-right corner for title block text
                    tb_roi = img[int(height*0.8):, int(width*0.6):]
                    title_text = pytesseract.image_to_string(tb_roi, config="--psm 6")
                except Exception:
                    pass  # OCR not available or failed

                # Classify by keywords (fast, no LLM cost)
                drawing_type, confidence = classify_by_keywords(title_text)

                result.drawing_type = drawing_type.value
                result.classification_confidence = confidence

                # Get pipeline config for this drawing type
                from aec_agent.mcp.tools.document_classifier import DRAWING_TYPE_CONFIG
                config = DRAWING_TYPE_CONFIG.get(drawing_type, {})
                result.drawing_discipline = config.get("discipline", "unknown")

                logger.info(
                    "Document classification complete",
                    drawing_type=drawing_type.value,
                    confidence=f"{confidence:.2f}",
                    discipline=result.drawing_discipline,
                )
                print(f"[vectorize] Document type: {drawing_type.value} "
                      f"(confidence: {confidence:.2f}, discipline: {result.drawing_discipline})")

            except ImportError:
                logger.warning("Document classifier not available")
            except Exception as e:
                logger.warning(f"Document classification failed: {e}")

        # =================================================================
        # PHASE A STAGE 2: REGION SEGMENTATION
        #
        # Detect title block, legend, notes, and main drawing area.
        # This allows specialized processing per region.
        # =================================================================
        drawing_area_mask = None
        if region_segmentation:
            try:
                from aec_agent.mcp.tools.region_segmenter import (
                    RegionType,
                    create_region_mask,
                    extract_title_block_text,
                    segment_regions,
                )

                seg_result = segment_regions(img)
                result.regions = seg_result.regions

                # Extract title block text if found
                if seg_result.title_block:
                    tb_text = extract_title_block_text(img, seg_result.title_block)
                    result.title_block_text = tb_text
                    logger.info(
                        "Title block text extracted",
                        sheet_number=tb_text.get("sheet_number"),
                    )

                # Create mask for drawing area if requested
                if process_drawing_area_only and seg_result.drawing_area:
                    drawing_area_mask = create_region_mask(
                        (height, width),
                        seg_result.regions,
                        include_types=[RegionType.DRAWING_AREA],
                    )
                    logger.info(
                        "Processing drawing area only",
                        bounds=seg_result.drawing_area.bounds,
                    )

                logger.info(
                    "Region segmentation complete",
                    regions_found=len(seg_result.regions),
                    has_title_block=seg_result.title_block is not None,
                    has_legend=seg_result.legend is not None,
                )
                print(f"[vectorize] Regions: {len(seg_result.regions)} detected "
                      f"(title_block: {seg_result.title_block is not None}, "
                      f"legend: {seg_result.legend is not None})")

            except ImportError:
                logger.warning("Region segmenter not available")
            except Exception as e:
                logger.warning(f"Region segmentation failed: {e}")

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
            print(f"[vectorize] Auto-inverted light-on-dark image (mean={mean_val:.1f})")
            _save_debug("02_auto_inverted", img)

        # 1. Gaussian blur: smooth scan grain and pixel noise BEFORE
        #    thresholding.  A 5x5 kernel at σ=0 (auto) removes
        #    high-frequency scan artifacts without blurring real geometry.
        blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
        _save_debug("03_gaussian_blur", blurred_img)

        # 2. Adaptive threshold: binarize based on local pixel
        #    neighbourhood.  blockSize=51 tolerates uneven scan lighting.
        #    C=12 balances noise rejection vs line preservation — lower
        #    than C=15 to keep faint scan lines that represent real
        #    geometry.
        binary = cv2.adaptiveThreshold(
            blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=51, C=12,
        )
        _save_debug("04_adaptive_threshold", binary)

        # 3. Morphological close: bridge tiny gaps in lines (1-2 px)
        #    caused by scan artifacts or threshold edge effects.
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        _save_debug("05_morph_close", binary)

        # 4. Morphological open (OPTIONAL): remove single-pixel noise.
        #    DISABLED by default because it erodes thin lines (2px wide
        #    at 300 DPI).  Only enable for very noisy scans with heavy
        #    speckling.
        if morph_open:
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
            _save_debug("06_morph_open", binary)

        ink_pixels = int(np.count_nonzero(binary))
        print(f"[vectorize] After threshold+morph: {ink_pixels} ink pixels "
              f"({100.0 * ink_pixels / (width * height):.1f}% of image)")

        # Apply drawing area mask if enabled
        if drawing_area_mask is not None:
            binary = cv2.bitwise_and(binary, drawing_area_mask)
            ink_after_mask = int(np.count_nonzero(binary))
            logger.info(
                "Applied drawing area mask",
                ink_before=ink_pixels,
                ink_after=ink_after_mask,
            )
            print(f"[vectorize] After drawing area mask: {ink_after_mask} ink pixels")
            _save_debug("04c_drawing_area_masked", binary)

        # =================================================================
        # SAVE REFERENCE BINARY for position refinement.
        #
        # The binary image BEFORE signal restoration and skeletonization
        # is the most positionally accurate representation of the ink.
        # Morphological operations (signal restore, skeletonize) improve
        # detection but accumulate positional drift (1-3px per step).
        # We save this clean binary to refine detected feature positions
        # back to their true locations after detection.
        # =================================================================
        binary_reference = binary.copy()

        # =================================================================
        # PHASE 2.5 STAGE 1: TEXT ISOLATION & MASKING (OCR)
        #
        # Detect text regions using Tesseract OCR and mask them from the
        # image BEFORE line detection. This prevents text from being
        # vectorized as geometry (the "bag of lines" problem).
        # =================================================================
        detected_texts = []
        if ocr_masking:
            try:
                from aec_agent.mcp.tools.ocr_masking import detect_and_mask_text

                binary, detected_texts = detect_and_mask_text(
                    binary,
                    scale=scale,
                    min_confidence=ocr_min_confidence,
                    min_text_height_px=ocr_min_text_height,
                    lang=ocr_lang,
                )
                result.texts = detected_texts
                logger.info(
                    "OCR masking complete",
                    texts_found=len(detected_texts),
                )
                print(f"[vectorize] OCR masking: {len(detected_texts)} text regions detected and masked")
                _save_debug("04a_ocr_masked", binary)
            except ImportError:
                logger.warning(
                    "pytesseract not installed, skipping OCR masking. "
                    "Install with: pip install pytesseract"
                )
            except Exception as e:
                logger.warning(f"OCR masking failed: {e}, continuing without text masking")

        # =================================================================
        # PHASE C: SYMBOL INTELLIGENCE (Two-Stage Classification)
        #
        # Two-stage symbol classification:
        # Stage 1: YOLO/template matching for fast detection
        # Stage 2: Vision LLM for detailed subtype classification
        #
        # Detected symbols are masked from the image to prevent them
        # from being traced as jagged polylines - instead they'll be
        # inserted as blocks.
        # =================================================================
        detected_blocks = []
        smart_symbols = []

        if symbol_detection:
            try:
                # Check if we should use the smart (Vision LLM) pipeline
                from aec_agent.mcp.tools.symbol_detection import detect_symbols

                use_smart_detection = vision_llm_classification
                if use_smart_detection:
                    try:
                        from aec_agent.mcp.tools.vision_llm import is_vision_llm_available
                        use_smart_detection = is_vision_llm_available()
                    except ImportError:
                        use_smart_detection = False

                if use_smart_detection:
                    # Use Phase C two-stage pipeline with Vision LLM
                    import asyncio
                    from aec_agent.mcp.tools.symbol_classifier import SymbolClassifier
                    from aec_agent.mcp.tools.document_classifier import DrawingType

                    # Get drawing type for context
                    drawing_type_enum = DrawingType.MEP_PLAN
                    if result.drawing_type:
                        try:
                            drawing_type_enum = DrawingType(result.drawing_type)
                        except ValueError:
                            drawing_type_enum = DrawingType.MEP_PLAN

                    classifier = SymbolClassifier(
                        yolo_confidence_threshold=vision_llm_confidence_threshold,
                        enable_vision_llm=True,
                        vision_provider=vision_llm_provider,
                    )

                    # Run async classifier (get or create event loop)
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # We're already in an async context
                            import concurrent.futures
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(
                                    asyncio.run,
                                    classifier.classify_symbols(
                                        image=binary,
                                        drawing_type=drawing_type_enum,
                                        scale=scale,
                                        parsed_annotations=result.parsed_annotations if semantic_ocr else None,
                                        yolo_backend=symbol_backend,
                                        yolo_confidence=yolo_confidence,
                                    )
                                )
                                binary, smart_symbols = future.result()
                        else:
                            binary, smart_symbols = loop.run_until_complete(
                                classifier.classify_symbols(
                                    image=binary,
                                    drawing_type=drawing_type_enum,
                                    scale=scale,
                                    parsed_annotations=result.parsed_annotations if semantic_ocr else None,
                                    yolo_backend=symbol_backend,
                                    yolo_confidence=yolo_confidence,
                                )
                            )
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        binary, smart_symbols = loop.run_until_complete(
                            classifier.classify_symbols(
                                image=binary,
                                drawing_type=drawing_type_enum,
                                scale=scale,
                                parsed_annotations=result.parsed_annotations if semantic_ocr else None,
                                yolo_backend=symbol_backend,
                                yolo_confidence=yolo_confidence,
                            )
                        )

                    # Convert SmartSymbols to DetectedBlocks for backwards compatibility
                    from aec_agent.mcp.tools.symbol_detection import DetectedBlock
                    detected_blocks = [
                        DetectedBlock(
                            block_name=sym.block_name,
                            position=sym.position,
                            scale=sym.scale,
                            rotation=sym.rotation,
                            confidence=sym.confidence,
                            category=sym.category,
                        )
                        for sym in smart_symbols
                    ]

                    result.smart_symbols = smart_symbols
                    result.vision_llm_used = any(s.classification_source == "vision_llm" for s in smart_symbols)

                    vision_classified = sum(1 for s in smart_symbols if s.classification_source == "vision_llm")
                    logger.info(
                        "Smart symbol detection complete (Phase C)",
                        backend=symbol_backend,
                        symbols_found=len(smart_symbols),
                        vision_llm_classified=vision_classified,
                    )
                    print(f"[vectorize] Smart symbol detection: {len(smart_symbols)} symbols, "
                          f"{vision_classified} classified by Vision LLM")

                else:
                    # Fall back to basic YOLO/template detection
                    binary, detected_blocks = detect_symbols(
                        binary,
                        scale=scale,
                        backend=symbol_backend,
                        # Template matching parameters
                        match_threshold=symbol_threshold,
                        nms_distance=symbol_nms_distance,
                        # YOLO parameters (Phase 2.5.1)
                        yolo_model_path=yolo_model_path,
                        yolo_confidence=yolo_confidence,
                        yolo_iou_threshold=yolo_iou_threshold,
                        mask_detections=True,
                    )
                    logger.info(
                        "Symbol detection complete (basic)",
                        backend=symbol_backend,
                        symbols_found=len(detected_blocks),
                    )
                    print(f"[vectorize] Symbol detection ({symbol_backend}): {len(detected_blocks)} symbols detected")

                result.blocks = detected_blocks
                _save_debug("04b_symbols_masked", binary)

            except ImportError as e:
                logger.warning(f"Symbol detection dependencies not available: {e}")
            except Exception as e:
                logger.warning(f"Symbol detection failed: {e}, continuing without symbol masking")

        # =================================================================
        # PHASE B STAGE 1: SEMANTIC OCR PARSING
        #
        # Parse raw OCR text into structured annotations. This transforms
        # text like "24x24 SA 200 CFM" into structured data:
        # {type: "supply_air_diffuser", width: 24, height: 24, cfm: 200}
        # =================================================================
        if semantic_ocr and detected_texts:
            try:
                from aec_agent.mcp.tools.document_classifier import DrawingType
                from aec_agent.mcp.tools.semantic_ocr import parse_annotation_by_patterns

                # Get drawing type for context
                drawing_type_enum = None
                if result.drawing_type:
                    try:
                        drawing_type_enum = DrawingType(result.drawing_type)
                    except ValueError:
                        drawing_type_enum = DrawingType.UNKNOWN

                parsed_annotations = []
                for text_item in detected_texts:
                    # Extract text content from DetectedText objects
                    text_content = getattr(text_item, 'text', str(text_item))
                    if not text_content or len(text_content.strip()) < 2:
                        continue

                    # Parse using pattern matching (fast, no LLM cost)
                    parsed = parse_annotation_by_patterns(
                        text_content,
                        drawing_type=drawing_type_enum,
                    )

                    if parsed:
                        # Transfer position from original detection
                        if hasattr(text_item, 'position'):
                            parsed.position = text_item.position
                        if hasattr(text_item, 'bounding_box'):
                            parsed.bounding_box = text_item.bounding_box
                        parsed_annotations.append(parsed)

                result.parsed_annotations = parsed_annotations

                # Count by type for logging
                type_counts = {}
                for pa in parsed_annotations:
                    type_name = pa.annotation_type.value
                    type_counts[type_name] = type_counts.get(type_name, 0) + 1

                logger.info(
                    "Semantic OCR parsing complete",
                    total_texts=len(detected_texts),
                    parsed=len(parsed_annotations),
                    types=type_counts,
                )
                print(f"[vectorize] Semantic OCR: {len(parsed_annotations)}/{len(detected_texts)} texts parsed")

            except ImportError:
                logger.warning("Semantic OCR module not available")
            except Exception as e:
                logger.warning(f"Semantic OCR parsing failed: {e}")

        # =================================================================
        # PHASE B STAGE 2: TEXT-ELEMENT ASSOCIATION
        #
        # Associate parsed text annotations with detected symbols.
        # Equipment tags → nearest symbol, specs → nearby equipment, etc.
        # =================================================================
        if text_association and result.parsed_annotations and detected_blocks:
            try:
                from aec_agent.mcp.tools.text_associator import (
                    DetectedElement,
                    associate_text_to_elements,
                    enrich_elements_with_text,
                )

                # Convert detected blocks to DetectedElement format
                elements = []
                for i, block in enumerate(detected_blocks):
                    # Handle different block formats
                    pos = getattr(block, 'position', (0, 0))
                    bounds = getattr(block, 'bounding_box', (pos[0], pos[1], 50, 50))
                    category = getattr(block, 'category', 'unknown')
                    subtype = getattr(block, 'class_name', getattr(block, 'block_name', 'symbol'))

                    elem = DetectedElement(
                        element_id=f"block-{i:04d}",
                        element_type="symbol",
                        position=pos,
                        bounds=bounds,
                        category=category,
                        subtype=subtype,
                        confidence=getattr(block, 'confidence', 0.5),
                    )
                    elements.append(elem)

                # Associate text with elements
                associations = associate_text_to_elements(
                    result.parsed_annotations,
                    elements,
                    max_distance=100.0 * scale,  # Scale to drawing units
                )
                result.text_associations = associations

                # Create enriched elements (symbols with their associated text)
                enriched = enrich_elements_with_text(
                    elements,
                    result.parsed_annotations,
                    max_distance=100.0 * scale,
                )
                result.enriched_elements = enriched

                # Count associations
                elements_with_text = len([e for e in enriched if e.get('annotations')])

                logger.info(
                    "Text-element association complete",
                    annotations=len(result.parsed_annotations),
                    elements=len(elements),
                    associations=len(associations),
                    elements_with_text=elements_with_text,
                )
                print(f"[vectorize] Text association: {len(associations)} associations, "
                      f"{elements_with_text}/{len(elements)} elements enriched")

            except ImportError:
                logger.warning("Text associator module not available")
            except Exception as e:
                logger.warning(f"Text-element association failed: {e}")

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
                k_center = k_len // 2
                angle_rad = math.radians(angle_deg)
                dx = math.cos(angle_rad)
                dy = math.sin(angle_rad)
                for t in range(-k_center, k_center + 1):
                    x = int(round(k_center + t * dx))
                    y = int(round(k_center + t * dy))
                    if 0 <= x < k_len and 0 <= y < k_len:
                        kern[y, x] = 1

                closed_dir = cv2.morphologyEx(
                    binary, cv2.MORPH_CLOSE, kern,
                    iterations=signal_close_iterations,
                )
                restored = cv2.bitwise_or(restored, closed_dir)

            # 5b. Gaussian blur + threshold to smooth jagged staircase
            #     edges introduced by the directional close and to
            #     re-binarize the result cleanly.
            #
            #     NOTE: We use a simple global threshold (not adaptive)
            #     because the input is already binary (ink=255, bg=0)
            #     that was only slightly blurred.  Adaptive thresholding
            #     fails here: in uniform dark background regions every
            #     pixel exceeds (local_mean - C), turning the entire
            #     background white and destroying the image.
            sk = signal_smooth_ksize if signal_smooth_ksize % 2 == 1 else signal_smooth_ksize + 1
            smoothed = cv2.GaussianBlur(restored, (sk, sk), 0)
            _, binary = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)
            _save_debug("07_signal_restored", binary)

            ink_after_sr = int(np.count_nonzero(binary))
            logger.info(
                "Signal restoration complete",
                kernel_length=signal_close_kernel_length,
                angle_step=signal_close_angle_step,
                directions=180 // signal_close_angle_step,
            )
            print(f"[vectorize] Signal restoration: {ink_after_sr} ink pixels after re-threshold")

        # 4. Remove small connected components (text, dots, annotations,
        #    scan artifacts).  Uses BOTH area AND bounding-box extent:
        #    - Small area AND small bbox → noise (dots, speckles) → remove
        #    - Small area BUT large bbox → thin line work → KEEP
        #    This preserves long thin contour lines that have small pixel
        #    area but span a significant distance across the drawing.
        # Lower thresholds to preserve thin curves
        min_component_area = max(min_contour_area, width * height * 0.0001)  # 0.01% not 0.1%
        min_component_dim = max(30, min(width, height) * 0.01)  # 1% of shorter side
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

        kept_count = sum(1 for lbl in range(1, n_labels) if _component_kept(lbl))
        removed_count = sum(1 for lbl in range(1, n_labels) if not _component_kept(lbl))
        logger.info(
            "Pre-processing complete",
            components_kept=kept_count,
            components_removed=removed_count,
        )
        print(f"[vectorize] Components: {kept_count} kept, {removed_count} removed")
        _save_debug("08_components_filtered", binary)

        # =================================================================
        # SKELETONIZATION: Reduce thick lines to 1px centerlines.
        #
        # Critical for drawings with thick lines (e.g. white lines on
        # dark background).  Without thinning, both FLD and HoughLinesP
        # detect the EDGES of thick features instead of the centerline,
        # producing two outlines per actual line.
        # =================================================================
        if skeletonize:
            try:
                from skimage.morphology import skeletonize as _skeletonize
                # skimage expects bool array (True = foreground)
                skel = _skeletonize(binary > 0)
                binary = (skel.astype(np.uint8)) * 255
                logger.info("Skeletonization complete (lines reduced to 1px centerlines)")
                print(f"[vectorize] Skeletonized: {int(np.count_nonzero(binary))} skeleton pixels")
                _save_debug("09_skeletonized", binary)
            except ImportError:
                # Fallback: iterative morphological thinning using OpenCV.
                # Zhang-Suen thinning via ximgproc, or repeated erosion
                # with hit-or-miss as a last resort.
                try:
                    thinned = cv2.ximgproc.thinning(
                        binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN,
                    )
                    binary = thinned
                    logger.info(
                        "Morphological thinning complete via cv2.ximgproc.thinning "
                        "(scikit-image not installed, using OpenCV fallback)"
                    )
                    _save_debug("09_skeletonized_ximgproc", binary)
                except AttributeError:
                    logger.warning(
                        "Neither scikit-image nor opencv-contrib available for "
                        "skeletonization. Thick lines may produce outline artifacts. "
                        "Install with: pip install scikit-image>=0.21.0 "
                        "or pip install opencv-contrib-python"
                    )

        # Helper: convert pixel coords to drawing units.
        def px_to_dwg(px_x: float, px_y: float) -> tuple[float, float]:
            return (px_x * scale, (height - px_y) * scale)

        def px_dist_to_dwg(px_dist: float) -> float:
            return px_dist * scale

        # =================================================================
        # LINE DETECTION (FastLineDetector + HoughLinesP) + Deduplication
        # =================================================================
        # Lines are detected FIRST.  Engineering drawings are predominantly
        # lines; detecting circles first causes false positives at line
        # intersections which then mask out actual line pixels.  By
        # detecting lines first and masking them, circle detection only
        # sees actual circular features.

        all_raw_lines = []

        # --- Primary: FastLineDetector (FLD) ---
        # FLD is superior to Hough for engineering drawings: it preserves
        # corners and straightness better and produces fewer fragments.
        if use_fast_line_detector:
            try:
                fld = cv2.ximgproc.createFastLineDetector(
                    fld_length_threshold,
                    fld_distance_threshold,
                    fld_canny_th1,
                    fld_canny_th2,
                    fld_canny_aperture,
                    do_merge=True,
                )
                fld_lines = fld.detect(binary)
                if fld_lines is not None:
                    for seg in fld_lines:
                        x1, y1, x2, y2 = seg[0]
                        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        if length >= min_line_length:
                            all_raw_lines.append([int(round(x1)), int(round(y1)),
                                                  int(round(x2)), int(round(y2))])
                    logger.info(f"FLD detected {len(fld_lines)} raw segments, "
                                f"{len(all_raw_lines)} after length filter")
            except AttributeError:
                logger.info("FastLineDetector not available (needs opencv-contrib-python), "
                            "falling back to HoughLinesP only")

        # --- Supplement: HoughLinesP ---
        # Pass binary directly — NOT Canny edges.  On thick lines, Canny
        # produces two edge outlines (one per side), causing HoughLinesP to
        # detect outlines instead of centerlines.  After skeletonization the
        # binary already contains 1px-wide centerlines, so Canny is redundant
        # and harmful.
        raw_hough = cv2.HoughLinesP(
            binary,
            rho=1,
            theta=np.pi / 180,
            threshold=hough_threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )
        hough_count = 0
        if raw_hough is not None:
            for line in raw_hough:
                x1, y1, x2, y2 = line[0]
                all_raw_lines.append([int(x1), int(y1), int(x2), int(y2)])
                hough_count += 1

        # Convert to the format _deduplicate_lines expects
        raw_lines_arr = None
        if all_raw_lines:
            raw_lines_arr = np.array(all_raw_lines).reshape(-1, 1, 4)

        # Deduplicate lines: merge nearly-parallel, closely-spaced segments
        deduped_lines = _deduplicate_lines(
            raw_lines_arr, line_merge_angle_tol, line_merge_dist_tol,
        )

        # --- Refine line positions against the reference binary ---
        # Detection runs on the processed image (signal restore +
        # skeletonize) for robust connectivity.  But those morphological
        # operations shift line positions.  Refine each line by fitting
        # to actual ink pixels in the clean reference binary.
        if refine_positions:
            refined_lines = []
            refined_count = 0
            for (x1, y1, x2, y2) in deduped_lines:
                rx1, ry1, rx2, ry2 = _refine_line_to_reference(
                    float(x1), float(y1), float(x2), float(y2),
                    binary_reference, refine_search_radius,
                )
                if (rx1, ry1, rx2, ry2) != (float(x1), float(y1), float(x2), float(y2)):
                    refined_count += 1
                refined_lines.append((rx1, ry1, rx2, ry2))
            deduped_lines = refined_lines
            logger.info(
                f"Position refinement: {refined_count}/{len(deduped_lines)} "
                f"lines refined against reference binary"
            )

        for (x1, y1, x2, y2) in deduped_lines:
            start = px_to_dwg(float(x1), float(y1))
            end = px_to_dwg(float(x2), float(y2))
            result.lines.append(DetectedLine(start=start, end=end))

        logger.info(
            f"Detected {len(result.lines)} lines "
            f"(FLD+Hough raw: {len(all_raw_lines)}, HoughLinesP: {hough_count}, "
            f"after dedup: {len(deduped_lines)})"
        )
        print(f"[vectorize] Lines: {len(result.lines)} (raw={len(all_raw_lines)}, dedup={len(deduped_lines)})")

        # Debug: show detected lines overlaid on binary
        if debug_output_dir and deduped_lines:
            line_anns = [
                {"type": "line", "p1": (int(x1), int(y1)), "p2": (int(x2), int(y2)), "color": (0, 255, 0), "thickness": 2}
                for (x1, y1, x2, y2) in deduped_lines
            ]
            _save_debug("10_detected_lines", binary, line_anns)

        # --- Mask detected lines from binary image ---
        # Masking lines BEFORE circle detection is critical: it prevents
        # line intersections from being misidentified as circles.
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
                "Masked detected lines from binary before circle detection",
                lines_masked=lines_masked,
                mask_thickness=mask_thickness,
            )
            _save_debug("11_lines_masked", binary)

        # =================================================================
        # CIRCLE DETECTION (HoughCircles) + Validation + Deduplication
        # =================================================================
        # Runs AFTER line detection and masking.  The binary image now has
        # line pixels removed, so only actual circular features remain.
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

        # --- Pixel-level circle validation ---
        # Verify each detected circle has actual ink pixels along its
        # circumference in the binary image.  This rejects false positives
        # from noise, text fragments, and line intersections that
        # HoughCircles mistakes for circles.
        if circle_pixel_validation and deduped_circles:
            validated_circles = _validate_circles_by_ink(
                deduped_circles, binary, circle_min_ink_ratio,
            )
            rejected = len(deduped_circles) - len(validated_circles)
            if rejected > 0:
                logger.info(
                    f"Circle validation: {len(deduped_circles)} candidates, "
                    f"{len(validated_circles)} validated, {rejected} rejected "
                    f"(min ink ratio: {circle_min_ink_ratio})"
                )
            deduped_circles = validated_circles

        # --- Refine circle positions against reference binary ---
        if refine_positions and deduped_circles:
            refined_circles = []
            refined_count = 0
            for (cx, cy, r) in deduped_circles:
                ncx, ncy, nr = _refine_circle_to_reference(
                    cx, cy, r, binary_reference, refine_search_radius,
                )
                if (ncx, ncy, nr) != (cx, cy, r):
                    refined_count += 1
                refined_circles.append((ncx, ncy, nr))
            deduped_circles = refined_circles
            logger.info(
                f"Position refinement: {refined_count}/{len(deduped_circles)} "
                f"circles refined against reference binary"
            )

        for (cx, cy, r) in deduped_circles:
            center = px_to_dwg(float(cx), float(cy))
            radius = px_dist_to_dwg(float(r))
            result.circles.append(DetectedCircle(center=center, radius=radius))

        logger.info(
            f"Detected {len(result.circles)} circles "
            f"(raw: {len(raw_circles[0]) if raw_circles is not None else 0}, "
            f"after dedup+validation: {len(deduped_circles)})"
        )

        # Debug: show detected circles
        if debug_output_dir and deduped_circles:
            circle_anns = [
                {"type": "circle", "center": (int(cx), int(cy)), "radius": int(r), "color": (255, 0, 0), "thickness": 2}
                for (cx, cy, r) in deduped_circles
            ]
            _save_debug("12_detected_circles", binary, circle_anns)

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
            _save_debug("13_circles_masked", binary)

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
        circle_centers_px: list[tuple[float, float, float]] = []
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
            semi_major = MA / 2.0
            semi_minor = ma / 2.0
            angle_rad = math.radians(angle)
            cos_ang = math.cos(angle_rad)
            sin_ang = math.sin(angle_rad)
            total_err = 0.0
            for pt in contour:
                dx = float(pt[0][0]) - cx
                dy = float(pt[0][1]) - cy
                lx = dx * cos_ang + dy * sin_ang
                ly = -dx * sin_ang + dy * cos_ang
                if semi_major > 0 and semi_minor > 0:
                    d = math.sqrt((lx / semi_major) ** 2 + (ly / semi_minor) ** 2)
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

        def _compute_arc_angles(contour, cx: float, cy: float) -> tuple[float, float]:
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
        print(f"[vectorize] Circles: {len(result.circles)}, Arcs: {len(result.arcs)}, "
              f"Ellipses: {len(result.ellipses)}, Polylines: {len(result.polylines)}")

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
            kept_lines: list[DetectedLine] = []
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
                f"Line overlap cleanup: {lines_before} -> {len(result.lines)} "
                f"(removed {lines_before - len(result.lines)} redundant lines)"
            )

            # --- Filter redundant circles ---
            circles_before = len(result.circles)
            kept_circles: list[DetectedCircle] = []
            for circle in result.circles:
                cx_px = circle.center[0] / scale if scale else 0
                cy_px = height - (circle.center[1] / scale if scale else 0)
                r_px = circle.radius / scale if scale else 0

                on_contour = False
                for contour in significant_contours:
                    # Sample 8 points around the circle perimeter
                    perimeter_on_contour = 0
                    for angle_i in range(8):
                        theta = angle_i * (2 * math.pi / 8)
                        px = cx_px + r_px * math.cos(theta)
                        py = cy_px + r_px * math.sin(theta)
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
                f"Circle overlap cleanup: {circles_before} -> {len(result.circles)} "
                f"(removed {circles_before - len(result.circles)} redundant circles)"
            )

        # =================================================================
        # OPTIONAL: Topology cleanup (merge degree-2, snap endpoints)
        # =================================================================
        if topology_cleanup:
            try:
                from aec_agent.mcp.tools.topology import (
                    build_segment_graph,
                    graph_to_vectorization_result,
                    merge_degree2_nodes,
                    snap_dangling_endpoints,
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

        # =================================================================
        # PHASE 2.5 STAGE 5: AEC GEOMETRIC HEURISTICS
        #
        # Apply engineering-specific post-processing to clean up geometry:
        # 1. Orthogonal snapping: Lines within ±tolerance of 0°/90° snap to exact
        # 2. Collinear merging: Fragmented segments merge into single lines
        # 3. Dashed line detection: Regular gaps → DASHED linetype
        # =================================================================
        if aec_heuristics and result.lines:
            try:
                from aec_agent.utils.geometry_cleanup import (
                    MergedLine,
                    merge_collinear_lines,
                    snap_to_orthogonal,
                )

                original_count = len(result.lines)

                # Convert DetectedLine to tuples for processing
                line_tuples = [
                    (ln.start[0], ln.start[1], ln.end[0], ln.end[1])
                    for ln in result.lines
                ]

                # Step 1: Orthogonal snapping
                if orthogonal_snap:
                    line_tuples = snap_to_orthogonal(
                        line_tuples,
                        angle_tolerance=orthogonal_angle_tolerance,
                    )
                    print(f"[vectorize] Orthogonal snapping: processed {len(line_tuples)} lines")

                # Step 2: Collinear merging
                if collinear_merge:
                    continuous, dashed = merge_collinear_lines(
                        line_tuples,
                        angle_tolerance=collinear_angle_tolerance,
                        distance_tolerance=collinear_distance_tolerance,
                        gap_tolerance=collinear_gap_tolerance,
                    )

                    # Convert MergedLine back to DetectedLine
                    result.lines = []
                    for ml in continuous:
                        result.lines.append(DetectedLine(
                            start=ml.start,
                            end=ml.end,
                            linetype="CONTINUOUS",
                        ))
                    for ml in dashed:
                        result.lines.append(DetectedLine(
                            start=ml.start,
                            end=ml.end,
                            linetype="DASHED",
                        ))

                    reduction = original_count - len(result.lines)
                    pct = (100.0 * reduction / original_count) if original_count > 0 else 0
                    logger.info(
                        "AEC heuristics complete",
                        original=original_count,
                        merged=len(result.lines),
                        reduction_pct=f"{pct:.1f}%",
                        dashed=len(dashed),
                    )
                    print(f"[vectorize] Collinear merge: {original_count} → {len(result.lines)} lines "
                          f"({pct:.1f}% reduction, {len(dashed)} dashed)")
                else:
                    # Just convert back with CONTINUOUS linetype
                    result.lines = [
                        DetectedLine(start=(x1, y1), end=(x2, y2), linetype="CONTINUOUS")
                        for x1, y1, x2, y2 in line_tuples
                    ]

            except ImportError as e:
                logger.warning(f"AEC heuristics module not available: {e}")
            except Exception as e:
                logger.warning(f"AEC heuristics failed: {e}, keeping original lines")

        total = (
            len(result.lines) + len(result.circles) + len(result.arcs)
            + len(result.ellipses) + len(result.polylines)
            + len(result.texts) + len(result.blocks)  # Phase 2.5 additions
        )
        logger.info(
            "Vectorization complete",
            total_features=total,
            lines=len(result.lines),
            circles=len(result.circles),
            arcs=len(result.arcs),
            ellipses=len(result.ellipses),
            polylines=len(result.polylines),
            texts=len(result.texts),
            blocks=len(result.blocks),
        )
        print(f"[vectorize] DONE — {total} features "
              f"(L={len(result.lines)} C={len(result.circles)} "
              f"A={len(result.arcs)} E={len(result.ellipses)} "
              f"P={len(result.polylines)} T={len(result.texts)} B={len(result.blocks)})")

        # Final debug: all detected features on original image
        if debug_output_dir:
            all_anns = []
            # Lines in green
            for ln in result.lines:
                # Convert back to pixel coords
                px1 = int(ln.start[0] / scale) if scale else 0
                py1 = int(height - ln.start[1] / scale) if scale else 0
                px2 = int(ln.end[0] / scale) if scale else 0
                py2 = int(height - ln.end[1] / scale) if scale else 0
                all_anns.append({"type": "line", "p1": (px1, py1), "p2": (px2, py2), "color": (0, 255, 0), "thickness": 2})
            # Circles in red
            for c in result.circles:
                cx = int(c.center[0] / scale) if scale else 0
                cy = int(height - c.center[1] / scale) if scale else 0
                r = int(c.radius / scale) if scale else 0
                all_anns.append({"type": "circle", "center": (cx, cy), "radius": r, "color": (0, 0, 255), "thickness": 2})
            # Arcs in cyan (draw as circles for simplicity)
            for a in result.arcs:
                cx = int(a.center[0] / scale) if scale else 0
                cy = int(height - a.center[1] / scale) if scale else 0
                r = int(a.radius / scale) if scale else 0
                all_anns.append({"type": "circle", "center": (cx, cy), "radius": r, "color": (255, 255, 0), "thickness": 2})
            _save_debug("99_final_all_features", img, all_anns)

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
) -> list[tuple[float, float, float, float]]:
    """
    Merge near-duplicate lines detected by HoughLinesP.

    Two lines are considered duplicates if:
    - Their angles differ by less than ``angle_tol`` degrees, AND
    - They are within ``dist_tol`` pixels perpendicular distance, AND
    - They actually OVERLAP along the line direction.

    Collinear but non-overlapping lines are kept as separate features.
    When duplicates are found, the longest line is kept.
    """
    import math

    import numpy as np

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

        # Line i direction unit vector
        line_dir = np.array([x2 - x1, y2 - y1], dtype=float)
        line_len = np.linalg.norm(line_dir)
        if line_len < 1:
            continue
        line_unit = line_dir / line_len
        perp = np.array([-line_unit[1], line_unit[0]])

        # Project line i endpoints onto its direction (for overlap check)
        origin = np.array([x1, y1], dtype=float)
        t_i_min = 0.0
        t_i_max = line_len

        # Mark near-duplicates as used
        for j in range(i + 1, len(lines_data)):
            if used[j]:
                continue
            x1_j, y1_j, x2_j, y2_j, ang_j, len_j, mx_j, my_j = lines_data[j]

            # Check angle similarity
            angle_diff = abs(ang - ang_j)
            if angle_diff > 90:
                angle_diff = 180 - angle_diff
            if angle_diff > angle_tol:
                continue

            # Check perpendicular distance between midpoints
            dmx = mx_j - x1
            dmy = my_j - y1
            perp_dist = abs(dmx * perp[0] + dmy * perp[1])

            if perp_dist >= dist_tol:
                continue

            # Check for OVERLAP along line direction
            # Project line j endpoints onto line i direction
            p1_j = np.array([x1_j, y1_j], dtype=float) - origin
            p2_j = np.array([x2_j, y2_j], dtype=float) - origin
            t_j1 = np.dot(p1_j, line_unit)
            t_j2 = np.dot(p2_j, line_unit)
            t_j_min = min(t_j1, t_j2)
            t_j_max = max(t_j1, t_j2)

            # Check if intervals overlap (with small tolerance)
            overlap_tol = dist_tol
            if t_j_max < t_i_min - overlap_tol or t_j_min > t_i_max + overlap_tol:
                # No overlap — these are collinear but separate lines
                continue

            # Lines overlap — mark j as duplicate
            used[j] = True

    return kept


def _deduplicate_circles(
    raw_circles,
    center_tol: float = 30.0,
    radius_tol: float = 20.0,
) -> list[tuple[float, float, float]]:
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


def _validate_circles_by_ink(
    circles: list[tuple[float, float, float]],
    binary_image: "np.ndarray",
    min_ink_ratio: float = 0.35,
    n_samples: int = 36,
) -> list[tuple[float, float, float]]:
    """
    Validate detected circles by checking for actual ink pixels along
    the circumference in the binary image.

    Samples ``n_samples`` evenly-spaced points around each circle's
    circumference and checks if the binary image has white (ink) pixels
    at those locations.  Circles where fewer than ``min_ink_ratio``
    fraction of samples have ink are rejected as false positives.

    Args:
        circles: List of (cx, cy, radius) tuples in pixel coordinates.
        binary_image: Binary image (white = ink, black = background).
        min_ink_ratio: Minimum fraction of samples with ink (default 0.35).
        n_samples: Number of points to sample around circumference (default 36).

    Returns:
        Filtered list of validated circles.
    """
    import math

    h, w = binary_image.shape[:2]
    validated = []

    for (cx, cy, r) in circles:
        ink_count = 0
        for i in range(n_samples):
            angle = 2.0 * math.pi * i / n_samples
            px = int(round(cx + r * math.cos(angle)))
            py = int(round(cy + r * math.sin(angle)))

            # Check a small neighborhood (3x3) around the sample point
            # to tolerate slight misalignment
            found_ink = False
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    sx, sy = px + dx, py + dy
                    if 0 <= sx < w and 0 <= sy < h:
                        if binary_image[sy, sx] > 0:
                            found_ink = True
                            break
                if found_ink:
                    break
            if found_ink:
                ink_count += 1

        ratio = ink_count / n_samples
        if ratio >= min_ink_ratio:
            validated.append((cx, cy, r))

    return validated


def _compute_bulges(
    original_contour,
    simplified: "np.ndarray",
    is_closed: bool,
) -> list[float]:
    """
    Compute bulge values for a simplified polyline by comparing to the
    original contour.  A bulge of 0 means a straight segment; non-zero
    means an arc segment (bulge = tan(included_angle / 4)).

    For each segment of the simplified polyline, we find the corresponding
    points on the original contour and measure the maximum deviation.  If
    the deviation is significant relative to the segment length, we
    compute a bulge value.
    """
    import math

    import numpy as np

    n = len(simplified)
    bulges: list[float] = [0.0] * n
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


# =========================================================================
# Position refinement helpers
# =========================================================================

def _refine_line_to_reference(
    x1: float, y1: float, x2: float, y2: float,
    binary_ref: "np.ndarray",
    search_radius: int = 10,
) -> tuple[float, float, float, float]:
    """
    Refine a detected line's position by fitting to actual ink pixels in
    the reference binary image.

    The detection pipeline uses heavy morphological pre-processing (signal
    restoration, skeletonization) that can shift line positions by several
    pixels.  This function finds the ink pixels in the *original* clean
    binary that lie near the detected line, fits a new line through them,
    and returns corrected endpoints.

    Args:
        x1, y1, x2, y2: Detected line endpoints in pixel coordinates.
        binary_ref: The reference binary image (before signal restore /
                    skeletonize).  White = ink, black = background.
        search_radius: Pixel radius to search for ink around the line.

    Returns:
        Refined (x1, y1, x2, y2) in pixel coordinates.
    """
    import cv2
    import numpy as np

    h, w = binary_ref.shape[:2]

    # Create a mask along the detected line with the search radius
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.line(
        mask,
        (int(round(x1)), int(round(y1))),
        (int(round(x2)), int(round(y2))),
        255, search_radius * 2,
    )

    # Find ink pixels in the reference binary within the search corridor
    ink_yx = np.column_stack(np.where((binary_ref > 0) & (mask > 0)))
    if len(ink_yx) < 4:
        return (x1, y1, x2, y2)  # Not enough ink pixels — keep original

    # Convert (row, col) → (x, y)
    pts = ink_yx[:, [1, 0]].astype(np.float32)

    # Fit a line through the ink pixels (L2 = least-squares)
    [vx, vy, cx, cy] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    dx, dy = float(vx[0]), float(vy[0])
    cx, cy = float(cx[0]), float(cy[0])

    # Project all ink pixels onto the fitted line direction to find
    # the actual extent (where ink starts and ends)
    projections = (pts[:, 0] - cx) * dx + (pts[:, 1] - cy) * dy

    # Use the 2nd and 98th percentile to avoid outlier ink pixels
    t_min = float(np.percentile(projections, 2))
    t_max = float(np.percentile(projections, 98))

    # Refined endpoints along the fitted line
    nx1 = float(cx + t_min * dx)
    ny1 = float(cy + t_min * dy)
    nx2 = float(cx + t_max * dx)
    ny2 = float(cy + t_max * dy)

    # Sanity check: refined line should be roughly the same length (±50%)
    orig_len = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    new_len = np.sqrt((nx2 - nx1) ** 2 + (ny2 - ny1) ** 2)
    if orig_len > 0 and (new_len / orig_len < 0.5 or new_len / orig_len > 1.5):
        return (x1, y1, x2, y2)  # Refinement is suspicious — keep original

    return (nx1, ny1, nx2, ny2)


def _refine_circle_to_reference(
    cx: float, cy: float, r: float,
    binary_ref: "np.ndarray",
    search_radius: int = 10,
    n_samples: int = 72,
) -> tuple[float, float, float]:
    """
    Refine a detected circle's position by fitting to actual ink pixels in
    the reference binary image.

    Samples points around the circumference, finds nearby ink pixels in the
    reference binary, and re-fits a circle through them using least-squares.

    Args:
        cx, cy, r: Detected circle center and radius in pixel coordinates.
        binary_ref: Reference binary image.
        search_radius: Pixel radius to search for ink around circumference.
        n_samples: Number of angular samples around circumference.

    Returns:
        Refined (cx, cy, r) in pixel coordinates.
    """
    import math

    import cv2
    import numpy as np

    h, w = binary_ref.shape[:2]

    # Create an annular mask around the detected circle
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))),
               int(round(r)) + search_radius, 255, search_radius * 2)

    # Find ink pixels within the annular mask
    ink_yx = np.column_stack(np.where((binary_ref > 0) & (mask > 0)))
    if len(ink_yx) < 8:
        return (cx, cy, r)  # Not enough ink

    # Convert (row, col) → (x, y)
    pts = ink_yx[:, [1, 0]].astype(np.float64)

    # Algebraic circle fit (Kasa method): minimize sum of (x²+y²-ax-by-c)²
    # Gives (a, b, c) where center = (a/2, b/2), r = sqrt(c + a²/4 + b²/4)
    A = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
    b_vec = pts[:, 0] ** 2 + pts[:, 1] ** 2

    try:
        result, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
    except np.linalg.LinAlgError:
        return (cx, cy, r)

    a, b, c = result
    ncx = a / 2.0
    ncy = b / 2.0
    nr = math.sqrt(max(c + ncx ** 2 + ncy ** 2, 0))

    # Sanity check: center shouldn't move too far, radius shouldn't change drastically
    center_shift = math.sqrt((ncx - cx) ** 2 + (ncy - cy) ** 2)
    if center_shift > search_radius * 2 or nr < r * 0.5 or nr > r * 1.5:
        return (cx, cy, r)

    return (float(ncx), float(ncy), float(nr))


# =========================================================================
# CLI entry point for standalone testing
# =========================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python image_vectorizer.py <image_path> [--debug <output_dir>] [dpi] [scale]")
        print("Example: python image_vectorizer.py C:/plans/floor1.tif --debug ./debug_output 300 1.0")
        print("\nThe --debug flag saves checkpoint images at each processing step.")
        sys.exit(1)

    # Parse args
    args = sys.argv[1:]
    debug_dir = None
    if "--debug" in args:
        idx = args.index("--debug")
        if idx + 1 < len(args):
            debug_dir = args[idx + 1]
            args = args[:idx] + args[idx + 2:]
        else:
            print("Error: --debug requires an output directory")
            sys.exit(1)

    path = args[0] if args else None
    if not path:
        print("Error: image_path is required")
        sys.exit(1)

    dpi_arg = int(args[1]) if len(args) > 1 else 300
    scale_arg = float(args[2]) if len(args) > 2 else 1.0

    if debug_dir:
        print(f"[debug] Checkpoint images will be saved to: {debug_dir}")

    result = vectorize_bitonal_image(path, dpi=dpi_arg, scale=scale_arg, debug_output_dir=debug_dir)

    print("\n=== Vectorization Summary ===")
    print(f"  Image: {path}")
    print(f"  Size:  {result.image_width_px} x {result.image_height_px} px")
    print(f"  Lines:     {len(result.lines)}")
    print(f"  Circles:   {len(result.circles)}")
    print(f"  Arcs:      {len(result.arcs)}")
    print(f"  Ellipses:  {len(result.ellipses)}")
    print(f"  Polylines: {len(result.polylines)}")
    total = (len(result.lines) + len(result.circles) + len(result.arcs)
             + len(result.ellipses) + len(result.polylines))
    print(f"  TOTAL:     {total}")

    if result.lines:
        print("\n--- Sample lines (first 5) ---")
        for i, ln in enumerate(result.lines[:5]):
            print(f"  [{i}] ({ln.start[0]:.1f}, {ln.start[1]:.1f}) -> "
                  f"({ln.end[0]:.1f}, {ln.end[1]:.1f})")
    if result.circles:
        print("\n--- Sample circles (first 5) ---")
        for i, c in enumerate(result.circles[:5]):
            print(f"  [{i}] center=({c.center[0]:.1f}, {c.center[1]:.1f}), r={c.radius:.1f}")
    if result.arcs:
        print("\n--- Sample arcs (first 5) ---")
        for i, a in enumerate(result.arcs[:5]):
            print(f"  [{i}] center=({a.center[0]:.1f}, {a.center[1]:.1f}), r={a.radius:.1f}, "
                  f"{a.start_angle:.1f}°-{a.end_angle:.1f}°")
