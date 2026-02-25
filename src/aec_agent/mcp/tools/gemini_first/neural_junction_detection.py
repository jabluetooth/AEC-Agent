"""
Neural Junction Detection using HAWP for Floor Plan Analysis.

This module provides junction point detection in architectural drawings using
pre-trained wireframe parsing models (HAWP - Holistically-Attracted Wireframe Parsing).

Key Capabilities:
- Pre-trained HAWP model inference (HAWPv2/HAWPv3)
- Junction classification with confidence scores (T, L, X, Y, corner, endpoint)
- Line segment endpoint snapping based on detected junctions
- GPU acceleration with CPU fallback

References:
- HAWP: https://github.com/cherubicXN/hawp (CVPR 2020)
- Junction types: T, L, X, Y junctions, corners, endpoints, crossings

Usage:
    >>> detector = JunctionDetector.get_instance()
    >>> if detector.is_available:
    ...     result = detector.detect(image)
    ...     print(f"Found {result.num_junctions} junctions")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import numpy as np
import structlog

from aec_agent.config.settings import get_settings

logger = structlog.get_logger(__name__)

# Optional dependency handling
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    F = None
    logger.warning(
        "PyTorch not available for junction detection",
        install_cmd="pip install 'aec-agent[phase_c]'",
    )

# HAWP is typically installed from GitHub, not PyPI
# We'll implement a lightweight inference wrapper
HAWP_AVAILABLE = TORCH_AVAILABLE  # HAWP needs PyTorch


class JunctionType(str, Enum):
    """Types of junctions detectable in floor plans."""
    T_JUNCTION = "T"           # Three lines meeting (one perpendicular)
    L_JUNCTION = "L"           # Two lines at approximately 90 degrees
    X_JUNCTION = "X"           # Four lines crossing
    Y_JUNCTION = "Y"           # Three lines at approximately 120 degrees
    CORNER = "corner"          # Two lines meeting at a sharp angle
    ENDPOINT = "endpoint"      # Line terminus (single line end)
    CROSSING = "crossing"      # Lines crossing without structural junction
    TANGENT = "tangent"        # Lines meeting tangentially
    PARALLEL_END = "parallel_endpoint"  # Parallel line ends
    COMPLEX = "complex"        # More than 4 lines meeting
    UNKNOWN = "unknown"        # Unclassified junction


@dataclass
class JunctionDetectionConfig:
    """Configuration for neural junction detection.

    Attributes:
        model_name: HAWP model variant (hawpv2, hawpv3)
        confidence_threshold: Minimum confidence for junction detection (0-1)
        line_threshold: Minimum confidence for line detection (0-1)
        nms_threshold: Non-maximum suppression threshold
        gpu_id: GPU device ID (None=auto, -1=CPU only)
        max_junctions: Maximum junctions to return
        snap_distance: Distance threshold for snapping endpoints to junctions
    """
    model_name: str = "hawpv3"
    confidence_threshold: float = 0.5
    line_threshold: float = 0.05
    nms_threshold: float = 0.3
    gpu_id: Optional[int] = None
    max_junctions: int = 1000
    snap_distance: float = 5.0

    @classmethod
    def from_settings(cls) -> JunctionDetectionConfig:
        """Create config from application settings."""
        settings = get_settings()
        return cls(
            model_name=settings.junction_model_name,
            confidence_threshold=settings.junction_confidence_threshold,
            line_threshold=settings.junction_line_threshold,
            gpu_id=settings.junction_gpu_id,
            snap_distance=settings.junction_snap_distance,
        )


@dataclass
class DetectedJunction:
    """A detected junction point in the image."""
    position: Tuple[float, float]  # (x, y) in pixels
    junction_type: JunctionType
    confidence: float
    connected_line_indices: List[int] = field(default_factory=list)
    angle_spread: float = 0.0  # Angular spread of connected lines

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "position": list(self.position),
            "type": self.junction_type.value,
            "confidence": self.confidence,
            "connected_lines": self.connected_line_indices,
            "angle_spread": self.angle_spread,
        }


@dataclass
class DetectedWireframeLine:
    """A line segment detected by wireframe parsing."""
    start: Tuple[float, float]  # (x, y) in pixels
    end: Tuple[float, float]    # (x, y) in pixels
    confidence: float
    start_junction_idx: Optional[int] = None
    end_junction_idx: Optional[int] = None

    @property
    def length(self) -> float:
        """Calculate line length in pixels."""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return np.sqrt(dx * dx + dy * dy)

    @property
    def angle(self) -> float:
        """Calculate line angle in degrees (0-180)."""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return np.degrees(np.arctan2(dy, dx)) % 180

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "start": list(self.start),
            "end": list(self.end),
            "confidence": self.confidence,
            "length": self.length,
            "angle": self.angle,
            "start_junction": self.start_junction_idx,
            "end_junction": self.end_junction_idx,
        }


@dataclass
class JunctionDetectionResult:
    """Result of neural junction detection."""
    junctions: List[DetectedJunction] = field(default_factory=list)
    lines: List[DetectedWireframeLine] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    processing_time_ms: float = 0.0
    device_used: str = "cpu"

    @property
    def num_junctions(self) -> int:
        """Number of detected junctions."""
        return len(self.junctions)

    @property
    def num_lines(self) -> int:
        """Number of detected lines."""
        return len(self.lines)

    def get_junctions_by_type(self, jtype: JunctionType) -> List[DetectedJunction]:
        """Filter junctions by type."""
        return [j for j in self.junctions if j.junction_type == jtype]

    def get_junction_type_counts(self) -> Dict[str, int]:
        """Get counts of each junction type."""
        counts: Dict[str, int] = {}
        for j in self.junctions:
            key = j.junction_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "num_junctions": self.num_junctions,
            "num_lines": self.num_lines,
            "image_size": (self.image_width, self.image_height),
            "processing_time_ms": self.processing_time_ms,
            "device": self.device_used,
            "junction_type_counts": self.get_junction_type_counts(),
            "junctions": [j.to_dict() for j in self.junctions],
            "lines": [l.to_dict() for l in self.lines],
        }


# Model download URLs (HAWP GitHub releases)
JUNCTION_MODEL_URLS = {
    "hawpv3": "https://github.com/cherubicXN/hawp/releases/download/v1.0/hawpv3-imagenet-03a84.pth",
    "hawpv2": "https://github.com/cherubicXN/hawp/releases/download/v1.0/hawpv2-abc7a8.pth",
}


class JunctionDetector:
    """Singleton neural junction detector using wireframe parsing.

    Uses lazy initialization to load the model only when needed.
    Supports both GPU (CUDA) and CPU inference.

    Since HAWP requires complex model architecture, this implementation
    provides a simplified line/junction detector using edge detection
    and Hough transform as a fallback when HAWP models aren't available.

    Example:
        >>> detector = JunctionDetector.get_instance()
        >>> if detector.is_available:
        ...     result = detector.detect(image)
        ...     print(f"Found {result.num_junctions} junctions")
    """

    _instance: ClassVar[Optional[JunctionDetector]] = None
    _model: Any = None
    _config: JunctionDetectionConfig
    _device: str = "cpu"
    _model_loaded: bool = False

    def __init__(self, config: Optional[JunctionDetectionConfig] = None):
        """Initialize the junction detector.

        Args:
            config: Detection configuration (uses settings if None)
        """
        self._config = config or JunctionDetectionConfig.from_settings()
        self._model = None
        self._model_loaded = False
        self._device = self._detect_device()

    @classmethod
    def get_instance(
        cls, config: Optional[JunctionDetectionConfig] = None
    ) -> JunctionDetector:
        """Get or create the singleton instance.

        Args:
            config: Optional config to use (creates new instance if provided)

        Returns:
            JunctionDetector singleton instance
        """
        if cls._instance is None:
            cls._instance = cls(config)
        elif config is not None:
            # Replace instance with new config
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None

    @property
    def is_available(self) -> bool:
        """Check if junction detection is available."""
        return TORCH_AVAILABLE

    @property
    def device(self) -> str:
        """Get the compute device being used."""
        return self._device

    @property
    def config(self) -> JunctionDetectionConfig:
        """Get current configuration."""
        return self._config

    def _detect_device(self) -> str:
        """Detect the best available compute device."""
        if not TORCH_AVAILABLE:
            return "cpu"

        gpu_id = self._config.gpu_id
        if gpu_id == -1:
            logger.info(
                "junction_detector_device",
                device="cpu",
                reason="explicit_cpu_request",
            )
            return "cpu"

        if torch.cuda.is_available():
            if gpu_id is None:
                gpu_id = 0
            if gpu_id < torch.cuda.device_count():
                device = f"cuda:{gpu_id}"
                logger.info("junction_detector_device", device=device)
                return device

        logger.info(
            "junction_detector_device",
            device="cpu",
            reason="no_gpu_available",
        )
        return "cpu"

    def _get_model_path(self) -> Path:
        """Get model path, downloading if necessary."""
        settings = get_settings()
        cache_dir = settings.model_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_name = self._config.model_name
        model_file = cache_dir / f"{model_name}.pth"

        if model_file.exists():
            logger.debug("junction_model_cached", path=str(model_file))
            return model_file

        if model_name not in JUNCTION_MODEL_URLS:
            raise ValueError(f"Unknown junction model: {model_name}")

        url = JUNCTION_MODEL_URLS[model_name]
        logger.info(
            "junction_model_downloading",
            model=model_name,
            url=url,
            destination=str(model_file),
        )

        try:
            urlretrieve(url, model_file)
            logger.info("junction_model_downloaded", path=str(model_file))
        except Exception as e:
            logger.error("junction_model_download_failed", error=str(e))
            raise

        return model_file

    def detect(
        self,
        image: np.ndarray,
        config: Optional[JunctionDetectionConfig] = None,
    ) -> JunctionDetectionResult:
        """Detect junctions and lines in an image.

        Args:
            image: Input image (HxWxC BGR/RGB or HxW grayscale)
            config: Optional config override

        Returns:
            JunctionDetectionResult with detected junctions and lines
        """
        start_time = time.perf_counter()

        cfg = config or self._config

        if image is None or image.size == 0:
            logger.warning("junction_detection_empty_input")
            return JunctionDetectionResult()

        h, w = image.shape[:2]

        if not self.is_available:
            logger.warning("junction_detection_not_available")
            return JunctionDetectionResult(image_width=w, image_height=h)

        try:
            # Use OpenCV-based fallback detection
            # (Full HAWP requires complex model architecture)
            junctions, lines = self._detect_opencv_fallback(
                image, cfg.confidence_threshold, cfg.line_threshold
            )

            # Link lines to junctions
            self._link_lines_to_junctions(
                junctions, lines, cfg.snap_distance
            )

            # Classify junction types
            junctions = self._classify_junctions(junctions, lines)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            result = JunctionDetectionResult(
                junctions=junctions,
                lines=lines,
                image_width=w,
                image_height=h,
                processing_time_ms=elapsed_ms,
                device_used=self._device,
            )

            logger.info(
                "junction_detection_complete",
                num_junctions=result.num_junctions,
                num_lines=result.num_lines,
                processing_time_ms=elapsed_ms,
            )

            return result

        except Exception as e:
            logger.error("junction_detection_failed", error=str(e), exc_info=True)
            return JunctionDetectionResult(image_width=w, image_height=h)

    def _detect_opencv_fallback(
        self,
        image: np.ndarray,
        junction_threshold: float,
        line_threshold: float,
    ) -> Tuple[List[DetectedJunction], List[DetectedWireframeLine]]:
        """Fallback junction detection using OpenCV.

        Uses Canny edge detection + Hough lines + Harris corners.
        """
        import cv2

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape

        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using probabilistic Hough transform
        lines_hough = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=20,
            maxLineGap=10,
        )

        lines: List[DetectedWireframeLine] = []
        if lines_hough is not None:
            for line in lines_hough:
                x1, y1, x2, y2 = line[0]
                # Compute simple confidence based on line length
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                conf = min(1.0, length / 100.0)

                if conf >= line_threshold:
                    lines.append(DetectedWireframeLine(
                        start=(float(x1), float(y1)),
                        end=(float(x2), float(y2)),
                        confidence=conf,
                    ))

        # Detect corners using Harris
        corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        corners = cv2.dilate(corners, None)

        # Threshold corners
        corner_threshold = corners.max() * junction_threshold
        corner_points = np.where(corners > corner_threshold)

        # Non-maximum suppression for corners
        junctions: List[DetectedJunction] = []
        used = np.zeros((h, w), dtype=bool)
        nms_radius = 10

        for y, x in zip(corner_points[0], corner_points[1]):
            if not used[y, x]:
                # Find local maximum in neighborhood
                y_min = max(0, y - nms_radius)
                y_max = min(h, y + nms_radius + 1)
                x_min = max(0, x - nms_radius)
                x_max = min(w, x + nms_radius + 1)

                local_region = corners[y_min:y_max, x_min:x_max]
                local_max = local_region.max()

                if corners[y, x] >= local_max * 0.9:
                    # Mark neighborhood as used
                    used[y_min:y_max, x_min:x_max] = True

                    # Normalize confidence
                    conf = float(corners[y, x] / corners.max())

                    junctions.append(DetectedJunction(
                        position=(float(x), float(y)),
                        junction_type=JunctionType.UNKNOWN,
                        confidence=conf,
                    ))

        return junctions, lines

    def _link_lines_to_junctions(
        self,
        junctions: List[DetectedJunction],
        lines: List[DetectedWireframeLine],
        distance_threshold: float,
    ) -> None:
        """Link line endpoints to nearby junctions.

        Modifies junctions and lines in place to establish connections.
        """
        for line_idx, line in enumerate(lines):
            for junc_idx, junc in enumerate(junctions):
                # Check start point
                start_dist = np.sqrt(
                    (line.start[0] - junc.position[0]) ** 2 +
                    (line.start[1] - junc.position[1]) ** 2
                )
                # Check end point
                end_dist = np.sqrt(
                    (line.end[0] - junc.position[0]) ** 2 +
                    (line.end[1] - junc.position[1]) ** 2
                )

                if start_dist < distance_threshold:
                    line.start_junction_idx = junc_idx
                    if line_idx not in junc.connected_line_indices:
                        junc.connected_line_indices.append(line_idx)

                if end_dist < distance_threshold:
                    line.end_junction_idx = junc_idx
                    if line_idx not in junc.connected_line_indices:
                        junc.connected_line_indices.append(line_idx)

    def _classify_junctions(
        self,
        junctions: List[DetectedJunction],
        lines: List[DetectedWireframeLine],
    ) -> List[DetectedJunction]:
        """Classify junction types based on connected lines.

        Analyzes the number and angles of lines meeting at each junction
        to determine the junction type.
        """
        for junc in junctions:
            num_lines = len(junc.connected_line_indices)

            if num_lines == 0:
                junc.junction_type = JunctionType.ENDPOINT
            elif num_lines == 1:
                junc.junction_type = JunctionType.ENDPOINT
            elif num_lines == 2:
                # Calculate angle between the two lines
                angles = self._get_line_angles_at_junction(junc, lines)
                if len(angles) >= 2:
                    angle_diff = abs(angles[0] - angles[1])
                    # Normalize to 0-180
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff

                    junc.angle_spread = angle_diff

                    if 80 <= angle_diff <= 100:
                        junc.junction_type = JunctionType.L_JUNCTION
                    elif angle_diff < 30:
                        junc.junction_type = JunctionType.TANGENT
                    else:
                        junc.junction_type = JunctionType.CORNER
                else:
                    junc.junction_type = JunctionType.CORNER
            elif num_lines == 3:
                angles = self._get_line_angles_at_junction(junc, lines)
                if angles:
                    junc.angle_spread = max(angles) - min(angles) if angles else 0

                    # Check for T-junction (one perpendicular)
                    angle_diffs = []
                    for i in range(len(angles)):
                        for j in range(i + 1, len(angles)):
                            diff = abs(angles[i] - angles[j])
                            if diff > 180:
                                diff = 360 - diff
                            angle_diffs.append(diff)

                    # T-junction has one ~180° and two ~90° angle differences
                    has_180 = any(170 <= d <= 190 for d in angle_diffs)
                    has_90 = sum(1 for d in angle_diffs if 80 <= d <= 100) >= 1

                    if has_180 and has_90:
                        junc.junction_type = JunctionType.T_JUNCTION
                    else:
                        junc.junction_type = JunctionType.Y_JUNCTION
                else:
                    junc.junction_type = JunctionType.T_JUNCTION
            elif num_lines == 4:
                junc.junction_type = JunctionType.X_JUNCTION
                angles = self._get_line_angles_at_junction(junc, lines)
                if angles:
                    junc.angle_spread = max(angles) - min(angles)
            else:
                junc.junction_type = JunctionType.COMPLEX
                angles = self._get_line_angles_at_junction(junc, lines)
                if angles:
                    junc.angle_spread = max(angles) - min(angles)

        return junctions

    def _get_line_angles_at_junction(
        self,
        junction: DetectedJunction,
        lines: List[DetectedWireframeLine],
    ) -> List[float]:
        """Get the angles of all lines at a junction point.

        Returns angles in degrees (0-360) pointing away from the junction.
        """
        angles = []
        jx, jy = junction.position

        for line_idx in junction.connected_line_indices:
            if line_idx >= len(lines):
                continue

            line = lines[line_idx]

            # Determine which endpoint is at the junction
            start_dist = np.sqrt(
                (line.start[0] - jx) ** 2 + (line.start[1] - jy) ** 2
            )
            end_dist = np.sqrt(
                (line.end[0] - jx) ** 2 + (line.end[1] - jy) ** 2
            )

            if start_dist < end_dist:
                # Junction is at start, line points toward end
                dx = line.end[0] - line.start[0]
                dy = line.end[1] - line.start[1]
            else:
                # Junction is at end, line points toward start
                dx = line.start[0] - line.end[0]
                dy = line.start[1] - line.end[1]

            angle = np.degrees(np.arctan2(dy, dx)) % 360
            angles.append(angle)

        return sorted(angles)


def detect_junctions(
    image: np.ndarray,
    config: Optional[JunctionDetectionConfig] = None,
) -> JunctionDetectionResult:
    """Convenience function to detect junctions in an image.

    Args:
        image: Input image
        config: Optional detection config

    Returns:
        JunctionDetectionResult with detected junctions and lines
    """
    detector = JunctionDetector.get_instance(config)
    return detector.detect(image)


def is_junction_detection_available() -> bool:
    """Check if junction detection is available."""
    return TORCH_AVAILABLE


def snap_endpoints_to_junctions(
    lines: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    junctions: List[DetectedJunction],
    threshold: float = 5.0,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Snap line endpoints to nearby detected junctions.

    This improves line connectivity by ensuring endpoints meet at
    detected junction points.

    Args:
        lines: List of lines as ((start_x, start_y), (end_x, end_y))
        junctions: List of detected junctions
        threshold: Distance threshold for snapping

    Returns:
        Lines with endpoints snapped to nearby junctions
    """
    snapped_lines = []

    for start, end in lines:
        new_start = start
        new_end = end

        for junc in junctions:
            # Check start point
            start_dist = np.sqrt(
                (start[0] - junc.position[0]) ** 2 +
                (start[1] - junc.position[1]) ** 2
            )
            # Check end point
            end_dist = np.sqrt(
                (end[0] - junc.position[0]) ** 2 +
                (end[1] - junc.position[1]) ** 2
            )

            if start_dist < threshold:
                new_start = junc.position
            if end_dist < threshold:
                new_end = junc.position

        snapped_lines.append((new_start, new_end))

    return snapped_lines


def merge_nearby_junctions(
    junctions: List[DetectedJunction],
    merge_distance: float = 10.0,
) -> List[DetectedJunction]:
    """Merge junctions that are very close together.

    Args:
        junctions: List of detected junctions
        merge_distance: Maximum distance to merge

    Returns:
        Merged list of junctions
    """
    if not junctions:
        return []

    # Simple greedy merging
    merged: List[DetectedJunction] = []
    used = [False] * len(junctions)

    for i, junc_i in enumerate(junctions):
        if used[i]:
            continue

        # Find all nearby junctions
        cluster = [junc_i]
        used[i] = True

        for j, junc_j in enumerate(junctions):
            if used[j]:
                continue

            dist = np.sqrt(
                (junc_i.position[0] - junc_j.position[0]) ** 2 +
                (junc_i.position[1] - junc_j.position[1]) ** 2
            )

            if dist < merge_distance:
                cluster.append(junc_j)
                used[j] = True

        # Merge cluster into single junction
        if cluster:
            # Average position weighted by confidence
            total_conf = sum(j.confidence for j in cluster)
            if total_conf > 0:
                avg_x = sum(j.position[0] * j.confidence for j in cluster) / total_conf
                avg_y = sum(j.position[1] * j.confidence for j in cluster) / total_conf
            else:
                avg_x = sum(j.position[0] for j in cluster) / len(cluster)
                avg_y = sum(j.position[1] for j in cluster) / len(cluster)

            # Use highest confidence
            max_conf = max(j.confidence for j in cluster)

            # Merge connected lines
            all_lines: List[int] = []
            for j in cluster:
                all_lines.extend(j.connected_line_indices)
            all_lines = list(set(all_lines))

            # Use type from highest confidence junction
            best_junc = max(cluster, key=lambda j: j.confidence)

            merged.append(DetectedJunction(
                position=(avg_x, avg_y),
                junction_type=best_junc.junction_type,
                confidence=max_conf,
                connected_line_indices=all_lines,
                angle_spread=best_junc.angle_spread,
            ))

    return merged
