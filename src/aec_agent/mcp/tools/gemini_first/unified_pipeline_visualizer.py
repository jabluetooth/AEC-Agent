"""
Debug Visualizer.

Extracted from unified_pipeline.py as part of a pure structural move refactor
(no logic changes). Helper class for saving debug screenshots at each
pipeline stage.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np
import structlog

if TYPE_CHECKING:
    from PIL import Image as PILImage
    from .gemini_understanding import DrawingAnalysis
    from .coordinate_calibration import ScaleCalibration
    from .adaptive_extraction import EntityToCreate

logger = structlog.get_logger(__name__)


class DebugVisualizer:
    """Helper class for saving debug screenshots at each pipeline stage."""

    def __init__(self, output_dir: Path, enabled: bool = True):
        self.output_dir = output_dir
        self.enabled = enabled
        self.step_counter = 0

        if enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info("debug_visualizer_initialized", output_dir=str(output_dir))

    def save_image(self, image: np.ndarray, stage: str, suffix: str = "") -> Optional[str]:
        """Save an image with stage name and step counter."""
        if not self.enabled:
            return None

        try:
            import cv2
            self.step_counter += 1
            filename = f"{self.step_counter:02d}_{stage}"
            if suffix:
                filename += f"_{suffix}"
            filename += ".png"

            filepath = self.output_dir / filename
            cv2.imwrite(str(filepath), image)
            logger.info("debug_image_saved", stage=stage, path=str(filepath))
            return str(filepath)
        except Exception as e:
            logger.warning("debug_image_save_failed", stage=stage, error=str(e))
            return None

    def save_with_overlay(
        self,
        image: np.ndarray,
        stage: str,
        lines: Optional[List] = None,
        circles: Optional[List] = None,
        text_regions: Optional[List] = None,
        junctions: Optional[List] = None,
    ) -> Optional[str]:
        """Save image with detected elements overlaid."""
        if not self.enabled:
            return None

        try:
            import cv2

            # Convert to color if grayscale
            if len(image.shape) == 2:
                vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                vis = image.copy()

            # Draw lines in green
            if lines:
                for line in lines:
                    if hasattr(line, 'start') and hasattr(line, 'end'):
                        start = (int(line.start[0]), int(line.start[1]))
                        end = (int(line.end[0]), int(line.end[1]))
                        # Color based on line type
                        color = (0, 255, 0)  # Green for continuous
                        if hasattr(line, 'line_type'):
                            lt = line.line_type.value if hasattr(line.line_type, 'value') else str(line.line_type)
                            if lt == "dashed":
                                color = (0, 165, 255)  # Orange
                            elif lt == "dotted":
                                color = (255, 0, 255)  # Magenta
                            elif lt == "center":
                                color = (255, 255, 0)  # Cyan
                        cv2.line(vis, start, end, color, 2)

            # Draw circles in blue
            if circles:
                for circle in circles:
                    if hasattr(circle, 'center') and hasattr(circle, 'radius'):
                        center = (int(circle.center[0]), int(circle.center[1]))
                        radius = int(circle.radius)
                        cv2.circle(vis, center, radius, (255, 0, 0), 2)

            # Draw text regions in yellow
            if text_regions:
                for region in text_regions:
                    if isinstance(region, dict) and 'bounds' in region:
                        x1, y1, x2, y2 = [int(v) for v in region['bounds']]
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    elif hasattr(region, 'position'):
                        pos = region.position
                        x, y = int(pos[0]), int(pos[1])
                        cv2.rectangle(vis, (x-5, y-5), (x+50, y+15), (0, 255, 255), 2)

            # Draw junctions in red
            if junctions:
                for junc in junctions:
                    if hasattr(junc, 'position'):
                        pos = (int(junc.position[0]), int(junc.position[1]))
                        cv2.circle(vis, pos, 5, (0, 0, 255), -1)

            return self.save_image(vis, stage, "overlay")
        except Exception as e:
            logger.warning("debug_overlay_save_failed", stage=stage, error=str(e))
            return None
