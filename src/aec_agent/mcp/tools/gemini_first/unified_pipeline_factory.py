"""
Extraction Factory.

Extracted from unified_pipeline.py as part of a pure structural move refactor
(no logic changes). Provides a unified interface to the different extraction
strategies (direct, hybrid, best, vtracer).
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

from .unified_pipeline import ExtractionMethod, PipelineConfig


class ExtractionFactory:
    """
    Factory for creating extraction strategies.

    Provides a unified interface to different extraction methods:
    - Direct: Gemini coordinates only
    - Hybrid: Gemini + OpenCV
    - Best: Optimal algorithm per entity type
    - VTracer: Raster-to-vector
    """

    @staticmethod
    async def extract(
        method: ExtractionMethod,
        image_path: str,
        analysis: Any,  # DrawingAnalysis
        calibration: Any,  # ScaleCalibration
        config: PipelineConfig,
    ) -> List[Any]:
        """
        Extract entities using the specified method.

        Args:
            method: Extraction method to use
            image_path: Path to the rendered PDF image
            analysis: Gemini drawing analysis
            calibration: Coordinate calibration
            config: Pipeline configuration

        Returns:
            List of EntityToCreate objects
        """
        from .adaptive_extraction import (
            extract_all,
            extract_direct_only,
            hybrid_extract_all,
            HybridExtractionConfig,
        )

        if method == ExtractionMethod.DIRECT:
            # Fast: Gemini coordinates only
            result = await extract_direct_only(
                analysis=analysis,
                calibration=calibration,
            )
            return result.entities

        elif method == ExtractionMethod.HYBRID:
            # Balanced: Gemini + OpenCV
            hybrid_config = HybridExtractionConfig(
                use_opencv_for_lines=True,
                use_opencv_for_circles=True,
                use_yolo_for_symbols=False,
                use_gemini_for_text=config.extract_text,
                use_gemini_for_semantic=True,
            )
            result = await hybrid_extract_all(
                image_path=image_path,
                analysis=analysis,
                calibration=calibration,
                config=hybrid_config,
            )
            return result.entities

        elif method == ExtractionMethod.BEST:
            # Highest quality: Best algorithm per entity type
            hybrid_config = HybridExtractionConfig(
                use_opencv_for_lines=True,
                use_opencv_for_circles=True,
                use_yolo_for_symbols=config.extract_symbols,
                use_gemini_for_text=config.extract_text,
                use_gemini_for_semantic=True,
                merge_opencv_gemini=True,
            )
            result = await hybrid_extract_all(
                image_path=image_path,
                analysis=analysis,
                calibration=calibration,
                config=hybrid_config,
            )
            return result.entities

        elif method == ExtractionMethod.VTRACER:
            # Raster-to-vector for scanned drawings
            from .vtracer_extraction import VTracerExtractor, is_vtracer_available

            if not is_vtracer_available():
                logger.warning("VTracer not available, falling back to hybrid")
                return await ExtractionFactory.extract(
                    ExtractionMethod.HYBRID,
                    image_path,
                    analysis,
                    calibration,
                    config,
                )

            from PIL import Image
            img = Image.open(image_path)
            extractor = VTracerExtractor(np.array(img))
            vtracer_result = extractor.extract()

            # Convert VTracer paths to entities
            entities = extractor.paths_to_entities(
                vtracer_result.paths,
                calibration,
            )
            return entities

        else:
            raise ValueError(f"Unknown extraction method: {method}")
