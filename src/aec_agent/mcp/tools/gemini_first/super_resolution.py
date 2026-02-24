"""
Real-ESRGAN Super-Resolution Module for Hybrid Pipeline.

This module provides neural super-resolution using Real-ESRGAN to enhance
low-DPI PDF renders before vectorization, improving both Gemini analysis
accuracy and OCR text detection.

Key Capabilities:
- 2x, 3x, or 4x upscaling with edge preservation
- GPU acceleration with CUDA auto-detection
- CPU fallback for environments without GPU
- Tile-based processing for memory efficiency
- Automatic model download and caching

When to Use:
- PDFs rendered below 200 DPI
- Scanned documents with poor quality
- Small text that's hard to detect
- Fine line details being lost

Usage:
    >>> upscaler = RealESRGANUpscaler.get_instance()
    >>> if upscaler.is_available:
    ...     enhanced = upscaler.upscale(image)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Optional, Tuple
from urllib.request import urlretrieve

import numpy as np
import structlog

from aec_agent.config.settings import get_settings

logger = structlog.get_logger(__name__)

# Try to import Real-ESRGAN and dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    logger.debug("PyTorch not available")

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False
    RealESRGANer = None  # type: ignore
    RRDBNet = None  # type: ignore
    logger.warning(
        "Real-ESRGAN not available. Install with: pip install realesrgan basicsr",
        install_cmd="pip install 'aec-agent[phase_b_gpu]'",
    )

# Model download URLs (official Real-ESRGAN releases)
MODEL_URLS = {
    "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "RealESRNet_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
}


@dataclass
class SuperResolutionConfig:
    """Configuration for Real-ESRGAN super-resolution.

    Attributes:
        scale: Upscale factor (2, 3, or 4)
        model_name: Model variant to use
        tile_size: Tile size for memory-efficient processing (0 = no tiling)
        tile_pad: Overlap between tiles to avoid seams
        gpu_id: GPU device ID (None = auto-detect, -1 = CPU only)
        denoise_strength: Denoising strength (0 = none, 1 = max)
        half_precision: Use FP16 for faster inference (GPU only)
    """
    scale: int = 4
    model_name: str = "RealESRGAN_x4plus"
    tile_size: int = 192
    tile_pad: int = 10
    gpu_id: Optional[int] = None  # None = auto-detect
    denoise_strength: float = 0.5
    half_precision: bool = True  # Use FP16 on GPU

    @classmethod
    def from_settings(cls) -> SuperResolutionConfig:
        """Create config from application settings."""
        settings = get_settings()
        return cls(
            scale=settings.super_resolution_scale,
            model_name=settings.super_resolution_model,
            tile_size=settings.super_resolution_tile_size,
            gpu_id=settings.super_resolution_gpu_id,
            denoise_strength=settings.super_resolution_denoise_strength,
        )


@dataclass
class SuperResolutionResult:
    """Result of super-resolution upscaling."""
    image: np.ndarray
    original_size: Tuple[int, int]  # (width, height)
    upscaled_size: Tuple[int, int]
    scale_factor: int
    device_used: str  # "cuda" or "cpu"
    processing_time_ms: float = 0.0


class RealESRGANUpscaler:
    """Singleton Real-ESRGAN model manager.

    Uses lazy initialization to load the model only when needed.
    Supports both GPU (CUDA) and CPU inference.

    Example:
        >>> upscaler = RealESRGANUpscaler.get_instance()
        >>> if upscaler.is_available:
        ...     result = upscaler.upscale(image)
        ...     print(f"Upscaled to {result.upscaled_size}")
    """

    _instance: ClassVar[Optional[RealESRGANUpscaler]] = None
    _model: Any = None
    _config: SuperResolutionConfig
    _device: str = "cpu"
    _model_loaded: bool = False

    def __init__(self, config: Optional[SuperResolutionConfig] = None):
        """Initialize the upscaler.

        Args:
            config: Super-resolution configuration
        """
        self._config = config or SuperResolutionConfig.from_settings()
        self._model = None
        self._model_loaded = False
        self._device = self._detect_device()

    @classmethod
    def get_instance(
        cls,
        config: Optional[SuperResolutionConfig] = None,
    ) -> RealESRGANUpscaler:
        """Get or create the singleton upscaler instance.

        Args:
            config: Optional config override

        Returns:
            RealESRGANUpscaler instance
        """
        if cls._instance is None:
            cls._instance = cls(config)
        elif config is not None:
            # Config changed, recreate
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None

    @property
    def is_available(self) -> bool:
        """Check if Real-ESRGAN can be used."""
        if not TORCH_AVAILABLE:
            return False
        if not REALESRGAN_AVAILABLE:
            return False
        return True

    @property
    def device(self) -> str:
        """Get the device being used."""
        return self._device

    @property
    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available."""
        if not TORCH_AVAILABLE:
            return False
        return torch.cuda.is_available()

    def _detect_device(self) -> str:
        """Detect the best available device."""
        if not TORCH_AVAILABLE:
            return "cpu"

        gpu_id = self._config.gpu_id

        # Explicit CPU request
        if gpu_id == -1:
            logger.info("super_resolution_device", device="cpu", reason="explicit_cpu_request")
            return "cpu"

        # Auto-detect or specific GPU
        if torch.cuda.is_available():
            if gpu_id is None:
                gpu_id = 0  # Default to first GPU
            if gpu_id < torch.cuda.device_count():
                device = f"cuda:{gpu_id}"
                logger.info(
                    "super_resolution_device",
                    device=device,
                    gpu_name=torch.cuda.get_device_name(gpu_id),
                )
                return device
            else:
                logger.warning(
                    "super_resolution_gpu_not_found",
                    requested_gpu=gpu_id,
                    available_gpus=torch.cuda.device_count(),
                )

        logger.info("super_resolution_device", device="cpu", reason="no_gpu_available")
        return "cpu"

    def _get_model_path(self) -> Path:
        """Get the path to the model file, downloading if necessary."""
        settings = get_settings()
        cache_dir = settings.model_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_name = self._config.model_name
        model_file = cache_dir / f"{model_name}.pth"

        if model_file.exists():
            logger.debug("super_resolution_model_cached", path=str(model_file))
            return model_file

        # Download model
        if model_name not in MODEL_URLS:
            logger.error("super_resolution_unknown_model", model=model_name)
            raise ValueError(f"Unknown model: {model_name}")

        url = MODEL_URLS[model_name]
        logger.info(
            "super_resolution_downloading_model",
            model=model_name,
            url=url,
            destination=str(model_file),
        )

        try:
            urlretrieve(url, model_file)
            logger.info("super_resolution_model_downloaded", path=str(model_file))
            return model_file
        except Exception as e:
            logger.error("super_resolution_download_failed", error=str(e))
            raise

    def _ensure_model_loaded(self) -> bool:
        """Ensure the model is loaded and ready."""
        if self._model_loaded and self._model is not None:
            return True

        if not self.is_available:
            logger.warning("super_resolution_not_available")
            return False

        try:
            model_path = self._get_model_path()

            # Create the network architecture
            scale = self._config.scale
            if scale == 4:
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=4,
                )
            elif scale == 2:
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=2,
                )
            else:
                # Default to 4x model
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=4,
                )

            # Determine if we should use half precision
            use_half = self._config.half_precision and self._device.startswith("cuda")

            # Create the upscaler
            self._model = RealESRGANer(
                scale=scale,
                model_path=str(model_path),
                dni_weight=self._config.denoise_strength,
                model=model,
                tile=self._config.tile_size,
                tile_pad=self._config.tile_pad,
                pre_pad=0,
                half=use_half,
                device=self._device,
            )

            self._model_loaded = True
            logger.info(
                "super_resolution_model_loaded",
                model=self._config.model_name,
                scale=scale,
                device=self._device,
                half_precision=use_half,
            )
            return True

        except Exception as e:
            logger.error("super_resolution_model_load_failed", error=str(e), exc_info=True)
            return False

    def upscale(
        self,
        image: np.ndarray,
        outscale: Optional[int] = None,
    ) -> SuperResolutionResult:
        """Upscale an image using Real-ESRGAN.

        Args:
            image: Input image as numpy array (HxWxC, RGB or grayscale)
            outscale: Optional output scale override

        Returns:
            SuperResolutionResult with upscaled image
        """
        import time

        start_time = time.perf_counter()
        scale = outscale or self._config.scale

        # Validate input
        if image is None or image.size == 0:
            logger.error("super_resolution_empty_input")
            return SuperResolutionResult(
                image=image,
                original_size=(0, 0),
                upscaled_size=(0, 0),
                scale_factor=scale,
                device_used=self._device,
            )

        # Get original size
        if len(image.shape) == 2:
            h, w = image.shape
            # Convert grayscale to RGB for Real-ESRGAN
            image_rgb = np.stack([image, image, image], axis=-1)
            was_grayscale = True
        elif len(image.shape) == 3 and image.shape[2] == 1:
            h, w = image.shape[:2]
            image_rgb = np.concatenate([image, image, image], axis=-1)
            was_grayscale = True
        elif len(image.shape) == 3 and image.shape[2] == 3:
            h, w = image.shape[:2]
            image_rgb = image
            was_grayscale = False
        elif len(image.shape) == 3 and image.shape[2] == 4:
            h, w = image.shape[:2]
            image_rgb = image[:, :, :3]  # Drop alpha
            was_grayscale = False
        else:
            logger.error("super_resolution_invalid_input", shape=image.shape)
            return SuperResolutionResult(
                image=image,
                original_size=(0, 0),
                upscaled_size=(0, 0),
                scale_factor=scale,
                device_used=self._device,
            )

        original_size = (w, h)

        # Ensure model is loaded
        if not self._ensure_model_loaded():
            logger.warning("super_resolution_model_not_loaded", returning_original=True)
            return SuperResolutionResult(
                image=image,
                original_size=original_size,
                upscaled_size=original_size,
                scale_factor=1,
                device_used=self._device,
            )

        try:
            # Run inference
            output, _ = self._model.enhance(image_rgb, outscale=scale)

            # Convert back to grayscale if input was grayscale
            if was_grayscale:
                output = np.mean(output, axis=-1).astype(np.uint8)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            new_h, new_w = output.shape[:2]
            upscaled_size = (new_w, new_h)

            logger.info(
                "super_resolution_complete",
                original_size=original_size,
                upscaled_size=upscaled_size,
                scale=scale,
                device=self._device,
                time_ms=round(elapsed_ms, 2),
            )

            return SuperResolutionResult(
                image=output,
                original_size=original_size,
                upscaled_size=upscaled_size,
                scale_factor=scale,
                device_used=self._device,
                processing_time_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error("super_resolution_failed", error=str(e), exc_info=True)
            return SuperResolutionResult(
                image=image,
                original_size=original_size,
                upscaled_size=original_size,
                scale_factor=1,
                device_used=self._device,
            )

    def upscale_file(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        outscale: Optional[int] = None,
    ) -> Path:
        """Upscale an image file.

        Args:
            input_path: Path to input image
            output_path: Path for output (auto-generated if None)
            outscale: Optional output scale override

        Returns:
            Path to the upscaled image
        """
        try:
            from PIL import Image
        except ImportError:
            logger.error("PIL required for file upscaling")
            return input_path

        # Load image
        img = Image.open(input_path)
        image = np.array(img)

        # Upscale
        result = self.upscale(image, outscale)

        # Generate output path if not provided
        if output_path is None:
            stem = input_path.stem
            suffix = input_path.suffix
            output_path = input_path.parent / f"{stem}_upscaled{result.scale_factor}x{suffix}"

        # Save result
        output_img = Image.fromarray(result.image)
        output_img.save(output_path)

        logger.info(
            "super_resolution_file_saved",
            input=str(input_path),
            output=str(output_path),
            scale=result.scale_factor,
        )

        return output_path


def upscale_image(
    image: np.ndarray,
    scale: int = 4,
    config: Optional[SuperResolutionConfig] = None,
) -> SuperResolutionResult:
    """Convenience function to upscale an image.

    Args:
        image: Input image
        scale: Upscale factor
        config: Optional configuration

    Returns:
        SuperResolutionResult
    """
    upscaler = RealESRGANUpscaler.get_instance(config)
    return upscaler.upscale(image, outscale=scale)


def is_super_resolution_available() -> bool:
    """Check if Real-ESRGAN is available."""
    return REALESRGAN_AVAILABLE and TORCH_AVAILABLE


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    if not TORCH_AVAILABLE:
        return False
    return torch.cuda.is_available()
