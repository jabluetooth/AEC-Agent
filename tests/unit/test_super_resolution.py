"""
Unit tests for Real-ESRGAN Super-Resolution Module.

Tests the neural super-resolution capabilities for the Gemini-First pipeline.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from aec_agent.mcp.tools.gemini_first.super_resolution import (
    SuperResolutionConfig,
    SuperResolutionResult,
    RealESRGANUpscaler,
    upscale_image,
    is_super_resolution_available,
    is_gpu_available,
    REALESRGAN_AVAILABLE,
    TORCH_AVAILABLE,
)


class TestSuperResolutionConfig:
    """Tests for SuperResolutionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SuperResolutionConfig()
        assert config.scale == 4
        assert config.model_name == "RealESRGAN_x4plus"
        assert config.tile_size == 192
        assert config.tile_pad == 10
        assert config.gpu_id is None  # Auto-detect
        assert config.denoise_strength == 0.5
        assert config.half_precision is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SuperResolutionConfig(
            scale=2,
            model_name="RealESRGAN_x2plus",
            tile_size=256,
            gpu_id=-1,  # CPU only
        )
        assert config.scale == 2
        assert config.model_name == "RealESRGAN_x2plus"
        assert config.tile_size == 256
        assert config.gpu_id == -1

    def test_config_from_settings(self):
        """Test creating config from application settings."""
        with patch("aec_agent.mcp.tools.gemini_first.super_resolution.get_settings") as mock_settings:
            mock_settings.return_value = Mock(
                super_resolution_scale=2,
                super_resolution_model="RealESRGAN_x2plus",
                super_resolution_tile_size=128,
                super_resolution_gpu_id=-1,
                super_resolution_denoise_strength=0.3,
            )
            config = SuperResolutionConfig.from_settings()
            assert config.scale == 2
            assert config.model_name == "RealESRGAN_x2plus"
            assert config.tile_size == 128
            assert config.gpu_id == -1


class TestSuperResolutionResult:
    """Tests for SuperResolutionResult dataclass."""

    def test_result_creation(self):
        """Test creating a result."""
        image = np.zeros((400, 400, 3), dtype=np.uint8)
        result = SuperResolutionResult(
            image=image,
            original_size=(100, 100),
            upscaled_size=(400, 400),
            scale_factor=4,
            device_used="cuda:0",
            processing_time_ms=1500.5,
        )
        assert result.original_size == (100, 100)
        assert result.upscaled_size == (400, 400)
        assert result.scale_factor == 4
        assert result.device_used == "cuda:0"
        assert result.processing_time_ms == 1500.5


class TestRealESRGANUpscaler:
    """Tests for RealESRGANUpscaler class."""

    def setup_method(self):
        """Reset singleton before each test."""
        RealESRGANUpscaler.reset_instance()

    def test_singleton_pattern(self):
        """Test that get_instance returns the same instance."""
        instance1 = RealESRGANUpscaler.get_instance()
        instance2 = RealESRGANUpscaler.get_instance()
        assert instance1 is instance2

    def test_singleton_with_different_config(self):
        """Test that different config creates new instance."""
        config1 = SuperResolutionConfig(scale=4)
        config2 = SuperResolutionConfig(scale=2)

        instance1 = RealESRGANUpscaler.get_instance(config1)
        instance2 = RealESRGANUpscaler.get_instance(config2)

        # Different config should create new instance
        assert instance1 is not instance2

    def test_is_available_no_deps(self):
        """Test availability when dependencies not installed."""
        with patch(
            "aec_agent.mcp.tools.gemini_first.super_resolution.TORCH_AVAILABLE",
            False,
        ):
            with patch(
                "aec_agent.mcp.tools.gemini_first.super_resolution.REALESRGAN_AVAILABLE",
                False,
            ):
                upscaler = RealESRGANUpscaler()
                assert upscaler.is_available is False

    def test_device_detection_cpu_explicit(self):
        """Test explicit CPU device selection."""
        config = SuperResolutionConfig(gpu_id=-1)
        upscaler = RealESRGANUpscaler(config)
        assert upscaler.device == "cpu"

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_device_detection_auto(self):
        """Test automatic device detection."""
        config = SuperResolutionConfig(gpu_id=None)
        upscaler = RealESRGANUpscaler(config)
        # Should return either "cpu" or "cuda:X"
        assert upscaler.device.startswith("cpu") or upscaler.device.startswith("cuda")

    def test_upscale_empty_image(self):
        """Test upscaling an empty image."""
        upscaler = RealESRGANUpscaler()
        result = upscaler.upscale(np.array([]))

        assert result.scale_factor == upscaler._config.scale
        assert result.original_size == (0, 0)

    def test_upscale_no_model(self):
        """Test upscaling when model not available."""
        with patch.object(
            RealESRGANUpscaler,
            "_ensure_model_loaded",
            return_value=False,
        ):
            upscaler = RealESRGANUpscaler()
            image = np.zeros((100, 100), dtype=np.uint8)
            result = upscaler.upscale(image)

            # Should return original image unchanged
            assert result.scale_factor == 1
            assert result.original_size == (100, 100)

    @pytest.mark.skipif(not REALESRGAN_AVAILABLE, reason="Real-ESRGAN not installed")
    @pytest.mark.requires_realesrgan
    def test_upscale_grayscale_image(self):
        """Test upscaling a grayscale image."""
        upscaler = RealESRGANUpscaler.get_instance()

        if not upscaler.is_available:
            pytest.skip("Real-ESRGAN not available")

        # Create a simple grayscale test image
        image = np.zeros((50, 50), dtype=np.uint8)
        image[10:40, 10:40] = 255  # White square

        result = upscaler.upscale(image, outscale=2)

        assert result.scale_factor == 2
        assert result.upscaled_size == (100, 100)
        # Output should still be grayscale
        assert len(result.image.shape) == 2

    @pytest.mark.skipif(not REALESRGAN_AVAILABLE, reason="Real-ESRGAN not installed")
    @pytest.mark.requires_realesrgan
    def test_upscale_rgb_image(self):
        """Test upscaling an RGB image."""
        upscaler = RealESRGANUpscaler.get_instance()

        if not upscaler.is_available:
            pytest.skip("Real-ESRGAN not available")

        # Create a simple RGB test image
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        image[10:40, 10:40, 0] = 255  # Red square

        result = upscaler.upscale(image, outscale=2)

        assert result.scale_factor == 2
        assert result.upscaled_size == (100, 100)
        assert result.image.shape[2] == 3


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_is_super_resolution_available(self):
        """Test the is_super_resolution_available function."""
        result = is_super_resolution_available()
        assert isinstance(result, bool)
        # Should be True only if both torch and realesrgan are available
        expected = REALESRGAN_AVAILABLE and TORCH_AVAILABLE
        assert result == expected

    def test_is_gpu_available_no_torch(self):
        """Test GPU availability when PyTorch not installed."""
        with patch(
            "aec_agent.mcp.tools.gemini_first.super_resolution.TORCH_AVAILABLE",
            False,
        ):
            from aec_agent.mcp.tools.gemini_first import super_resolution
            # Need to reimport to pick up the patched value
            result = super_resolution.is_gpu_available()
            assert result is False

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_is_gpu_available_with_torch(self):
        """Test GPU availability with PyTorch installed."""
        result = is_gpu_available()
        # Returns True if CUDA is available, False otherwise
        assert isinstance(result, bool)

    def test_upscale_image_convenience(self):
        """Test the upscale_image convenience function."""
        with patch.object(
            RealESRGANUpscaler,
            "upscale",
            return_value=SuperResolutionResult(
                image=np.zeros((200, 200, 3), dtype=np.uint8),
                original_size=(100, 100),
                upscaled_size=(200, 200),
                scale_factor=2,
                device_used="cpu",
            ),
        ):
            RealESRGANUpscaler.reset_instance()
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            result = upscale_image(image, scale=2)

            assert result.scale_factor == 2
            assert result.upscaled_size == (200, 200)


class TestModelDownload:
    """Tests for model download functionality."""

    def setup_method(self):
        """Reset singleton before each test."""
        RealESRGANUpscaler.reset_instance()

    def test_get_model_path_cached(self, tmp_path):
        """Test getting model path when model is cached."""
        with patch("aec_agent.mcp.tools.gemini_first.super_resolution.get_settings") as mock_settings:
            mock_settings.return_value = Mock(
                model_cache_dir=tmp_path,
                super_resolution_scale=4,
                super_resolution_model="RealESRGAN_x4plus",
                super_resolution_tile_size=192,
                super_resolution_gpu_id=-1,
                super_resolution_denoise_strength=0.5,
            )

            # Create fake model file
            model_file = tmp_path / "RealESRGAN_x4plus.pth"
            model_file.touch()

            upscaler = RealESRGANUpscaler()
            path = upscaler._get_model_path()

            assert path == model_file

    def test_unknown_model_error(self):
        """Test error for unknown model name."""
        config = SuperResolutionConfig(model_name="UnknownModel")
        upscaler = RealESRGANUpscaler(config)

        with pytest.raises(ValueError, match="Unknown model"):
            upscaler._get_model_path()


class TestImageFormats:
    """Tests for handling different image formats."""

    def setup_method(self):
        """Reset singleton before each test."""
        RealESRGANUpscaler.reset_instance()

    def test_grayscale_2d_input(self):
        """Test handling 2D grayscale input."""
        upscaler = RealESRGANUpscaler()

        with patch.object(upscaler, "_ensure_model_loaded", return_value=False):
            image = np.zeros((100, 100), dtype=np.uint8)
            result = upscaler.upscale(image)
            # Should handle gracefully
            assert result.original_size == (100, 100)

    def test_grayscale_3d_input(self):
        """Test handling 3D grayscale input (H, W, 1)."""
        upscaler = RealESRGANUpscaler()

        with patch.object(upscaler, "_ensure_model_loaded", return_value=False):
            image = np.zeros((100, 100, 1), dtype=np.uint8)
            result = upscaler.upscale(image)
            assert result.original_size == (100, 100)

    def test_rgba_input(self):
        """Test handling RGBA input (drops alpha)."""
        upscaler = RealESRGANUpscaler()

        with patch.object(upscaler, "_ensure_model_loaded", return_value=False):
            image = np.zeros((100, 100, 4), dtype=np.uint8)
            result = upscaler.upscale(image)
            assert result.original_size == (100, 100)

    def test_invalid_shape(self):
        """Test handling invalid image shape."""
        upscaler = RealESRGANUpscaler()

        with patch.object(upscaler, "_ensure_model_loaded", return_value=True):
            # 5-channel image is invalid
            image = np.zeros((100, 100, 5), dtype=np.uint8)
            result = upscaler.upscale(image)
            assert result.original_size == (0, 0)
