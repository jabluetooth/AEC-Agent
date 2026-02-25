"""
Pipeline Visualizer for Phase C Advanced Vectorization.

Creates visual comparison screenshots showing:
1. Original input image
2. Junction detection overlay
3. Bezier Splatting result
4. LIVE vectorization result
5. Side-by-side comparison

Usage:
    from aec_agent.mcp.tools.gemini_first.pipeline_visualizer import visualize_pipeline

    # Run visualization on a test image
    result = await visualize_pipeline(
        image_path="path/to/floor_plan.png",
        output_dir="path/to/output",
    )
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class VisualizationResult:
    """Result of pipeline visualization."""

    input_image_path: str
    output_dir: str

    # Individual outputs
    junction_overlay_path: str | None = None
    bezier_svg_path: str | None = None
    bezier_png_path: str | None = None
    live_svg_path: str | None = None
    live_png_path: str | None = None

    # Comparison output
    comparison_path: str | None = None

    # Statistics
    junction_count: int = 0
    junction_time_ms: float = 0.0
    bezier_curve_count: int = 0
    bezier_time_ms: float = 0.0
    live_layer_count: int = 0
    live_path_count: int = 0
    live_time_ms: float = 0.0

    # Errors
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_image": self.input_image_path,
            "output_dir": self.output_dir,
            "outputs": {
                "junction_overlay": self.junction_overlay_path,
                "bezier_svg": self.bezier_svg_path,
                "bezier_png": self.bezier_png_path,
                "live_svg": self.live_svg_path,
                "live_png": self.live_png_path,
                "comparison": self.comparison_path,
            },
            "statistics": {
                "junction_count": self.junction_count,
                "junction_time_ms": self.junction_time_ms,
                "bezier_curve_count": self.bezier_curve_count,
                "bezier_time_ms": self.bezier_time_ms,
                "live_layer_count": self.live_layer_count,
                "live_path_count": self.live_path_count,
                "live_time_ms": self.live_time_ms,
            },
            "errors": self.errors,
        }


def _draw_junctions_on_image(
    image: "Image.Image",
    junctions: list,
    lines: list,
) -> "Image.Image":
    """Draw detected junctions and lines on image.

    Args:
        image: PIL Image to draw on
        junctions: List of DetectedJunction objects
        lines: List of DetectedWireframeLine objects

    Returns:
        Image with overlays
    """
    # Create a copy to draw on
    overlay = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay)

    # Color mapping for junction types
    junction_colors = {
        "T": (255, 0, 0, 200),      # Red
        "L": (0, 255, 0, 200),      # Green
        "X": (0, 0, 255, 200),      # Blue
        "Y": (255, 255, 0, 200),    # Yellow
        "corner": (255, 0, 255, 200),  # Magenta
        "endpoint": (0, 255, 255, 200),  # Cyan
        "crossing": (255, 128, 0, 200),  # Orange
    }

    # Draw lines first (underneath junctions)
    for line in lines:
        start = line.start
        end = line.end
        # Draw line with transparency
        draw.line([start, end], fill=(100, 100, 255, 150), width=2)

    # Draw junctions
    for junction in junctions:
        x, y = junction.position
        jtype = junction.junction_type.value
        color = junction_colors.get(jtype, (128, 128, 128, 200))

        # Draw circle for junction
        radius = 6
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=color,
            outline=(255, 255, 255, 255),
            width=1,
        )

        # Draw junction type label
        try:
            draw.text((x + 8, y - 6), jtype[:1], fill=(255, 255, 255, 255))
        except Exception:
            pass  # Skip if font not available

    return overlay


def _svg_to_png(svg_path: Path, output_path: Path, size: tuple[int, int]) -> bool:
    """Convert SVG to PNG using cairosvg or PIL fallback.

    Args:
        svg_path: Path to SVG file
        output_path: Path for PNG output
        size: (width, height) for output

    Returns:
        True if successful
    """
    try:
        import cairosvg
        cairosvg.svg2png(
            url=str(svg_path),
            write_to=str(output_path),
            output_width=size[0],
            output_height=size[1],
        )
        return True
    except ImportError:
        pass

    # Fallback: Create a placeholder with text
    if PIL_AVAILABLE:
        img = Image.new("RGB", size, "white")
        draw = ImageDraw.Draw(img)
        draw.text(
            (size[0] // 4, size[1] // 2),
            f"SVG: {svg_path.name}\n(Install cairosvg to render)",
            fill="gray",
        )
        img.save(output_path)
        return True

    return False


def _create_comparison_image(
    original: "Image.Image",
    junction_overlay: "Image.Image | None",
    bezier_img: "Image.Image | None",
    live_img: "Image.Image | None",
    output_path: Path,
) -> None:
    """Create a side-by-side comparison image.

    Layout:
    +----------------+----------------+
    |   Original     | Junction Det.  |
    +----------------+----------------+
    | Bezier Splat.  | LIVE Vector.   |
    +----------------+----------------+
    """
    # Get dimensions
    w, h = original.size

    # Create 2x2 grid
    comparison = Image.new("RGB", (w * 2 + 20, h * 2 + 60), "white")
    draw = ImageDraw.Draw(comparison)

    # Paste images
    comparison.paste(original.convert("RGB"), (0, 30))

    if junction_overlay:
        comparison.paste(junction_overlay.convert("RGB"), (w + 20, 30))
    else:
        draw.rectangle([w + 20, 30, w * 2 + 20, h + 30], fill="lightgray")
        draw.text((w + 40, h // 2), "Junction Detection\n(Not available)", fill="gray")

    if bezier_img:
        comparison.paste(bezier_img.convert("RGB"), (0, h + 50))
    else:
        draw.rectangle([0, h + 50, w, h * 2 + 50], fill="lightgray")
        draw.text((20, h + h // 2 + 30), "Bezier Splatting\n(Not available)", fill="gray")

    if live_img:
        comparison.paste(live_img.convert("RGB"), (w + 20, h + 50))
    else:
        draw.rectangle([w + 20, h + 50, w * 2 + 20, h * 2 + 50], fill="lightgray")
        draw.text((w + 40, h + h // 2 + 30), "LIVE Vectorization\n(Not available)", fill="gray")

    # Add labels
    try:
        draw.text((10, 5), "Original", fill="black")
        draw.text((w + 30, 5), "Junction Detection", fill="black")
        draw.text((10, h + 35), "Bezier Splatting", fill="black")
        draw.text((w + 30, h + 35), "LIVE Vectorization", fill="black")
    except Exception:
        pass

    comparison.save(output_path)


async def visualize_pipeline(
    image_path: str | Path,
    output_dir: str | Path | None = None,
    run_junction_detection: bool = True,
    run_bezier_splatting: bool = True,
    run_live_vectorization: bool = True,
    bezier_num_curves: int = 64,
    bezier_iterations: int = 300,
    live_num_layers: int = 4,
    live_iterations: int = 300,
) -> VisualizationResult:
    """
    Visualize the Phase C vectorization pipeline on an image.

    Creates visual outputs showing each stage of the pipeline:
    1. Junction detection overlay
    2. Bezier Splatting SVG/PNG
    3. LIVE vectorization SVG/PNG
    4. Side-by-side comparison

    Args:
        image_path: Path to input image (PNG, JPG, etc.)
        output_dir: Directory for output files (default: same as input)
        run_junction_detection: Run C.1 Junction Detection
        run_bezier_splatting: Run C.2 Bezier Splatting
        run_live_vectorization: Run C.3 LIVE Vectorization
        bezier_num_curves: Number of curves for Bezier Splatting
        bezier_iterations: Iterations for Bezier optimization
        live_num_layers: Number of layers for LIVE
        live_iterations: Iterations per layer for LIVE

    Returns:
        VisualizationResult with paths to all outputs
    """
    if not PIL_AVAILABLE:
        raise ImportError("PIL/Pillow is required for visualization")

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Setup output directory
    if output_dir is None:
        output_dir = image_path.parent / "phase_c_output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load original image
    original = Image.open(image_path)
    width, height = original.size

    logger.info(
        "visualize_pipeline_start",
        image=str(image_path),
        size=(width, height),
        output_dir=str(output_dir),
    )

    result = VisualizationResult(
        input_image_path=str(image_path),
        output_dir=str(output_dir),
    )

    junction_overlay = None
    bezier_img = None
    live_img = None

    # =========================================================================
    # C.1: Junction Detection
    # =========================================================================
    if run_junction_detection:
        try:
            from .neural_junction_detection import (
                is_junction_detection_available,
                JunctionDetector,
                JunctionDetectionConfig,
            )

            if is_junction_detection_available():
                logger.info("running_junction_detection")

                config = JunctionDetectionConfig(gpu_id=-1)  # Use CPU
                detector = JunctionDetector.get_instance(config)

                start_time = time.time()
                detection_result = detector.detect(str(image_path))
                result.junction_time_ms = (time.time() - start_time) * 1000

                result.junction_count = len(detection_result.junctions)

                # Create overlay
                junction_overlay = _draw_junctions_on_image(
                    original,
                    detection_result.junctions,
                    detection_result.lines,
                )

                # Save overlay
                overlay_path = output_dir / f"{image_path.stem}_junctions.png"
                junction_overlay.save(overlay_path)
                result.junction_overlay_path = str(overlay_path)

                logger.info(
                    "junction_detection_complete",
                    junctions=result.junction_count,
                    lines=len(detection_result.lines),
                    time_ms=result.junction_time_ms,
                )
            else:
                result.errors.append("Junction detection not available (PyTorch required)")

        except Exception as e:
            logger.exception("junction_detection_failed", error=str(e))
            result.errors.append(f"Junction detection failed: {e}")

    # =========================================================================
    # C.2: Bezier Splatting
    # =========================================================================
    if run_bezier_splatting:
        try:
            from .bezier_splatting import (
                is_bezier_splatting_available,
                bezier_splat,
                BezierSplattingConfig,
            )

            if is_bezier_splatting_available():
                logger.info("running_bezier_splatting", curves=bezier_num_curves)

                config = BezierSplattingConfig(
                    num_curves=bezier_num_curves,
                    iterations=bezier_iterations,
                    gpu_id=-1,  # Use CPU
                )

                svg_path = output_dir / f"{image_path.stem}_bezier.svg"

                start_time = time.time()
                bezier_result = bezier_splat(image_path, config, svg_path)
                result.bezier_time_ms = (time.time() - start_time) * 1000

                result.bezier_curve_count = len(bezier_result.curves)
                result.bezier_svg_path = str(svg_path)

                # Convert SVG to PNG for comparison
                png_path = output_dir / f"{image_path.stem}_bezier.png"
                if _svg_to_png(svg_path, png_path, (width, height)):
                    result.bezier_png_path = str(png_path)
                    bezier_img = Image.open(png_path)

                logger.info(
                    "bezier_splatting_complete",
                    curves=result.bezier_curve_count,
                    loss=bezier_result.final_loss,
                    time_ms=result.bezier_time_ms,
                )
            else:
                result.errors.append("Bezier Splatting not available (PyTorch required)")

        except Exception as e:
            logger.exception("bezier_splatting_failed", error=str(e))
            result.errors.append(f"Bezier Splatting failed: {e}")

    # =========================================================================
    # C.3: LIVE Vectorization
    # =========================================================================
    if run_live_vectorization:
        try:
            from .live_vectorization import (
                is_live_available,
                live_vectorize,
                LIVEConfig,
            )

            if is_live_available():
                logger.info("running_live_vectorization", layers=live_num_layers)

                config = LIVEConfig(
                    num_layers=live_num_layers,
                    iterations_per_layer=live_iterations,
                    gpu_id=-1,  # Use CPU
                )

                svg_path = output_dir / f"{image_path.stem}_live.svg"

                start_time = time.time()
                live_result = live_vectorize(image_path, config, svg_path)
                result.live_time_ms = (time.time() - start_time) * 1000

                result.live_layer_count = len(live_result.layers)
                result.live_path_count = sum(len(l.paths) for l in live_result.layers)
                result.live_svg_path = str(svg_path)

                # Convert SVG to PNG for comparison
                png_path = output_dir / f"{image_path.stem}_live.png"
                if _svg_to_png(svg_path, png_path, (width, height)):
                    result.live_png_path = str(png_path)
                    live_img = Image.open(png_path)

                logger.info(
                    "live_vectorization_complete",
                    layers=result.live_layer_count,
                    paths=result.live_path_count,
                    loss=live_result.final_loss,
                    time_ms=result.live_time_ms,
                )
            else:
                result.errors.append("LIVE vectorization not available (PyTorch required)")

        except Exception as e:
            logger.exception("live_vectorization_failed", error=str(e))
            result.errors.append(f"LIVE vectorization failed: {e}")

    # =========================================================================
    # Create Comparison Image
    # =========================================================================
    try:
        comparison_path = output_dir / f"{image_path.stem}_comparison.png"
        _create_comparison_image(
            original,
            junction_overlay,
            bezier_img,
            live_img,
            comparison_path,
        )
        result.comparison_path = str(comparison_path)
        logger.info("comparison_image_created", path=str(comparison_path))
    except Exception as e:
        logger.exception("comparison_creation_failed", error=str(e))
        result.errors.append(f"Comparison image failed: {e}")

    logger.info(
        "visualize_pipeline_complete",
        junction_count=result.junction_count,
        bezier_curves=result.bezier_curve_count,
        live_layers=result.live_layer_count,
        errors=len(result.errors),
    )

    return result


def visualize_pipeline_sync(
    image_path: str | Path,
    output_dir: str | Path | None = None,
    **kwargs,
) -> VisualizationResult:
    """Synchronous wrapper for visualize_pipeline."""
    return asyncio.run(visualize_pipeline(image_path, output_dir, **kwargs))


# CLI entry point for quick testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pipeline_visualizer.py <image_path> [output_dir]")
        print("\nExample:")
        print("  python pipeline_visualizer.py floor_plan.png ./output")
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Processing: {image_path}")
    result = visualize_pipeline_sync(image_path, output_dir)

    print("\n=== Phase C Pipeline Visualization ===")
    print(f"Input: {result.input_image_path}")
    print(f"Output: {result.output_dir}")
    print()
    print("Results:")
    print(f"  Junction Detection: {result.junction_count} junctions ({result.junction_time_ms:.0f}ms)")
    print(f"  Bezier Splatting: {result.bezier_curve_count} curves ({result.bezier_time_ms:.0f}ms)")
    print(f"  LIVE: {result.live_layer_count} layers, {result.live_path_count} paths ({result.live_time_ms:.0f}ms)")
    print()
    print("Output Files:")
    if result.junction_overlay_path:
        print(f"  Junction overlay: {result.junction_overlay_path}")
    if result.bezier_svg_path:
        print(f"  Bezier SVG: {result.bezier_svg_path}")
    if result.live_svg_path:
        print(f"  LIVE SVG: {result.live_svg_path}")
    if result.comparison_path:
        print(f"  Comparison: {result.comparison_path}")

    if result.errors:
        print("\nErrors:")
        for err in result.errors:
            print(f"  - {err}")
