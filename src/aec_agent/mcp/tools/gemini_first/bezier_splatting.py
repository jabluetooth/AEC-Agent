"""
Bezier Splatting: Fast Differentiable Curve Fitting via 2D Gaussian Splatting.

This module implements vectorization using Bezier curves optimized through
differentiable 2D Gaussian splatting, achieving 150x faster convergence
compared to traditional optimization methods.

Reference: arxiv 2503.16424 "Bezier Splatting" (NeurIPS 2025)

Key idea: Instead of rendering full curves, sample points along Bezier curves
and splat 2D Gaussians at those locations. This enables efficient gradient
computation and rapid curve parameter optimization.

Phase C.2 of the Gemini-First vectorization pipeline.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

# Optional PyTorch dependency
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore

logger = logging.getLogger(__name__)


def is_bezier_splatting_available() -> bool:
    """Check if Bezier Splatting is available (requires PyTorch)."""
    return TORCH_AVAILABLE


class CurveType(Enum):
    """Type of Bezier curve based on number of control points."""

    LINEAR = "linear"  # 2 control points (P0, P1)
    QUADRATIC = "quadratic"  # 3 control points (P0, P1, P2)
    CUBIC = "cubic"  # 4 control points (P0, P1, P2, P3)


@dataclass
class BezierSplattingConfig:
    """Configuration for Bezier Splatting optimization.

    Attributes:
        num_curves: Number of Bezier curves to fit
        curve_type: Type of curves (linear, quadratic, cubic)
        points_per_curve: Number of sample points per curve for splatting
        iterations: Number of optimization iterations
        learning_rate: Learning rate for Adam optimizer
        gaussian_sigma: Standard deviation of 2D Gaussians
        stroke_width: Initial stroke width for curves
        pruning_threshold: Remove curves with opacity below this
        densification_grad_threshold: Add curves where gradient is high
        pruning_interval: Prune/densify every N iterations
        adaptive_control: Enable adaptive pruning and densification
        use_color: Optimize per-curve colors (vs black/white)
        gpu_id: GPU device ID (None=auto, -1=CPU only)
        seed: Random seed for reproducibility
    """

    num_curves: int = 64
    curve_type: CurveType = CurveType.CUBIC
    points_per_curve: int = 20
    iterations: int = 500
    learning_rate: float = 0.01
    gaussian_sigma: float = 1.0
    stroke_width: float = 2.0
    pruning_threshold: float = 0.01
    densification_grad_threshold: float = 0.1
    pruning_interval: int = 100
    adaptive_control: bool = True
    use_color: bool = False
    gpu_id: int | None = None
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_curves": self.num_curves,
            "curve_type": self.curve_type.value,
            "points_per_curve": self.points_per_curve,
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "gaussian_sigma": self.gaussian_sigma,
            "stroke_width": self.stroke_width,
            "pruning_threshold": self.pruning_threshold,
            "densification_grad_threshold": self.densification_grad_threshold,
            "pruning_interval": self.pruning_interval,
            "adaptive_control": self.adaptive_control,
            "use_color": self.use_color,
            "gpu_id": self.gpu_id,
            "seed": self.seed,
        }


@dataclass
class OptimizedCurve:
    """A single optimized Bezier curve.

    Attributes:
        control_points: List of (x, y) control points
        curve_type: Type of curve
        stroke_width: Width of the stroke
        color: RGB color tuple (0-255)
        opacity: Opacity (0-1)
    """

    control_points: list[tuple[float, float]]
    curve_type: CurveType
    stroke_width: float = 2.0
    color: tuple[int, int, int] = (0, 0, 0)  # Black
    opacity: float = 1.0

    def to_svg_path(self) -> str:
        """Convert to SVG path data string.

        Returns:
            SVG path d attribute value
        """
        if not self.control_points:
            return ""

        # Start point
        x0, y0 = self.control_points[0]
        path_data = f"M {x0:.3f} {y0:.3f}"

        if self.curve_type == CurveType.LINEAR:
            # Line to end point
            x1, y1 = self.control_points[1]
            path_data += f" L {x1:.3f} {y1:.3f}"

        elif self.curve_type == CurveType.QUADRATIC:
            # Quadratic Bezier
            x1, y1 = self.control_points[1]
            x2, y2 = self.control_points[2]
            path_data += f" Q {x1:.3f} {y1:.3f} {x2:.3f} {y2:.3f}"

        elif self.curve_type == CurveType.CUBIC:
            # Cubic Bezier
            x1, y1 = self.control_points[1]
            x2, y2 = self.control_points[2]
            x3, y3 = self.control_points[3]
            path_data += f" C {x1:.3f} {y1:.3f} {x2:.3f} {y2:.3f} {x3:.3f} {y3:.3f}"

        return path_data

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "control_points": self.control_points,
            "curve_type": self.curve_type.value,
            "stroke_width": self.stroke_width,
            "color": self.color,
            "opacity": self.opacity,
            "svg_path": self.to_svg_path(),
        }


@dataclass
class BezierSplattingResult:
    """Result of Bezier Splatting optimization.

    Attributes:
        curves: List of optimized curves
        loss_history: Loss values at each iteration
        final_loss: Final loss value
        image_size: (width, height) of target image
        processing_time_ms: Time taken for optimization
        config: Configuration used
        device: Device used (cpu/cuda)
    """

    curves: list[OptimizedCurve]
    loss_history: list[float]
    final_loss: float
    image_size: tuple[int, int]
    processing_time_ms: float
    config: BezierSplattingConfig
    device: str = "cpu"

    def to_svg(
        self,
        background_color: str = "white",
        include_style: bool = True,
    ) -> str:
        """Generate SVG from optimized curves.

        Args:
            background_color: Background fill color
            include_style: Include CSS styling

        Returns:
            Complete SVG document string
        """
        width, height = self.image_size

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ]

        if include_style:
            svg_parts.append("""
<style>
  path { fill: none; stroke-linecap: round; stroke-linejoin: round; }
</style>""")

        # Background
        svg_parts.append(
            f'<rect width="{width}" height="{height}" fill="{background_color}"/>'
        )

        # Curves
        for curve in self.curves:
            path_data = curve.to_svg_path()
            if not path_data:
                continue

            r, g, b = curve.color
            stroke = f"rgb({r},{g},{b})"
            opacity = f' opacity="{curve.opacity:.3f}"' if curve.opacity < 1.0 else ""

            svg_parts.append(
                f'<path d="{path_data}" '
                f'stroke="{stroke}" '
                f'stroke-width="{curve.stroke_width:.2f}"'
                f'{opacity}/>'
            )

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    def save_svg(self, path: str | Path) -> None:
        """Save SVG to file."""
        path = Path(path)
        path.write_text(self.to_svg(), encoding="utf-8")
        logger.info(f"Saved SVG to {path}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "curves": [c.to_dict() for c in self.curves],
            "num_curves": len(self.curves),
            "loss_history": self.loss_history,
            "final_loss": self.final_loss,
            "image_size": self.image_size,
            "processing_time_ms": self.processing_time_ms,
            "config": self.config.to_dict(),
            "device": self.device,
        }


# PyTorch modules (only defined if torch is available)
if TORCH_AVAILABLE:

    class BezierCurveModule(nn.Module):
        """PyTorch module representing a set of Bezier curves.

        This module stores control points as learnable parameters and
        provides methods for evaluating points along the curves using
        De Casteljau's algorithm.
        """

        def __init__(
            self,
            num_curves: int,
            curve_type: CurveType,
            image_size: tuple[int, int],
            points_per_curve: int = 20,
            stroke_width: float = 2.0,
            use_color: bool = False,
            device: torch.device | None = None,
        ):
            """Initialize Bezier curves with random control points.

            Args:
                num_curves: Number of curves
                curve_type: Type of curves (linear, quadratic, cubic)
                image_size: (width, height) for initialization bounds
                points_per_curve: Number of sample points per curve
                stroke_width: Initial stroke width
                use_color: Whether to optimize colors
                device: Torch device
            """
            super().__init__()

            self.num_curves = num_curves
            self.curve_type = curve_type
            self.image_size = image_size
            self.points_per_curve = points_per_curve
            self.device = device or torch.device("cpu")

            # Number of control points per curve
            if curve_type == CurveType.LINEAR:
                num_control_points = 2
            elif curve_type == CurveType.QUADRATIC:
                num_control_points = 3
            else:  # CUBIC
                num_control_points = 4

            self.num_control_points = num_control_points
            width, height = image_size

            # Initialize control points randomly within image bounds
            # Shape: (num_curves, num_control_points, 2)
            init_points = torch.rand(num_curves, num_control_points, 2, device=self.device)
            init_points[:, :, 0] *= width
            init_points[:, :, 1] *= height

            self.control_points = nn.Parameter(init_points)

            # Stroke widths (learnable)
            init_widths = torch.full((num_curves,), stroke_width, device=self.device)
            self.stroke_widths = nn.Parameter(init_widths)

            # Opacities (learnable, sigmoid will be applied)
            init_opacities = torch.zeros(num_curves, device=self.device)
            self.opacities_logit = nn.Parameter(init_opacities)

            # Colors (optional)
            self.use_color = use_color
            if use_color:
                # RGB values in [0, 1]
                init_colors = torch.zeros(num_curves, 3, device=self.device)
                self.colors = nn.Parameter(init_colors)
            else:
                self.register_buffer(
                    "colors",
                    torch.zeros(num_curves, 3, device=self.device)
                )

            # Pre-compute t values for De Casteljau evaluation
            t_values = torch.linspace(0, 1, points_per_curve, device=self.device)
            self.register_buffer("t_values", t_values)

        @property
        def opacities(self) -> torch.Tensor:
            """Get opacities (sigmoid of logits)."""
            return torch.sigmoid(self.opacities_logit)

        def evaluate_curves(self) -> torch.Tensor:
            """Evaluate all curves at sample points using De Casteljau.

            Returns:
                Points tensor of shape (num_curves, points_per_curve, 2)
            """
            # t shape: (points_per_curve,)
            t = self.t_values

            if self.curve_type == CurveType.LINEAR:
                return self._evaluate_linear(t)
            elif self.curve_type == CurveType.QUADRATIC:
                return self._evaluate_quadratic(t)
            else:
                return self._evaluate_cubic(t)

        def _evaluate_linear(self, t: torch.Tensor) -> torch.Tensor:
            """Evaluate linear Bezier (line segment)."""
            # P(t) = (1-t)*P0 + t*P1
            P0 = self.control_points[:, 0, :]  # (num_curves, 2)
            P1 = self.control_points[:, 1, :]

            t = t.view(1, -1, 1)  # (1, points_per_curve, 1)
            P0 = P0.unsqueeze(1)  # (num_curves, 1, 2)
            P1 = P1.unsqueeze(1)

            return (1 - t) * P0 + t * P1  # (num_curves, points_per_curve, 2)

        def _evaluate_quadratic(self, t: torch.Tensor) -> torch.Tensor:
            """Evaluate quadratic Bezier curve."""
            # P(t) = (1-t)^2*P0 + 2*(1-t)*t*P1 + t^2*P2
            P0 = self.control_points[:, 0, :].unsqueeze(1)
            P1 = self.control_points[:, 1, :].unsqueeze(1)
            P2 = self.control_points[:, 2, :].unsqueeze(1)

            t = t.view(1, -1, 1)
            t2 = t * t
            mt = 1 - t
            mt2 = mt * mt

            return mt2 * P0 + 2 * mt * t * P1 + t2 * P2

        def _evaluate_cubic(self, t: torch.Tensor) -> torch.Tensor:
            """Evaluate cubic Bezier curve using De Casteljau."""
            # P(t) = (1-t)^3*P0 + 3*(1-t)^2*t*P1 + 3*(1-t)*t^2*P2 + t^3*P3
            P0 = self.control_points[:, 0, :].unsqueeze(1)
            P1 = self.control_points[:, 1, :].unsqueeze(1)
            P2 = self.control_points[:, 2, :].unsqueeze(1)
            P3 = self.control_points[:, 3, :].unsqueeze(1)

            t = t.view(1, -1, 1)
            t2 = t * t
            t3 = t2 * t
            mt = 1 - t
            mt2 = mt * mt
            mt3 = mt2 * mt

            return mt3 * P0 + 3 * mt2 * t * P1 + 3 * mt * t2 * P2 + t3 * P3

        def render(
            self,
            sigma: float = 1.0,
            return_per_curve: bool = False,
        ) -> torch.Tensor:
            """Render curves using 2D Gaussian splatting.

            Args:
                sigma: Standard deviation of Gaussians
                return_per_curve: If True, return per-curve renders

            Returns:
                Rendered image tensor of shape (H, W) or (num_curves, H, W)
            """
            width, height = self.image_size

            # Get sample points: (num_curves, points_per_curve, 2)
            points = self.evaluate_curves()

            # Create pixel grid: (H, W, 2)
            y_coords = torch.arange(height, device=self.device, dtype=torch.float32)
            x_coords = torch.arange(width, device=self.device, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
            grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)

            # Compute Gaussian weights
            # points: (num_curves, points_per_curve, 2)
            # grid: (H, W, 2)
            # We want distance from each grid point to each sample point

            # Reshape for broadcasting
            points_reshaped = points.view(
                self.num_curves, self.points_per_curve, 1, 1, 2
            )
            grid_reshaped = grid.view(1, 1, height, width, 2)

            # Squared distances: (num_curves, points_per_curve, H, W)
            diff = points_reshaped - grid_reshaped
            sq_dist = (diff ** 2).sum(dim=-1)

            # Scale sigma by stroke width
            stroke_sigma = sigma * self.stroke_widths.view(-1, 1, 1, 1).clamp(min=0.5)

            # Gaussian weights
            weights = torch.exp(-sq_dist / (2 * stroke_sigma ** 2))

            # Sum over sample points: (num_curves, H, W)
            curve_renders = weights.sum(dim=1)

            # Apply opacities
            opacities = self.opacities.view(-1, 1, 1)
            curve_renders = curve_renders * opacities

            if return_per_curve:
                return curve_renders

            # Combine all curves (alpha compositing approximation)
            combined = curve_renders.sum(dim=0).clamp(0, 1)

            return combined

        def render_color(self, sigma: float = 1.0) -> torch.Tensor:
            """Render curves with colors.

            Returns:
                RGB image tensor of shape (H, W, 3) in [0, 1]
            """
            # Get per-curve renders: (num_curves, H, W)
            curve_renders = self.render(sigma, return_per_curve=True)

            # Get colors: (num_curves, 3)
            if self.use_color:
                colors = torch.sigmoid(self.colors)
            else:
                colors = self.colors  # All zeros (black)

            # Weight by curve intensity: (num_curves, H, W, 1) * (num_curves, 1, 1, 3)
            colored = curve_renders.unsqueeze(-1) * colors.view(-1, 1, 1, 3)

            # Sum and normalize: (H, W, 3)
            total_intensity = curve_renders.sum(dim=0, keepdim=True).unsqueeze(-1).clamp(min=1e-6)
            result = colored.sum(dim=0) / total_intensity.squeeze(0)

            # Where no curves, use white background
            mask = curve_renders.sum(dim=0) < 0.01
            result[mask] = 1.0  # White

            return result.clamp(0, 1)

        def get_curves(self) -> list[OptimizedCurve]:
            """Extract optimized curves as OptimizedCurve objects."""
            curves = []

            control_points_np = self.control_points.detach().cpu().numpy()
            stroke_widths_np = self.stroke_widths.detach().cpu().numpy()
            opacities_np = self.opacities.detach().cpu().numpy()

            if self.use_color:
                colors_np = torch.sigmoid(self.colors).detach().cpu().numpy()
            else:
                colors_np = np.zeros((self.num_curves, 3))

            for i in range(self.num_curves):
                # Skip low-opacity curves
                if opacities_np[i] < 0.01:
                    continue

                points = [
                    (float(control_points_np[i, j, 0]), float(control_points_np[i, j, 1]))
                    for j in range(self.num_control_points)
                ]

                color = tuple(int(c * 255) for c in colors_np[i])

                curves.append(OptimizedCurve(
                    control_points=points,
                    curve_type=self.curve_type,
                    stroke_width=float(stroke_widths_np[i]),
                    color=color,  # type: ignore
                    opacity=float(opacities_np[i]),
                ))

            return curves

        def prune_curves(self, threshold: float = 0.01) -> int:
            """Remove curves with low opacity.

            Args:
                threshold: Opacity threshold below which to prune

            Returns:
                Number of curves pruned
            """
            opacities = self.opacities.detach()
            mask = opacities >= threshold

            if mask.all():
                return 0

            num_kept = mask.sum().item()
            num_pruned = self.num_curves - num_kept

            if num_kept == 0:
                logger.warning("All curves would be pruned, keeping at least one")
                # Keep the highest opacity curve
                idx = opacities.argmax().item()
                mask[idx] = True
                num_kept = 1
                num_pruned = self.num_curves - 1

            # Update parameters
            self.control_points.data = self.control_points.data[mask]
            self.stroke_widths.data = self.stroke_widths.data[mask]
            self.opacities_logit.data = self.opacities_logit.data[mask]

            if self.use_color:
                self.colors.data = self.colors.data[mask]
            else:
                self.colors = self.colors[mask]

            self.num_curves = int(num_kept)

            logger.debug(f"Pruned {num_pruned} curves, {num_kept} remaining")
            return int(num_pruned)

        def densify(
            self,
            target_image: torch.Tensor,
            num_new_curves: int = 4,
            grad_threshold: float = 0.1,
        ) -> int:
            """Add new curves in high-gradient regions.

            Args:
                target_image: Target image tensor (H, W)
                num_new_curves: Maximum number of new curves to add
                grad_threshold: Minimum gradient magnitude to consider

            Returns:
                Number of curves added
            """
            # Compute error gradient
            with torch.no_grad():
                rendered = self.render()
                error = (target_image - rendered).abs()

                # Find high-error regions
                # Use max pooling to find local maxima
                error_pooled = F.max_pool2d(
                    error.unsqueeze(0).unsqueeze(0),
                    kernel_size=16,
                    stride=16,
                    return_indices=True,
                )[0].squeeze()

                # Find top-k error regions
                error_flat = error_pooled.flatten()
                if error_flat.max() < grad_threshold:
                    return 0

                num_add = min(num_new_curves, (error_flat > grad_threshold).sum().item())
                if num_add == 0:
                    return 0

                # Get top error positions
                topk_values, topk_indices = error_flat.topk(num_add)

                # Convert to 2D coordinates
                pool_h, pool_w = error_pooled.shape
                new_y = (topk_indices // pool_w) * 16 + 8
                new_x = (topk_indices % pool_w) * 16 + 8

                # Create new curves centered at these positions
                width, height = self.image_size
                new_points = torch.zeros(
                    num_add, self.num_control_points, 2, device=self.device
                )

                for i in range(num_add):
                    cx, cy = new_x[i].item(), new_y[i].item()

                    # Create control points around center
                    for j in range(self.num_control_points):
                        angle = (j / self.num_control_points) * 2 * math.pi
                        radius = 20  # Initial curve size
                        px = cx + radius * math.cos(angle)
                        py = cy + radius * math.sin(angle)
                        new_points[i, j, 0] = max(0, min(width - 1, px))
                        new_points[i, j, 1] = max(0, min(height - 1, py))

                # Append new parameters
                self.control_points.data = torch.cat([
                    self.control_points.data, new_points
                ], dim=0)

                new_widths = torch.full(
                    (num_add,), 2.0, device=self.device
                )
                self.stroke_widths.data = torch.cat([
                    self.stroke_widths.data, new_widths
                ], dim=0)

                new_opacities = torch.zeros(num_add, device=self.device)
                self.opacities_logit.data = torch.cat([
                    self.opacities_logit.data, new_opacities
                ], dim=0)

                if self.use_color:
                    new_colors = torch.zeros(num_add, 3, device=self.device)
                    self.colors.data = torch.cat([
                        self.colors.data, new_colors
                    ], dim=0)
                else:
                    new_colors = torch.zeros(num_add, 3, device=self.device)
                    self.colors = torch.cat([self.colors, new_colors], dim=0)

                self.num_curves += num_add

                logger.debug(f"Densified by adding {num_add} curves")
                return num_add


class BezierSplattingOptimizer:
    """Main optimizer for Bezier Splatting vectorization.

    This class handles the complete optimization loop:
    1. Initialize random Bezier curves
    2. Render via 2D Gaussian splatting
    3. Compute MSE loss against target
    4. Backpropagate and update control points
    5. Periodically prune/densify curves
    """

    _instance: "BezierSplattingOptimizer | None" = None

    def __init__(self, config: BezierSplattingConfig | None = None):
        """Initialize optimizer.

        Args:
            config: Configuration (uses defaults if None)
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for Bezier Splatting. "
                "Install with: pip install torch torchvision"
            )

        self.config = config or BezierSplattingConfig()
        self.device = self._get_device()
        self._curves_module: BezierCurveModule | None = None

        logger.info(f"BezierSplattingOptimizer initialized on {self.device}")

    @classmethod
    def get_instance(
        cls, config: BezierSplattingConfig | None = None
    ) -> "BezierSplattingOptimizer":
        """Get singleton instance.

        Args:
            config: Configuration (only used if creating new instance)

        Returns:
            Singleton instance
        """
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance."""
        cls._instance = None

    def _get_device(self) -> torch.device:
        """Get compute device based on config."""
        if self.config.gpu_id == -1:
            return torch.device("cpu")

        if not torch.cuda.is_available():
            logger.info("CUDA not available, using CPU")
            return torch.device("cpu")

        if self.config.gpu_id is not None:
            device_id = self.config.gpu_id
        else:
            device_id = 0

        if device_id >= torch.cuda.device_count():
            logger.warning(
                f"GPU {device_id} not available, using CPU"
            )
            return torch.device("cpu")

        return torch.device(f"cuda:{device_id}")

    def _load_image(self, image_path: str | Path) -> torch.Tensor:
        """Load image and convert to grayscale tensor.

        Args:
            image_path: Path to image file

        Returns:
            Grayscale image tensor of shape (H, W) in [0, 1]
        """
        from PIL import Image

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load and convert to grayscale
        img = Image.open(image_path).convert("L")
        img_np = np.array(img, dtype=np.float32) / 255.0

        # Invert so lines are 1.0 and background is 0.0
        img_np = 1.0 - img_np

        return torch.from_numpy(img_np).to(self.device)

    def _load_color_image(self, image_path: str | Path) -> torch.Tensor:
        """Load color image as tensor.

        Args:
            image_path: Path to image file

        Returns:
            RGB image tensor of shape (H, W, 3) in [0, 1]
        """
        from PIL import Image

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img, dtype=np.float32) / 255.0

        return torch.from_numpy(img_np).to(self.device)

    def optimize(
        self,
        image_path: str | Path,
        config: BezierSplattingConfig | None = None,
        progress_callback: callable | None = None,
    ) -> BezierSplattingResult:
        """Run Bezier Splatting optimization.

        Args:
            image_path: Path to target image
            config: Override configuration
            progress_callback: Optional callback(iteration, loss) for progress

        Returns:
            BezierSplattingResult with optimized curves
        """
        config = config or self.config
        start_time = time.time()

        # Set random seed
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)

        # Load target image
        if config.use_color:
            target = self._load_color_image(image_path)
            height, width = target.shape[:2]
        else:
            target = self._load_image(image_path)
            height, width = target.shape

        image_size = (width, height)
        logger.info(f"Loaded target image: {width}x{height}")

        # Initialize curves
        curves_module = BezierCurveModule(
            num_curves=config.num_curves,
            curve_type=config.curve_type,
            image_size=image_size,
            points_per_curve=config.points_per_curve,
            stroke_width=config.stroke_width,
            use_color=config.use_color,
            device=self.device,
        )

        # Setup optimizer
        optimizer = Adam(curves_module.parameters(), lr=config.learning_rate)

        loss_history: list[float] = []

        # Optimization loop
        for iteration in range(config.iterations):
            optimizer.zero_grad()

            # Render
            if config.use_color:
                rendered = curves_module.render_color(sigma=config.gaussian_sigma)
                loss = F.mse_loss(rendered, target)
            else:
                rendered = curves_module.render(sigma=config.gaussian_sigma)
                loss = F.mse_loss(rendered, target)

            # Backprop
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            loss_history.append(loss_val)

            # Progress callback
            if progress_callback is not None:
                progress_callback(iteration, loss_val)

            # Adaptive control
            if config.adaptive_control and (iteration + 1) % config.pruning_interval == 0:
                # Prune low-opacity curves
                curves_module.prune_curves(config.pruning_threshold)

                # Densify in high-error regions
                if not config.use_color:
                    curves_module.densify(
                        target,
                        num_new_curves=4,
                        grad_threshold=config.densification_grad_threshold,
                    )

                # Recreate optimizer with updated parameters
                optimizer = Adam(curves_module.parameters(), lr=config.learning_rate)

            # Log progress
            if (iteration + 1) % 100 == 0:
                logger.info(
                    f"Iteration {iteration + 1}/{config.iterations}: "
                    f"loss={loss_val:.6f}, curves={curves_module.num_curves}"
                )

        # Extract final curves
        final_curves = curves_module.get_curves()

        processing_time = (time.time() - start_time) * 1000

        result = BezierSplattingResult(
            curves=final_curves,
            loss_history=loss_history,
            final_loss=loss_history[-1] if loss_history else 0.0,
            image_size=image_size,
            processing_time_ms=processing_time,
            config=config,
            device=str(self.device),
        )

        logger.info(
            f"Optimization complete: {len(final_curves)} curves, "
            f"final loss={result.final_loss:.6f}, "
            f"time={processing_time:.1f}ms"
        )

        return result


def bezier_splat(
    image_path: str | Path,
    config: BezierSplattingConfig | None = None,
    output_svg: str | Path | None = None,
) -> BezierSplattingResult:
    """Convenience function for Bezier Splatting vectorization.

    Args:
        image_path: Path to input image
        config: Configuration (uses defaults if None)
        output_svg: Optional path to save SVG output

    Returns:
        BezierSplattingResult with optimized curves
    """
    if not is_bezier_splatting_available():
        raise ImportError(
            "Bezier Splatting requires PyTorch. "
            "Install with: pip install torch torchvision"
        )

    optimizer = BezierSplattingOptimizer(config)
    result = optimizer.optimize(image_path, config)

    if output_svg is not None:
        result.save_svg(output_svg)

    return result


__all__ = [
    "is_bezier_splatting_available",
    "CurveType",
    "BezierSplattingConfig",
    "OptimizedCurve",
    "BezierSplattingResult",
    "BezierSplattingOptimizer",
    "bezier_splat",
]
