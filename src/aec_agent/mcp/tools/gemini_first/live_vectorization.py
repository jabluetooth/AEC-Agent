"""
LIVE: Towards Layer-wise Image Vectorization.

This module implements progressive coarse-to-fine vectorization using
closed Bezier paths, optimizing each layer to capture image structure
at different levels of detail.

Reference: "Towards Layer-wise Image Vectorization" (CVPR 2022)
https://ma-xu.github.io/LIVE/

Key idea: Build vector representation layer-by-layer, where each layer
uses minimal paths to approximate the residual from previous layers.
The result is a compact SVG with semantic layering.

Phase C.3 of the Gemini-First vectorization pipeline.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
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


def is_live_available() -> bool:
    """Check if LIVE vectorization is available (requires PyTorch)."""
    return TORCH_AVAILABLE


@dataclass
class LIVEConfig:
    """Configuration for LIVE vectorization.

    Attributes:
        num_layers: Number of vectorization layers
        paths_per_layer: Number of closed paths per layer
        control_points_per_path: Number of control points per path
        iterations_per_layer: Optimization iterations per layer
        learning_rate: Learning rate for Adam optimizer
        xing_loss_weight: Weight for self-crossing loss
        udf_loss_weight: Weight for unsigned distance field loss
        color_loss_weight: Weight for color matching loss
        path_init_scale: Initial path scale relative to image
        regularization_weight: Weight for path smoothness regularization
        min_path_area: Minimum area threshold for paths
        use_color: Optimize colors (vs grayscale)
        gpu_id: GPU device ID (None=auto, -1=CPU only)
        seed: Random seed for reproducibility
    """

    num_layers: int = 5
    paths_per_layer: int = 1
    control_points_per_path: int = 8
    iterations_per_layer: int = 500
    learning_rate: float = 0.01
    xing_loss_weight: float = 0.1
    udf_loss_weight: float = 1.0
    color_loss_weight: float = 1.0
    path_init_scale: float = 0.3
    regularization_weight: float = 0.01
    min_path_area: float = 100.0
    use_color: bool = True
    gpu_id: int | None = None
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_layers": self.num_layers,
            "paths_per_layer": self.paths_per_layer,
            "control_points_per_path": self.control_points_per_path,
            "iterations_per_layer": self.iterations_per_layer,
            "learning_rate": self.learning_rate,
            "xing_loss_weight": self.xing_loss_weight,
            "udf_loss_weight": self.udf_loss_weight,
            "color_loss_weight": self.color_loss_weight,
            "path_init_scale": self.path_init_scale,
            "regularization_weight": self.regularization_weight,
            "min_path_area": self.min_path_area,
            "use_color": self.use_color,
            "gpu_id": self.gpu_id,
            "seed": self.seed,
        }


@dataclass
class VectorPath:
    """A single closed Bezier path.

    Attributes:
        control_points: List of (x, y) control points
        fill_color: RGB fill color (0-255)
        opacity: Path opacity (0-1)
    """

    control_points: list[tuple[float, float]]
    fill_color: tuple[int, int, int] = (0, 0, 0)
    opacity: float = 1.0

    def to_svg_path(self) -> str:
        """Convert to SVG path data string (closed cubic Bezier).

        Returns:
            SVG path d attribute value
        """
        if len(self.control_points) < 4:
            return ""

        n = len(self.control_points)
        x0, y0 = self.control_points[0]
        path_parts = [f"M {x0:.3f} {y0:.3f}"]

        # Create cubic Bezier segments
        # Each segment uses 3 control points from the list
        for i in range(0, n, 3):
            if i + 3 > n:
                # Last segment, wrap around
                p1 = self.control_points[(i + 1) % n]
                p2 = self.control_points[(i + 2) % n]
                p3 = self.control_points[0]  # Close to start
            else:
                p1 = self.control_points[i + 1]
                p2 = self.control_points[i + 2]
                p3 = self.control_points[(i + 3) % n]

            path_parts.append(
                f"C {p1[0]:.3f} {p1[1]:.3f} "
                f"{p2[0]:.3f} {p2[1]:.3f} "
                f"{p3[0]:.3f} {p3[1]:.3f}"
            )

        path_parts.append("Z")  # Close path
        return " ".join(path_parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "control_points": self.control_points,
            "fill_color": self.fill_color,
            "opacity": self.opacity,
            "svg_path": self.to_svg_path(),
        }


@dataclass
class VectorLayer:
    """A single layer of vector paths.

    Attributes:
        layer_index: Layer index (0 = bottom/background)
        paths: List of paths in this layer
        fill_color: Default fill color for the layer
    """

    layer_index: int
    paths: list[VectorPath]
    fill_color: tuple[int, int, int] = (255, 255, 255)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "layer_index": self.layer_index,
            "paths": [p.to_dict() for p in self.paths],
            "fill_color": self.fill_color,
            "num_paths": len(self.paths),
        }


@dataclass
class LIVEResult:
    """Result of LIVE vectorization.

    Attributes:
        layers: List of vector layers (bottom to top)
        loss_history: Loss values during optimization
        final_loss: Final loss value
        image_size: (width, height) of source image
        processing_time_ms: Total processing time
        config: Configuration used
        device: Device used (cpu/cuda)
    """

    layers: list[VectorLayer]
    loss_history: list[list[float]]  # Per-layer loss histories
    final_loss: float
    image_size: tuple[int, int]
    processing_time_ms: float
    config: LIVEConfig
    device: str = "cpu"

    def to_svg(
        self,
        background_color: str = "white",
        include_style: bool = True,
    ) -> str:
        """Generate SVG from vector layers.

        Args:
            background_color: Background color
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
  path { stroke: none; }
</style>""")

        # Background
        svg_parts.append(
            f'<rect width="{width}" height="{height}" fill="{background_color}"/>'
        )

        # Render layers bottom to top
        for layer in self.layers:
            for path in layer.paths:
                path_data = path.to_svg_path()
                if not path_data:
                    continue

                r, g, b = path.fill_color
                fill = f"rgb({r},{g},{b})"
                opacity = f' opacity="{path.opacity:.3f}"' if path.opacity < 1.0 else ""

                svg_parts.append(f'<path d="{path_data}" fill="{fill}"{opacity}/>')

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
            "layers": [layer.to_dict() for layer in self.layers],
            "num_layers": len(self.layers),
            "total_paths": sum(len(layer.paths) for layer in self.layers),
            "loss_history": self.loss_history,
            "final_loss": self.final_loss,
            "image_size": self.image_size,
            "processing_time_ms": self.processing_time_ms,
            "config": self.config.to_dict(),
            "device": self.device,
        }


# PyTorch modules (only defined if torch is available)
if TORCH_AVAILABLE:

    class ClosedBezierPath(nn.Module):
        """A single closed Bezier path with learnable control points.

        The path is defined by N control points that form a closed
        cubic Bezier curve.
        """

        def __init__(
            self,
            num_control_points: int,
            image_size: tuple[int, int],
            init_center: tuple[float, float] | None = None,
            init_scale: float = 0.3,
            device: torch.device | None = None,
        ):
            """Initialize closed Bezier path.

            Args:
                num_control_points: Number of control points
                image_size: (width, height) for initialization
                init_center: Initial center position (random if None)
                init_scale: Initial path scale relative to image
                device: Torch device
            """
            super().__init__()

            self.num_control_points = num_control_points
            self.image_size = image_size
            self.device = device or torch.device("cpu")

            width, height = image_size

            # Initialize center
            if init_center is None:
                cx = width * (0.2 + 0.6 * np.random.random())
                cy = height * (0.2 + 0.6 * np.random.random())
            else:
                cx, cy = init_center

            # Initialize control points in a circle around center
            radius = min(width, height) * init_scale / 2
            angles = np.linspace(0, 2 * np.pi, num_control_points, endpoint=False)

            init_points = np.zeros((num_control_points, 2))
            for i, angle in enumerate(angles):
                # Add some randomness to initial positions
                r = radius * (0.8 + 0.4 * np.random.random())
                init_points[i, 0] = cx + r * np.cos(angle)
                init_points[i, 1] = cy + r * np.sin(angle)

            # Clamp to image bounds
            init_points[:, 0] = np.clip(init_points[:, 0], 0, width - 1)
            init_points[:, 1] = np.clip(init_points[:, 1], 0, height - 1)

            self.control_points = nn.Parameter(
                torch.from_numpy(init_points).float().to(self.device)
            )

            # Fill color (RGB in [0, 1])
            init_color = torch.rand(3, device=self.device) * 0.5 + 0.25
            self.color = nn.Parameter(init_color)

            # Opacity (logit space)
            self.opacity_logit = nn.Parameter(torch.tensor(2.0, device=self.device))

        @property
        def opacity(self) -> torch.Tensor:
            """Get opacity (sigmoid of logit)."""
            return torch.sigmoid(self.opacity_logit)

        def get_color_rgb(self) -> tuple[int, int, int]:
            """Get color as RGB tuple (0-255)."""
            color = torch.sigmoid(self.color).detach().cpu().numpy()
            return tuple(int(c * 255) for c in color)  # type: ignore

        def sample_points(self, num_samples: int = 100) -> torch.Tensor:
            """Sample points along the closed Bezier curve.

            Args:
                num_samples: Number of points to sample

            Returns:
                Points tensor of shape (num_samples, 2)
            """
            t = torch.linspace(0, 1, num_samples, device=self.device)
            n = self.num_control_points

            # Use Catmull-Rom to cubic Bezier conversion for smooth closed curve
            # This creates a smooth closed curve through the control points

            points_list = []
            for i in range(n):
                # Get 4 points for this segment
                p0 = self.control_points[(i - 1) % n]
                p1 = self.control_points[i]
                p2 = self.control_points[(i + 1) % n]
                p3 = self.control_points[(i + 2) % n]

                # Catmull-Rom to Bezier control points
                # Tension = 0.5 for standard Catmull-Rom
                tension = 0.5
                c1 = p1 + (p2 - p0) * tension / 3
                c2 = p2 - (p3 - p1) * tension / 3

                # Sample this segment
                segment_t = torch.linspace(0, 1, num_samples // n + 1, device=self.device)[:-1]

                for st in segment_t:
                    # Cubic Bezier evaluation
                    mt = 1 - st
                    point = (
                        mt ** 3 * p1 +
                        3 * mt ** 2 * st * c1 +
                        3 * mt * st ** 2 * c2 +
                        st ** 3 * p2
                    )
                    points_list.append(point)

            if not points_list:
                return self.control_points

            return torch.stack(points_list)

        def compute_area(self) -> torch.Tensor:
            """Compute approximate area using shoelace formula."""
            points = self.sample_points(50)
            n = points.shape[0]

            # Shoelace formula
            x = points[:, 0]
            y = points[:, 1]

            area = torch.abs(
                torch.sum(x * torch.roll(y, -1) - torch.roll(x, -1) * y)
            ) / 2

            return area

        def render(self, resolution: int = 100) -> torch.Tensor:
            """Render path as a binary mask.

            Args:
                resolution: Number of sample points for rendering

            Returns:
                Binary mask tensor of shape (H, W)
            """
            width, height = self.image_size
            points = self.sample_points(resolution)

            # Create pixel grid
            y_coords = torch.arange(height, device=self.device, dtype=torch.float32)
            x_coords = torch.arange(width, device=self.device, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

            # Use winding number for point-in-polygon test
            # Approximate with soft boundary
            n = points.shape[0]

            # Compute distances to all edges
            min_dist = torch.full((height, width), float("inf"), device=self.device)

            for i in range(n):
                p1 = points[i]
                p2 = points[(i + 1) % n]

                # Distance from each pixel to this edge
                edge = p2 - p1
                edge_len_sq = (edge ** 2).sum() + 1e-6

                # Project pixel onto edge
                t = ((grid_x - p1[0]) * edge[0] + (grid_y - p1[1]) * edge[1]) / edge_len_sq
                t = t.clamp(0, 1)

                # Closest point on edge
                closest_x = p1[0] + t * edge[0]
                closest_y = p1[1] + t * edge[1]

                # Distance to closest point
                dist = torch.sqrt(
                    (grid_x - closest_x) ** 2 + (grid_y - closest_y) ** 2
                )
                min_dist = torch.minimum(min_dist, dist)

            # Soft inside/outside using signed distance (approximation)
            # Use winding number for sign
            winding = torch.zeros((height, width), device=self.device)

            for i in range(n):
                p1 = points[i]
                p2 = points[(i + 1) % n]

                # Cross product to determine winding
                cross = (p2[0] - p1[0]) * (grid_y - p1[1]) - (p2[1] - p1[1]) * (grid_x - p1[0])

                # Add contribution based on edge crossing
                cond_up = (p1[1] <= grid_y) & (p2[1] > grid_y) & (cross > 0)
                cond_down = (p1[1] > grid_y) & (p2[1] <= grid_y) & (cross < 0)

                winding = winding + cond_up.float() - cond_down.float()

            # Inside if winding number is non-zero
            inside = (winding != 0).float()

            # Soft boundary
            boundary_width = 2.0
            soft_mask = inside * torch.sigmoid((boundary_width - min_dist) * 2)

            return soft_mask * self.opacity

        def get_vector_path(self) -> VectorPath:
            """Extract as VectorPath object."""
            points = self.control_points.detach().cpu().numpy()
            control_points = [(float(p[0]), float(p[1])) for p in points]

            return VectorPath(
                control_points=control_points,
                fill_color=self.get_color_rgb(),
                opacity=float(self.opacity.item()),
            )


    class LIVELayer(nn.Module):
        """A single LIVE layer containing multiple paths."""

        def __init__(
            self,
            num_paths: int,
            control_points_per_path: int,
            image_size: tuple[int, int],
            init_scale: float = 0.3,
            device: torch.device | None = None,
        ):
            """Initialize LIVE layer.

            Args:
                num_paths: Number of paths in this layer
                control_points_per_path: Control points per path
                image_size: (width, height)
                init_scale: Initial path scale
                device: Torch device
            """
            super().__init__()

            self.num_paths = num_paths
            self.image_size = image_size
            self.device = device or torch.device("cpu")

            # Create paths
            self.paths = nn.ModuleList([
                ClosedBezierPath(
                    num_control_points=control_points_per_path,
                    image_size=image_size,
                    init_scale=init_scale,
                    device=self.device,
                )
                for _ in range(num_paths)
            ])

        def render(self, target: torch.Tensor | None = None) -> torch.Tensor:
            """Render layer as colored image.

            Args:
                target: Optional target image for adaptive rendering

            Returns:
                RGB image tensor of shape (H, W, 3) in [0, 1]
            """
            width, height = self.image_size
            result = torch.zeros((height, width, 3), device=self.device)
            total_alpha = torch.zeros((height, width), device=self.device)

            for path in self.paths:
                mask = path.render()  # (H, W)
                color = torch.sigmoid(path.color)  # (3,)

                # Alpha compositing
                alpha = mask * path.opacity
                result = result + alpha.unsqueeze(-1) * color.view(1, 1, 3)
                total_alpha = total_alpha + alpha

            # Normalize where alpha > 0
            mask = total_alpha > 0.01
            result[mask] = result[mask] / total_alpha[mask].unsqueeze(-1)

            return result, total_alpha

        def compute_xing_loss(self) -> torch.Tensor:
            """Compute self-crossing loss to prevent path self-intersection.

            Returns:
                Crossing loss scalar
            """
            total_loss = torch.tensor(0.0, device=self.device)

            for path in self.paths:
                points = path.sample_points(50)
                n = points.shape[0]

                # Check for edge crossings
                for i in range(n):
                    p1, p2 = points[i], points[(i + 1) % n]

                    for j in range(i + 2, n):
                        if j == (i - 1) % n:
                            continue  # Skip adjacent edges

                        p3, p4 = points[j], points[(j + 1) % n]

                        # Line segment intersection test (soft version)
                        d1 = p2 - p1
                        d2 = p4 - p3
                        d3 = p3 - p1

                        cross = d1[0] * d2[1] - d1[1] * d2[0]

                        if torch.abs(cross) < 1e-6:
                            continue

                        t = (d3[0] * d2[1] - d3[1] * d2[0]) / cross
                        u = (d3[0] * d1[1] - d3[1] * d1[0]) / cross

                        # Soft penalty for intersection
                        if 0 < t < 1 and 0 < u < 1:
                            penalty = torch.exp(-((t - 0.5) ** 2 + (u - 0.5) ** 2))
                            total_loss = total_loss + penalty

            return total_loss

        def get_vector_layer(self, layer_index: int) -> VectorLayer:
            """Extract as VectorLayer object."""
            paths = [path.get_vector_path() for path in self.paths]

            # Use average color as layer color
            if paths:
                avg_r = sum(p.fill_color[0] for p in paths) // len(paths)
                avg_g = sum(p.fill_color[1] for p in paths) // len(paths)
                avg_b = sum(p.fill_color[2] for p in paths) // len(paths)
                fill_color = (avg_r, avg_g, avg_b)
            else:
                fill_color = (255, 255, 255)

            return VectorLayer(
                layer_index=layer_index,
                paths=paths,
                fill_color=fill_color,
            )


class LIVEVectorizer:
    """Main LIVE vectorization class.

    Implements layer-by-layer optimization:
    1. Initialize first layer with rough approximation
    2. Optimize using UDF + Xing losses
    3. Compute residual (target - rendered)
    4. Add next layer to capture residual
    5. Repeat for all layers
    """

    _instance: "LIVEVectorizer | None" = None

    def __init__(self, config: LIVEConfig | None = None):
        """Initialize vectorizer.

        Args:
            config: Configuration (uses defaults if None)
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for LIVE vectorization. "
                "Install with: pip install torch torchvision"
            )

        self.config = config or LIVEConfig()
        self.device = self._get_device()
        self._layers: list[LIVELayer] = []

        logger.info(f"LIVEVectorizer initialized on {self.device}")

    @classmethod
    def get_instance(cls, config: LIVEConfig | None = None) -> "LIVEVectorizer":
        """Get singleton instance."""
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
            logger.warning(f"GPU {device_id} not available, using CPU")
            return torch.device("cpu")

        return torch.device(f"cuda:{device_id}")

    def _load_image(self, image_path: str | Path) -> torch.Tensor:
        """Load image as RGB tensor.

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

    def _compute_udf_loss(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute unsigned distance field loss.

        This measures how well the rendered result matches the target
        in terms of spatial structure.

        Args:
            rendered: Rendered image (H, W, 3)
            target: Target image (H, W, 3)

        Returns:
            UDF loss scalar
        """
        # Simple MSE as UDF approximation
        return F.mse_loss(rendered, target)

    def _compute_color_loss(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Compute color matching loss in covered regions.

        Args:
            rendered: Rendered image (H, W, 3)
            target: Target image (H, W, 3)
            alpha: Coverage mask (H, W)

        Returns:
            Color loss scalar
        """
        # Weight by coverage
        if alpha.sum() < 1:
            return torch.tensor(0.0, device=self.device)

        diff = (rendered - target) ** 2
        weighted_diff = diff * alpha.unsqueeze(-1)

        return weighted_diff.sum() / (alpha.sum() * 3 + 1e-6)

    def vectorize(
        self,
        image_path: str | Path,
        config: LIVEConfig | None = None,
        progress_callback: callable | None = None,
    ) -> LIVEResult:
        """Run LIVE vectorization.

        Args:
            image_path: Path to target image
            config: Override configuration
            progress_callback: Optional callback(layer, iteration, loss)

        Returns:
            LIVEResult with vector layers
        """
        config = config or self.config
        start_time = time.time()

        # Set random seed
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)

        # Load target image
        target = self._load_image(image_path)
        height, width = target.shape[:2]
        image_size = (width, height)

        logger.info(f"Loaded target image: {width}x{height}")

        # Initialize storage
        layers: list[LIVELayer] = []
        all_loss_history: list[list[float]] = []
        current_rendered = torch.ones_like(target)  # Start with white

        # Layer-by-layer optimization
        for layer_idx in range(config.num_layers):
            logger.info(f"Optimizing layer {layer_idx + 1}/{config.num_layers}")

            # Compute residual (what needs to be captured)
            residual = target - current_rendered

            # Initialize new layer
            layer = LIVELayer(
                num_paths=config.paths_per_layer,
                control_points_per_path=config.control_points_per_path,
                image_size=image_size,
                init_scale=config.path_init_scale,
                device=self.device,
            )

            # Setup optimizer
            optimizer = Adam(layer.parameters(), lr=config.learning_rate)

            layer_loss_history: list[float] = []

            # Optimize this layer
            for iteration in range(config.iterations_per_layer):
                optimizer.zero_grad()

                # Render current layer
                layer_render, layer_alpha = layer.render()

                # Composite with previous layers
                composite = current_rendered * (1 - layer_alpha.unsqueeze(-1)) + layer_render

                # Compute losses
                udf_loss = self._compute_udf_loss(composite, target)
                color_loss = self._compute_color_loss(layer_render, target, layer_alpha)
                xing_loss = layer.compute_xing_loss()

                # Regularization: prefer smooth paths
                reg_loss = torch.tensor(0.0, device=self.device)
                for path in layer.paths:
                    points = path.control_points
                    n = points.shape[0]
                    for i in range(n):
                        p0 = points[(i - 1) % n]
                        p1 = points[i]
                        p2 = points[(i + 1) % n]

                        # Penalize sharp angles
                        v1 = p1 - p0
                        v2 = p2 - p1
                        angle_penalty = 1 - F.cosine_similarity(
                            v1.unsqueeze(0), v2.unsqueeze(0)
                        )
                        reg_loss = reg_loss + angle_penalty.squeeze()

                # Total loss
                total_loss = (
                    config.udf_loss_weight * udf_loss +
                    config.color_loss_weight * color_loss +
                    config.xing_loss_weight * xing_loss +
                    config.regularization_weight * reg_loss
                )

                # Backprop
                total_loss.backward()
                optimizer.step()

                loss_val = total_loss.item()
                layer_loss_history.append(loss_val)

                # Progress callback
                if progress_callback is not None:
                    progress_callback(layer_idx, iteration, loss_val)

                # Log progress
                if (iteration + 1) % 100 == 0:
                    logger.info(
                        f"  Layer {layer_idx + 1}, Iter {iteration + 1}: "
                        f"loss={loss_val:.6f}"
                    )

            # Update composite for next layer
            with torch.no_grad():
                layer_render, layer_alpha = layer.render()
                current_rendered = (
                    current_rendered * (1 - layer_alpha.unsqueeze(-1)) +
                    layer_render
                )

            layers.append(layer)
            all_loss_history.append(layer_loss_history)

        # Extract vector layers
        vector_layers = [
            layer.get_vector_layer(i) for i, layer in enumerate(layers)
        ]

        processing_time = (time.time() - start_time) * 1000

        # Final loss
        final_loss = all_loss_history[-1][-1] if all_loss_history and all_loss_history[-1] else 0.0

        result = LIVEResult(
            layers=vector_layers,
            loss_history=all_loss_history,
            final_loss=final_loss,
            image_size=image_size,
            processing_time_ms=processing_time,
            config=config,
            device=str(self.device),
        )

        total_paths = sum(len(layer.paths) for layer in vector_layers)
        logger.info(
            f"LIVE complete: {len(vector_layers)} layers, {total_paths} paths, "
            f"final loss={final_loss:.6f}, time={processing_time:.1f}ms"
        )

        return result


def live_vectorize(
    image_path: str | Path,
    config: LIVEConfig | None = None,
    output_svg: str | Path | None = None,
) -> LIVEResult:
    """Convenience function for LIVE vectorization.

    Args:
        image_path: Path to input image
        config: Configuration (uses defaults if None)
        output_svg: Optional path to save SVG output

    Returns:
        LIVEResult with vector layers
    """
    if not is_live_available():
        raise ImportError(
            "LIVE vectorization requires PyTorch. "
            "Install with: pip install torch torchvision"
        )

    vectorizer = LIVEVectorizer(config)
    result = vectorizer.vectorize(image_path, config)

    if output_svg is not None:
        result.save_svg(output_svg)

    return result


__all__ = [
    "is_live_available",
    "LIVEConfig",
    "VectorPath",
    "VectorLayer",
    "LIVEResult",
    "LIVEVectorizer",
    "live_vectorize",
]
