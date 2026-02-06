#!/usr/bin/env python3
"""
YOLOv8 Training Script for MEP Symbol Detection.

This script trains a YOLOv8 model on MEP (Mechanical, Electrical, Plumbing)
symbols extracted from AutoCAD block libraries. The trained model can then
be exported to ONNX format for deployment.

Part of Phase 2.5.1: LLM-Enhanced Vectorization Pipeline.

Usage:
    # Prepare dataset from DWG blocks
    python scripts/train_yolo_symbols.py prepare --dwg-dir path/to/blocks

    # Train model
    python scripts/train_yolo_symbols.py train --data datasets/mep_symbols/data.yaml

    # Export to ONNX
    python scripts/train_yolo_symbols.py export --model runs/detect/train/weights/best.pt

    # Validate model
    python scripts/train_yolo_symbols.py validate --model models/mep_symbols.onnx

Requirements:
    pip install ultralytics opencv-python-headless numpy pillow
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def create_synthetic_symbol(
    symbol_type: str,
    size: int = 64,
    line_width: int = 2,
) -> "np.ndarray":
    """
    Create a synthetic symbol image for training data augmentation.

    This generates simple geometric representations of MEP symbols
    when actual DWG blocks are not available.

    Args:
        symbol_type: Type of symbol to create (e.g., "valve", "outlet", "detector")
        size: Image size in pixels (square)
        line_width: Line thickness

    Returns:
        Grayscale numpy array (white background, black symbol)
    """
    import cv2

    img = np.ones((size, size), dtype=np.uint8) * 255
    center = size // 2
    half = size // 3

    if "valve" in symbol_type.lower():
        # Bowtie shape for valves
        pts = np.array(
            [
                [center - half, center - half],
                [center + half, center],
                [center - half, center + half],
            ],
            np.int32,
        )
        cv2.polylines(img, [pts], True, 0, line_width)
        pts2 = np.array(
            [
                [center + half, center - half],
                [center - half, center],
                [center + half, center + half],
            ],
            np.int32,
        )
        cv2.polylines(img, [pts2], True, 0, line_width)

    elif "outlet" in symbol_type.lower() or "receptacle" in symbol_type.lower():
        # Circle with parallel lines
        cv2.circle(img, (center, center), half, 0, line_width)
        cv2.line(img, (center - half // 2, center - half // 2), (center - half // 2, center + half // 2), 0, line_width)
        cv2.line(img, (center + half // 2, center - half // 2), (center + half // 2, center + half // 2), 0, line_width)

    elif "switch" in symbol_type.lower():
        # S-shape in circle
        cv2.circle(img, (center, center), half, 0, line_width)
        cv2.line(img, (center - half // 2, center), (center, center - half // 2), 0, line_width)
        cv2.line(img, (center, center - half // 2), (center + half // 2, center), 0, line_width)

    elif "detector" in symbol_type.lower() or "smoke" in symbol_type.lower():
        # Circle with dot
        cv2.circle(img, (center, center), half, 0, line_width)
        cv2.circle(img, (center, center), half // 3, 0, -1)

    elif "sprinkler" in symbol_type.lower():
        # Circle with cross
        cv2.circle(img, (center, center), half, 0, line_width)
        cv2.line(img, (center - half, center), (center + half, center), 0, line_width)
        cv2.line(img, (center, center - half), (center, center + half), 0, line_width)

    elif "diffuser" in symbol_type.lower():
        # Square with X
        cv2.rectangle(img, (center - half, center - half), (center + half, center + half), 0, line_width)
        cv2.line(img, (center - half, center - half), (center + half, center + half), 0, line_width)
        cv2.line(img, (center + half, center - half), (center - half, center + half), 0, line_width)

    elif "drain" in symbol_type.lower():
        # Circle with grid
        cv2.circle(img, (center, center), half, 0, line_width)
        for i in range(-half + half // 3, half, half // 3):
            cv2.line(img, (center + i, center - half), (center + i, center + half), 0, 1)
            cv2.line(img, (center - half, center + i), (center + half, center + i), 0, 1)

    elif "panel" in symbol_type.lower():
        # Rectangle with internal divisions
        cv2.rectangle(img, (center - half, center - half), (center + half, center + half), 0, line_width)
        cv2.line(img, (center, center - half), (center, center + half), 0, line_width)

    elif "light" in symbol_type.lower():
        # Circle with rays
        cv2.circle(img, (center, center), half // 2, 0, line_width)
        for angle in range(0, 360, 45):
            rad = np.radians(angle)
            x1 = int(center + (half // 2 + 2) * np.cos(rad))
            y1 = int(center + (half // 2 + 2) * np.sin(rad))
            x2 = int(center + half * np.cos(rad))
            y2 = int(center + half * np.sin(rad))
            cv2.line(img, (x1, y1), (x2, y2), 0, line_width)

    else:
        # Generic symbol: circle with cross
        cv2.circle(img, (center, center), half, 0, line_width)
        cv2.line(img, (center - half // 2, center), (center + half // 2, center), 0, line_width)

    return img


def augment_symbol(
    img: "np.ndarray",
    rotation: int = 0,
    scale: float = 1.0,
    noise_level: float = 0.0,
    blur_kernel: int = 0,
) -> "np.ndarray":
    """
    Apply augmentation to a symbol image.

    Args:
        img: Input grayscale image
        rotation: Rotation angle in degrees (0, 90, 180, 270)
        scale: Scale factor
        noise_level: Standard deviation of Gaussian noise (0-50)
        blur_kernel: Blur kernel size (0 = no blur)

    Returns:
        Augmented image
    """
    import cv2

    result = img.copy()

    # Rotation
    if rotation != 0:
        if rotation == 90:
            result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation == 180:
            result = cv2.rotate(result, cv2.ROTATE_180)
        elif rotation == 270:
            result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
        else:
            h, w = result.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, rotation, 1.0)
            result = cv2.warpAffine(result, M, (w, h), borderValue=255)

    # Scale
    if scale != 1.0:
        h, w = result.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        result = cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad or crop to original size
        if new_h < h or new_w < w:
            padded = np.ones((h, w), dtype=np.uint8) * 255
            y_off = (h - new_h) // 2
            x_off = (w - new_w) // 2
            padded[y_off : y_off + new_h, x_off : x_off + new_w] = result
            result = padded
        elif new_h > h or new_w > w:
            y_off = (new_h - h) // 2
            x_off = (new_w - w) // 2
            result = result[y_off : y_off + h, x_off : x_off + w]

    # Noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, result.shape).astype(np.float32)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Blur
    if blur_kernel > 0:
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        result = cv2.GaussianBlur(result, (blur_kernel, blur_kernel), 0)

    return result


def create_yolo_annotation(
    img_width: int,
    img_height: int,
    symbol_bbox: Tuple[int, int, int, int],
    class_id: int,
) -> str:
    """
    Create YOLO format annotation string.

    YOLO format: class_id center_x center_y width height
    All values normalized to [0, 1]

    Args:
        img_width: Image width in pixels
        img_height: Image height in pixels
        symbol_bbox: Bounding box (x1, y1, x2, y2) in pixels
        class_id: Class index

    Returns:
        YOLO annotation string
    """
    x1, y1, x2, y2 = symbol_bbox

    # Calculate center and size
    cx = (x1 + x2) / 2 / img_width
    cy = (y1 + y2) / 2 / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height

    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def prepare_synthetic_dataset(
    output_dir: str,
    num_samples_per_class: int = 500,
    image_size: int = 640,
    symbol_sizes: List[int] = [32, 48, 64],
    train_split: float = 0.8,
    val_split: float = 0.1,
) -> None:
    """
    Prepare a synthetic training dataset.

    Creates images with randomly placed symbols and corresponding YOLO annotations.

    Args:
        output_dir: Output directory for dataset
        num_samples_per_class: Number of training samples per class
        image_size: Training image size
        symbol_sizes: List of symbol sizes to use
        train_split: Fraction for training
        val_split: Fraction for validation (rest is test)
    """
    import cv2

    output_path = Path(output_dir)

    # Class definitions matching DEFAULT_CLASS_MAP in yolo_detection.py
    classes = [
        ("valve_gate", "mechanical"),
        ("valve_ball", "mechanical"),
        ("valve_butterfly", "mechanical"),
        ("valve_check", "mechanical"),
        ("diffuser_supply", "mechanical"),
        ("diffuser_return", "mechanical"),
        ("diffuser_exhaust", "mechanical"),
        ("duct_elbow", "mechanical"),
        ("duct_tee", "mechanical"),
        ("pump", "mechanical"),
        ("ahu", "mechanical"),
        ("fcu", "mechanical"),
        ("vav_box", "mechanical"),
        ("outlet_duplex", "electrical"),
        ("outlet_gfci", "electrical"),
        ("outlet_quad", "electrical"),
        ("switch_single", "electrical"),
        ("switch_3way", "electrical"),
        ("switch_dimmer", "electrical"),
        ("panel_electrical", "electrical"),
        ("light_fixture", "electrical"),
        ("light_recessed", "electrical"),
        ("light_exit", "electrical"),
        ("junction_box", "electrical"),
        ("smoke_detector", "fire"),
        ("heat_detector", "fire"),
        ("pull_station", "fire"),
        ("horn_strobe", "fire"),
        ("sprinkler_head", "fire"),
        ("fire_extinguisher", "fire"),
        ("floor_drain", "plumbing"),
        ("cleanout", "plumbing"),
        ("fixture_sink", "plumbing"),
        ("fixture_toilet", "plumbing"),
        ("fixture_urinal", "plumbing"),
        ("water_heater", "plumbing"),
        ("data_outlet", "low_voltage"),
        ("camera_security", "low_voltage"),
        ("card_reader", "low_voltage"),
        ("speaker", "low_voltage"),
        ("thermostat", "low_voltage"),
        ("unknown", "unknown"),
    ]

    # Create directory structure
    for split in ["train", "val", "test"]:
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Create data.yaml
    data_yaml = {
        "path": str(output_path.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(classes),
        "names": [c[0].upper().replace("_", "-") for c in classes],
    }

    with open(output_path / "data.yaml", "w") as f:
        import yaml  # type: ignore

        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"Creating synthetic dataset with {len(classes)} classes...")
    print(f"Samples per class: {num_samples_per_class}")

    total_images = 0

    for class_id, (class_name, category) in enumerate(classes):
        print(f"  [{class_id + 1}/{len(classes)}] Generating {class_name}...")

        for i in range(num_samples_per_class):
            # Determine split
            r = random.random()
            if r < train_split:
                split = "train"
            elif r < train_split + val_split:
                split = "val"
            else:
                split = "test"

            # Create background image
            img = np.ones((image_size, image_size), dtype=np.uint8) * 255

            # Add some noise/texture to background
            noise = np.random.randint(240, 256, (image_size, image_size), dtype=np.uint8)
            img = np.minimum(img, noise)

            annotations = []

            # Add 1-5 symbols per image
            num_symbols = random.randint(1, 5)

            for _ in range(num_symbols):
                # Random symbol size
                sym_size = random.choice(symbol_sizes)

                # Generate symbol
                base_symbol = create_synthetic_symbol(class_name, sym_size)

                # Apply augmentation
                rotation = random.choice([0, 90, 180, 270])
                scale = random.uniform(0.8, 1.2)
                noise_level = random.uniform(0, 10)
                blur = random.choice([0, 0, 0, 3])

                symbol = augment_symbol(base_symbol, rotation, scale, noise_level, blur)
                sym_h, sym_w = symbol.shape[:2]

                # Random position
                max_x = image_size - sym_w - 10
                max_y = image_size - sym_h - 10
                if max_x <= 10 or max_y <= 10:
                    continue

                x = random.randint(10, max_x)
                y = random.randint(10, max_y)

                # Paste symbol (minimum blend for overlap)
                region = img[y : y + sym_h, x : x + sym_w]
                img[y : y + sym_h, x : x + sym_w] = np.minimum(region, symbol)

                # Add annotation
                ann = create_yolo_annotation(image_size, image_size, (x, y, x + sym_w, y + sym_h), class_id)
                annotations.append(ann)

            # Save image and annotation
            img_name = f"{class_name}_{i:04d}"
            cv2.imwrite(str(output_path / "images" / split / f"{img_name}.png"), img)

            with open(output_path / "labels" / split / f"{img_name}.txt", "w") as f:
                f.write("\n".join(annotations))

            total_images += 1

    print(f"\nDataset created: {total_images} images")
    print(f"Output: {output_path}")
    print(f"Config: {output_path / 'data.yaml'}")


def train_yolo_model(
    data_yaml: str,
    model_size: str = "n",
    epochs: int = 100,
    batch_size: int = 16,
    image_size: int = 640,
    device: Optional[str] = None,
    project: str = "runs/detect",
    name: str = "mep_symbols",
) -> str:
    """
    Train YOLOv8 model on MEP symbol dataset.

    Args:
        data_yaml: Path to data.yaml file
        model_size: Model size ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        batch_size: Batch size
        image_size: Training image size
        device: Device to train on
        project: Project directory
        name: Run name

    Returns:
        Path to best model weights
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Load pretrained model
    model = YOLO(f"yolov8{model_size}.pt")

    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        verbose=True,
    )

    # Return path to best weights
    best_path = Path(project) / name / "weights" / "best.pt"
    print(f"\nTraining complete!")
    print(f"Best model: {best_path}")

    return str(best_path)


def export_to_onnx(
    model_path: str,
    output_path: Optional[str] = None,
    image_size: int = 640,
    simplify: bool = True,
    dynamic: bool = False,
) -> str:
    """
    Export YOLOv8 model to ONNX format.

    Args:
        model_path: Path to .pt model file
        output_path: Output ONNX path (default: same as input with .onnx)
        image_size: Export image size
        simplify: Simplify ONNX model
        dynamic: Enable dynamic input shapes

    Returns:
        Path to exported ONNX model
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    model = YOLO(model_path)

    # Export
    onnx_path = model.export(
        format="onnx",
        imgsz=image_size,
        simplify=simplify,
        dynamic=dynamic,
    )

    # Move to desired location if specified
    if output_path:
        onnx_path = Path(onnx_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(onnx_path), str(output_path))
        onnx_path = output_path

    print(f"\nONNX export complete: {onnx_path}")
    return str(onnx_path)


def validate_model(
    model_path: str,
    test_image: Optional[str] = None,
    conf_threshold: float = 0.5,
) -> None:
    """
    Validate a trained model.

    Args:
        model_path: Path to model (.pt or .onnx)
        test_image: Optional test image path
        conf_threshold: Confidence threshold
    """
    import cv2

    # Add project root to path for imports
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root / "src"))

    from aec_agent.mcp.tools.yolo_detection import YOLOSymbolDetector

    print(f"Loading model: {model_path}")
    detector = YOLOSymbolDetector(model_path, conf_threshold=conf_threshold)

    if not detector.is_available:
        print("Error: Failed to load model")
        sys.exit(1)

    print(f"Backend: {detector._backend}")
    print(f"Confidence threshold: {conf_threshold}")

    if test_image:
        print(f"\nTesting on: {test_image}")
        img = cv2.imread(test_image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not load image: {test_image}")
            sys.exit(1)

        _, blocks = detector.detect(img, scale=1 / 300)

        print(f"\nDetected {len(blocks)} symbols:")
        for block in blocks:
            print(f"  - {block.block_name} at {block.position} (conf: {block.confidence:.2f})")
    else:
        print("\nModel loaded successfully. Use --test-image to run inference.")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="YOLOv8 Training Script for MEP Symbol Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Prepare command
    prepare_parser = subparsers.add_parser("prepare", help="Prepare training dataset")
    prepare_parser.add_argument("--output-dir", default="datasets/mep_symbols", help="Output directory")
    prepare_parser.add_argument("--samples-per-class", type=int, default=500, help="Samples per class")
    prepare_parser.add_argument("--image-size", type=int, default=640, help="Image size")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train YOLOv8 model")
    train_parser.add_argument("--data", required=True, help="Path to data.yaml")
    train_parser.add_argument("--model-size", default="n", choices=["n", "s", "m", "l", "x"], help="Model size")
    train_parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    train_parser.add_argument("--image-size", type=int, default=640, help="Image size")
    train_parser.add_argument("--device", help="Device (cpu, cuda, mps)")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export model to ONNX")
    export_parser.add_argument("--model", required=True, help="Path to .pt model")
    export_parser.add_argument("--output", help="Output ONNX path")
    export_parser.add_argument("--image-size", type=int, default=640, help="Export image size")
    export_parser.add_argument("--no-simplify", action="store_true", help="Disable ONNX simplification")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate trained model")
    validate_parser.add_argument("--model", required=True, help="Path to model")
    validate_parser.add_argument("--test-image", help="Test image path")
    validate_parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")

    args = parser.parse_args()

    if args.command == "prepare":
        prepare_synthetic_dataset(
            args.output_dir,
            num_samples_per_class=args.samples_per_class,
            image_size=args.image_size,
        )

    elif args.command == "train":
        train_yolo_model(
            args.data,
            model_size=args.model_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            device=args.device,
        )

    elif args.command == "export":
        export_to_onnx(
            args.model,
            output_path=args.output,
            image_size=args.image_size,
            simplify=not args.no_simplify,
        )

    elif args.command == "validate":
        validate_model(args.model, test_image=args.test_image, conf_threshold=args.conf)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
