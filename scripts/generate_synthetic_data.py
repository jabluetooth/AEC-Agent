#!/usr/bin/env python3
"""
Comprehensive Synthetic Data Generator for MEP Symbol YOLO Training.

This script generates realistic synthetic training data for YOLOv8 MEP symbol
detection. It creates detailed symbol representations that closely match
actual CAD drawing symbols.

Features:
- Detailed renderers for all 42 MEP symbol classes
- Realistic AEC drawing backgrounds (grids, dimension lines, text)
- Advanced augmentations (perspective, line quality variation, scan artifacts)
- Multiple symbols per image with proper spacing
- Configurable noise, blur, and distortion levels

Usage:
    python scripts/generate_synthetic_data.py --output datasets/mep_symbols_v2
    python scripts/generate_synthetic_data.py --output datasets/mep_symbols_v2 --samples 1000 --workers 4

Requirements:
    pip install opencv-python-headless numpy pillow pyyaml tqdm
"""

import argparse
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


# =============================================================================
# Symbol Class Definitions (matching yolo_detection.py DEFAULT_CLASS_MAP)
# =============================================================================

@dataclass
class SymbolClass:
    """Definition of a symbol class."""
    class_id: int
    name: str
    category: str
    renderer: str  # Name of renderer function


SYMBOL_CLASSES: List[SymbolClass] = [
    # Mechanical - Valves
    SymbolClass(0, "VALVE-GATE", "mechanical", "render_gate_valve"),
    SymbolClass(1, "VALVE-BALL", "mechanical", "render_ball_valve"),
    SymbolClass(2, "VALVE-BUTTERFLY", "mechanical", "render_butterfly_valve"),
    SymbolClass(3, "VALVE-CHECK", "mechanical", "render_check_valve"),
    # Mechanical - Diffusers
    SymbolClass(4, "DIFFUSER-SUPPLY", "mechanical", "render_supply_diffuser"),
    SymbolClass(5, "DIFFUSER-RETURN", "mechanical", "render_return_diffuser"),
    SymbolClass(6, "DIFFUSER-EXHAUST", "mechanical", "render_exhaust_diffuser"),
    # Mechanical - Duct Fittings
    SymbolClass(7, "DUCT-ELBOW", "mechanical", "render_duct_elbow"),
    SymbolClass(8, "DUCT-TEE", "mechanical", "render_duct_tee"),
    # Mechanical - Equipment
    SymbolClass(9, "PUMP", "mechanical", "render_pump"),
    SymbolClass(10, "AHU", "mechanical", "render_ahu"),
    SymbolClass(11, "FCU", "mechanical", "render_fcu"),
    SymbolClass(12, "VAV-BOX", "mechanical", "render_vav_box"),
    # Electrical - Outlets
    SymbolClass(13, "OUTLET-DUPLEX", "electrical", "render_duplex_outlet"),
    SymbolClass(14, "OUTLET-GFCI", "electrical", "render_gfci_outlet"),
    SymbolClass(15, "OUTLET-QUAD", "electrical", "render_quad_outlet"),
    # Electrical - Switches
    SymbolClass(16, "SWITCH-SINGLE", "electrical", "render_single_switch"),
    SymbolClass(17, "SWITCH-3WAY", "electrical", "render_3way_switch"),
    SymbolClass(18, "SWITCH-DIMMER", "electrical", "render_dimmer_switch"),
    # Electrical - Panels & Lights
    SymbolClass(19, "PANEL-ELECTRICAL", "electrical", "render_electrical_panel"),
    SymbolClass(20, "LIGHT-FIXTURE", "electrical", "render_light_fixture"),
    SymbolClass(21, "LIGHT-RECESSED", "electrical", "render_recessed_light"),
    SymbolClass(22, "LIGHT-EXIT", "electrical", "render_exit_light"),
    SymbolClass(23, "JUNCTION-BOX", "electrical", "render_junction_box"),
    # Fire Alarm
    SymbolClass(24, "SMOKE-DETECTOR", "fire", "render_smoke_detector"),
    SymbolClass(25, "HEAT-DETECTOR", "fire", "render_heat_detector"),
    SymbolClass(26, "PULL-STATION", "fire", "render_pull_station"),
    SymbolClass(27, "HORN-STROBE", "fire", "render_horn_strobe"),
    SymbolClass(28, "SPRINKLER-HEAD", "fire", "render_sprinkler_head"),
    SymbolClass(29, "FIRE-EXTINGUISHER", "fire", "render_fire_extinguisher"),
    # Plumbing
    SymbolClass(30, "FLOOR-DRAIN", "plumbing", "render_floor_drain"),
    SymbolClass(31, "CLEANOUT", "plumbing", "render_cleanout"),
    SymbolClass(32, "FIXTURE-SINK", "plumbing", "render_sink"),
    SymbolClass(33, "FIXTURE-TOILET", "plumbing", "render_toilet"),
    SymbolClass(34, "FIXTURE-URINAL", "plumbing", "render_urinal"),
    SymbolClass(35, "WATER-HEATER", "plumbing", "render_water_heater"),
    # Low Voltage
    SymbolClass(36, "DATA-OUTLET", "low_voltage", "render_data_outlet"),
    SymbolClass(37, "CAMERA-SECURITY", "low_voltage", "render_security_camera"),
    SymbolClass(38, "CARD-READER", "low_voltage", "render_card_reader"),
    SymbolClass(39, "SPEAKER", "low_voltage", "render_speaker"),
    SymbolClass(40, "THERMOSTAT", "low_voltage", "render_thermostat"),
    # Unknown
    SymbolClass(41, "UNKNOWN", "unknown", "render_unknown"),
]


# =============================================================================
# Symbol Renderers - Detailed implementations for each symbol type
# =============================================================================

def _draw_text(img: np.ndarray, text: str, pos: Tuple[int, int],
               font_scale: float = 0.3, thickness: int = 1) -> None:
    """Draw text on image."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 0, thickness)


def render_gate_valve(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Gate valve - bowtie/hourglass shape."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    h = size // 3

    # Left triangle
    pts1 = np.array([[c - h, c - h], [c, c], [c - h, c + h]], np.int32)
    cv2.fillPoly(img, [pts1], 200)
    cv2.polylines(img, [pts1], True, 0, line_width)

    # Right triangle
    pts2 = np.array([[c + h, c - h], [c, c], [c + h, c + h]], np.int32)
    cv2.fillPoly(img, [pts2], 200)
    cv2.polylines(img, [pts2], True, 0, line_width)

    # Stem
    cv2.line(img, (c, c - h - 5), (c, c - h - 15), 0, line_width)
    cv2.line(img, (c - 5, c - h - 15), (c + 5, c - h - 15), 0, line_width)

    return img


def render_ball_valve(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Ball valve - circle with line through center."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    r = size // 4

    # Circle body
    cv2.circle(img, (c, c), r, 0, line_width)

    # Line through center (handle position)
    cv2.line(img, (c - r - 5, c), (c + r + 5, c), 0, line_width)

    # Handle indicator
    cv2.line(img, (c, c - r), (c, c - r - 10), 0, line_width)

    return img


def render_butterfly_valve(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Butterfly valve - circle with curved wings."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    r = size // 4

    # Outer circle
    cv2.circle(img, (c, c), r, 0, line_width)

    # Butterfly wings (curved lines)
    # Left wing
    cv2.ellipse(img, (c - r//2, c), (r//2, r//2), 0, 90, 270, 0, line_width)
    # Right wing
    cv2.ellipse(img, (c + r//2, c), (r//2, r//2), 0, -90, 90, 0, line_width)

    # Center line
    cv2.line(img, (c, c - r), (c, c + r), 0, line_width)

    return img


def render_check_valve(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Check valve - triangle/arrow indicating flow direction."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    h = size // 3

    # Arrow pointing right
    pts = np.array([
        [c - h, c - h//2],
        [c + h//2, c],
        [c - h, c + h//2],
    ], np.int32)
    cv2.polylines(img, [pts], True, 0, line_width)

    # Vertical stop line
    cv2.line(img, (c + h//2, c - h), (c + h//2, c + h), 0, line_width)

    # Inlet/outlet lines
    cv2.line(img, (c - h - 10, c), (c - h, c), 0, line_width)
    cv2.line(img, (c + h//2, c), (c + h//2 + 10, c), 0, line_width)

    return img


def render_supply_diffuser(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Supply air diffuser - square with internal pattern."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    h = size // 3

    # Outer square
    cv2.rectangle(img, (c - h, c - h), (c + h, c + h), 0, line_width)

    # Inner concentric squares (4-way throw pattern)
    for i in range(1, 4):
        offset = h - (i * h // 4)
        cv2.rectangle(img, (c - offset, c - offset), (c + offset, c + offset), 0, 1)

    # Center dot
    cv2.circle(img, (c, c), 3, 0, -1)

    return img


def render_return_diffuser(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Return air grille - square with horizontal lines (louvers)."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    h = size // 3

    # Outer square
    cv2.rectangle(img, (c - h, c - h), (c + h, c + h), 0, line_width)

    # Horizontal louver lines
    num_lines = 5
    for i in range(num_lines):
        y = c - h + (i + 1) * (2 * h) // (num_lines + 1)
        cv2.line(img, (c - h + 3, y), (c + h - 3, y), 0, 1)

    return img


def render_exhaust_diffuser(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Exhaust grille - square with X pattern and E label."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    h = size // 3

    # Outer square
    cv2.rectangle(img, (c - h, c - h), (c + h, c + h), 0, line_width)

    # X pattern
    cv2.line(img, (c - h, c - h), (c + h, c + h), 0, 1)
    cv2.line(img, (c + h, c - h), (c - h, c + h), 0, 1)

    # "E" label
    _draw_text(img, "E", (c - 4, c + 4), 0.4, 1)

    return img


def render_duct_elbow(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Duct elbow - 90 degree turn."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    h = size // 3

    # Two parallel arcs
    cv2.ellipse(img, (c - h//2, c + h//2), (h, h), 0, -90, 0, 0, line_width)
    cv2.ellipse(img, (c - h//2, c + h//2), (h//2, h//2), 0, -90, 0, 0, line_width)

    # Connecting lines
    cv2.line(img, (c - h//2, c - h//2), (c - h//2, c + h//2 - h//2), 0, line_width)
    cv2.line(img, (c + h//2, c + h//2), (c - h//2 + h, c + h//2), 0, line_width)

    return img


def render_duct_tee(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Duct tee - T-junction."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    h = size // 3
    w = h // 2

    # Main duct (horizontal)
    cv2.rectangle(img, (c - h, c - w), (c + h, c + w), 0, line_width)

    # Branch duct (vertical down)
    cv2.rectangle(img, (c - w, c + w), (c + w, c + h), 0, line_width)

    # Remove overlapping line
    cv2.line(img, (c - w + 1, c + w), (c + w - 1, c + w), 255, line_width)

    return img


def render_pump(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Pump - circle with inlet/outlet."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    r = size // 4

    # Pump body (circle)
    cv2.circle(img, (c, c), r, 0, line_width)

    # Inlet (left)
    cv2.line(img, (c - r - 10, c), (c - r, c), 0, line_width)

    # Outlet (top with arrow)
    cv2.line(img, (c, c - r), (c, c - r - 10), 0, line_width)
    # Arrow head
    cv2.line(img, (c - 4, c - r - 6), (c, c - r - 10), 0, line_width)
    cv2.line(img, (c + 4, c - r - 6), (c, c - r - 10), 0, line_width)

    # Impeller indicator
    cv2.line(img, (c - r//2, c), (c + r//2, c), 0, 1)

    return img


def render_ahu(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Air Handling Unit - rectangle with sections."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    h = size // 3
    w = h * 2

    # Main rectangle
    cv2.rectangle(img, (c - w//2, c - h//2), (c + w//2, c + h//2), 0, line_width)

    # Internal sections
    cv2.line(img, (c - w//4, c - h//2), (c - w//4, c + h//2), 0, 1)
    cv2.line(img, (c + w//4, c - h//2), (c + w//4, c + h//2), 0, 1)

    # Fan symbol in center
    cv2.circle(img, (c, c), h//4, 0, 1)

    # Label
    _draw_text(img, "AHU", (c - 10, c + h//2 + 12), 0.3, 1)

    return img


def render_fcu(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Fan Coil Unit - smaller rectangle with fan."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    h = size // 4
    w = h * 2

    # Main rectangle
    cv2.rectangle(img, (c - w//2, c - h), (c + w//2, c + h), 0, line_width)

    # Fan circle
    cv2.circle(img, (c - w//4, c), h//2, 0, 1)

    # Coil lines
    for i in range(-h//2, h//2 + 1, 4):
        cv2.line(img, (c + 2, c + i), (c + w//2 - 4, c + i), 0, 1)

    return img


def render_vav_box(size: int = 64, line_width: int = 2) -> np.ndarray:
    """VAV Box - rectangle with damper symbol."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    h = size // 4
    w = h * 2

    # Main rectangle
    cv2.rectangle(img, (c - w//2, c - h), (c + w//2, c + h), 0, line_width)

    # Damper blade (diagonal line)
    cv2.line(img, (c - w//4, c - h//2), (c + w//4, c + h//2), 0, line_width)

    # Actuator circle
    cv2.circle(img, (c, c - h - 5), 5, 0, line_width)

    # Label
    _draw_text(img, "VAV", (c - 10, c + 4), 0.3, 1)

    return img


def render_duplex_outlet(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Duplex outlet - circle with two parallel vertical lines."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    r = size // 3

    # Outer circle
    cv2.circle(img, (c, c), r, 0, line_width)

    # Two parallel vertical lines (slots)
    gap = r // 3
    slot_h = r // 2
    cv2.line(img, (c - gap, c - slot_h), (c - gap, c + slot_h), 0, line_width)
    cv2.line(img, (c + gap, c - slot_h), (c + gap, c + slot_h), 0, line_width)

    return img


def render_gfci_outlet(size: int = 64, line_width: int = 2) -> np.ndarray:
    """GFCI outlet - circle with wavy line or GFI text."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    r = size // 3

    # Outer circle
    cv2.circle(img, (c, c), r, 0, line_width)

    # Wavy line (sine wave pattern)
    pts = []
    for x in range(-r + 5, r - 5):
        y = int(5 * math.sin(x * 0.3))
        pts.append([c + x, c + y])
    pts = np.array(pts, np.int32)
    cv2.polylines(img, [pts], False, 0, line_width)

    # GFI text below
    _draw_text(img, "GFI", (c - 8, c + r - 5), 0.25, 1)

    return img


def render_quad_outlet(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Quad outlet - circle with four lines."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    r = size // 3

    # Outer circle
    cv2.circle(img, (c, c), r, 0, line_width)

    # Four vertical lines (quad slots)
    gap = r // 4
    slot_h = r // 3
    for i in [-2, -1, 1, 2]:
        x = c + i * gap
        cv2.line(img, (x, c - slot_h), (x, c + slot_h), 0, line_width)

    return img


def render_single_switch(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Single pole switch - circle with S."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    r = size // 3

    # Circle
    cv2.circle(img, (c, c), r, 0, line_width)

    # S letter
    _draw_text(img, "S", (c - 5, c + 5), 0.5, line_width)

    return img


def render_3way_switch(size: int = 64, line_width: int = 2) -> np.ndarray:
    """3-way switch - circle with S3."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    r = size // 3

    cv2.circle(img, (c, c), r, 0, line_width)
    _draw_text(img, "S3", (c - 8, c + 5), 0.4, line_width)

    return img


def render_dimmer_switch(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Dimmer switch - circle with SD."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    r = size // 3

    cv2.circle(img, (c, c), r, 0, line_width)
    _draw_text(img, "SD", (c - 8, c + 5), 0.4, line_width)

    return img


def render_electrical_panel(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Electrical panel - rectangle with busbar symbol."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    h = size // 3
    w = size // 4

    # Main rectangle
    cv2.rectangle(img, (c - w, c - h), (c + w, c + h), 0, line_width)

    # Busbar (vertical line)
    cv2.line(img, (c, c - h + 3), (c, c + h - 3), 0, line_width)

    # Branch lines
    for i in range(-h + 10, h - 5, 10):
        cv2.line(img, (c, c + i), (c + w - 5, c + i), 0, 1)

    return img


def render_light_fixture(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Light fixture - circle with rays."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    r = size // 5

    # Central circle
    cv2.circle(img, (c, c), r, 0, line_width)
    cv2.circle(img, (c, c), r - 3, 0, -1)  # Filled center

    # Light rays
    for angle in range(0, 360, 30):
        rad = math.radians(angle)
        x1 = int(c + (r + 3) * math.cos(rad))
        y1 = int(c + (r + 3) * math.sin(rad))
        x2 = int(c + (r + 10) * math.cos(rad))
        y2 = int(c + (r + 10) * math.sin(rad))
        cv2.line(img, (x1, y1), (x2, y2), 0, 1)

    return img


def render_recessed_light(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Recessed downlight - concentric circles."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2

    # Outer circle
    cv2.circle(img, (c, c), size // 3, 0, line_width)
    # Inner circle
    cv2.circle(img, (c, c), size // 5, 0, line_width)
    # Center dot
    cv2.circle(img, (c, c), 3, 0, -1)

    return img


def render_exit_light(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Exit sign - rectangle with EXIT text."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    h = size // 5
    w = size // 3

    # Rectangle
    cv2.rectangle(img, (c - w, c - h), (c + w, c + h), 0, line_width)

    # EXIT text
    _draw_text(img, "EXIT", (c - 12, c + 4), 0.35, 1)

    # Arrow above
    cv2.line(img, (c, c - h - 5), (c, c - h - 12), 0, line_width)
    cv2.line(img, (c - 4, c - h - 9), (c, c - h - 12), 0, line_width)
    cv2.line(img, (c + 4, c - h - 9), (c, c - h - 12), 0, line_width)

    return img


def render_junction_box(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Junction box - square or octagon."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    r = size // 4

    # Octagon
    pts = []
    for i in range(8):
        angle = math.radians(i * 45 + 22.5)
        x = int(c + r * math.cos(angle))
        y = int(c + r * math.sin(angle))
        pts.append([x, y])
    pts = np.array(pts, np.int32)
    cv2.polylines(img, [pts], True, 0, line_width)

    # J label
    _draw_text(img, "J", (c - 4, c + 5), 0.4, 1)

    return img


def render_smoke_detector(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Smoke detector - circle with S or dots."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    r = size // 3

    # Outer circle
    cv2.circle(img, (c, c), r, 0, line_width)

    # Dot pattern (smoke pattern)
    for _ in range(5):
        dx = random.randint(-r//2, r//2)
        dy = random.randint(-r//2, r//2)
        cv2.circle(img, (c + dx, c + dy), 2, 0, -1)

    # S label
    _draw_text(img, "S", (c - 4, c + r - 2), 0.3, 1)

    return img


def render_heat_detector(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Heat detector - circle with H."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    r = size // 3

    cv2.circle(img, (c, c), r, 0, line_width)
    _draw_text(img, "H", (c - 5, c + 5), 0.5, line_width)

    # Temperature indicator
    cv2.line(img, (c - r + 5, c + r - 8), (c + r - 5, c + r - 8), 0, 1)

    return img


def render_pull_station(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Pull station - rectangle with handle."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    h = size // 3
    w = size // 4

    # Rectangle body
    cv2.rectangle(img, (c - w, c - h), (c + w, c + h), 0, line_width)

    # Handle (T-shape)
    cv2.rectangle(img, (c - w//2, c - h//3), (c + w//2, c + h//3), 0, line_width)
    cv2.line(img, (c, c + h//3), (c, c + h - 3), 0, line_width)

    return img


def render_horn_strobe(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Horn/Strobe - rectangle with speaker and flash symbol."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    h = size // 3
    w = size // 3

    # Main rectangle
    cv2.rectangle(img, (c - w, c - h), (c + w, c + h), 0, line_width)

    # Speaker cone (left half)
    pts = np.array([
        [c - w + 5, c - 5],
        [c - w + 15, c - 10],
        [c - w + 15, c + 10],
        [c - w + 5, c + 5],
    ], np.int32)
    cv2.polylines(img, [pts], True, 0, 1)

    # Flash rays (right half)
    for angle in [0, 45, -45]:
        rad = math.radians(angle)
        x1 = c + 5
        y1 = c
        x2 = int(c + 15 * math.cos(rad) + 5)
        y2 = int(c + 15 * math.sin(rad))
        cv2.line(img, (x1, y1), (x2, y2), 0, 1)

    return img


def render_sprinkler_head(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Sprinkler head - circle with spray pattern."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    r = size // 5

    # Head circle
    cv2.circle(img, (c, c), r, 0, line_width)

    # Spray pattern (fan shape below)
    for angle in range(-60, 61, 20):
        rad = math.radians(angle + 90)
        x = int(c + (r + 15) * math.cos(rad))
        y = int(c + (r + 15) * math.sin(rad))
        cv2.line(img, (c, c + r), (x, y), 0, 1)

    # Pipe connection (above)
    cv2.line(img, (c, c - r), (c, c - r - 10), 0, line_width)

    return img


def render_fire_extinguisher(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Fire extinguisher - cylinder shape with handle."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    h = size // 3
    w = size // 5

    # Body (rounded rectangle)
    cv2.ellipse(img, (c, c - h + w), (w, w), 0, 180, 360, 0, line_width)
    cv2.ellipse(img, (c, c + h - w), (w, w), 0, 0, 180, 0, line_width)
    cv2.line(img, (c - w, c - h + w), (c - w, c + h - w), 0, line_width)
    cv2.line(img, (c + w, c - h + w), (c + w, c + h - w), 0, line_width)

    # Handle
    cv2.line(img, (c - 3, c - h), (c + 3, c - h), 0, line_width)
    cv2.line(img, (c, c - h), (c, c - h - 8), 0, line_width)
    cv2.line(img, (c, c - h - 8), (c + 8, c - h - 8), 0, line_width)

    # "FE" label
    _draw_text(img, "FE", (c - 6, c + 4), 0.3, 1)

    return img


def render_floor_drain(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Floor drain - circle with grid pattern."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    r = size // 3

    # Outer circle
    cv2.circle(img, (c, c), r, 0, line_width)

    # Grid pattern
    for i in range(-r + 5, r, 8):
        # Clip to circle
        half_len = int(math.sqrt(r*r - i*i)) if abs(i) < r else 0
        cv2.line(img, (c - half_len, c + i), (c + half_len, c + i), 0, 1)
        cv2.line(img, (c + i, c - half_len), (c + i, c + half_len), 0, 1)

    # FD label
    _draw_text(img, "FD", (c - 6, c + r + 10), 0.25, 1)

    return img


def render_cleanout(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Cleanout - circle with CO or cross."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    r = size // 3

    # Circle
    cv2.circle(img, (c, c), r, 0, line_width)

    # Crosshairs
    cv2.line(img, (c - r, c), (c + r, c), 0, line_width)
    cv2.line(img, (c, c - r), (c, c + r), 0, line_width)

    # CO label
    _draw_text(img, "CO", (c - 6, c + r + 10), 0.25, 1)

    return img


def render_sink(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Sink fixture - oval/rectangle shape."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    h = size // 4
    w = size // 3

    # Oval basin
    cv2.ellipse(img, (c, c), (w, h), 0, 0, 360, 0, line_width)

    # Faucet (top)
    cv2.circle(img, (c, c - h - 5), 3, 0, -1)
    cv2.line(img, (c, c - h - 5), (c, c - h - 12), 0, line_width)

    # Drain (center)
    cv2.circle(img, (c, c), 3, 0, -1)

    return img


def render_toilet(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Toilet/WC fixture - characteristic shape."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2

    # Tank (rectangle at top)
    tank_w = size // 5
    tank_h = size // 6
    cv2.rectangle(img, (c - tank_w, c - size//3), (c + tank_w, c - size//3 + tank_h), 0, line_width)

    # Bowl (oval)
    cv2.ellipse(img, (c, c + 5), (size//4, size//3), 0, 0, 360, 0, line_width)

    return img


def render_urinal(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Urinal fixture."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    w = size // 5
    h = size // 3

    # U-shape
    cv2.ellipse(img, (c, c + h//2), (w, h//2), 0, 0, 180, 0, line_width)
    cv2.line(img, (c - w, c - h + h//2), (c - w, c + h//2), 0, line_width)
    cv2.line(img, (c + w, c - h + h//2), (c + w, c + h//2), 0, line_width)
    cv2.line(img, (c - w, c - h + h//2), (c + w, c - h + h//2), 0, line_width)

    return img


def render_water_heater(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Water heater - cylinder with WH label."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    r = size // 4
    h = size // 3

    # Cylinder (circle top and bottom, lines on sides)
    cv2.ellipse(img, (c, c - h + 5), (r, 5), 0, 0, 360, 0, line_width)
    cv2.ellipse(img, (c, c + h - 5), (r, 5), 0, 180, 360, 0, line_width)
    cv2.line(img, (c - r, c - h + 5), (c - r, c + h - 5), 0, line_width)
    cv2.line(img, (c + r, c - h + 5), (c + r, c + h - 5), 0, line_width)

    # WH label
    _draw_text(img, "WH", (c - 8, c + 5), 0.35, 1)

    return img


def render_data_outlet(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Data outlet - triangle or diamond shape."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    h = size // 3

    # Triangle pointing up
    pts = np.array([
        [c, c - h],
        [c - h, c + h//2],
        [c + h, c + h//2],
    ], np.int32)
    cv2.polylines(img, [pts], True, 0, line_width)

    # D label
    _draw_text(img, "D", (c - 4, c + 4), 0.4, 1)

    return img


def render_security_camera(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Security camera - camera shape."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2

    # Camera body (trapezoid)
    pts = np.array([
        [c - 15, c - 8],
        [c + 5, c - 12],
        [c + 5, c + 12],
        [c - 15, c + 8],
    ], np.int32)
    cv2.polylines(img, [pts], True, 0, line_width)

    # Lens
    cv2.circle(img, (c + 10, c), 8, 0, line_width)
    cv2.circle(img, (c + 10, c), 4, 0, -1)

    return img


def render_card_reader(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Card reader - rectangle with slot."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    h = size // 3
    w = size // 5

    # Main rectangle
    cv2.rectangle(img, (c - w, c - h), (c + w, c + h), 0, line_width)

    # Card slot
    cv2.rectangle(img, (c - w + 3, c - h + 8), (c + w - 3, c - h + 12), 0, 1)

    # Indicator LED
    cv2.circle(img, (c, c + h - 8), 3, 0, -1)

    return img


def render_speaker(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Speaker - speaker cone symbol."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2

    # Speaker cone
    pts = np.array([
        [c - 10, c - 5],
        [c, c - 15],
        [c, c + 15],
        [c - 10, c + 5],
    ], np.int32)
    cv2.polylines(img, [pts], True, 0, line_width)

    # Mounting plate
    cv2.rectangle(img, (c - 15, c - 8), (c - 10, c + 8), 0, line_width)

    # Sound waves
    for r in [12, 18]:
        cv2.ellipse(img, (c + 5, c), (r, r), 0, -45, 45, 0, 1)

    return img


def render_thermostat(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Thermostat - circle with T."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    r = size // 3

    # Circle
    cv2.circle(img, (c, c), r, 0, line_width)

    # T label
    _draw_text(img, "T", (c - 5, c + 6), 0.5, line_width)

    # Temperature indicator
    cv2.line(img, (c - r + 5, c - r + 8), (c - r + 5, c + r - 8), 0, 1)

    return img


def render_unknown(size: int = 64, line_width: int = 2) -> np.ndarray:
    """Unknown symbol - question mark in circle."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    c = size // 2
    r = size // 3

    cv2.circle(img, (c, c), r, 0, line_width)
    _draw_text(img, "?", (c - 5, c + 6), 0.6, line_width)

    return img


# Map renderer names to functions
RENDERERS: Dict[str, Callable] = {
    "render_gate_valve": render_gate_valve,
    "render_ball_valve": render_ball_valve,
    "render_butterfly_valve": render_butterfly_valve,
    "render_check_valve": render_check_valve,
    "render_supply_diffuser": render_supply_diffuser,
    "render_return_diffuser": render_return_diffuser,
    "render_exhaust_diffuser": render_exhaust_diffuser,
    "render_duct_elbow": render_duct_elbow,
    "render_duct_tee": render_duct_tee,
    "render_pump": render_pump,
    "render_ahu": render_ahu,
    "render_fcu": render_fcu,
    "render_vav_box": render_vav_box,
    "render_duplex_outlet": render_duplex_outlet,
    "render_gfci_outlet": render_gfci_outlet,
    "render_quad_outlet": render_quad_outlet,
    "render_single_switch": render_single_switch,
    "render_3way_switch": render_3way_switch,
    "render_dimmer_switch": render_dimmer_switch,
    "render_electrical_panel": render_electrical_panel,
    "render_light_fixture": render_light_fixture,
    "render_recessed_light": render_recessed_light,
    "render_exit_light": render_exit_light,
    "render_junction_box": render_junction_box,
    "render_smoke_detector": render_smoke_detector,
    "render_heat_detector": render_heat_detector,
    "render_pull_station": render_pull_station,
    "render_horn_strobe": render_horn_strobe,
    "render_sprinkler_head": render_sprinkler_head,
    "render_fire_extinguisher": render_fire_extinguisher,
    "render_floor_drain": render_floor_drain,
    "render_cleanout": render_cleanout,
    "render_sink": render_sink,
    "render_toilet": render_toilet,
    "render_urinal": render_urinal,
    "render_water_heater": render_water_heater,
    "render_data_outlet": render_data_outlet,
    "render_security_camera": render_security_camera,
    "render_card_reader": render_card_reader,
    "render_speaker": render_speaker,
    "render_thermostat": render_thermostat,
    "render_unknown": render_unknown,
}


# =============================================================================
# Background Generators - Create realistic AEC drawing backgrounds
# =============================================================================

def create_blank_background(size: int) -> np.ndarray:
    """Plain white background with subtle noise."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    # Add subtle noise
    noise = np.random.randint(250, 256, (size, size), dtype=np.uint8)
    return np.minimum(img, noise)


def create_grid_background(size: int, grid_spacing: int = 50) -> np.ndarray:
    """Grid paper background."""
    img = np.ones((size, size), dtype=np.uint8) * 255

    # Light grid lines
    for i in range(0, size, grid_spacing):
        cv2.line(img, (i, 0), (i, size), 240, 1)
        cv2.line(img, (0, i), (size, i), 240, 1)

    return img


def create_drawing_background(size: int) -> np.ndarray:
    """Background with random lines simulating a drawing."""
    img = np.ones((size, size), dtype=np.uint8) * 255

    # Add some random horizontal and vertical lines (like walls/ducts)
    num_lines = random.randint(3, 8)
    for _ in range(num_lines):
        if random.random() > 0.5:
            # Horizontal line
            y = random.randint(20, size - 20)
            x1 = random.randint(0, size // 3)
            x2 = random.randint(2 * size // 3, size)
            cv2.line(img, (x1, y), (x2, y), 200, 1)
        else:
            # Vertical line
            x = random.randint(20, size - 20)
            y1 = random.randint(0, size // 3)
            y2 = random.randint(2 * size // 3, size)
            cv2.line(img, (x, y1), (x, y2), 200, 1)

    return img


def create_cluttered_background(size: int) -> np.ndarray:
    """Background with various drawing elements."""
    img = np.ones((size, size), dtype=np.uint8) * 255

    # Random lines
    for _ in range(random.randint(5, 15)):
        x1, y1 = random.randint(0, size), random.randint(0, size)
        x2, y2 = random.randint(0, size), random.randint(0, size)
        cv2.line(img, (x1, y1), (x2, y2), random.randint(180, 230), 1)

    # Random rectangles
    for _ in range(random.randint(1, 3)):
        x, y = random.randint(0, size - 50), random.randint(0, size - 50)
        w, h = random.randint(30, 100), random.randint(30, 100)
        cv2.rectangle(img, (x, y), (x + w, y + h), random.randint(180, 230), 1)

    # Random text-like patterns
    for _ in range(random.randint(0, 3)):
        x, y = random.randint(10, size - 50), random.randint(10, size - 20)
        text_len = random.randint(3, 8)
        text = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=text_len))
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 220, 1)

    return img


BACKGROUND_GENERATORS = [
    create_blank_background,
    create_grid_background,
    create_drawing_background,
    create_cluttered_background,
]


# =============================================================================
# Augmentation Functions
# =============================================================================

def augment_symbol(
    img: np.ndarray,
    rotation: float = 0,
    scale: float = 1.0,
    noise_level: float = 0,
    blur_kernel: int = 0,
    perspective_strength: float = 0,
    line_thickness_var: float = 0,
) -> np.ndarray:
    """Apply augmentations to a symbol image."""
    result = img.copy()
    h, w = result.shape[:2]

    # Rotation
    if rotation != 0:
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotation, 1.0)
        result = cv2.warpAffine(result, M, (w, h), borderValue=255)

    # Scale
    if scale != 1.0:
        new_h, new_w = int(h * scale), int(w * scale)
        result = cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad or crop to original size
        if new_h < h or new_w < w:
            padded = np.ones((h, w), dtype=np.uint8) * 255
            y_off = (h - new_h) // 2
            x_off = (w - new_w) // 2
            padded[y_off:y_off + new_h, x_off:x_off + new_w] = result
            result = padded
        elif new_h > h or new_w > w:
            y_off = (new_h - h) // 2
            x_off = (new_w - w) // 2
            result = result[y_off:y_off + h, x_off:x_off + w]

    # Perspective transform
    if perspective_strength > 0:
        h, w = result.shape[:2]
        strength = perspective_strength * 0.1  # Max 10% distortion
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([
            [random.uniform(0, strength * w), random.uniform(0, strength * h)],
            [w - random.uniform(0, strength * w), random.uniform(0, strength * h)],
            [random.uniform(0, strength * w), h - random.uniform(0, strength * h)],
            [w - random.uniform(0, strength * w), h - random.uniform(0, strength * h)],
        ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(result, M, (w, h), borderValue=255)

    # Line thickness variation (erosion/dilation)
    if line_thickness_var != 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        if line_thickness_var > 0:
            result = cv2.erode(result, kernel, iterations=1)  # Thicker lines
        else:
            result = cv2.dilate(result, kernel, iterations=1)  # Thinner lines

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


# =============================================================================
# Dataset Generation
# =============================================================================

def create_yolo_annotation(
    img_width: int,
    img_height: int,
    bbox: Tuple[int, int, int, int],
    class_id: int,
) -> str:
    """Create YOLO format annotation."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2 / img_width
    cy = (y1 + y2) / 2 / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def generate_single_image(
    args: Tuple[int, int, str, List[SymbolClass], int, List[int], Dict],
) -> Tuple[str, str, str]:
    """Generate a single training image with annotations."""
    image_id, class_id, split, symbol_classes, image_size, symbol_sizes, aug_config = args

    symbol_class = symbol_classes[class_id]
    renderer = RENDERERS.get(symbol_class.renderer)
    if renderer is None:
        renderer = render_unknown

    # Create background
    bg_generator = random.choice(BACKGROUND_GENERATORS)
    img = bg_generator(image_size)

    annotations = []

    # Place 1-5 symbols per image
    num_symbols = random.randint(1, 5)
    placed_boxes = []

    for _ in range(num_symbols):
        sym_size = random.choice(symbol_sizes)

        # Render base symbol
        base_symbol = renderer(sym_size, line_width=random.choice([1, 2]))

        # Apply augmentations
        rotation = random.choice([0, 90, 180, 270]) if aug_config.get("rotate_90", True) else 0
        if aug_config.get("random_rotation", False):
            rotation = random.uniform(-15, 15)

        scale = random.uniform(
            aug_config.get("scale_min", 0.8),
            aug_config.get("scale_max", 1.2)
        )
        noise = random.uniform(0, aug_config.get("noise_max", 10))
        blur = random.choice([0, 0, 0, 3]) if aug_config.get("blur", True) else 0
        perspective = random.uniform(0, aug_config.get("perspective_max", 0.3))
        thickness_var = random.choice([-1, 0, 0, 0, 1]) if aug_config.get("thickness_var", True) else 0

        symbol = augment_symbol(
            base_symbol,
            rotation=rotation,
            scale=scale,
            noise_level=noise,
            blur_kernel=blur,
            perspective_strength=perspective,
            line_thickness_var=thickness_var,
        )

        sym_h, sym_w = symbol.shape[:2]

        # Find valid position (non-overlapping)
        max_attempts = 50
        for _ in range(max_attempts):
            x = random.randint(10, image_size - sym_w - 10)
            y = random.randint(10, image_size - sym_h - 10)

            # Check overlap
            new_box = (x, y, x + sym_w, y + sym_h)
            overlap = False
            for box in placed_boxes:
                if (new_box[0] < box[2] and new_box[2] > box[0] and
                    new_box[1] < box[3] and new_box[3] > box[1]):
                    overlap = True
                    break

            if not overlap:
                break
        else:
            continue  # Skip if no valid position found

        # Paste symbol
        region = img[y:y + sym_h, x:x + sym_w]
        img[y:y + sym_h, x:x + sym_w] = np.minimum(region, symbol)

        placed_boxes.append((x, y, x + sym_w, y + sym_h))

        # Add annotation
        ann = create_yolo_annotation(image_size, image_size, (x, y, x + sym_w, y + sym_h), class_id)
        annotations.append(ann)

    # Generate filename
    img_name = f"{symbol_class.name.lower().replace('-', '_')}_{image_id:06d}"

    return img_name, split, (img, annotations)


def generate_dataset(
    output_dir: str,
    samples_per_class: int = 500,
    image_size: int = 640,
    symbol_sizes: List[int] = [32, 48, 64, 80],
    train_split: float = 0.8,
    val_split: float = 0.1,
    num_workers: int = 4,
    augmentation_config: Optional[Dict] = None,
) -> None:
    """Generate the complete synthetic dataset."""
    output_path = Path(output_dir)

    # Default augmentation config
    aug_config = {
        "rotate_90": True,
        "random_rotation": True,
        "scale_min": 0.7,
        "scale_max": 1.3,
        "noise_max": 15,
        "blur": True,
        "perspective_max": 0.3,
        "thickness_var": True,
    }
    if augmentation_config:
        aug_config.update(augmentation_config)

    # Create directory structure
    for split in ["train", "val", "test"]:
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Create data.yaml
    import yaml
    data_yaml = {
        "path": str(output_path.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(SYMBOL_CLASSES),
        "names": [sc.name for sc in SYMBOL_CLASSES],
    }

    with open(output_path / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    # Create class_names.txt for reference
    with open(output_path / "class_names.txt", "w") as f:
        for sc in SYMBOL_CLASSES:
            f.write(f"{sc.class_id}: {sc.name} ({sc.category})\n")

    print(f"Generating synthetic dataset...")
    print(f"  Output: {output_path}")
    print(f"  Classes: {len(SYMBOL_CLASSES)}")
    print(f"  Samples per class: {samples_per_class}")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Symbol sizes: {symbol_sizes}")
    print(f"  Workers: {num_workers}")
    print()

    # Prepare tasks
    tasks = []
    image_id = 0
    for class_id in range(len(SYMBOL_CLASSES)):
        for _ in range(samples_per_class):
            # Determine split
            r = random.random()
            if r < train_split:
                split = "train"
            elif r < train_split + val_split:
                split = "val"
            else:
                split = "test"

            tasks.append((image_id, class_id, split, SYMBOL_CLASSES, image_size, symbol_sizes, aug_config))
            image_id += 1

    # Generate images
    total = len(tasks)

    if num_workers > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(generate_single_image, task): task for task in tasks}

            with tqdm(total=total, desc="Generating images") as pbar:
                for future in as_completed(futures):
                    img_name, split, (img, annotations) = future.result()

                    # Save image
                    cv2.imwrite(str(output_path / "images" / split / f"{img_name}.png"), img)

                    # Save annotations
                    with open(output_path / "labels" / split / f"{img_name}.txt", "w") as f:
                        f.write("\n".join(annotations))

                    pbar.update(1)
    else:
        # Sequential processing
        with tqdm(total=total, desc="Generating images") as pbar:
            for task in tasks:
                img_name, split, (img, annotations) = generate_single_image(task)

                cv2.imwrite(str(output_path / "images" / split / f"{img_name}.png"), img)
                with open(output_path / "labels" / split / f"{img_name}.txt", "w") as f:
                    f.write("\n".join(annotations))

                pbar.update(1)

    # Print summary
    train_count = len(list((output_path / "images" / "train").glob("*.png")))
    val_count = len(list((output_path / "images" / "val").glob("*.png")))
    test_count = len(list((output_path / "images" / "test").glob("*.png")))

    print(f"\nDataset generated successfully!")
    print(f"  Train: {train_count} images")
    print(f"  Val: {val_count} images")
    print(f"  Test: {test_count} images")
    print(f"  Total: {train_count + val_count + test_count} images")
    print(f"\nTo train: python scripts/train_yolo_symbols.py train --data {output_path / 'data.yaml'}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for MEP symbol detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output", "-o",
        default="datasets/mep_symbols_v2",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=500,
        help="Number of samples per class"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Training image size"
    )
    parser.add_argument(
        "--symbol-sizes",
        type=int,
        nargs="+",
        default=[32, 48, 64, 80],
        help="Symbol sizes to generate"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction for training"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction for validation"
    )
    parser.add_argument(
        "--workers", "-j",
        type=int,
        default=1,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Generate preview of all symbol types"
    )

    args = parser.parse_args()

    if args.preview:
        # Generate preview grid
        print("Generating symbol preview...")
        preview_size = 80
        cols = 7
        rows = (len(SYMBOL_CLASSES) + cols - 1) // cols

        grid = np.ones((rows * (preview_size + 20), cols * (preview_size + 10)), dtype=np.uint8) * 255

        for i, sc in enumerate(SYMBOL_CLASSES):
            row = i // cols
            col = i % cols

            renderer = RENDERERS.get(sc.renderer, render_unknown)
            symbol = renderer(preview_size - 10)

            y = row * (preview_size + 20) + 5
            x = col * (preview_size + 10) + 5

            grid[y:y + preview_size - 10, x:x + preview_size - 10] = symbol

            # Add label
            label = sc.name[:12]
            cv2.putText(grid, label, (x, y + preview_size - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, 0, 1)

        preview_path = Path(args.output).parent / "symbol_preview.png"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview_path), grid)
        print(f"Preview saved to: {preview_path}")
        return

    generate_dataset(
        output_dir=args.output,
        samples_per_class=args.samples,
        image_size=args.image_size,
        symbol_sizes=args.symbol_sizes,
        train_split=args.train_split,
        val_split=args.val_split,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
