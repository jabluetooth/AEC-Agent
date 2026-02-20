
import asyncio
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

# Mocking the structures needed for the test
@dataclass
class DetectedElement:
    position: Any
    bounding_box: Optional[List[int]] = None
    confidence: float = 1.0

@dataclass
class DrawingElements:
    text: List[DetectedElement] = field(default_factory=list)
    symbols: List[DetectedElement] = field(default_factory=list)

@dataclass
class SpecialRegion:
    bounds: List[int]
    reason: str
    strategy: str
    expected_pattern: Optional[str] = None
    layer_suggestion: Optional[str] = None

@dataclass
class ExtractionStrategy:
    special_regions: List[SpecialRegion] = field(default_factory=list)

@dataclass
class DrawingAnalysis:
    drawing_type: str = "floor_plan"
    elements: DrawingElements = field(default_factory=DrawingElements)
    extraction_strategy: ExtractionStrategy = field(default_factory=ExtractionStrategy)
    total_elements: int = 0

@dataclass
class ScaleCalibration:
    dpi: int = 100
    
    def to_dwg(self, x, y):
        return x / self.dpi, y / self.dpi
    
    def scale_length(self, length):
        return length / self.dpi

@dataclass
class HybridExtractionConfig:
    use_opencv_for_lines: bool = True
    use_opencv_for_circles: bool = True
    opencv_line_min_length: int = 10
    opencv_circle_min_radius: int = 5
    opencv_circle_max_radius: int = 100

# Mock classes for Entities
class EntityType:
    LINE = "line"
    CIRCLE = "circle"

class ExtractionSource:
    DIRECT = "direct"
    SELECTIVE_OPENCV = "selective_opencv"

@dataclass
class EntityToCreate:
    entity_type: str
    layer: str = "0"
    properties: Dict = field(default_factory=dict)
    source: str = ExtractionSource.DIRECT
    confidence: float = 1.0
    source_element: str = ""

import sys
import os
# Add project root to path (one level up from tests/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.aec_agent.mcp.tools.gemini_first.adaptive_extraction import (
        hybrid_opencv_extraction,
        _merge_duplicate_entities
    )
    from src.aec_agent.mcp.tools.gemini_first.opencv_extraction import OpenCVExtractor
except ImportError as e:
    print(f"Error importing modules: {e}")
    pass

async def test_masking():
    print("\n--- Testing Masking ---")
    
    # Create an image with a black line and a black 'text' block
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    
    # Draw a line (should be detected)
    cv2.line(img, (20, 20), (180, 20), (0, 0, 0), 2)
    
    # Draw a "text" block (should be masked)
    # Rectangle from 50,50 to 150,80
    cv2.rectangle(img, (50, 50), (150, 80), (0, 0, 0), -1)
    
    # Setup analysis reporting the text block
    analysis = DrawingAnalysis()
    
    # Mask using bounding box (explicitly covering the rectangle)
    # Rectangle is 50,50 to 150,80. Masking 45,45 to 155,85.
    symbol_bbox = [45, 45, 155, 85]
    analysis.elements.symbols.append(DetectedElement(position=(100, 65), bounding_box=symbol_bbox))
    
    calibration = ScaleCalibration()
    config = HybridExtractionConfig()
    
    # Run extraction
    entities = await hybrid_opencv_extraction(img, analysis, calibration, config)
    
    # Check results
    lines = [e for e in entities if e.entity_type == EntityType.LINE]
    
    # We expect the top line to be found
    found_top_line = any(l.properties['start'][1] < 0.35 for l in lines) # 20px / 100dpi = 0.2
    
    # We expect NO lines from the symbol area (approx y=0.5 to 0.8)
    found_symbol_lines = any(0.4 < l.properties['start'][1] < 0.9 for l in lines)
    
    print(f"Top line detected: {found_top_line}")
    print(f"Entities in symbol area: {[l.properties['start'][1] for l in lines if 0.4 < l.properties['start'][1] < 0.9]}")
    print(f"Symbol lines detected (should be False): {found_symbol_lines}")
    
    if found_top_line and not found_symbol_lines:
        print("PASS: Masking success")
    else:
        print("FAIL: Masking failed")
        print(f"Total entities: {len(entities)}")


async def test_default_circles():
    print("\n--- Testing Default Circles ---")
    
    # Create image with a circle
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    # Thicker circle, larger radius
    cv2.circle(img, (100, 100), 40, (0, 0, 0), 3)
    
    analysis = DrawingAnalysis() # Empty strategies -> should trigger default full_image
    calibration = ScaleCalibration()
    config = HybridExtractionConfig(use_opencv_for_circles=True)
    
    entities = await hybrid_opencv_extraction(img, analysis, calibration, config)
    
    circles = [e for e in entities if e.entity_type == EntityType.CIRCLE]
    
    print(f"Circles detected: {len(circles)}")
    
    if len(circles) > 0:
         print("PASS: Default circle detection success")
    else:
         print("FAIL: No circles detected with default strategy")


def test_metadata_transfer():
    print("\n--- Testing Metadata Transfer ---")
    
    # Create a Gemini Line (Semantic, Layer A-WALL)
    gemini_line = EntityToCreate(
        entity_type=EntityType.LINE,
        layer="A-WALL",
        properties={"start": (0, 0), "end": (10, 0), "linetype": "Dashed"},
        source=ExtractionSource.DIRECT
    )
    
    # Create an OpenCV Line (Geometric, Layer 0, Continuous) - overlaps perfectly
    opencv_line = EntityToCreate(
        entity_type=EntityType.LINE,
        layer="0",
        properties={"start": (0.01, 0), "end": (10.01, 0), "linetype": "Continuous"},
        source=ExtractionSource.SELECTIVE_OPENCV
    )
    
    entities = [gemini_line, opencv_line]
    
    # Merge
    merged, removed = _merge_duplicate_entities(entities, tolerance=0.1, prefer_opencv=True)
    
    print(f"Entities before: {len(entities)}")
    print(f"Entities after: {len(merged)}")
    print(f"Removed: {removed}")
    
    if len(merged) == 1:
        result = merged[0]
        print(f"Result Source: {result.source}")
        print(f"Result Layer: {result.layer}")
        print(f"Result Linetype: {result.properties.get('linetype')}")
        
        if (result.source == ExtractionSource.SELECTIVE_OPENCV and 
            result.layer == "A-WALL" and 
            result.properties.get('linetype') == "Dashed"):
            print("PASS: Metadata transfer success")
        else:
             print("FAIL: Metadata transfer failed")
    else:
        print("FAIL: Merge failed (count check)")

async def main():
    # Redirect stdout to file to avoid encoding issues
    with open("verification_results.txt", "w", encoding="utf-8") as f:
        sys.stdout = f
        await test_masking()
        await test_default_circles()
        test_metadata_transfer()
        print("\nDone.")

if __name__ == "__main__":
    asyncio.run(main())
