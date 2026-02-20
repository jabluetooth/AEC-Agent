AEC Repository Audit: Raster PDF to Vector Conversion
1. Executive Summary
The repository currently contains two distinct pipelines for converting raster PDFs to AutoCAD vector entities:

Legacy Pipeline (OpenCV-based): Relies on traditional computer vision techniques (morphological operations, Hough transforms) on bitonal (black & white) images.
Gemini-First Pipeline (Modern): Leverages Gemini Vision AI for semantic understanding, coordinate extraction, and hybrid processing strategies.
For deep research into improving output, the Gemini-First pipeline represents the state-of-the-art approach in this codebase, while the legacy pipeline offers fallback capabilities for specific geometric tasks.

2. Legacy Pipeline (OpenCV-based)
This pipeline treats the drawing as a "bag of lines" and attempts to reconstruct geometry from pixels using mathematical transforms.

Key Files
src/aec_agent/mcp/tools/pdf_converter.py
:
Function: Renders PDF to Bitonal TIFF (1-bit depth).
Limitation: Lossy conversion. Discards color, anti-aliasing, and grayscale information, making it harder to distinguish faint lines or overlapping elements.
src/aec_agent/mcp/tools/image_vectorizer.py
:
Function: Core vectorization engine using opencv-python.
Logic:
Preprocessing: Gaussian blur, adaptive thresholding, morphological closing/opening.
Line Detection: Uses FastLineDetector (FLD) and HoughLinesP.
Circle/Arc Detection: Uses HoughCircles.
Topology: Merges collinear lines and snaps endpoints.
Pros: Fast, runs locally, no API costs.
Cons:
Fragile parameter tuning (thresholds, kernel sizes).
Struggles with noise, text overlaps, and complex geometry.
"Dumb" conversion: Doesn't understand that a rectangle is a "room" or a circle is a "column".
3. Gemini-First Pipeline (Modern)
This pipeline prioritizes semantic understanding before extraction. It allows the AI to "see" the drawing like a human engineer would, identifying systems and components before attempting to draw them.

Workflow Phases
Phase 1: Intake (
pdf_intake.py
)
Goal: Fidelity.
Method: Renders PDF to high-quality PNG (RGB or Grayscale) at 300+ DPI.
Improvement: Preserves anti-aliasing and fine details lost in the legacy bitonal conversion.
Phase 2: Understanding (
gemini_understanding.py
)
Goal: Context & Inventory.
Method: Sends the high-quality image to Gemini 1.5 Pro / 2.0 Flash.
Output: Structured JSON containing:
Drawing Type (e.g., "Electrical Plan").
Element Inventory (lines, text, symbols) with exact pixel coordinates.
Scale & Calibration hints.
Extraction Strategy: Recommendation on how to extract each region (Direct vs. Guided).
Phase 3: Calibration (coordinate_calibration.py)
Goal: Precision.
Method: Maps pixel coordinates to AutoCAD Drawing Units (DWG).
Sources: Uses dimension text ("20'-0""), scale notation ("1/4"=1'"), or standard sheet sizes to calculate the scale factor.
Phase 4: Adaptive Extraction (
adaptive_extraction.py
)
Goal: Flexibility.
Strategies:
Direct: Converts Gemini-detected coordinates directly to AutoCAD entities.
Guided Rasterization: Generates commands for AutoCAD Raster Design (VTools) to trace complex curves where AI coordinates might be approximate.
Selective OpenCV: Runs OpenCV only on specific, small regions (e.g., to find a hatching pattern) defined by Gemini.
Phase 5: Creation (
autocad_creation.py
)
Goal: Execution.
Method: Batches the creation of entities in AutoCAD.
Features:
NCS Layering: Automatically assigns layers (e.g., A-WALL, E-POWR) based on element type.
Block Insertion: Maps detected symbols (e.g., "duplex outlet") to CAD blocks.
4. Key Technical Comparison
Feature	Legacy Pipeline	Gemini-First Pipeline
Input	Bitonal TIFF (1-bit)	High-Quality PNG (24-bit/8-bit)
Core Tech	OpenCV (Hough Transforms)	Gemini Vision + Hybrid Strategies
Understanding	None (Geometric primitives on layer 0)	Semantic (Walls, Ducts, Devices)
Layering	Manual/None	Automated (NCS Standards)
Text Handling	Tesseract OCR (often noisy)	Gemini Vision (Context-aware correction)
Configuration	Many hardcoded parameters	Adaptive / AI-driven
5. Recommendations for Research
To improve the output of raster-to-vector conversion, focusing on the Gemini-First pipeline is recommended. Specific areas for deep research:

Refining "Direct" Extraction:

Gemini's coordinate precision can vary. Research post-processing algorithms (like the "Topology Cleanup" in 
image_vectorizer.py
) to snap Gemini's approximate coordinates to a rigid orthogonal grid.
Hybridization:

Investigate the 
HybridExtractionConfig
 in 
adaptive_extraction.py
. Can we use Gemini to draw the bounding box of a wall, and then use OpenCV to find the exact edges within that box? This combines semantic accuracy with pixel-perfect precision.
Symbol Intelligence:

The detected_symbols logic in 
gemini_understanding.py
 is powerful. Research how to expand the Block Mapping in 
adaptive_extraction.py
 to support dynamic block attributes (e.g., transferring the text label "RTU-1" into the block attribute directly).
Raster Design Integration:

The Guided Rasterization strategy is unique. Research how to programmatically control AutoCAD Raster Design (IBIM) more effectively to handle "dirty" scans that confuse both OpenCV and Gemini.