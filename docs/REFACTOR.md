# 📄 PRD: Phase 2.5 — Semantic AEC Vectorization Pipeline

## 1. Executive Summary

**Problem:** The current OpenCV pipeline uses purely geometric detection (FastLineDetector, HoughLinesP), resulting in highly fragmented, non-semantic AutoCAD entities. Text is rendered as stray lines, dashed lines are disconnected segments, and wall thicknesses generate double-lines.
**Objective:** Evolve `image_vectorizer.py` into a multi-stage semantic pipeline that isolates AEC components (Text, Symbols, Geometry) using masking, skeletonization, and geometric heuristics before generating AutoCAD entities.

## 2. Success Metrics

* **Entity Reduction:** ≥ 60% reduction in total line count on standard MEP PDFs (achieved by collinear merging and text masking).
* **Text Accuracy:** ≥ 85% of raster text converted to AutoCAD `MText` rather than line geometry.
* **Geometric Precision:** 100% of pseudo-orthogonal lines (88° - 92°) snapped to perfect 0° or 90°.
* **Processing Time:** Under 15 seconds per PDF page (keeping within standard MCP tool timeouts).

---

## 3. Technical Pipeline Requirements

### Stage 1: Text Isolation & Masking (The "De-noising" Step)

Before any line detection runs, text must be extracted and removed to prevent geometric noise.

* **Method:** Integrate Tesseract OCR via `pytesseract` to detect text bounding boxes.
* **Action:** For each detected box, extract the text string and coordinates.
* **Masking:** Fill the bounding box region in the working TIFF with white (background color) to erase it from the image.
* **AutoCAD Output:** Queue `draw_text` sidecar commands.

### Stage 2: Symbol Detection (Template Matching)

Identify standard MEP/Architectural symbols (valves, doors, diffusers) before detecting lines.

* **Method:** OpenCV Template Matching (`cv2.matchTemplate`) against a small library of standard AEC icons.
* **Action:** Detect centroid coordinates of high-confidence matches.
* **Masking:** Erase the symbol footprint from the working TIFF.
* **AutoCAD Output:** Queue `draw_block` sidecar commands (e.g., insert "MEP_VALVE" block).

### Stage 3: Skeletonization (Thickness Reduction)

Eliminate the "double-line" problem caused by thick walls/ducts in the raster.

* **Method:** Use `skimage.morphology.skeletonize` to reduce all remaining black pixels to a 1-pixel width centerline.
* **Result:** Thick walls become single, central lines.

### Stage 4: Geometric Detection (Existing Stack + Refinement)

Run the existing FLD/Hough algorithms on the now-cleaned, skeletonized image.

* **Order of Operations:**
1. `HoughCircles` (Mask out resulting circles).
2. `FastLineDetector` (Extract remaining lines).



### Stage 5: AEC Geometric Heuristics (Post-Processing)

Apply engineering logic to the raw line array *before* sending to AutoCAD.

* **Orthogonal Snapping:** Calculate line angles. Force lines within ±2 degrees of 0°, 90°, 180°, or 270° to snap to the exact angle by modifying their endpoints.
* **Collinear Line Merging (Dashed Line Fix):** Group lines by angle. If lines share the same trajectory  and their endpoints are within a minimum distance threshold, merge them into a single line entity. Apply AutoCAD "DASHED" linetype if gaps are detected.

---

## 4. Architectural Updates & New Files

| Component | File Path | New/Modify | Description |
| --- | --- | --- | --- |
| **Vectorizer** | `src/aec_agent/mcp/tools/image_vectorizer.py` | **Modify** | Major refactor to include masking, OCR pipeline, and heuristic classes. |
| **Heuristics** | `src/aec_agent/utils/geometry_cleanup.py` | **New** | Math utilities for orthogonal snapping and collinear merging algorithms. |
| **Symbol DB** | `src/assets/templates/` | **New** | Directory holding 5-10 standard bitonal templates for test matching (valves, diffusers). |
| **Sidecar API** | `src/sidecars/autocad/Commands.cs` | **Modify** | Ensure `draw_text` and `draw_block` can accept lists of entities asynchronously. |

## 5. Implementation Phases (Sprints)

### Sprint 1: Pipeline Reordering & OCR Masking

* Implement `pytesseract` bounding box detection.
* Implement the masking function to erase text from the OpenCV array.
* Verify FLD runs on the cleaned image, observing the drop in artifact lines.

### Sprint 2: Skeletonization & Orthogonal Snapping

* Introduce `scikit-image` for the thinning/skeletonization step.
* Write the Orthogonal Snapping algorithm in Python.
* **Test:** Visual confirmation in AutoCAD that walls are now single, straight 90° lines.

### Sprint 3: Collinear Merging (The Dashed-Line Fix)

* Write the math to calculate line intersections and proximity.
* Implement merging logic for segmented lines.
* Add mapping to AutoCAD linetypes (CONTINUOUS vs. DASHED).

### Sprint 4: Symbol Template Matching (Bonus/stretch)

* Implement `cv2.matchTemplate` with a test valve symbol.
* Connect to AutoCAD sidecar `InsertBlock` command.

---

**Would you like me to start by writing the Python code for `geometry_cleanup.py` (the orthogonal snapping and collinear merging logic) so you can plug it into your existing vectorizer today?**