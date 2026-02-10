# YOLOv8 Training Datasets for MEP & Architecture

This document tracks public datasets and resources for training the AEC Agent's symbol detection models.

## 🏆 Top Recommended Datasets

### 1. Roboflow Universe (MEP Specific)
Roboflow hosts user-uploaded datasets that are often already annotated in YOLO format.
*   **Search Query:** "MEP", "Electrical Symbols", "Floor Plan"
*   **Key Datasets:**
    *   **"Detection electric symbols"**: ~75 images (Good for fine-tuning validation).
    *   **"SkeySpot" / DELP**: Scanned electrical layout plans with 2,450 annotated instances.
    *   **"G-FloorPlan"**: Structural elements (walls, doors, windows).
*   **URL:** [Roboflow Universe](https://universe.roboflow.com/search?q=mep%20symbols)

### 2. FloorPlanCAD (Huge Scale)
A large-scale dataset with 15k+ annotated floor plans.
*   **Content:** 30 categories (doors, windows, furniture, some fixtures).
*   **Format:** Vector-based annotations converted to images.
*   **Use Case:** Excellent for "background" training—teaching the model to distinguish *walls/furniture* from *MEP symbols*.
*   **Source:** [Hugging Object Detection](https://huggingface.co/datasets) (search for FloorPlanCAD)

### 3. SESYD (Synthetic Engineering)
The "Systems Evaluation SYnthetic Documents" dataset.
*   **Content:** Floor plans with electrical and fluid power symbols.
*   **Pros:** Clean, perfect ground truth.
*   **Cons:** Older, lower resolution, specific symbol sets (European standards).
*   **URL:** [SESYD Dataset](http://mathieu.delalandre.free.fr/projects/sesyd/)

### 4. SFPI (Synthetic Floor Plan Images)
10,000 synthetic images used for object detection reliability.
*   **Use Case:** Augmenting the "negative" class or general architectural understanding.

---

## 🏗️ CAD Block Libraries (For Synthetic Generation)

Since we have a synthetic generation script (`scripts/train_yolo_symbols.py`), we can use raw CAD blocks to generate unlimited training data.

*   **CAD Block Sources:**
    *   **CAD-Blocks.net**: Free DWG downloads of valves, diffusers, etc.
    *   **Bibliocad**: Huge library of user-submitted blocks.
    *   **FastrackCAD**: Manufacturer-specific MEP blocks (very high quality).

**Strategy:**
1.  Download `.dwg` block libraries.
2.  Convert blocks to transparent `.png` images.
3.  Place them into `datasets/source_images/`.
4.  Run `generate_synthetic_data.py` to create thousands of "perfect" YOLO labels.

---

## 🧠 Recommended Training Strategy: "Hybrid"

Don't rely on just one source.

1.  **Synthetic (80% of data):**
    *   **Use your existing script:** `scripts/generate_synthetic_data.py`.
    *   **Why:** Infinite variation, perfect bounding boxes, covers *your* specific project standards.
    *   **Action:** Add more "source images" to your generator from the CAD Block libraries above.

2.  **Real-World (20% of data):**
    *   Use **Roboflow** or **FloorPlanCAD** images.
    *   **Why:** Teaches the model to handle noise, scanning artifacts, and hand-drawn lines.
    *   **Action:** Download ~100-200 real images, manually label them (or fix auto-labels), and mix them into the `train` folder.

## 🚀 How to Add These to Your Pipeline

1.  **Download** a dataset export (YOLOv8 format) from Roboflow.
2.  **Unzip** into `datasets/external/`.
3.  **Update** `data.yaml` to include paths to both your synthetic data and this external data.
    ```yaml
    train:
      - ./datasets/synthetic/images/train
      - ./datasets/external/roboflow_mep/images/train
    val:
      - ./datasets/synthetic/images/val
      - ./datasets/external/roboflow_mep/images/val
    ```
