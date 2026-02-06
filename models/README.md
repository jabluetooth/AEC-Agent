# Models Directory

This directory contains trained neural network models for the AEC Agent.

## MEP Symbol Detection (Phase 2.5.1)

The `mep_symbols.onnx` model is used for detecting MEP (Mechanical, Electrical, Plumbing) symbols in architectural drawings.

### Model Files

| File | Description | Size |
|------|-------------|------|
| `mep_symbols.onnx` | YOLOv8n model exported to ONNX | ~12 MB |
| `mep_symbols.pt` | Native ultralytics model (training) | ~6 MB |

### Training

To train a new model:

```bash
# 1. Prepare synthetic dataset
python scripts/train_yolo_symbols.py prepare --output-dir datasets/mep_symbols

# 2. Train model
python scripts/train_yolo_symbols.py train --data datasets/mep_symbols/data.yaml --epochs 100

# 3. Export to ONNX
python scripts/train_yolo_symbols.py export --model runs/detect/mep_symbols/weights/best.pt --output models/mep_symbols.onnx

# 4. Validate
python scripts/train_yolo_symbols.py validate --model models/mep_symbols.onnx
```

### Supported Symbol Classes

The model supports 42 symbol classes across 5 categories:

**Mechanical (13 classes):**
- VALVE-GATE, VALVE-BALL, VALVE-BUTTERFLY, VALVE-CHECK
- DIFFUSER-SUPPLY, DIFFUSER-RETURN, DIFFUSER-EXHAUST
- DUCT-ELBOW, DUCT-TEE
- PUMP, AHU, FCU, VAV-BOX

**Electrical (11 classes):**
- OUTLET-DUPLEX, OUTLET-GFCI, OUTLET-QUAD
- SWITCH-SINGLE, SWITCH-3WAY, SWITCH-DIMMER
- PANEL-ELECTRICAL
- LIGHT-FIXTURE, LIGHT-RECESSED, LIGHT-EXIT
- JUNCTION-BOX

**Fire (6 classes):**
- SMOKE-DETECTOR, HEAT-DETECTOR
- PULL-STATION, HORN-STROBE
- SPRINKLER-HEAD, FIRE-EXTINGUISHER

**Plumbing (6 classes):**
- FLOOR-DRAIN, CLEANOUT
- FIXTURE-SINK, FIXTURE-TOILET, FIXTURE-URINAL
- WATER-HEATER

**Low Voltage (5 classes):**
- DATA-OUTLET, CAMERA-SECURITY, CARD-READER
- SPEAKER, THERMOSTAT

**Other (1 class):**
- UNKNOWN (catch-all)

### Usage

```python
from aec_agent.mcp.tools.symbol_detection import detect_symbols

# Auto-select YOLO if model exists, else template matching
masked, blocks = detect_symbols(image, scale=1/300, backend="auto")

# Force YOLO
masked, blocks = detect_symbols(image, scale=1/300, backend="yolo")

# Specify custom model
masked, blocks = detect_symbols(
    image,
    backend="yolo",
    yolo_model_path="models/custom_symbols.onnx",
)
```

### Requirements

Install YOLO dependencies:

```bash
pip install aec-agent[yolo]
# or
pip install ultralytics onnxruntime
```

### Notes

- Model not included in repository (too large for git)
- Download from releases or train your own
- Falls back to template matching if model not found
