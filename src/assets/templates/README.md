# Symbol Templates for AEC Vectorization

This directory contains template images for detecting standard AEC (Architecture, Engineering, Construction) symbols during PDF/raster vectorization.

## Directory Structure

```
templates/
    mechanical/     # HVAC, piping symbols
    electrical/     # Outlets, switches, panels
    fire/           # Smoke detectors, pull stations
    plumbing/       # Floor drains, cleanouts
    low_voltage/    # Data ports, cameras, card readers
```

## Template Requirements

### Image Format
- **Format:** PNG (grayscale)
- **Size:** 32x32 to 64x64 pixels recommended
- **Colors:** White background (255), Black symbol (0)
- **Naming:** lowercase_with_underscores.png

### Naming Convention
The filename (without extension) becomes the AutoCAD block name:
- `valve_gate.png` → Block name: `VALVE-GATE`
- `smoke_detector.png` → Block name: `SMOKE-DETECTOR`

### Creating Templates

1. **From CAD:** Export block to PNG at 300 DPI, crop to symbol bounds
2. **From Scan:** Crop symbol from high-quality scan, convert to bitonal
3. **Manual:** Create in image editor with clean lines

### Tips for Good Templates
- Use clean, representative examples of each symbol
- Avoid anti-aliasing (use hard edges)
- Include the symbol boundary, not just the icon
- Test with rotated versions if symbol is not rotationally symmetric

## Example Templates to Add

### Mechanical (mechanical/)
- valve_gate.png
- valve_ball.png
- valve_butterfly.png
- diffuser_4way.png
- diffuser_linear.png
- duct_elbow.png
- vav_box.png

### Electrical (electrical/)
- outlet_duplex.png
- outlet_gfci.png
- switch_single.png
- switch_3way.png
- panel_main.png
- junction_box.png

### Fire (fire/)
- smoke_detector.png
- heat_detector.png
- pull_station.png
- horn_strobe.png
- sprinkler_pendent.png
- sprinkler_upright.png

### Plumbing (plumbing/)
- floor_drain.png
- cleanout.png
- hose_bibb.png
- water_heater.png
- fixture_sink.png

### Low Voltage (low_voltage/)
- data_outlet.png
- camera_dome.png
- card_reader.png
- motion_sensor.png
- speaker.png

## Usage

Templates are automatically loaded by `symbol_detection.py` during vectorization:

```python
from aec_agent.mcp.tools.symbol_detection import load_symbol_templates, detect_and_mask_symbols

templates = load_symbol_templates()
masked_image, detected_blocks = detect_and_mask_symbols(binary_image, templates)

for block in detected_blocks:
    print(f"Found {block.block_name} at {block.position}")
```

## Notes

- Templates must match the style of symbols in your source drawings
- Consider creating multiple templates for the same symbol at different scales
- For better detection of varied drawing styles, consider upgrading to YOLOv8 (Phase 2.5 roadmap)
