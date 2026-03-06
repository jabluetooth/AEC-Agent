# Error Log

No active errors.

## Resolved (2026-03-06)
- `denoise() got an unexpected keyword argument 'h'` - Fixed. All denoise calls now use `strength=` parameter.
- `'ScaleCalibration' object has no attribute 'from_dwg'` - Fixed. Replaced `from_dwg()` with `to_pixels()` in unified_pipeline.py and adaptive_extraction.py. Also fixed `calibration.scale` to `calibration.scale_factor`.
- **Missing MText, linetype, and lineweight detection** - Fixed preprocessing pipeline:
  - Keep original image separate from preprocessed
  - Use original (not preprocessed) image for Gemini analysis → better text/MText detection
  - Added `_stage_detect_line_properties()` for linetype/lineweight detection on original image
  - Disabled binarization by default (destroys grayscale info)
  - Reduced denoise_strength from 10 to 5 (preserve dashed line patterns)
- **MText positions misaligned** - Fixed coordinate transformation:
  - Preprocessing was updating `result.image_width/height` after deskew
  - But Gemini uses original image dimensions for text positions
  - Calibration Y-flip `(height - y)` used wrong height → text shifted
  - Fix: Keep original dimensions for calibration, don't update after preprocessing
