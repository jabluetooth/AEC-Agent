
import sys
import cv2
import numpy as np
import asyncio
from aec_agent.mcp.tools.symbol_detection import detect_symbols, load_symbol_templates
from aec_agent.mcp.tools.symbol_classifier import SymbolClassifier, DetectedBlock

# Force flush
sys.stdout.reconfigure(line_buffering=True)

async def verify():
    print("--- Verifying Template Detection & Classification Logic ---")

    # 1. Load a template to create a synthetic image
    templates = load_symbol_templates()
    if not templates:
        print("ERROR: No templates loaded!")
        return
    
    # Find valve-gate template
    template = next((t for t in templates if "valve_gate" in t.name), None)
    if not template:
        print("ERROR: valve_gate template not found")
        return
        
    print(f"Found template: {template.name} ({template.image.shape})")
    
    # 2. Create synthetic image (white background)
    # 500x500 white image
    img = np.full((500, 500), 255, dtype=np.uint8)
    
    # Paste template in center
    h, w = template.image.shape
    y, x = 250 - h//2, 250 - w//2
    img[y:y+h, x:x+w] = template.image
    
    print("Created synthetic image with valve_gate in center")
    
    # 3. Run detection (Template Matching)
    # Use scale=1.0 since we pasted it 1:1
    print("Running detect_symbols...")
    masked, blocks = detect_symbols(img, scale=1.0, backend="template")
    
    print(f"Detections found: {len(blocks)}")
    for b in blocks:
        print(f" - {b.block_name}: conf={b.confidence:.2f}, pos={b.position}")
        
    if not blocks:
        print("FAILURE: Template detection failed on perfect match!")
        return

    # 4. Run Classification Logic
    # We want to see if this detection would trigger Vision LLM
    print("\nChecking SymbolClassifier logic...")
    classifier = SymbolClassifier(enable_vision_llm=True)
    
    for b in blocks:
        needs_vision = classifier._needs_vision_llm(b)
        print(f"Block '{b.block_name}' needs Vision LLM? {needs_vision}")
        
        # Determine why
        type_name = b.block_name.split("-")[0].lower() if "-" in b.block_name else b.block_name.lower()
        # Access GENERIC_CLASSES directly if possible, or infer
        print(f"  Type: {type_name}")
    
if __name__ == "__main__":
    asyncio.run(verify())
