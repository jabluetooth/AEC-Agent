
import asyncio
import os
from aec_agent.config.settings import get_settings
from aec_agent.mcp.tools.vision_llm import is_vision_llm_available
from aec_agent.mcp.tools.symbol_classifier import SymbolClassifier
from aec_agent.mcp.tools.symbol_detection import DetectedBlock, load_symbol_templates

async def verify():
    print("--- Verifying Gemini Setup ---")
    settings = get_settings()
    
    # Check keys
    print(f"Gemini API Key configured: {bool(settings.gemini_api_key)}")
    print(f"OpenAI API Key configured: {bool(settings.openai_api_key)}")
    
    # Check is_vision_llm_available
    available = is_vision_llm_available()
    print(f"is_vision_llm_available(): {available}")
    
    # Check templates
    templates = load_symbol_templates()
    print(f"Templates loaded: {len(templates)}")
    if templates:
        print(f"First template: {templates[0].name} -> {templates[0].block_name}")

    # Check SymbolClassifier logic
    classifier = SymbolClassifier(enable_vision_llm=True)
    print(f"Classifier enable_vision_llm: {classifier.enable_vision_llm}")
    
    block = DetectedBlock(
        block_name="VALVE-GATE",
        position=(0,0),
        confidence=0.9
    )
    needs_vision = classifier._needs_vision_llm(block)
    print(f"Needs Vision LLM (VALVE-GATE, conf=0.9): {needs_vision}")
    
    block_unknown = DetectedBlock(
        block_name="UNKNOWN",
        position=(0,0),
        confidence=0.9
    )
    needs_vision_unk = classifier._needs_vision_llm(block_unknown)
    print(f"Needs Vision LLM (UNKNOWN, conf=0.9): {needs_vision_unk}")

    block_low_conf = DetectedBlock(
        block_name="UNKNOWN",
        position=(0,0),
        confidence=0.4
    )
    needs_vision_low = classifier._needs_vision_llm(block_low_conf)
    print(f"Needs Vision LLM (UNKNOWN, conf=0.4): {needs_vision_low}")

if __name__ == "__main__":
    asyncio.run(verify())
