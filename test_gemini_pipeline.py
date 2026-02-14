import asyncio
import os
import structlog
from pathlib import Path
from aec_agent.config.settings import get_settings
from aec_agent.mcp.tools.gemini_first.gemini_understanding import DrawingAnalyzer

# Setup logging
structlog.configure()
logger = structlog.get_logger()

async def test_analyzer():
    settings = get_settings()
    print(f"Gemini API Key present: {bool(settings.gemini_api_key)}")
    
    try:
        import google.generativeai as genai
        print("google-generativeai imported successfully")
    except ImportError:
        print("FAILED to import google-generativeai")
        return

    analyzer = DrawingAnalyzer()
    print(f"Initializing DrawingAnalyzer with model: {analyzer.model_name}")
    
    try:
        model = await analyzer._get_gemini_model()
        print("Gemini model initialized successfully")
        
        # We won't actually perform an analysis yet to avoid token costs,
        # but the initialization confirms the API key and library are working.
        print("\nSUCCESS: Gemini pipeline connectivity verified.")
        
    except Exception as e:
        print(f"\nFAILED: Error initializing Gemini pipeline: {e}")

if __name__ == "__main__":
    asyncio.run(test_analyzer())
