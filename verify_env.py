
import os
import sys

# Force immediate flush of output
sys.stdout.reconfigure(line_buffering=True)

print("--- Checking Environment ---")

# Check if .env file exists
env_path = os.path.join(os.getcwd(), ".env")
print(f".env exists: {os.path.exists(env_path)}")

# Try loading .env
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    print("Loaded .env")
except ImportError:
    print("python-dotenv not installed")

# Check environment variables directly
gemini_key = os.getenv("GEMINI_API_KEY")
print(f"GEMINI_API_KEY env var: {'SET' if gemini_key else 'NOT SET'}")

if gemini_key:
    # Print first few chars to verify it's loaded correctly
    print(f"GEMINI_API_KEY prefix: {gemini_key[:4]}...")

# Check settings via pydantic
try:
    from aec_agent.config.settings import get_settings
    settings = get_settings()
    print(f"Settings.gemini_api_key: {'SET' if settings.gemini_api_key else 'NOT SET'}")
    print(f"Settings.gemini_model: {settings.gemini_model}")
except Exception as e:
    print(f"Failed to load settings: {e}")

from aec_agent.mcp.tools.vision_llm import is_vision_llm_available
print(f"is_vision_llm_available(): {is_vision_llm_available()}")
