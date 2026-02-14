from aec_agent.config.settings import get_settings
import os

def check_gemini():
    settings = get_settings()
    print(f"GEMINI_API_KEY in settings: {'Set' if settings.gemini_api_key else 'NOT set'}")
    if settings.gemini_api_key:
        print(f"Starts with: {settings.gemini_api_key[:5]}...")
        print(f"Length: {len(settings.gemini_api_key)}")
    
    print(f"GEMINI_API_KEY in os.environ: {'Set' if 'GEMINI_API_KEY' in os.environ else 'NOT set'}")

if __name__ == '__main__':
    check_gemini()
