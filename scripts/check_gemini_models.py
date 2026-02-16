"""
Quick script to check available Gemini models.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

import google.generativeai as genai

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY not found in environment")
    exit(1)

print(f"API Key found: {api_key[:10]}...")

genai.configure(api_key=api_key)

print("\nAvailable Gemini models that support generateContent:\n")
models = genai.list_models()
for m in models:
    if 'gemini' in m.name.lower() and 'generateContent' in m.supported_generation_methods:
        print(f"  - {m.name}")
        print(f"    Methods: {m.supported_generation_methods}")
        print()
