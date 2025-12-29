# scripts/list_gemini_models.py
import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Set GEMINI_API_KEY in your environment first.")

genai.configure(api_key=api_key)

print("Available models that support generateContent:\n")

for m in genai.list_models():
    # m.name looks like "models/gemini-1.5-flash"
    methods = getattr(m, "supported_generation_methods", []) or []
    if "generateContent" in methods:
        print(f"- {m.name}")
