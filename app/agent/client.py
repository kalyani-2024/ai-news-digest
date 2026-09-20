import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

DIGEST_MODEL = os.getenv("DIGEST_MODEL", "gemini-3.5-flash-lite")
CURATOR_MODEL = os.getenv("CURATOR_MODEL", "gemini-3.5-flash")
EMAIL_MODEL = os.getenv("EMAIL_MODEL", "gemini-3.5-flash")


def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    return genai.Client(api_key=api_key)
