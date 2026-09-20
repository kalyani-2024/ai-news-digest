import logging
import os
import random
import time

from google import genai
from google.genai import errors
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DIGEST_MODEL = os.getenv("DIGEST_MODEL", "gemini-3.5-flash-lite")
CURATOR_MODEL = os.getenv("CURATOR_MODEL", "gemini-3.5-flash")
EMAIL_MODEL = os.getenv("EMAIL_MODEL", "gemini-3.5-flash")

FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
MAX_ATTEMPTS = int(os.getenv("GEMINI_MAX_ATTEMPTS", "4"))

# Transient server-side conditions. 429 is included because Gemini uses it for
# per-minute rate limits as well as hard quota exhaustion, and the former clears.
RETRY_CODES = {429, 500, 502, 503, 504}


def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    return genai.Client(api_key=api_key)


def generate(client: genai.Client, model: str, contents, config):
    """Call Gemini, retrying transient failures and falling back to another model.

    A 503 on the ranking call used to lose an entire pipeline run, so each model
    is retried with exponential backoff before the fallback model is tried.
    """
    models = [model] if model == FALLBACK_MODEL else [model, FALLBACK_MODEL]
    last_error = None

    for model_name in models:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return client.models.generate_content(
                    model=model_name, contents=contents, config=config
                )
            except errors.APIError as exc:
                if exc.code not in RETRY_CODES:
                    raise
                last_error = exc
                if attempt == MAX_ATTEMPTS:
                    break
                delay = min(2 ** attempt, 30) + random.uniform(0, 1)
                logger.warning(
                    "Gemini %s returned %s, retrying in %.1fs (attempt %d/%d)",
                    model_name, exc.code, delay, attempt, MAX_ATTEMPTS,
                )
                time.sleep(delay)

        if model_name != models[-1]:
            logger.warning(
                "Gemini %s exhausted retries, falling back to %s", model_name, models[-1]
            )

    raise last_error
