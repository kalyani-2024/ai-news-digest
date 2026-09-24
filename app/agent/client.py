"""LangChain model factory shared by every agent.

Each agent builds an LCEL chain: prompt | structured_llm(Schema). This module owns
the model side of that chain: which Gemini model, how it retries, and what it
falls back to when the primary model is unavailable.
"""
import os
from typing import Type

from dotenv import load_dotenv
from google import genai
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

load_dotenv()

DIGEST_MODEL = os.getenv("DIGEST_MODEL", "gemini-3.5-flash-lite")
CURATOR_MODEL = os.getenv("CURATOR_MODEL", "gemini-3.5-flash")
EMAIL_MODEL = os.getenv("EMAIL_MODEL", "gemini-3.5-flash")

FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
MAX_ATTEMPTS = int(os.getenv("GEMINI_MAX_ATTEMPTS", "4"))


def _api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    return api_key


def get_client() -> genai.Client:
    """Raw Gemini SDK client, kept for utilities such as listing available models."""
    return genai.Client(api_key=_api_key())


def chat_model(model: str, temperature: float) -> ChatGoogleGenerativeAI:
    # max_retries is passed down to the Google SDK's HTTP retry policy, which
    # retries 408/429/500/502/503/504 with jittered exponential backoff. That is
    # the same set of transient errors the old hand-rolled retry loop handled.
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_retries=MAX_ATTEMPTS,
        google_api_key=_api_key(),
    )


def structured_llm(schema: Type[BaseModel], model: str, temperature: float) -> Runnable:
    """A model that returns a validated `schema` instance, with model fallback.

    Retries happen per model inside the SDK. Only once the primary model has
    exhausted them does with_fallbacks() rerun the same request on FALLBACK_MODEL,
    so a 503 spike on one model tier does not lose the whole run.
    """
    primary = chat_model(model, temperature).with_structured_output(schema)
    if model == FALLBACK_MODEL:
        return primary
    fallback = chat_model(FALLBACK_MODEL, temperature).with_structured_output(schema)
    return primary.with_fallbacks([fallback])
