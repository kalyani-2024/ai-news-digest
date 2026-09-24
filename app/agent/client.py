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
from langchain_groq import ChatGroq
from pydantic import BaseModel

load_dotenv()

DIGEST_MODEL = os.getenv("DIGEST_MODEL", "gemini-3.5-flash-lite")
CURATOR_MODEL = os.getenv("CURATOR_MODEL", "gemini-3.5-flash")
EMAIL_MODEL = os.getenv("EMAIL_MODEL", "gemini-3.5-flash")

FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
MAX_ATTEMPTS = int(os.getenv("GEMINI_MAX_ATTEMPTS", "4"))

# Last-resort fallback on a different provider. Gemini's free tier caps requests
# per day per model, so every Gemini model can be exhausted at once; a second
# provider is the only fallback that still works when that happens.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def _api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    return api_key


def get_client() -> genai.Client:
    """Raw Gemini SDK client, kept for utilities such as listing available models."""
    return genai.Client(api_key=_api_key())


def groq_model(temperature: float) -> ChatGroq:
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY"),
        max_retries=MAX_ATTEMPTS,
    )


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
    """A model that returns a validated `schema` instance, with layered fallback.

    Retries happen per model inside the SDK. Once a model has exhausted them,
    with_fallbacks() reruns the same request on the next one: first the cheaper
    Gemini tier, then Groq if a key is configured. The cross-provider step is
    what survives Gemini's per-day free-tier quota, which takes out every Gemini
    model at once and so cannot be escaped by switching Gemini models alone.
    """
    primary = chat_model(model, temperature).with_structured_output(schema)

    fallbacks = []
    if model != FALLBACK_MODEL:
        fallbacks.append(chat_model(FALLBACK_MODEL, temperature).with_structured_output(schema))
    if os.getenv("GROQ_API_KEY"):
        # json_schema, not the default tool-calling path: the gpt-oss models
        # answer in prose often enough that tool calling fails with
        # "Tool choice is required, but model did not call a tool".
        fallbacks.append(
            groq_model(temperature).with_structured_output(schema, method="json_schema")
        )

    return primary.with_fallbacks(fallbacks) if fallbacks else primary
