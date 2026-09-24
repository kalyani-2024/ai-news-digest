"""The state object that flows through the pipeline graph.

Every node receives the current state and returns a partial update (a dict
holding only the keys it changed). LangGraph merges that update into the state.
By default a key is overwritten; a key annotated with a reducer is combined
with the old value instead.
"""
from typing import Annotated, Optional, TypedDict

from app.agent.email_agent import EmailDigestResponse


def merge_dicts(left: Optional[dict], right: Optional[dict]) -> dict:
    """Reducer for keys written by parallel branches.

    extract_anthropic and extract_youtube run in the same step and both write
    `processing`. Without a reducer LangGraph rejects two writes to one key in
    one step; with it, {"anthropic": ...} and {"youtube": ...} are combined.
    """
    return {**(left or {}), **(right or {})}


class PipelineState(TypedDict, total=False):
    # Inputs
    hours: int
    top_n: int
    dry_run: bool

    # Per-stage stats, reported in the run summary
    scraping: dict
    processing: Annotated[dict, merge_dicts]
    digests: dict

    # Curation
    candidates: list[dict]        # digests in the lookback window, newest first
    ranked: list[dict]            # RankedArticle dicts, best first
    ranking_problems: list[str]   # why the last ranking was rejected; fed back to the LLM
    ranking_attempts: int
    ranking_method: str           # "llm", or "fallback" when the LLM never produced a valid ranking

    # Delivery
    email_digest: EmailDigestResponse
    email: dict
