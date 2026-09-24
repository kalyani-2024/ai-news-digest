"""Graph nodes and routing functions.

A node takes the current PipelineState and returns a dict of the keys it
updates. A router takes the state and returns the name of the next node.
The ingestion nodes wrap the existing service functions unchanged; the
curation nodes add validation, a feedback retry loop, and a fallback.
"""
import logging
import os

from langgraph.graph import END

from app.agent.curator_agent import CuratorAgent, RankedArticle, validate_ranking
from app.agent.email_agent import EmailAgent
from app.database.repository import Repository
from app.profiles.user_profile import USER_PROFILE
from app.runner import run_scrapers
from app.services.process_anthropic import process_anthropic_markdown
from app.services.process_digest import process_digests
from app.services.process_email import build_article_details, deliver_digest
from app.services.process_youtube import process_youtube_transcripts
from app.graph.state import PipelineState

logger = logging.getLogger(__name__)

# How many times the LLM may try to produce a complete, valid ranking before
# the deterministic fallback takes over. Each attempt already includes the SDK's
# own HTTP retries and the model fallback, so this guards content, not transport.
MAX_RANK_ATTEMPTS = int(os.getenv("MAX_RANK_ATTEMPTS", "2"))


# --- Ingestion ---------------------------------------------------------------

def scrape(state: PipelineState) -> dict:
    logger.info("[scrape] Scraping articles from sources...")
    results = run_scrapers(hours=state["hours"])
    counts = {source: len(results.get(source, [])) for source in ("youtube", "openai", "anthropic")}
    logger.info(f"[scrape] ✓ {counts}")
    return {"scraping": counts}


def extract_anthropic(state: PipelineState) -> dict:
    logger.info("[extract_anthropic] Converting Anthropic articles to markdown...")
    result = process_anthropic_markdown()
    logger.info(f"[extract_anthropic] ✓ {result['processed']} processed, {result['failed']} failed")
    return {"processing": {"anthropic": result}}


def extract_youtube(state: PipelineState) -> dict:
    logger.info("[extract_youtube] Fetching YouTube transcripts...")
    result = process_youtube_transcripts()
    logger.info(f"[extract_youtube] ✓ {result['processed']} processed, {result['unavailable']} unavailable")
    return {"processing": {"youtube": result}}


def summarize(state: PipelineState) -> dict:
    logger.info("[summarize] Creating digests...")
    result = process_digests()
    logger.info(f"[summarize] ✓ {result['processed']} created, {result['failed']} failed of {result['total']}")
    return {"digests": result}


# --- Curation and delivery ---------------------------------------------------

def load_candidates(state: PipelineState) -> dict:
    hours = state["hours"]
    candidates = Repository().get_recent_digests(hours=hours)
    logger.info(f"[load_candidates] {len(candidates)} digests published in the last {hours} hours")
    update = {"candidates": candidates, "ranking_attempts": 0, "ranking_problems": []}
    if not candidates:
        # A quiet news day is a normal outcome: report it as a successful skip.
        update["email"] = {
            "success": True,
            "skipped": True,
            "reason": f"No articles published in the last {hours} hours",
            "articles_count": 0,
        }
    return update


def route_after_load(state: PipelineState) -> str:
    return "rank" if state["candidates"] else END


def rank(state: PipelineState) -> dict:
    attempt = state.get("ranking_attempts", 0) + 1
    feedback = state.get("ranking_problems") or None
    logger.info(f"[rank] Attempt {attempt}/{MAX_RANK_ATTEMPTS}"
                + (f" with feedback: {feedback}" if feedback else ""))
    ranked = CuratorAgent(USER_PROFILE).rank_digests(state["candidates"], feedback=feedback)
    return {"ranked": [a.model_dump() for a in ranked], "ranking_attempts": attempt}


def validate(state: PipelineState) -> dict:
    ranked = [RankedArticle(**r) for r in state.get("ranked", [])]
    candidate_ids = [d["id"] for d in state["candidates"]]
    usable, problems = validate_ranking(ranked, candidate_ids)
    if not ranked:
        problems = ["the ranking call returned no articles"]
    if problems:
        logger.warning(f"[validate] Ranking rejected: {problems}")
    else:
        logger.info(f"[validate] ✓ Ranking covers all {len(candidate_ids)} digests")
    return {
        "ranked": [a.model_dump() for a in usable],
        "ranking_problems": problems,
        "ranking_method": "llm",
    }


def route_after_validate(state: PipelineState) -> str:
    if not state["ranking_problems"]:
        return "compose_email"
    if state["ranking_attempts"] < MAX_RANK_ATTEMPTS:
        return "rank"
    return "fallback_rank"


def fallback_rank(state: PipelineState) -> dict:
    """Keep whatever the LLM ranked validly, then append the rest newest first.

    Better to send a digest in a slightly worse order than to send nothing.
    """
    ranked = list(state.get("ranked", []))
    ranked_ids = {r["digest_id"] for r in ranked}
    for digest in state["candidates"]:  # already ordered newest first
        if digest["id"] not in ranked_ids:
            ranked.append({
                "digest_id": digest["id"],
                "relevance_score": 0.0,
                "rank": len(ranked) + 1,
                "reasoning": "Not ranked by the curator; ordered by recency",
            })
    logger.warning(f"[fallback_rank] Using fallback ranking: {len(ranked_ids)} LLM-ranked, "
                   f"{len(ranked) - len(ranked_ids)} added by recency")
    return {"ranked": ranked, "ranking_method": "fallback"}


def compose_email(state: PipelineState) -> dict:
    ranked = [RankedArticle(**r) for r in state["ranked"]]
    details = build_article_details(ranked, state["candidates"])
    email_digest = EmailAgent(USER_PROFILE).create_email_digest_response(
        ranked_articles=details,
        total_ranked=len(details),
        limit=state["top_n"],
    )
    logger.info(f"[compose_email] ✓ {email_digest.introduction.greeting}")
    return {"email_digest": email_digest}


def send(state: PipelineState) -> dict:
    result = deliver_digest(state["email_digest"], dry_run=state.get("dry_run", False))
    result["ranking_method"] = state.get("ranking_method")
    return {"email": result}
