"""Offline tests for the LangGraph pipeline.

No network, LLM, database, or SMTP: the agents and services are replaced with
fakes, so these tests check graph behavior itself (routing, the retry cycle,
the fallback, and how parallel branches merge state).
"""
import pytest

from app.agent.curator_agent import RankedArticle, validate_ranking
from app.agent.email_agent import EmailDigestResponse, EmailIntroduction
from app.graph import nodes
from app.graph.pipeline import build_delivery_graph, build_pipeline_graph
from app.graph.state import merge_dicts

CANDIDATES = [
    {"id": "openai:1", "article_type": "openai", "article_id": "1", "url": "https://a/1",
     "title": "Newest", "summary": "s1", "created_at": None},
    {"id": "anthropic:2", "article_type": "anthropic", "article_id": "2", "url": "https://a/2",
     "title": "Middle", "summary": "s2", "created_at": None},
    {"id": "youtube:3", "article_type": "youtube", "article_id": "3", "url": "https://a/3",
     "title": "Oldest", "summary": "s3", "created_at": None},
]


def ranked(digest_id, score, rank):
    return RankedArticle(digest_id=digest_id, relevance_score=score, rank=rank, reasoning="r")


COMPLETE = [ranked("anthropic:2", 9.0, 1), ranked("youtube:3", 7.0, 2), ranked("openai:1", 3.0, 3)]
INCOMPLETE = [ranked("anthropic:2", 9.0, 1), ranked("made-up:99", 8.0, 2)]


class FakeCurator:
    """Returns one scripted ranking per call and records the feedback it was given."""
    responses = []
    feedback_seen = []

    def __init__(self, profile):
        pass

    def rank_digests(self, digests, feedback=None):
        FakeCurator.feedback_seen.append(feedback)
        return FakeCurator.responses.pop(0)


class FakeEmailAgent:
    def __init__(self, profile):
        pass

    def create_email_digest_response(self, ranked_articles, total_ranked, limit):
        return EmailDigestResponse(
            introduction=EmailIntroduction(greeting="Hey Test", introduction="Intro"),
            articles=ranked_articles[:limit],
            total_ranked=total_ranked,
            top_n=limit,
        )


@pytest.fixture
def delivery(monkeypatch):
    """Patch every external dependency of the delivery subgraph."""
    FakeCurator.responses = []
    FakeCurator.feedback_seen = []
    sent = []
    candidates = list(CANDIDATES)

    class FakeRepository:
        def get_recent_digests(self, hours):
            return candidates

    monkeypatch.setattr(nodes, "Repository", FakeRepository)
    monkeypatch.setattr(nodes, "CuratorAgent", FakeCurator)
    monkeypatch.setattr(nodes, "EmailAgent", FakeEmailAgent)
    monkeypatch.setattr(nodes, "deliver_digest",
                        lambda digest, dry_run: sent.append(digest) or {"success": True, "articles_count": len(digest.articles)})
    return {"sent": sent, "candidates": candidates}


def run_delivery():
    return build_delivery_graph().invoke({"hours": 24, "top_n": 10, "dry_run": True})


# --- validate_ranking --------------------------------------------------------

def test_validate_accepts_complete_ranking_and_orders_by_score():
    shuffled = [ranked("openai:1", 3.0, 1), ranked("anthropic:2", 9.0, 1), ranked("youtube:3", 7.0, 5)]
    usable, problems = validate_ranking(shuffled, [c["id"] for c in CANDIDATES])
    assert problems == []
    assert [a.digest_id for a in usable] == ["anthropic:2", "youtube:3", "openai:1"]
    assert [a.rank for a in usable] == [1, 2, 3]


def test_validate_flags_unknown_duplicate_and_missing_ids():
    bad = [ranked("anthropic:2", 9.0, 1), ranked("anthropic:2", 8.0, 2), ranked("made-up:99", 7.0, 3)]
    usable, problems = validate_ranking(bad, [c["id"] for c in CANDIDATES])
    assert [a.digest_id for a in usable] == ["anthropic:2"]
    assert any("unknown ID 'made-up:99'" in p for p in problems)
    assert any("duplicate ID 'anthropic:2'" in p for p in problems)
    assert any("missing IDs: openai:1, youtube:3" in p for p in problems)


def test_merge_dicts_combines_parallel_writes():
    assert merge_dicts({"anthropic": 1}, {"youtube": 2}) == {"anthropic": 1, "youtube": 2}
    assert merge_dicts(None, {"youtube": 2}) == {"youtube": 2}


# --- delivery subgraph routing -----------------------------------------------

def test_valid_ranking_goes_straight_to_email(delivery):
    FakeCurator.responses = [COMPLETE]
    state = run_delivery()
    assert state["ranking_attempts"] == 1
    assert state["ranking_method"] == "llm"
    assert [a.title for a in delivery["sent"][0].articles] == ["Middle", "Oldest", "Newest"]


def test_invalid_ranking_retries_with_feedback(delivery):
    FakeCurator.responses = [INCOMPLETE, COMPLETE]
    state = run_delivery()
    assert state["ranking_attempts"] == 2
    assert state["ranking_method"] == "llm"
    assert FakeCurator.feedback_seen[0] is None
    assert any("made-up:99" in p for p in FakeCurator.feedback_seen[1])


def test_repeated_failure_falls_back_without_losing_articles(delivery):
    FakeCurator.responses = [INCOMPLETE, INCOMPLETE]
    state = run_delivery()
    assert state["ranking_method"] == "fallback"
    titles = [a.title for a in delivery["sent"][0].articles]
    # The one valid LLM pick stays first; the rest follow newest first.
    assert titles == ["Middle", "Newest", "Oldest"]


def test_quiet_day_skips_ranking_and_email(delivery):
    delivery["candidates"].clear()
    state = run_delivery()
    assert state["email"]["skipped"] is True
    assert FakeCurator.feedback_seen == []
    assert delivery["sent"] == []


# --- full pipeline -----------------------------------------------------------

def test_full_pipeline_merges_parallel_extract_results(delivery, monkeypatch):
    FakeCurator.responses = [COMPLETE]
    monkeypatch.setattr(nodes, "run_scrapers", lambda hours: {"openai": [1, 2], "anthropic": [], "youtube": [3]})
    monkeypatch.setattr(nodes, "process_anthropic_markdown", lambda: {"total": 1, "processed": 1, "failed": 0})
    monkeypatch.setattr(nodes, "process_youtube_transcripts",
                        lambda: {"total": 1, "processed": 1, "unavailable": 0, "failed": 0})
    monkeypatch.setattr(nodes, "process_digests", lambda: {"total": 3, "processed": 3, "failed": 0})

    state = build_pipeline_graph().invoke({"hours": 24, "top_n": 2, "dry_run": True})

    assert state["scraping"] == {"youtube": 1, "openai": 2, "anthropic": 0}
    assert set(state["processing"]) == {"anthropic", "youtube"}
    assert state["email"]["success"] is True
    assert len(delivery["sent"][0].articles) == 2
