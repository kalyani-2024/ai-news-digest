import logging
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()

from app import PROJECT_ROOT
from app.agent.email_agent import RankedArticleDetail, EmailDigestResponse
from app.agent.curator_agent import RankedArticle
from app.services.email import send_email, digest_to_html

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

PREVIEW_PATH = PROJECT_ROOT / "output" / "digest_preview.html"


def build_article_details(ranked: List[RankedArticle], digests: List[dict]) -> List[RankedArticleDetail]:
    """Join the curator's ranking back onto the stored digests by digest ID."""
    by_id = {d["id"]: d for d in digests}
    return [
        RankedArticleDetail(
            digest_id=a.digest_id,
            rank=a.rank,
            relevance_score=a.relevance_score,
            reasoning=a.reasoning,
            title=by_id[a.digest_id]["title"],
            summary=by_id[a.digest_id]["summary"],
            url=by_id[a.digest_id]["url"],
            article_type=by_id[a.digest_id]["article_type"],
        )
        for a in ranked
        if a.digest_id in by_id
    ]


def deliver_digest(email_digest: EmailDigestResponse, dry_run: bool = False) -> dict:
    """Render the digest and email it, or write an HTML preview when dry_run is set."""
    markdown_content = email_digest.to_markdown()
    html_content = digest_to_html(email_digest)

    greeting = email_digest.introduction.greeting
    subject = f"Daily AI News Digest - {greeting.split('for ')[-1] if 'for ' in greeting else 'Today'}"

    if dry_run:
        PREVIEW_PATH.parent.mkdir(exist_ok=True)
        PREVIEW_PATH.write_text(html_content, encoding="utf-8")
        logger.info(f"Dry run: email not sent, preview written to {PREVIEW_PATH}")
        return {
            "success": True,
            "dry_run": True,
            "subject": subject,
            "preview_path": str(PREVIEW_PATH),
            "articles_count": len(email_digest.articles),
        }

    try:
        send_email(subject=subject, body_text=markdown_content, body_html=html_content)
    except ValueError as e:
        logger.error(f"Error sending email: {e}")
        return {"success": False, "error": str(e)}

    logger.info("Email sent successfully!")
    return {
        "success": True,
        "subject": subject,
        "articles_count": len(email_digest.articles),
    }


def send_digest_email(hours: int = 24, top_n: int = 10, dry_run: bool = False) -> dict:
    """Rank recent digests and deliver the email, without scraping first.

    Runs the curate-and-deliver LangGraph subgraph on its own, so ranking gets
    the same validation, retry, and fallback behavior as the full pipeline.
    """
    from app.graph.pipeline import RECURSION_LIMIT, build_delivery_graph

    final_state = build_delivery_graph().invoke(
        {"hours": hours, "top_n": top_n, "dry_run": dry_run},
        config={"run_name": "curate_and_deliver", "recursion_limit": RECURSION_LIMIT},
    )
    return final_state["email"]


if __name__ == "__main__":
    import sys

    result = send_digest_email(hours=24, top_n=10, dry_run="--dry-run" in sys.argv)
    if result.get("skipped"):
        print(f"\nNothing to send: {result['reason']}")
    elif result["success"]:
        print("\n=== Email Digest Sent ===" if not result.get("dry_run") else "\n=== Dry Run ===")
        print(f"Subject: {result['subject']}")
        print(f"Articles: {result['articles_count']}")
    else:
        print(f"Error: {result['error']}")
