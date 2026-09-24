import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from app.graph.pipeline import RECURSION_LIMIT, build_pipeline_graph

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_daily_pipeline(hours: int = 24, top_n: int = 10, dry_run: bool = False) -> dict:
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("Starting Daily AI News Aggregator Pipeline (LangGraph)")
    logger.info("=" * 60)

    results = {
        "start_time": start_time.isoformat(),
        "scraping": {},
        "processing": {},
        "digests": {},
        "email": {},
        "success": False
    }

    try:
        # run_name and tags label the run in LangSmith when tracing is enabled.
        final_state = build_pipeline_graph().invoke(
            {"hours": hours, "top_n": top_n, "dry_run": dry_run},
            config={"run_name": "daily_digest", "tags": ["dry-run" if dry_run else "scheduled"], "recursion_limit": RECURSION_LIMIT},
        )
        for key in ("scraping", "processing", "digests", "email"):
            results[key] = final_state.get(key, {})

        email_result = results["email"]
        if email_result.get("skipped"):
            logger.info(f"✓ {email_result['reason']} - no digest to send today")
            results["success"] = True
        elif email_result.get("success"):
            action = "Preview written" if email_result.get("dry_run") else "Email sent successfully"
            logger.info(f"✓ {action} with {email_result['articles_count']} articles "
                        f"(ranking: {email_result.get('ranking_method')})")
            results["success"] = True
        else:
            logger.error(f"✗ Failed to send email: {email_result.get('error', 'Unknown error')}")

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        results["error"] = str(e)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    results["end_time"] = end_time.isoformat()
    results["duration_seconds"] = duration

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Summary")
    logger.info("=" * 60)
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info(f"Scraped: {results['scraping']}")
    logger.info(f"Processed: {results['processing']}")
    logger.info(f"Digests: {results['digests']}")
    if results.get("email", {}).get("skipped"):
        email_status = "Nothing to send"
    elif results.get("email", {}).get("dry_run"):
        email_status = f"Dry run, preview at {results['email']['preview_path']}"
    else:
        email_status = "Sent" if results["success"] else "Failed"
    logger.info(f"Email: {email_status}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    result = run_daily_pipeline(hours=24, top_n=10)
    exit(0 if result["success"] else 1)
