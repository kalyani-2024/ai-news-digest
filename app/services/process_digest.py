from typing import Optional
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.agent.digest_agent import DigestAgent
from app.database.repository import Repository

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def process_digests(limit: Optional[int] = None) -> dict:
    agent = DigestAgent()
    repo = Repository()
    
    articles = repo.get_articles_without_digest(limit=limit)
    total = len(articles)
    processed = 0
    failed = 0
    
    logger.info(f"Starting digest processing for {total} articles")
    
    # Summaries are generated concurrently; database writes stay sequential
    # because a SQLAlchemy session is not safe to share across threads.
    digest_results = agent.generate_digests(articles)
    
    for idx, (article, digest_result) in enumerate(zip(articles, digest_results), 1):
        article_type = article["type"]
        article_id = article["id"]
        article_title = article["title"][:60] + "..." if len(article["title"]) > 60 else article["title"]
        
        logger.info(f"[{idx}/{total}] {article_type}: {article_title} (ID: {article_id})")
        
        try:
            if digest_result:
                repo.create_digest(
                    article_type=article_type,
                    article_id=article_id,
                    url=article["url"],
                    title=digest_result.title,
                    summary=digest_result.summary,
                    published_at=article.get("published_at")
                )
                processed += 1
                logger.info(f"✓ Successfully created digest for {article_type} {article_id}")
            else:
                failed += 1
                logger.warning(f"✗ Failed to generate digest for {article_type} {article_id}")
        except Exception as e:
            failed += 1
            logger.error(f"✗ Error processing {article_type} {article_id}: {e}")
    
    logger.info(f"Processing complete: {processed} processed, {failed} failed out of {total} total")
    
    return {
        "total": total,
        "processed": processed,
        "failed": failed
    }


if __name__ == "__main__":
    result = process_digests()
    print(f"Total articles: {result['total']}")
    print(f"Processed: {result['processed']}")
    print(f"Failed: {result['failed']}")

