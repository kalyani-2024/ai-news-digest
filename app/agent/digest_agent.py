import os
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from app.agent.client import structured_llm, DIGEST_MODEL

# Articles are independent, so they are summarized concurrently. Kept low by
# default because the free Gemini tier enforces a requests-per-minute limit.
DIGEST_CONCURRENCY = int(os.getenv("DIGEST_CONCURRENCY", "4"))
MAX_CONTENT_CHARS = 8000


class DigestOutput(BaseModel):
    title: str
    summary: str

PROMPT = """You are an expert AI news analyst specializing in summarizing technical articles, research papers, and video content about artificial intelligence.

Your role is to create concise, informative digests that help readers quickly understand the key points and significance of AI-related content.

Guidelines:
- Create a compelling title (5-10 words) that captures the essence of the content
- Write a 2-3 sentence summary that highlights the main points and why they matter
- Focus on actionable insights and implications
- Use clear, accessible language while maintaining technical accuracy
- Avoid marketing fluff - focus on substance"""

# Article text is passed as a template variable, never formatted into the
# template string, so braces inside scraped content cannot break the prompt.
DIGEST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", PROMPT),
    ("human", "Create a digest for this {article_type}: \n Title: {title} \n Content: {content}"),
])


class DigestAgent:
    def __init__(self):
        self.model = DIGEST_MODEL
        self.chain = DIGEST_PROMPT | structured_llm(DigestOutput, self.model, temperature=0.7)

    @staticmethod
    def _inputs(title: str, content: str, article_type: str) -> dict:
        return {"title": title, "content": content[:MAX_CONTENT_CHARS], "article_type": article_type}

    def generate_digest(self, title: str, content: str, article_type: str) -> Optional[DigestOutput]:
        try:
            return self.chain.invoke(self._inputs(title, content, article_type))
        except Exception as e:
            print(f"Error generating digest: {e}")
            return None

    def generate_digests(self, articles: List[dict]) -> List[Optional[DigestOutput]]:
        """Summarize many articles concurrently; one failure does not sink the batch.

        Returns results in the same order as `articles`, with None for failures.
        """
        inputs = [self._inputs(a["title"], a["content"], a["type"]) for a in articles]
        results = self.chain.batch(
            inputs,
            config={"max_concurrency": DIGEST_CONCURRENCY, "run_name": "digest_batch"},
            return_exceptions=True,
        )
        digests = []
        for article, result in zip(articles, results):
            if isinstance(result, Exception):
                print(f"Error generating digest for {article['type']} {article['id']}: {result}")
                digests.append(None)
            else:
                digests.append(result)
        return digests
