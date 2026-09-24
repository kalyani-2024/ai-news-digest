from typing import List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from app.agent.client import structured_llm, CURATOR_MODEL


class RankedArticle(BaseModel):
    digest_id: str = Field(description="The ID of the digest (article_type:article_id)")
    relevance_score: float = Field(description="Relevance score from 0.0 to 10.0", ge=0.0, le=10.0)
    rank: int = Field(description="Rank position (1 = most relevant)", ge=1)
    reasoning: str = Field(description="Brief explanation of why this article is ranked here")


class RankedDigestList(BaseModel):
    articles: List[RankedArticle] = Field(description="List of ranked articles")


CURATOR_PROMPT = """You are an expert AI news curator specializing in personalized content ranking for AI professionals.

Your role is to analyze and rank AI-related news articles, research papers, and video content based on a user's specific profile, interests, and background.

Ranking Criteria:
1. Relevance to user's stated interests and background
2. Technical depth and practical value
3. Novelty and significance of the content
4. Alignment with user's expertise level
5. Actionability and real-world applicability

Scoring Guidelines:
- 9.0-10.0: Highly relevant, directly aligns with user interests, significant value
- 7.0-8.9: Very relevant, strong alignment with interests, good value
- 5.0-6.9: Moderately relevant, some alignment, decent value
- 3.0-4.9: Somewhat relevant, limited alignment, lower value
- 0.0-2.9: Low relevance, minimal alignment, little value

Rank articles from most relevant (rank 1) to least relevant. Ensure each article has a unique rank."""

RANKING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    ("human", """Rank these {count} AI news digests based on the user profile:

{digest_list}

Provide a relevance score (0.0-10.0) and rank (1-{count}) for each article, ordered from most to least relevant.
Use each ID exactly as given, and include every ID exactly once.{feedback}"""),
])


def validate_ranking(
    ranked: List[RankedArticle], candidate_ids: List[str]
) -> Tuple[List[RankedArticle], List[str]]:
    """Check an LLM ranking against the digests it was asked to rank.

    Returns the usable entries, re-ordered and re-numbered, plus a list of
    problems. An empty problem list means the ranking is complete and trustworthy.
    Structured output guarantees the JSON shape; it cannot guarantee the model
    ranked every article or did not invent an ID, so that is checked here.
    """
    expected = set(candidate_ids)
    problems = []
    seen = set()
    usable = []

    for article in ranked:
        if article.digest_id not in expected:
            problems.append(f"unknown ID {article.digest_id!r} (not in the list you were given)")
        elif article.digest_id in seen:
            problems.append(f"duplicate ID {article.digest_id!r}")
        else:
            seen.add(article.digest_id)
            usable.append(article)

    missing = [digest_id for digest_id in candidate_ids if digest_id not in seen]
    if missing:
        problems.append(f"missing IDs: {', '.join(missing)}")

    # The score is the model's actual judgement; its rank numbers can collide or
    # skip, so order by score (rank breaks ties) and number the result 1..n.
    usable.sort(key=lambda a: (-a.relevance_score, a.rank))
    usable = [a.model_copy(update={"rank": i}) for i, a in enumerate(usable, 1)]
    return usable, problems


class CuratorAgent:
    def __init__(self, user_profile: dict):
        self.model = CURATOR_MODEL
        self.user_profile = user_profile
        self.system_prompt = self._build_system_prompt()
        # temperature 0.3: ranking should be stable run to run, not creative.
        self.chain = RANKING_PROMPT | structured_llm(RankedDigestList, self.model, temperature=0.3)

    def _build_system_prompt(self) -> str:
        interests = "\n".join(f"- {interest}" for interest in self.user_profile["interests"])
        preferences = self.user_profile["preferences"]
        pref_text = "\n".join(f"- {k}: {v}" for k, v in preferences.items())

        return f"""{CURATOR_PROMPT}

User Profile:
Name: {self.user_profile["name"]}
Background: {self.user_profile["background"]}
Expertise Level: {self.user_profile["expertise_level"]}

Interests:
{interests}

Preferences:
{pref_text}"""

    def rank_digests(self, digests: List[dict], feedback: Optional[List[str]] = None) -> List[RankedArticle]:
        """Rank digests against the profile.

        `feedback` lists what was wrong with a previous attempt, so a retry can
        correct it rather than repeat the same mistake.
        """
        if not digests:
            return []

        digest_list = "\n\n".join([
            f"ID: {d['id']}\nTitle: {d['title']}\nSummary: {d['summary']}\nType: {d['article_type']}"
            for d in digests
        ])
        feedback_text = ""
        if feedback:
            feedback_text = "\n\nYour previous ranking was rejected because of: " + "; ".join(feedback)

        try:
            ranked_list = self.chain.invoke({
                "system_prompt": self.system_prompt,
                "count": len(digests),
                "digest_list": digest_list,
                "feedback": feedback_text,
            })
            return ranked_list.articles if ranked_list else []
        except Exception as e:
            print(f"Error ranking digests: {e}")
            return []
