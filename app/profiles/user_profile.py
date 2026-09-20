import os

USER_PROFILE = {
    "name": os.getenv("USER_NAME", "Reader"),
    "title": os.getenv("USER_TITLE", "AI Engineer"),
    "background": os.getenv(
        "USER_BACKGROUND",
        "Engineer following applied AI, research breakthroughs, and production-ready systems",
    ),
    "interests": [
        "Large Language Models (LLMs) and their applications",
        "Retrieval-Augmented Generation (RAG) systems",
        "AI agent architectures and frameworks",
        "Multimodal AI and vision-language models",
        "AI safety and alignment research",
        "Production AI systems and MLOps",
        "Real-world AI applications and case studies",
        "Research papers with practical implications",
        "AI infrastructure and scaling challenges",
    ],
    "preferences": {
        "prefer_practical": True,
        "prefer_technical_depth": True,
        "prefer_research_breakthroughs": True,
        "prefer_production_focus": True,
        "avoid_marketing_hype": True,
    },
    "expertise_level": os.getenv("USER_EXPERTISE_LEVEL", "Advanced"),
}
