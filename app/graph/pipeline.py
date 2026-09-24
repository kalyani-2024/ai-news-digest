"""LangGraph wiring for the daily pipeline.

    START -> scrape -> extract_anthropic -+-> summarize -> curate_and_deliver -> END
                    \\-> extract_youtube --/

curate_and_deliver is itself a compiled graph (a subgraph):

    START -> load_candidates -> (no digests) -> END
                             -> rank -> validate -> compose_email -> send_email -> END
                                 ^         |
                                 +- retry -+-> fallback_rank -> compose_email

It is built separately so the ranking-and-email stage can also run on its own
(python -m app.services.process_email) with the same validation and fallback.
"""
from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.state import PipelineState

# Backstop for the rank/validate cycle. MAX_RANK_ATTEMPTS is what normally ends
# it; this caps total steps if a routing bug ever made the loop unbounded.
# (LangGraph's own default is 10,007 steps, far too high for a batch job.)
RECURSION_LIMIT = 50


def build_delivery_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("load_candidates", nodes.load_candidates)
    graph.add_node("rank", nodes.rank)
    graph.add_node("validate", nodes.validate)
    graph.add_node("fallback_rank", nodes.fallback_rank)
    graph.add_node("compose_email", nodes.compose_email)
    graph.add_node("send_email", nodes.send)

    graph.add_edge(START, "load_candidates")
    graph.add_conditional_edges("load_candidates", nodes.route_after_load, ["rank", END])
    graph.add_edge("rank", "validate")
    # The cycle: an invalid ranking loops back to rank with the problems as feedback.
    graph.add_conditional_edges(
        "validate", nodes.route_after_validate, ["compose_email", "rank", "fallback_rank"]
    )
    graph.add_edge("fallback_rank", "compose_email")
    graph.add_edge("compose_email", "send_email")
    graph.add_edge("send_email", END)

    return graph.compile()


def build_pipeline_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("scrape", nodes.scrape)
    graph.add_node("extract_anthropic", nodes.extract_anthropic)
    graph.add_node("extract_youtube", nodes.extract_youtube)
    graph.add_node("summarize", nodes.summarize)
    graph.add_node("curate_and_deliver", build_delivery_graph())

    graph.add_edge(START, "scrape")
    # Fan out: both extractors run in the same step, in parallel.
    graph.add_edge("scrape", "extract_anthropic")
    graph.add_edge("scrape", "extract_youtube")
    # Fan in: summarize waits for both extractors to finish.
    graph.add_edge(["extract_anthropic", "extract_youtube"], "summarize")
    graph.add_edge("summarize", "curate_and_deliver")
    graph.add_edge("curate_and_deliver", END)

    return graph.compile()
