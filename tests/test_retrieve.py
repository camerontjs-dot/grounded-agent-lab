"""Fixture retrieval preserves the trust split and weak-fit band."""

from __future__ import annotations

from grounded_agent.models import ResearchRequest
from grounded_agent.retrieve import overlap, retrieve
from grounded_agent.router import route_intent


def test_knowledge_retrieval_does_not_return_project_items() -> None:
    request = ResearchRequest(
        request_id="r1",
        question="What must Harbor do when combining retrieval results from different indexes?",
    )
    bundle = retrieve(request, route_intent(request))
    assert bundle.project == ()
    assert bundle.knowledge
    assert all(item.trust_profile == "durable_knowledge" for item in bundle.knowledge)
    assert bundle.citable()


def test_single_token_overlap_is_weak_fit_not_citable() -> None:
    score, shared = overlap(
        "What is Harbor's CEO salary?",
        "Harbor stores durable notes and live project status in two separate indexes.",
    )
    assert shared == 1
    assert score > 0
    request = ResearchRequest(request_id="r2", question="What is Harbor's CEO salary?")
    bundle = retrieve(request, route_intent(request))
    assert all(item.weak_fit for item in bundle.knowledge)
    assert bundle.citable() == ()
