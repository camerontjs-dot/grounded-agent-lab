"""Intent routing keeps knowledge and project scopes labelled."""

from __future__ import annotations

from grounded_agent.models import ResearchRequest
from grounded_agent.router import route_intent


def test_knowledge_route() -> None:
    route = route_intent(
        ResearchRequest(
            request_id="r1",
            question="What must Harbor do when combining retrieval results from different indexes?",
        )
    )
    assert route.intent == "knowledge"
    assert route.scopes == ("durable_knowledge",)


def test_project_route() -> None:
    route = route_intent(
        ResearchRequest(request_id="r2", question="What is the next action for lantern-agent?")
    )
    assert route.intent == "project"
    assert route.scopes == ("project_status",)


def test_federated_route_keeps_both_labels() -> None:
    route = route_intent(
        ResearchRequest(
            request_id="r3",
            question="How does the lantern-agent next action relate to Harbor trust labels?",
        )
    )
    assert route.intent == "federated"
    assert route.scopes == ("durable_knowledge", "project_status")
