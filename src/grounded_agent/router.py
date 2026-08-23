"""Deterministic intent router over fixture questions."""

from __future__ import annotations

from grounded_agent.models import INTENT_SCOPES, ResearchRequest, RouteDecision

PROJECT_CUES = (
    "next action",
    "status",
    "lantern-agent",
    "atlas-search",
    "paused",
    "active project",
    "wip",
)
KNOWLEDGE_CUES = (
    "trust label",
    "indexes",
    "retrieval",
    "receipt",
    "nominate",
    "durable",
    "proof",
    "harbor",
    "snippet",
    "citation",
)


def _contains_any(text: str, cues: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(cue in lowered for cue in cues)


def route_intent(request: ResearchRequest) -> RouteDecision:
    """Classify a question into knowledge, project, or labelled federated scopes."""

    question = request.question
    project_hit = _contains_any(question, PROJECT_CUES)
    knowledge_hit = _contains_any(question, KNOWLEDGE_CUES)

    if project_hit and knowledge_hit:
        intent = "federated"
        reason = "question hits both durable-knowledge and project-status cues"
    elif project_hit:
        intent = "project"
        reason = "question hits project-status cues only"
    elif knowledge_hit:
        intent = "knowledge"
        reason = "question hits durable-knowledge cues only"
    else:
        intent = "federated"
        reason = "no strong cue; query both indexes and keep trust labels"

    return RouteDecision(intent=intent, scopes=INTENT_SCOPES[intent], reason=reason)
