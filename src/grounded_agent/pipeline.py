"""Framework-free research loop: route → retrieve → draft or abstain → receipt."""

from __future__ import annotations

from grounded_agent.draft import draft_or_abstain
from grounded_agent.models import ResearchRequest, ResearchResult
from grounded_agent.receipt import build_receipt
from grounded_agent.retrieve import retrieve
from grounded_agent.router import route_intent


def run_research(request: ResearchRequest) -> ResearchResult:
    route = route_intent(request)
    evidence = retrieve(request, route)
    answer = draft_or_abstain(request, evidence)
    receipt = build_receipt(request, route, answer)
    return ResearchResult(
        request=request,
        route=route,
        evidence=evidence,
        answer=answer,
        receipt=receipt,
    )
