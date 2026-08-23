"""Framework-free research loop: route → retrieve → draft or abstain → receipt."""

from __future__ import annotations

from grounded_agent.draft import draft_or_abstain
from grounded_agent.models import ResearchRequest, ResearchResult
from grounded_agent.receipt import build_receipt
from grounded_agent.router import route_intent
from grounded_agent.tools import retrieve_traced


def run_research(request: ResearchRequest) -> ResearchResult:
    route = route_intent(request)
    trace = retrieve_traced(request, route)
    answer = draft_or_abstain(request, trace.bundle)
    receipt = build_receipt(
        request,
        route,
        answer,
        tools_used=trace.tools_used,
        tool_errors=trace.tool_errors,
    )
    return ResearchResult(
        request=request,
        route=route,
        evidence=trace.bundle,
        answer=answer,
        receipt=receipt,
    )
