"""Framework-free research loop: route → retrieve → draft or abstain → receipt."""

from __future__ import annotations

from grounded_agent.models import ResearchRequest, ResearchResult
from grounded_agent.provider import DraftProvider, ExtractiveProvider, guarded_draft
from grounded_agent.receipt import build_receipt
from grounded_agent.router import route_intent
from grounded_agent.tools import retrieve_traced
from grounded_agent.trace import MemoryTracer, NullTracer


def run_research(
    request: ResearchRequest,
    *,
    tracer: MemoryTracer | NullTracer | None = None,
    provider: DraftProvider | None = None,
) -> ResearchResult:
    drafter = provider if provider is not None else ExtractiveProvider()
    log = tracer if tracer is not None else NullTracer()
    route = route_intent(request)
    log.emit("route", intent=route.intent, scopes=list(route.scopes))
    retrieval = retrieve_traced(request, route)
    log.emit(
        "retrieve",
        tools_used=list(retrieval.tools_used),
        knowledge_paths=[item.source_path for item in retrieval.bundle.knowledge],
        project_paths=[item.source_path for item in retrieval.bundle.project],
        tool_errors=list(retrieval.tool_errors),
    )
    answer = guarded_draft(drafter, request, retrieval.bundle)
    log.emit(
        "draft",
        outcome=answer.outcome,
        abstain_reason=answer.abstain_reason,
        citation_paths=[citation.source_path for citation in answer.citations],
        provider=drafter.name,
    )
    receipt = build_receipt(
        request,
        route,
        answer,
        tools_used=retrieval.tools_used,
        tool_errors=retrieval.tool_errors,
        provider=drafter.name,
    )
    log.emit(
        "receipt",
        receipt_hash=receipt.receipt_hash,
        tools_used=list(receipt.tools_used),
        content_redacted=receipt.content_redacted,
    )
    return ResearchResult(
        request=request,
        route=route,
        evidence=retrieval.bundle,
        answer=answer,
        receipt=receipt,
    )
