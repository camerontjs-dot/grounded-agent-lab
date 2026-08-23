"""LangGraph wrap of the framework-free research loop.

Nodes call the same functions as `pipeline.run_research`. State is JSON-shaped
dicts so a checkpointer can serialize it. Review is optional: without a
checkpointer the review node auto-approves and the graph is equivalent to the
baseline.
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from grounded_agent.models import (
    Answer,
    EvidenceBundle,
    Receipt,
    ResearchRequest,
    ResearchResult,
    RouteDecision,
)
from grounded_agent.provider import DraftProvider, ExtractiveProvider, guarded_draft
from grounded_agent.receipt import build_receipt
from grounded_agent.router import route_intent
from grounded_agent.tools import retrieve_traced

ReviewDecision = Literal["approve", "reject"]

GRAPH_NODES = ("route", "retrieve", "draft", "review", "emit_receipt")


class ResearchGraphState(TypedDict):
    request: dict[str, Any]
    require_review: bool
    route: NotRequired[dict[str, Any]]
    evidence: NotRequired[dict[str, Any]]
    answer: NotRequired[dict[str, Any]]
    review_decision: NotRequired[ReviewDecision]
    tools_used: NotRequired[list[str]]
    tool_errors: NotRequired[list[str]]
    provider: NotRequired[str]
    receipt: NotRequired[dict[str, Any]]


def _json(model: Any) -> dict[str, Any]:
    return model.model_dump(mode="json")


def _route(state: ResearchGraphState) -> dict[str, Any]:
    request = ResearchRequest.model_validate(state["request"])
    return {"route": _json(route_intent(request))}


def _retrieve(state: ResearchGraphState) -> dict[str, Any]:
    request = ResearchRequest.model_validate(state["request"])
    route = RouteDecision.model_validate(state["route"])
    trace = retrieve_traced(request, route)
    return {
        "evidence": _json(trace.bundle),
        "tools_used": list(trace.tools_used),
        "tool_errors": list(trace.tool_errors),
    }


def _draft_with(provider: DraftProvider):
    def _draft(state: ResearchGraphState) -> dict[str, Any]:
        request = ResearchRequest.model_validate(state["request"])
        evidence = EvidenceBundle.model_validate(state["evidence"])
        answer = guarded_draft(provider, request, evidence)
        return {"answer": _json(answer), "provider": provider.name}

    return _draft


def _review(state: ResearchGraphState) -> dict[str, Any]:
    if not state.get("require_review"):
        return {"review_decision": "approve"}
    answer = state["answer"]
    decision = interrupt(
        {
            "kind": "review_draft",
            "request_id": state["request"]["request_id"],
            "outcome": answer["outcome"],
            "abstain_reason": answer.get("abstain_reason"),
            "citation_paths": [item["source_path"] for item in answer.get("citations", [])],
        }
    )
    if decision not in {"approve", "reject"}:
        raise ValueError("review decision must be 'approve' or 'reject'")
    return {"review_decision": decision}


def _emit_receipt(state: ResearchGraphState) -> dict[str, Any]:
    if state.get("receipt"):
        return {}
    request = ResearchRequest.model_validate(state["request"])
    route = RouteDecision.model_validate(state["route"])
    answer = Answer.model_validate(state["answer"])
    if state.get("review_decision") == "reject":
        answer = Answer(
            outcome="abstained",
            text="Refused: reviewer rejected the draft.",
            abstain_reason="review_rejected",
        )
    receipt = build_receipt(
        request,
        route,
        answer,
        tools_used=tuple(state.get("tools_used") or ()),
        tool_errors=tuple(state.get("tool_errors") or ()),
        provider=str(state.get("provider") or "extractive"),
    )
    payload: dict[str, Any] = {"receipt": _json(receipt)}
    if state.get("review_decision") == "reject":
        payload["answer"] = _json(answer)
    return payload


def compile_research_graph(
    *, checkpointer: Any | None = None, provider: DraftProvider | None = None
) -> Any:
    drafter = provider if provider is not None else ExtractiveProvider()
    builder = StateGraph(ResearchGraphState)
    builder.add_node("route", _route)
    builder.add_node("retrieve", _retrieve)
    builder.add_node("draft", _draft_with(drafter))
    builder.add_node("review", _review)
    builder.add_node("emit_receipt", _emit_receipt)
    builder.add_edge(START, "route")
    builder.add_edge("route", "retrieve")
    builder.add_edge("retrieve", "draft")
    builder.add_edge("draft", "review")
    builder.add_edge("review", "emit_receipt")
    builder.add_edge("emit_receipt", END)
    return builder.compile(checkpointer=checkpointer)


def result_from_state(state: ResearchGraphState) -> ResearchResult:
    return ResearchResult(
        request=ResearchRequest.model_validate(state["request"]),
        route=RouteDecision.model_validate(state["route"]),
        evidence=EvidenceBundle.model_validate(state["evidence"]),
        answer=Answer.model_validate(state["answer"]),
        receipt=Receipt.model_validate(state["receipt"]),
    )


def run_research_graph(
    request: ResearchRequest, *, provider: DraftProvider | None = None
) -> ResearchResult:
    """End-to-end graph run with no human review. Must match `run_research`."""

    graph = compile_research_graph(provider=provider)
    state = graph.invoke({"request": _json(request), "require_review": False})
    return result_from_state(state)


def thread_config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def start_reviewed_run(
    request: ResearchRequest,
    *,
    checkpointer: Any,
    provider: DraftProvider | None = None,
) -> Any:
    """Run until the review interrupt. No receipt is written yet."""

    graph = compile_research_graph(checkpointer=checkpointer, provider=provider)
    config = thread_config(request.request_id)
    graph.invoke(
        {"request": _json(request), "require_review": True},
        config,
    )
    return graph, config


def resume_reviewed_run(
    graph: Any, config: dict[str, Any], decision: ReviewDecision
) -> ResearchResult:
    state = graph.invoke(Command(resume=decision), config)
    return result_from_state(state)
