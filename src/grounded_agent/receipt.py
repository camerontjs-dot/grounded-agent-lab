"""Redacted run receipts. Snippets and prompts never enter the hashed payload."""

from __future__ import annotations

import hashlib
import json

from grounded_agent.models import Answer, Receipt, ResearchRequest, RouteDecision
from grounded_agent.redact import assert_no_sensitive


def _digest(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_receipt(
    request: ResearchRequest,
    route: RouteDecision,
    answer: Answer,
    *,
    tools_used: tuple[str, ...] = (),
    tool_errors: tuple[str, ...] = (),
) -> Receipt:
    citation_paths = tuple(citation.source_path for citation in answer.citations)
    trust_profiles = tuple(dict.fromkeys(citation.trust_profile for citation in answer.citations))
    payload = {
        "request_id": request.request_id,
        "intent": route.intent,
        "scopes": list(route.scopes),
        "outcome": answer.outcome,
        "citation_paths": list(citation_paths),
        "trust_profiles_used": list(trust_profiles),
        "abstain_reason": answer.abstain_reason,
        "tools_used": list(tools_used),
        "tool_errors": list(tool_errors),
        "content_redacted": True,
    }
    receipt = Receipt(
        request_id=request.request_id,
        intent=route.intent,
        scopes=route.scopes,
        outcome=answer.outcome,
        citation_paths=citation_paths,
        trust_profiles_used=trust_profiles,
        abstain_reason=answer.abstain_reason,
        tools_used=tools_used,
        tool_errors=tool_errors,
        content_redacted=True,
        receipt_hash=_digest(payload),
    )
    serialized = receipt.model_dump_json()
    if request.question in serialized:
        raise ValueError("receipt leaked the question text")
    if answer.text in serialized and answer.outcome == "answered":
        raise ValueError("receipt leaked answer text")
    assert_no_sensitive(serialized, label="receipt")
    return receipt
