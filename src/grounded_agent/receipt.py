"""Redacted run receipts. Snippets and prompts never enter the hashed payload.

The hash binds the bytes behind every citation, the manifest of each corpus that
was searched, and the declared implementation version, so a cited file cannot
change in place while the digest stands still.
"""

from __future__ import annotations

import hashlib
import json
import re

import grounded_agent
from grounded_agent.models import (
    RECEIPT_SCHEMA,
    Answer,
    EvidenceBundle,
    EvidenceItem,
    Receipt,
    ResearchRequest,
    RouteDecision,
    TrustProfile,
)
from grounded_agent.redact import assert_no_sensitive

MIN_LEAK_CHARS = 16
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_PATH_SUFFIX = re.compile(r"\s+\([^()\s]+\)$")


def _digest(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _cited_digests(answer: Answer, evidence: EvidenceBundle) -> tuple[str, ...]:
    """The source-byte digest behind each citation, in citation order."""

    pool = (*evidence.knowledge, *evidence.project)
    bound: list[str] = []
    for citation in answer.citations:
        matches = [
            item
            for item in pool
            if item.source_path == citation.source_path
            and item.trust_profile == citation.trust_profile
        ]
        digests = {item.source_sha256 for item in matches}
        digest = next(iter(digests)) if len(digests) == 1 else None
        if digest is None:
            raise ValueError(
                f"cannot bind {citation.source_path}: no unique source digest in the evidence"
            )
        bound.append(digest)
    return tuple(bound)


def _sentences(text: str) -> tuple[str, ...]:
    """Sentences long enough to be distinctive, with the extractive scaffolding removed.

    The extractive drafter writes `- <sentence> (<path>)` lines under a header, so
    the whole text never appears in a receipt. The leading `- ` and the trailing
    `(path)` are dropped and each remaining sentence is checked on its own.
    """

    found: list[str] = []
    for raw_line in text.splitlines():
        line = _PATH_SUFFIX.sub("", raw_line.strip().removeprefix("- "))
        for piece in _SENTENCE_BREAK.split(line):
            sentence = piece.strip()
            if len(sentence) >= MIN_LEAK_CHARS:
                found.append(sentence)
    return tuple(found)


def _appears_in(serialized: str, needle: str) -> bool:
    """Match raw or JSON-escaped, since the receipt is checked in serialized form."""

    escaped = json.dumps(needle, ensure_ascii=False)[1:-1]
    return needle in serialized or escaped in serialized


def _assert_redacted(
    serialized: str,
    request: ResearchRequest,
    answer: Answer,
    retrieved: tuple[EvidenceItem, ...],
) -> None:
    if _appears_in(serialized, request.question):
        raise ValueError("receipt leaked the question text")
    # Every retrieved snippet, cited or not: a model's free-text abstain_reason could
    # otherwise carry an uncited note into the receipt.
    for item in retrieved:
        if any(_appears_in(serialized, sentence) for sentence in _sentences(item.snippet)):
            raise ValueError("receipt leaked evidence snippet text")
    if answer.outcome == "answered" and any(
        _appears_in(serialized, sentence) for sentence in _sentences(answer.text)
    ):
        raise ValueError("receipt leaked answer text")
    assert_no_sensitive(serialized, label="receipt")


def build_receipt(
    request: ResearchRequest,
    route: RouteDecision,
    answer: Answer,
    *,
    evidence: EvidenceBundle,
    corpus_digests: tuple[tuple[TrustProfile, str], ...] = (),
    tools_used: tuple[str, ...] = (),
    tool_errors: tuple[str, ...] = (),
    provider: str = "extractive",
) -> Receipt:
    cited_digests = _cited_digests(answer, evidence)
    citation_paths = tuple(citation.source_path for citation in answer.citations)
    trust_profiles = tuple(dict.fromkeys(citation.trust_profile for citation in answer.citations))
    implementation_version = grounded_agent.__version__
    payload = {
        "receipt_schema": RECEIPT_SCHEMA,
        "implementation_version": implementation_version,
        "request_id": request.request_id,
        "intent": route.intent,
        "scopes": list(route.scopes),
        "outcome": answer.outcome,
        "citation_paths": list(citation_paths),
        "cited_source_sha256": list(cited_digests),
        "corpus_digests": [list(pair) for pair in corpus_digests],
        "trust_profiles_used": list(trust_profiles),
        "abstain_reason": answer.abstain_reason,
        "tools_used": list(tools_used),
        "tool_errors": list(tool_errors),
        "provider": provider,
        "content_redacted": True,
    }
    receipt = Receipt(
        receipt_schema=RECEIPT_SCHEMA,
        implementation_version=implementation_version,
        request_id=request.request_id,
        intent=route.intent,
        scopes=route.scopes,
        outcome=answer.outcome,
        citation_paths=citation_paths,
        cited_source_sha256=cited_digests,
        corpus_digests=corpus_digests,
        trust_profiles_used=trust_profiles,
        abstain_reason=answer.abstain_reason,
        tools_used=tools_used,
        tool_errors=tool_errors,
        provider=provider,
        content_redacted=True,
        receipt_hash=_digest(payload),
    )
    retrieved = (*evidence.knowledge, *evidence.project)
    _assert_redacted(receipt.model_dump_json(), request, answer, retrieved)
    return receipt
