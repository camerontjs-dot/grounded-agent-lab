"""Stage-level traces. Paths and hashes only; no snippets or secrets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from grounded_agent.models import ResearchResult
from grounded_agent.redact import assert_no_sensitive, redact_text


class MemoryTracer:
    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []

    def emit(self, stage: str, **fields: object) -> None:
        event = {"stage": stage}
        for key, value in fields.items():
            if isinstance(value, str):
                event[key] = redact_text(value)
            else:
                event[key] = value
        blob = json.dumps(event, sort_keys=True)
        assert_no_sensitive(blob, label=f"trace.{stage}")
        self._events.append(event)

    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._events)

    def dump_jsonl(self, path: Path) -> None:
        path.write_text(
            "".join(json.dumps(event, sort_keys=True) + "\n" for event in self._events),
            encoding="utf-8",
        )


class NullTracer:
    def emit(self, stage: str, **fields: object) -> None:
        return None


def trace_result(tracer: MemoryTracer, result: ResearchResult) -> None:
    """Record redacted stages from a finished run (used by the graph CLI path)."""

    tracer.emit("route", intent=result.route.intent, scopes=list(result.route.scopes))
    tracer.emit(
        "retrieve",
        tools_used=list(result.receipt.tools_used),
        knowledge_paths=[item.source_path for item in result.evidence.knowledge],
        project_paths=[item.source_path for item in result.evidence.project],
        tool_errors=list(result.receipt.tool_errors),
    )
    tracer.emit(
        "draft",
        outcome=result.answer.outcome,
        abstain_reason=result.answer.abstain_reason,
        citation_paths=[citation.source_path for citation in result.answer.citations],
        provider=result.receipt.provider,
    )
    tracer.emit(
        "receipt",
        receipt_hash=result.receipt.receipt_hash,
        tools_used=list(result.receipt.tools_used),
        content_redacted=result.receipt.content_redacted,
    )
