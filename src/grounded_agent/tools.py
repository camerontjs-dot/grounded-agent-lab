"""Read-only dual-index tool boundary.

The agent may call only `query_knowledge` and `query_projects`. Unknown names,
write-shaped names, extra arguments, and over-budget calls fail closed.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from grounded_agent.models import (
    EvidenceBundle,
    EvidenceItem,
    FrozenModel,
    NonBlankStr,
    ResearchRequest,
    RouteDecision,
    TrustProfile,
)
from grounded_agent.retrieve import search_corpus

ALLOWED_TOOLS = frozenset({"query_knowledge", "query_projects"})
DENIED_WRITE_TOOLS = frozenset({"write_index", "delete_note", "grant_write", "merge_indexes"})
SCOPE_TOOL: dict[TrustProfile, str] = {
    "durable_knowledge": "query_knowledge",
    "project_status": "query_projects",
}
TOOL_TRUST: dict[str, TrustProfile] = {
    "query_knowledge": "durable_knowledge",
    "query_projects": "project_status",
}
DEFAULT_TIMEOUT_S = 2.0


class ToolDenied(Exception):
    """Tool is missing from the allowlist or is a write capability."""


class ToolTimeout(Exception):
    """Tool exceeded its time budget."""


class ToolSchemaError(Exception):
    """Arguments failed the tool schema."""


class ToolUnavailable(Exception):
    """Optional live backend is not configured."""


class QueryArgs(FrozenModel):
    question: NonBlankStr


class ToolObservation(FrozenModel):
    tool: str
    trust_profile: TrustProfile
    items: tuple[EvidenceItem, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class RetrievalTrace:
    bundle: EvidenceBundle
    tools_used: tuple[str, ...]
    tool_errors: tuple[str, ...] = ()


class FixtureToolClient:
    """In-process Harbor corpus backend behind the same allowlist as MCP."""

    def __init__(
        self, *, min_delay_s: float = 0.0, timeout_s: float = DEFAULT_TIMEOUT_S
    ) -> None:
        self.min_delay_s = min_delay_s
        self.timeout_s = timeout_s

    def allowed_tools(self) -> frozenset[str]:
        return ALLOWED_TOOLS

    def invoke(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> ToolObservation:
        budget = self.timeout_s if timeout_s is None else timeout_s
        if name in DENIED_WRITE_TOOLS or name not in ALLOWED_TOOLS:
            raise ToolDenied(f"{name} is not a read-only allowlisted tool")
        try:
            args = QueryArgs.model_validate(arguments)
        except ValidationError as exc:
            raise ToolSchemaError(str(exc)) from exc

        def _run() -> ToolObservation:
            if self.min_delay_s:
                time.sleep(self.min_delay_s)
            items = search_corpus(args.question, TOOL_TRUST[name])
            return ToolObservation(tool=name, trust_profile=TOOL_TRUST[name], items=items)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run)
            try:
                return future.result(timeout=budget)
            except FuturesTimeout as exc:
                raise ToolTimeout(name) from exc


class LiveMindgraphAdapter:
    """Fail-closed live adapter. Tests never require a running daemon."""

    def __init__(self, url: str | None = None) -> None:
        self.url = url if url is not None else os.environ.get("GROUNDED_AGENT_MINDGRAPH_URL")

    def allowed_tools(self) -> frozenset[str]:
        return ALLOWED_TOOLS

    def invoke(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> ToolObservation:
        if name in DENIED_WRITE_TOOLS or name not in ALLOWED_TOOLS:
            raise ToolDenied(f"{name} is not a read-only allowlisted tool")
        QueryArgs.model_validate(arguments)
        if not self.url:
            raise ToolUnavailable("GROUNDED_AGENT_MINDGRAPH_URL is not set")
        raise ToolUnavailable("live MindGraph HTTP transport is not enabled in this phase")


def retrieve_traced(
    request: ResearchRequest,
    route: RouteDecision,
    client: FixtureToolClient | None = None,
) -> RetrievalTrace:
    client = client or FixtureToolClient()
    knowledge: list[EvidenceItem] = []
    project: list[EvidenceItem] = []
    tools_used: list[str] = []
    errors: list[str] = []
    buckets: dict[TrustProfile, list[EvidenceItem]] = {
        "durable_knowledge": knowledge,
        "project_status": project,
    }
    for scope in route.scopes:
        tool = SCOPE_TOOL[scope]
        tools_used.append(tool)
        try:
            observation = client.invoke(tool, {"question": request.question})
            buckets[scope].extend(observation.items)
        except (ToolDenied, ToolTimeout, ToolSchemaError, ToolUnavailable) as exc:
            errors.append(f"{tool}:{type(exc).__name__}")
    return RetrievalTrace(
        bundle=EvidenceBundle(knowledge=tuple(knowledge), project=tuple(project)),
        tools_used=tuple(tools_used),
        tool_errors=tuple(errors),
    )
