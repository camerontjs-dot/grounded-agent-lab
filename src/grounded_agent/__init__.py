"""Grounded research agent for the Harbor fixture corpus."""

from __future__ import annotations

from typing import Any

__version__ = "0.6.0"

from grounded_agent.models import (
    Answer,
    Citation,
    EvidenceBundle,
    EvidenceItem,
    Receipt,
    ResearchRequest,
    ResearchResult,
    RouteDecision,
)
from grounded_agent.pipeline import run_research

__all__ = [
    "__version__",
    "Answer",
    "Citation",
    "EvidenceBundle",
    "EvidenceItem",
    "Receipt",
    "ResearchRequest",
    "ResearchResult",
    "RouteDecision",
    "run_research",
    "run_research_graph",
]


def __getattr__(name: str) -> Any:
    if name == "run_research_graph":
        from grounded_agent.graph import run_research_graph

        return run_research_graph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
