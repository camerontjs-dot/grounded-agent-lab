"""Framework-free grounded research agent."""

__version__ = "0.2.0"

from grounded_agent.graph import run_research_graph
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
