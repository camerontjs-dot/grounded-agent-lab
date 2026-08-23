"""Command-line entry for fixture-mode research runs."""

from __future__ import annotations

import argparse
import json
import sys
import uuid

from grounded_agent.graph import run_research_graph
from grounded_agent.models import ResearchRequest
from grounded_agent.pipeline import run_research


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grounded-agent",
        description="Grounded research agent (fixture corpus, no API key).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="Answer or abstain from the fixture corpus")
    ask.add_argument("question", help="Question to route, retrieve, and answer")
    ask.add_argument("--json", action="store_true", help="Print machine-readable output")
    ask.add_argument(
        "--runtime",
        choices=("baseline", "graph"),
        default="baseline",
        help="baseline is the framework-free loop; graph is the LangGraph wrap",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "ask":
        return 1
    request = ResearchRequest(request_id=str(uuid.uuid4()), question=args.question)
    result = run_research_graph(request) if args.runtime == "graph" else run_research(request)
    if args.json:
        payload = {
            "outcome": result.answer.outcome,
            "intent": result.route.intent,
            "text": result.answer.text,
            "citations": [citation.model_dump() for citation in result.answer.citations],
            "receipt": result.receipt.model_dump(),
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return 0
    sys.stdout.write(f"outcome: {result.answer.outcome}\n")
    sys.stdout.write(f"intent: {result.route.intent}\n")
    if result.answer.abstain_reason:
        sys.stdout.write(f"abstain_reason: {result.answer.abstain_reason}\n")
    sys.stdout.write(f"\n{result.answer.text}\n")
    if result.answer.citations:
        sys.stdout.write("\ncitations:\n")
        for citation in result.answer.citations:
            sys.stdout.write(f"- {citation.trust_profile}: {citation.source_path}\n")
    if result.receipt.tools_used:
        sys.stdout.write(f"tools_used: {', '.join(result.receipt.tools_used)}\n")
    sys.stdout.write(f"receipt_hash: {result.receipt.receipt_hash}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
