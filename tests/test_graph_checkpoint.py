"""Checkpointed review: pause, resume, reject, and one receipt per thread."""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver

from grounded_agent.graph import resume_reviewed_run, start_reviewed_run
from grounded_agent.models import ResearchRequest

QUESTION = "What is the next action for lantern-agent?"


def test_review_interrupt_has_no_receipt_until_resume() -> None:
    request = ResearchRequest(request_id="review-1", question=QUESTION)
    graph, config = start_reviewed_run(request, checkpointer=InMemorySaver())
    snapshot = graph.get_state(config)
    assert snapshot.values.get("answer") is not None
    assert snapshot.values.get("receipt") is None
    assert snapshot.values["answer"]["outcome"] == "answered"

    result = resume_reviewed_run(graph, config, "approve")
    assert result.answer.outcome == "answered"
    assert result.receipt.receipt_hash
    assert "projects/lantern-agent.md" in result.receipt.citation_paths
    done = graph.get_state(config)
    assert not done.next
    assert done.values["receipt"]["receipt_hash"] == result.receipt.receipt_hash


def test_reject_abstains_and_does_not_cite() -> None:
    request = ResearchRequest(request_id="review-2", question=QUESTION)
    graph, config = start_reviewed_run(request, checkpointer=InMemorySaver())
    result = resume_reviewed_run(graph, config, "reject")
    assert result.answer.outcome == "abstained"
    assert result.answer.abstain_reason == "review_rejected"
    assert result.answer.citations == ()
    assert result.receipt.outcome == "abstained"


def test_resume_does_not_duplicate_receipt() -> None:
    request = ResearchRequest(request_id="review-3", question=QUESTION)
    graph, config = start_reviewed_run(request, checkpointer=InMemorySaver())
    first = resume_reviewed_run(graph, config, "approve")
    snapshot = graph.get_state(config)
    second = graph.invoke(None, config)
    assert second["receipt"]["receipt_hash"] == first.receipt.receipt_hash
    assert snapshot.values["receipt"]["receipt_hash"] == first.receipt.receipt_hash
    histories = list(graph.get_state_history(config))
    hashes = [
        item.values["receipt"]["receipt_hash"]
        for item in histories
        if item.values.get("receipt")
    ]
    assert hashes
    assert set(hashes) == {first.receipt.receipt_hash}
