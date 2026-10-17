"""Support gate: a nomination or a valid citation path is not warrant (#11).

These cases pin what grounded_agent.support checks: a quantity question needs a
cited sentence that states the quantity, and a model claim may not add numbers or
names its cited notes lack. They do not show that the gate judges meaning. The
strict xfails at the end record known gaps of this deterministic gate.
"""

from __future__ import annotations

import pytest

from grounded_agent.draft import draft_or_abstain
from grounded_agent.graph import run_research_graph
from grounded_agent.models import EvidenceBundle, EvidenceItem, ResearchRequest
from grounded_agent.pipeline import run_research
from grounded_agent.provider import ModelDraft, answer_from_model_draft
from grounded_agent.retrieve import retrieve
from grounded_agent.router import route_intent
from grounded_agent.support import (
    QuantityRequest,
    numbers,
    quantities_for,
    requested_quantity,
    unsupported_specifics,
)

COUNT_QUESTION = "How many Harbor notes are in the durable knowledge index?"
TRUST_QUESTION = "What must Harbor do when combining retrieval results from different indexes?"
TRUST_PATH = "knowledge/trust-labels.md"
TWO_INDEXES = "Harbor stores durable notes and live project status in two separate indexes."


def _request(question: str, key: str = "support") -> ResearchRequest:
    return ResearchRequest(request_id=key, question=question)


def _note(snippet: str, path: str = "knowledge/count-note.md") -> EvidenceBundle:
    return EvidenceBundle(
        knowledge=(
            EvidenceItem(
                source_path=path,
                trust_profile="durable_knowledge",
                title="Harbor durable knowledge index",
                snippet=snippet,
                score=0.9,
                weak_fit=False,
                citation_class="citable",
            ),
        )
    )


def _trust_evidence() -> EvidenceBundle:
    request = _request(TRUST_QUESTION, "support-evidence")
    return retrieve(request, route_intent(request))


def _model(text: str, path: str = TRUST_PATH) -> ModelDraft:
    return ModelDraft(outcome="answered", text=text, citation_paths=(path,))


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (COUNT_QUESTION, QuantityRequest(noun="note")),
        ("How many separate indexes does Harbor use?", QuantityRequest(noun="index")),
        ("How many of the projects are paused?", QuantityRequest(noun="project")),
        ("What is the number of Harbor projects?", QuantityRequest(noun="project")),
        ("How much does Harbor pay its operator?", QuantityRequest(noun=None)),
        (TRUST_QUESTION, None),
        ("When a Harbor receipt omits snippet text, what remains unproven?", None),
        ("What is Harbor's CEO salary?", None),
    ],
)
def test_requested_quantity_reads_the_counted_noun(
    question: str, expected: QuantityRequest | None
) -> None:
    assert requested_quantity(question) == expected


def test_number_must_attach_to_the_counted_noun() -> None:
    assert quantities_for("The Harbor durable knowledge index contains six notes.", "note") == {"6"}
    assert quantities_for("It lists 12 notes.", "note") == {"12"}
    assert quantities_for("It lists twenty-five Harbor notes.", "note") == {"25"}
    assert quantities_for(TWO_INDEXES, "note") == set()
    assert quantities_for(TWO_INDEXES, "index") == {"2"}
    # "a separate index" is not a count of notes.
    assert (
        quantities_for(
            "Harbor durable knowledge notes live in a separate index from project status.",
            "note",
        )
        == set()
    )


def test_numbers_compare_digits_and_number_words() -> None:
    assert numbers("six notes and 6 notes") == {"6"}
    assert numbers("$10 million") == {"10", "1000000"}
    assert numbers("1,000 notes, 2.50 percent") == {"1000", "2.5"}
    assert numbers("version v2 of lantern-agent") == set()


def test_unsupported_specifics_lists_numbers_and_names() -> None:
    assert unsupported_specifics("Harbor stores notes in two separate indexes.", TWO_INDEXES) == ()
    assert unsupported_specifics(
        "Harbor stores notes in seven separate indexes.", TWO_INDEXES
    ) == ("number 7",)
    salary = unsupported_specifics("Harbor's CEO salary is $10 million.", TWO_INDEXES)
    assert {"number 10", "number 1000000", "name ceo"} <= set(salary)
    assert unsupported_specifics("Ask Alice before combining.", TWO_INDEXES) == ("name alice",)
    assert unsupported_specifics("Write to ops@example.invalid now.", TWO_INDEXES) == (
        "email ops@example.invalid",
    )
    assert unsupported_specifics("The lantern-agent keeps notes.", TWO_INDEXES) == (
        "name lantern agent",
    )


def test_extractive_count_answer_uses_the_sentence_that_states_the_count() -> None:
    evidence = _note("Harbor keeps one durable index. It currently holds six notes.")
    answer = draft_or_abstain(_request(COUNT_QUESTION), evidence)
    assert answer.outcome == "answered"
    assert "It currently holds six notes." in answer.text
    assert "one durable index" not in answer.text


def test_extractive_count_for_another_noun_abstains() -> None:
    answer = draft_or_abstain(_request(COUNT_QUESTION), _note(TWO_INDEXES))
    assert answer.outcome == "abstained"
    assert answer.abstain_reason == "unsupported_claim"
    assert answer.citations == ()


def test_open_questions_still_get_first_sentence_nominations() -> None:
    # Unchanged on purpose: for open questions the gate adds no check, and the
    # first sentence of each citable note is a nomination, not a verified answer.
    answer = draft_or_abstain(_request(TRUST_QUESTION), _trust_evidence())
    assert answer.outcome == "answered"
    assert TWO_INDEXES in answer.text


def test_count_question_abstains_for_the_same_reason_in_both_runtimes() -> None:
    request = _request(COUNT_QUESTION, "support-count")
    baseline = run_research(request)
    graphed = run_research_graph(request)
    assert baseline.answer.abstain_reason == "unsupported_claim"
    assert graphed.answer.abstain_reason == "unsupported_claim"
    assert baseline.receipt.abstain_reason == "unsupported_claim"
    assert baseline.receipt.receipt_hash == graphed.receipt.receipt_hash


def test_model_paraphrase_without_new_numbers_or_names_is_answered() -> None:
    answer = answer_from_model_draft(_model("Harbor keeps the groups labelled."), _trust_evidence())
    assert answer.outcome == "answered"
    assert tuple(c.source_path for c in answer.citations) == (TRUST_PATH,)


def test_model_name_absent_from_cited_note_abstains() -> None:
    answer = answer_from_model_draft(
        _model("Alice Smith maintains the Harbor indexes."), _trust_evidence()
    )
    assert answer.outcome == "abstained"
    assert answer.abstain_reason == "unsupported_claim"
    assert answer.citations == ()


def test_model_count_answer_needs_a_count_the_cited_note_states() -> None:
    request = _request(COUNT_QUESTION)
    supported = answer_from_model_draft(
        _model("The durable knowledge index contains six notes.", "knowledge/count-note.md"),
        _note("The Harbor durable knowledge index contains six notes."),
        request=request,
    )
    assert supported.outcome == "answered"

    for text in (
        TWO_INDEXES,  # true of the note, but it does not state a count of notes
        "Harbor keeps two notes.",  # the note's "two" counts indexes, not notes
    ):
        answer = answer_from_model_draft(_model(text), _trust_evidence(), request=request)
        assert answer.outcome == "abstained", text
        assert answer.abstain_reason == "unsupported_claim", text
        assert answer.citations == (), text


def test_refusal_text_does_not_repeat_the_claim() -> None:
    answer = answer_from_model_draft(
        _model("Harbor's CEO salary is $10 million."), _trust_evidence()
    )
    assert answer.outcome == "abstained"
    assert "salary" not in answer.text
    assert "10" not in answer.text


# Known gaps. These are limits of a deterministic gate, recorded as strict
# xfails so a future check that closes one has to update this record.


@pytest.mark.xfail(strict=True, reason="support gate gap: a dropped negation is not detected")
def test_known_gap_dropped_negation() -> None:
    answer = answer_from_model_draft(
        _model("A labelled group is proof that a nominated note is true."), _trust_evidence()
    )
    assert answer.outcome == "abstained"


@pytest.mark.xfail(strict=True, reason="support gate gap: ordinary-word fabrication passes")
def test_known_gap_fabrication_without_numbers_or_names() -> None:
    answer = answer_from_model_draft(
        _model("Harbor pays its chief executive a salary."), _trust_evidence()
    )
    assert answer.outcome == "abstained"


@pytest.mark.xfail(strict=True, reason="support gate gap: the count is not tied to its index")
def test_known_gap_count_from_the_wrong_index() -> None:
    answer = draft_or_abstain(
        _request(COUNT_QUESTION), _note("The project status index contains two notes.")
    )
    assert answer.outcome == "abstained"


@pytest.mark.xfail(strict=True, reason="support gate gap: open-question extracts are unchecked")
def test_known_gap_open_question_first_sentence() -> None:
    # The lantern-agent note states its next action in its second sentence. The
    # extractive drafter still returns first sentences for open questions.
    result = run_research(_request("What is the next action for lantern-agent?", "gap-open"))
    assert (
        result.answer.outcome == "abstained"
        or "implement the deterministic intent router" in result.answer.text
    )
