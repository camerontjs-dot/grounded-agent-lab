"""Receipts bind the bytes they cite, the corpus they searched and the code that ran (#14).

Every mutation happens in a private copy of the fixture corpora. The real fixture
files are only ever read.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

import grounded_agent
import grounded_agent.retrieve as retrieve_module
from grounded_agent.cli import main
from grounded_agent.graph import resume_reviewed_run, run_research_graph, start_reviewed_run
from grounded_agent.models import (
    Answer,
    Citation,
    EvidenceBundle,
    EvidenceItem,
    Receipt,
    ResearchRequest,
    ResearchResult,
)
from grounded_agent.paths import QUESTIONS_PATH
from grounded_agent.pipeline import run_research
from grounded_agent.receipt import build_receipt
from grounded_agent.redact import SENSITIVE_MARKERS
from grounded_agent.retrieve import (
    corpus_digest,
    load_corpus,
    parse_note,
    search_corpus,
    search_corpus_with_digest,
)
from grounded_agent.router import route_intent
from grounded_agent.tools import FixtureToolClient, retrieve_traced

TRUST_QUESTION = "What must Harbor do when combining retrieval results from different indexes?"
LANTERN_QUESTION = "What is the next action for lantern-agent?"
FEDERATED_QUESTION = "How does the lantern-agent next action relate to Harbor trust labels?"
SALARY_QUESTION = "What is Harbor's CEO salary?"
CONTACT_QUESTION = "What is the Harbor operator contact alias?"
TRUST_PATH = "knowledge/trust-labels.md"
UNCITED_PATH = "knowledge/neighbor-only.md"
REVISION = "\nAn unrelated fixture-only revision.\n"
DIGEST = "c" * 64
HARMLESS = "A separate and harmless answer."
BARE_RECEIPT = {
    "implementation_version": "0.0.0-test",
    "request_id": "r1",
    "intent": "knowledge",
    "scopes": ("durable_knowledge",),
    "outcome": "abstained",
    "abstain_reason": "insufficient_evidence",
    "receipt_hash": "h",
}


def _request(key: str, question: str = TRUST_QUESTION) -> ResearchRequest:
    return ResearchRequest(request_id=key, question=question)


def _questions() -> list[tuple[str, str]]:
    cases = json.loads(Path(QUESTIONS_PATH).read_text(encoding="utf-8"))
    table = [(str(case["id"]), str(case["question"])) for case in cases]
    return [*table, ("q-contact", CONTACT_QUESTION)]


def _sentences(text: str) -> list[str]:
    pieces = (piece.strip() for piece in re.split(r"(?<=[.!?])\s+", text))
    return [piece for piece in pieces if len(piece) >= 16]


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_of(directory: Path) -> str:
    """The documented manifest hash, recomputed from the files without the product code."""

    pairs = sorted(
        [f"{directory.name}/{path.name}", _sha256_of(path)] for path in directory.glob("*.md")
    )
    canonical = json.dumps(pairs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _append(path: Path, text: str) -> None:
    path.write_text(path.read_text(encoding="utf-8") + text, encoding="utf-8")


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private copy of both corpora. Copy first, then point the module at the copies."""

    knowledge = tmp_path / retrieve_module.KNOWLEDGE_CORPUS.name
    projects = tmp_path / retrieve_module.PROJECT_CORPUS.name
    shutil.copytree(retrieve_module.KNOWLEDGE_CORPUS, knowledge)
    shutil.copytree(retrieve_module.PROJECT_CORPUS, projects)
    monkeypatch.setattr(retrieve_module, "KNOWLEDGE_CORPUS", knowledge)
    monkeypatch.setattr(retrieve_module, "PROJECT_CORPUS", projects)
    return tmp_path


def _custom_evidence(snippet: str, *, digest: str | None = DIGEST) -> EvidenceBundle:
    return EvidenceBundle(
        knowledge=(
            EvidenceItem(
                source_path="knowledge/x.md",
                trust_profile="durable_knowledge",
                title="X",
                snippet=snippet,
                score=0.9,
                weak_fit=False,
                citation_class="citable",
                source_sha256=digest,
            ),
        )
    )


def _custom_answer(text: str) -> Answer:
    citation = Citation(source_path="knowledge/x.md", trust_profile="durable_knowledge", title="X")
    return Answer(outcome="answered", text=text, citations=(citation,))


def _custom_receipt(request_id: str, snippet: str) -> Receipt:
    request = _request(request_id)
    return build_receipt(
        request,
        route_intent(request),
        _custom_answer(HARMLESS),
        evidence=_custom_evidence(snippet),
    )


def _rebuild(
    request: ResearchRequest, result: ResearchResult, *, answer: Answer | None = None
) -> Receipt:
    """Build a receipt again from a finished run's own evidence."""

    return build_receipt(
        request,
        result.route,
        answer or result.answer,
        evidence=result.evidence,
        corpus_digests=result.receipt.corpus_digests,
        tools_used=result.receipt.tools_used,
        tool_errors=result.receipt.tool_errors,
        provider=result.receipt.provider,
    )


# What the receipt binds


def test_cited_digest_is_the_sha256_of_the_cited_file_bytes(corpus: Path) -> None:
    result = run_research(_request("bind-bytes"))
    receipt = result.receipt
    assert TRUST_PATH in receipt.citation_paths
    assert len(receipt.cited_source_sha256) == len(receipt.citation_paths)
    for path, digest in zip(receipt.citation_paths, receipt.cited_source_sha256, strict=True):
        assert digest == _sha256_of(corpus / path), path
    items = {item.source_path: item for item in result.evidence.knowledge}
    assert [items[path].source_sha256 for path in receipt.citation_paths] == list(
        receipt.cited_source_sha256
    )


def test_corpus_digest_is_the_documented_manifest_hash(corpus: Path) -> None:
    result = run_research(_request("bind-manifest", FEDERATED_QUESTION))
    assert result.route.intent == "federated"
    assert result.receipt.corpus_digests == (
        ("durable_knowledge", _manifest_of(retrieve_module.KNOWLEDGE_CORPUS)),
        ("project_status", _manifest_of(retrieve_module.PROJECT_CORPUS)),
    )


def test_changing_an_uncited_note_moves_the_corpus_digest_not_the_cited_digest(
    corpus: Path,
) -> None:
    request = _request("bind-uncited")
    before = run_research(request)
    assert UNCITED_PATH not in before.receipt.citation_paths

    _append(corpus / UNCITED_PATH, REVISION)
    after = run_research(request)

    assert after.answer == before.answer
    assert after.receipt.citation_paths == before.receipt.citation_paths
    assert after.receipt.cited_source_sha256 == before.receipt.cited_source_sha256
    assert after.receipt.corpus_digests != before.receipt.corpus_digests
    assert after.receipt.receipt_hash != before.receipt.receipt_hash


def test_changing_the_cited_note_moves_its_cited_digest(corpus: Path) -> None:
    request = _request("bind-cited")
    before = run_research(request)

    _append(corpus / TRUST_PATH, REVISION)
    after = run_research(request)

    assert after.answer.citations == before.answer.citations
    assert after.receipt.citation_paths == before.receipt.citation_paths
    was = dict(zip(before.receipt.citation_paths, before.receipt.cited_source_sha256, strict=True))
    now = dict(zip(after.receipt.citation_paths, after.receipt.cited_source_sha256, strict=True))
    assert now[TRUST_PATH] != was[TRUST_PATH]
    assert now[TRUST_PATH] == _sha256_of(corpus / TRUST_PATH)
    # Any other cited note keeps its digest: only the edited file moves.
    assert all(now[path] == was[path] for path in was if path != TRUST_PATH)
    assert after.receipt.receipt_hash != before.receipt.receipt_hash


def test_implementation_version_is_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request("bind-version")
    declared = grounded_agent.__version__
    before = run_research(request)
    monkeypatch.setattr(grounded_agent, "__version__", "0.0.0-test")
    after = run_research(request)

    assert before.receipt.implementation_version == declared
    assert after.receipt.implementation_version == "0.0.0-test"
    assert after.receipt.receipt_hash != before.receipt.receipt_hash
    held = {"implementation_version", "receipt_hash"}
    assert after.receipt.model_dump(exclude=held) == before.receipt.model_dump(exclude=held)


def test_receipt_hash_covers_every_field() -> None:
    for case_id, question in _questions():
        receipt = run_research(_request(case_id, question)).receipt
        hashed = receipt.model_dump(mode="json", exclude={"receipt_hash"})
        canonical = json.dumps(hashed, sort_keys=True, separators=(",", ":"))
        assert receipt.receipt_hash == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert hashed["receipt_schema"] == "grounded-agent-receipt-v2"


# What the receipt still must not contain


def test_receipts_stay_redacted_across_the_question_table() -> None:
    for case_id, question in _questions():
        result = run_research(_request(case_id, question))
        serialized = result.receipt.model_dump_json()
        assert question not in serialized, case_id
        assert result.answer.text not in serialized, case_id
        for item in (*result.evidence.knowledge, *result.evidence.project):
            assert item.snippet not in serialized, case_id
            for sentence in _sentences(item.snippet):
                assert sentence not in serialized, (case_id, sentence)
        for marker in SENSITIVE_MARKERS:
            assert marker not in serialized, case_id


def test_contact_receipt_binds_a_note_that_holds_the_planted_secret() -> None:
    result = run_research(_request("bind-contact", CONTACT_QUESTION))
    assert result.answer.outcome == "answered"
    assert "knowledge/operator-contact.md" in result.receipt.citation_paths
    raw = " ".join(item.snippet for item in result.evidence.knowledge)
    assert all(marker in raw for marker in SENSITIVE_MARKERS), "the fixture should plant both"
    serialized = result.receipt.model_dump_json()
    assert not any(marker in serialized for marker in SENSITIVE_MARKERS)


# The two runtimes agree


def test_baseline_and_graph_receipts_match_field_for_field() -> None:
    for case_id, question in _questions():
        request = _request(case_id, question)
        baseline = run_research(request).receipt
        graphed = run_research_graph(request).receipt
        for name in Receipt.model_fields:
            assert getattr(baseline, name) == getattr(graphed, name), (case_id, name)


def test_graph_state_stays_json_shaped_and_a_rejected_run_keeps_the_corpus_binding() -> None:
    request = _request("bind-reject", LANTERN_QUESTION)
    baseline = run_research(request)
    assert baseline.receipt.corpus_digests

    graph, config = start_reviewed_run(request, checkpointer=InMemorySaver())
    paused = graph.get_state(config).values
    json.dumps(paused)
    assert paused["corpus_digests"] == [list(pair) for pair in baseline.receipt.corpus_digests]

    rejected = resume_reviewed_run(graph, config, "reject")
    assert rejected.answer.abstain_reason == "review_rejected"
    assert rejected.receipt.citation_paths == ()
    assert rejected.receipt.cited_source_sha256 == ()
    assert rejected.receipt.corpus_digests == baseline.receipt.corpus_digests


# Fail closed


def test_a_cited_item_without_a_digest_fails_closed() -> None:
    request = _request("bind-missing")
    result = run_research(request)
    assert result.receipt.citation_paths

    def undigested(items: tuple[EvidenceItem, ...]) -> tuple[EvidenceItem, ...]:
        return tuple(item.model_copy(update={"source_sha256": None}) for item in items)

    stripped = EvidenceBundle(
        knowledge=undigested(result.evidence.knowledge),
        project=undigested(result.evidence.project),
    )
    with pytest.raises(ValueError, match="source digest"):
        build_receipt(request, result.route, result.answer, evidence=stripped)
    with pytest.raises(ValueError, match="source digest"):
        build_receipt(request, result.route, result.answer, evidence=EvidenceBundle())


def test_a_citation_is_bound_only_to_an_item_of_the_same_trust_profile() -> None:
    request = _request("bind-profile")
    mislabelled = Citation(source_path="knowledge/x.md", trust_profile="project_status", title="X")
    answer = Answer(outcome="answered", text=HARMLESS, citations=(mislabelled,))
    with pytest.raises(ValueError, match="source digest"):
        build_receipt(request, route_intent(request), answer, evidence=_custom_evidence("A note."))


def test_an_abstention_needs_no_digest() -> None:
    request = _request("bind-abstain", SALARY_QUESTION)
    result = run_research(request)
    assert result.answer.outcome == "abstained"
    receipt = build_receipt(request, result.route, result.answer, evidence=EvidenceBundle())
    assert receipt.citation_paths == ()
    assert receipt.cited_source_sha256 == ()
    assert result.receipt.corpus_digests, "the searched corpus is still bound"


# The leak guard can fire


def test_leak_guard_fires_on_a_cited_snippet_sentence() -> None:
    request = _request("guard-ok")
    result = run_research(request)
    cited = next(item for item in result.evidence.knowledge if item.source_path == TRUST_PATH)
    sentence = _sentences(cited.snippet)[0]

    assert _rebuild(request, result).receipt_hash == result.receipt.receipt_hash
    with pytest.raises(ValueError, match="snippet"):
        _rebuild(_request(sentence), result)


def test_leak_guard_fires_on_the_question() -> None:
    request = _request("guard-ok")
    result = run_research(request)
    with pytest.raises(ValueError, match="question"):
        _rebuild(_request(TRUST_QUESTION), result)


@pytest.mark.parametrize(
    "line",
    ["Harbor keeps both indexes labelled and apart.", "Harbor keeps both indexes labelled apart"],
)
def test_leak_guard_sees_a_line_of_a_multiline_answer_the_old_check_could_not(line: str) -> None:
    request = _request("guard-ok")
    result = run_research(request)
    answer = Answer(
        outcome="answered",
        text=f"First, a short lead.\n- {line} ({TRUST_PATH})",
        citations=result.answer.citations,
    )
    assert all(line not in item.snippet for item in result.evidence.knowledge)

    leaked = result.receipt.model_copy(update={"request_id": line}).model_dump_json()
    assert line in leaked
    assert answer.text not in leaked, "the old whole-text comparison was blind to this"

    assert _rebuild(request, result, answer=answer).receipt_hash
    with pytest.raises(ValueError, match="answer text"):
        _rebuild(_request(line), result, answer=answer)


def test_leak_guard_matches_text_that_json_escapes() -> None:
    quoted = 'The "labelled" group is not proof of anything.'
    escaped_form = Receipt(**{**BARE_RECEIPT, "request_id": quoted}).model_dump_json()
    assert quoted not in escaped_form, "premise: JSON escapes the quotes"
    with pytest.raises(ValueError, match="snippet"):
        _custom_receipt(quoted, snippet=quoted)


def test_the_answer_check_leaves_abstentions_alone() -> None:
    # A model-built abstention carries its reason in the receipt by design, and a small
    # model may repeat its sentence in both fields. That must not crash the run.
    request = _request("guard-abstain")
    sentence = "The notes do not say anything on this."
    answer = Answer(outcome="abstained", text=sentence, abstain_reason=sentence)
    receipt = build_receipt(request, route_intent(request), answer, evidence=EvidenceBundle())
    assert receipt.abstain_reason == sentence


def test_leak_guard_covers_retrieved_snippets_that_were_not_cited() -> None:
    # An abstaining model writes abstain_reason as free text; nothing is cited.
    request = _request("guard-uncited")
    sentence = "This retrieved sentence was never cited."
    answer = Answer(outcome="abstained", text="No answer.", abstain_reason=sentence)
    with pytest.raises(ValueError, match="snippet"):
        build_receipt(request, route_intent(request), answer, evidence=_custom_evidence(sentence))


def test_leak_guard_threshold_is_sixteen_characters() -> None:
    short = "x" * 14 + "."
    long = "x" * 15 + "."
    assert (len(short), len(long)) == (15, 16)
    assert _custom_receipt(short, snippet=short).request_id == short
    with pytest.raises(ValueError, match="snippet"):
        _custom_receipt(long, snippet=long)


# Validation


@pytest.mark.parametrize(
    "bad", ["", "abc", "A" * 64, "g" * 64, "a" * 63, "a" * 65, "a" * 64 + "\n"]
)
def test_evidence_item_rejects_a_malformed_source_digest(bad: str) -> None:
    with pytest.raises(ValidationError):
        _custom_evidence("A snippet.", digest=bad)


def test_receipt_rejects_misaligned_or_malformed_bindings() -> None:
    assert Receipt(**BARE_RECEIPT).cited_source_sha256 == ()
    with pytest.raises(ValidationError, match="align"):
        Receipt(**BARE_RECEIPT, citation_paths=("knowledge/x.md",))
    with pytest.raises(ValidationError):
        Receipt(**BARE_RECEIPT, corpus_digests=(("durable_knowledge", "not-hex"),))
    with pytest.raises(ValidationError):
        Receipt(**BARE_RECEIPT, corpus_digests=(("both", DIGEST),))
    unversioned = {k: v for k, v in BARE_RECEIPT.items() if k != "implementation_version"}
    with pytest.raises(ValidationError, match="implementation_version"):
        Receipt(**unversioned)


# Plumbing below the receipt


def test_tool_observation_and_trace_carry_the_corpus_digest(corpus: Path) -> None:
    expected = corpus_digest(load_corpus(retrieve_module.KNOWLEDGE_CORPUS, "durable_knowledge"))

    observation = FixtureToolClient().invoke("query_knowledge", {"question": TRUST_QUESTION})
    assert observation.corpus_sha256 == expected

    items, digest = search_corpus_with_digest(TRUST_QUESTION, "durable_knowledge")
    assert digest == expected
    assert items == search_corpus(TRUST_QUESTION, "durable_knowledge")

    request = _request("bind-trace", FEDERATED_QUESTION)
    trace = retrieve_traced(request, route_intent(request))
    assert [scope for scope, _ in trace.corpus_digests] == ["durable_knowledge", "project_status"]
    assert trace.corpus_digests[0][1] == expected


def test_a_failed_tool_call_contributes_no_corpus_digest() -> None:
    request = _request("bind-timeout", TRUST_QUESTION)
    client = FixtureToolClient(min_delay_s=0.3, timeout_s=0.05)
    trace = retrieve_traced(request, route_intent(request), client=client)
    assert trace.tool_errors == ("query_knowledge:ToolTimeout",)
    assert trace.corpus_digests == ()


def test_an_observation_without_a_corpus_digest_contributes_none() -> None:
    class DigestlessClient(FixtureToolClient):
        def invoke(self, name: str, arguments: dict, *, timeout_s: float | None = None):
            observation = super().invoke(name, arguments, timeout_s=timeout_s)
            return observation.model_copy(update={"corpus_sha256": None})

    request = _request("bind-digestless", FEDERATED_QUESTION)
    trace = retrieve_traced(request, route_intent(request), client=DigestlessClient())
    assert trace.tool_errors == ()
    assert trace.bundle.citable()
    assert trace.corpus_digests == ()


def test_parse_note_hashes_the_bytes_it_decoded(tmp_path: Path) -> None:
    raw = (
        b"---\r\nid: kn-crlf\r\ntrust_profile: durable_knowledge\r\ntitle: Crlf note\r\n---\r\n"
        b"\r\nBody line one.\r\nBody line two.\r\n"
    )
    path = tmp_path / "knowledge" / "crlf.md"
    path.parent.mkdir()
    path.write_bytes(raw)

    note = parse_note(path, "durable_knowledge")

    assert note.source_sha256 == hashlib.sha256(raw).hexdigest()
    assert note.title == "Crlf note"
    assert note.body.split() == ["Body", "line", "one.", "Body", "line", "two."]


def test_corpus_digest_binds_paths_and_bytes_not_load_order(corpus: Path) -> None:
    directory = retrieve_module.KNOWLEDGE_CORPUS
    notes = load_corpus(directory, "durable_knowledge")
    original = corpus_digest(notes)
    assert corpus_digest(tuple(reversed(notes))) == original

    _append(corpus / UNCITED_PATH, REVISION)
    edited = corpus_digest(load_corpus(directory, "durable_knowledge"))

    (corpus / UNCITED_PATH).rename(directory / "renamed.md")
    renamed = corpus_digest(load_corpus(directory, "durable_knowledge"))

    assert len({original, edited, renamed}) == 3


def test_cli_json_receipt_carries_the_bindings(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ask", "--json", TRUST_QUESTION]) == 0
    receipt = json.loads(capsys.readouterr().out)["receipt"]
    assert receipt["receipt_schema"] == "grounded-agent-receipt-v2"
    assert receipt["implementation_version"] == grounded_agent.__version__
    assert len(receipt["cited_source_sha256"]) == len(receipt["citation_paths"]) > 0
    assert [scope for scope, _ in receipt["corpus_digests"]] == ["durable_knowledge"]
