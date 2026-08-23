"""Reviewer demo is a live checklist, not a hand-written scoreboard."""

from __future__ import annotations

from grounded_agent.cli import main
from grounded_agent.demo import _check, format_demo, run_demo
from grounded_agent.paths import WORKBENCH_ROOT
from grounded_agent.redact import contains_sensitive


def test_demo_all_expected_outcomes_hold() -> None:
    payload = run_demo()
    assert payload["n"] == 7
    assert payload["ok"] is True, payload
    assert payload["passed"] == 7
    ids = [row["id"] for row in payload["rows"]]
    assert ids == [
        "answer-knowledge",
        "answer-graph-matches",
        "salary-abstain",
        "injection-refuse",
        "contact-redaction",
        "ollama-closed-port",
        "hosted-missing-credentials",
    ]


def test_demo_mismatch_is_visible() -> None:
    row = _check(
        {"outcome": "answered", "secret_leaked": False},
        outcome="abstained",
        secret_leaked=False,
    )
    assert row["ok"] is False
    assert any("outcome" in item for item in row["failures"])


def test_demo_output_has_no_secrets_or_machine_paths() -> None:
    payload = run_demo()
    text = format_demo(payload) + str(payload)
    assert contains_sensitive(text) is False
    assert "/Users/" not in text
    assert "/home/" not in text
    assert "PASS" in format_demo(payload)


def test_cli_demo_exits_zero(capsys) -> None:
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "7/7 passed" in out
    assert "Demo green" in out


def test_packaging_docs_exist() -> None:
    docs = WORKBENCH_ROOT / "docs"
    for name in ("architecture.md", "case-study.md", "limitations.md", "workflow-portability.md"):
        path = docs / name
        assert path.is_file(), path
        body = path.read_text(encoding="utf-8")
        assert "—" not in body
        assert "/Users/" not in body
