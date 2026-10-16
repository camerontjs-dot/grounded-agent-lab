"""Run the open-question candidates on one split and write a receipt. See PROTOCOL.md.

    python experiments/open-question-support/run_eval.py --split dev
    python experiments/open-question-support/run_eval.py --split test --final

The learned candidates need torch and transformers plus the two pinned models in
the local Hugging Face cache. Without them those candidates are reported as
blocked. A test run needs --final, a clean git tree, an intact data manifest and
no earlier test receipt, so it happens once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from math import comb
from typing import Any

from candidates import DETERMINISTIC, Judge, select
from evalset import HERE, KINDS, NEGATIVE_TYPES, TAGS, Item, load_items

RESULTS = HERE / "results"
PROTOCOL_COMMIT = "069da322a575f71df0be176244554c2199129466"
DATA_COMMIT = "1b21a6eaf2bf8f52f7aa3d66310fc1ac80ea1031"
ORDER = ("B0", "B1", "C1", "C2", "C3a", "C3b", "C4", "C5")
CLASSES = {
    "B0": "reference",
    "B1": "reference",
    "C1": "deterministic",
    "C2": "deterministic",
    "C3a": "local model",
    "C3b": "local model",
    "C4": "hybrid",
    "C5": "local model",
}
PREFERENCE = {"deterministic": 0, "hybrid": 1, "local model": 2}
# The pass bar from PROTOCOL.md, for the 40 + 40 test split.
MAX_FALSE_ACCEPTS = 0
MAX_WRONG_ANSWERS = 2
MIN_SUPPORTED = 26
ALPHA = 0.05


def _binomial_cdf(x: int, n: int, p: float) -> float:
    return sum(comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(x + 1))


def upper_bound(x: int, n: int) -> float:
    """One-sided Clopper-Pearson upper bound on a rate of x in n."""

    if n == 0 or x >= n:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(100):
        middle = (low + high) / 2
        low, high = (middle, high) if _binomial_cdf(x, n, middle) > ALPHA else (low, middle)
    return round(low, 4)


def lower_bound(x: int, n: int) -> float:
    """One-sided Clopper-Pearson lower bound on a rate of x in n."""

    if n == 0 or x <= 0:
        return 0.0
    low, high = 0.0, 1.0
    for _ in range(100):
        middle = (low + high) / 2
        tail = 1 - _binomial_cdf(x - 1, n, middle)
        low, high = (middle, high) if tail < ALPHA else (low, middle)
    return round(high, 4)


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(HERE), *args], capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def manifest_problems() -> list[str]:
    problems = []
    for line in (HERE / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        path = HERE / name.strip()
        if not path.is_file():
            problems.append(f"missing {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            problems.append(f"changed {name}")
    return problems


def _outcome(item: Item, selected: list[tuple[str, str]]) -> str:
    if not selected:
        return "abstained"
    if item.label == "negative":
        return "false_accept"
    if all(choice in item.gold for choice in selected):
        return "supported"
    return "wrong_answer"


def run_candidate(judge: Judge, items: tuple[Item, ...]) -> dict[str, Any]:
    records = []
    pairs = Counter()
    for item in items:
        notes = []
        selected: list[tuple[str, str]] = []
        for note in item.notes:
            judgements = judge(item.question, note)
            for judgement in judgements:
                gold = (note.path, judgement.sentence) in item.gold
                pairs[(gold, judgement.accepted)] += 1
            choice = select(judgements)
            if choice is not None:
                selected.append((note.path, choice.sentence))
            notes.append(
                {
                    "note": note.path,
                    "selected": choice.sentence if choice else None,
                    "detail": choice.detail if choice else None,
                    "accepted": [j.sentence for j in judgements if j.accepted],
                }
            )
        records.append(
            {
                "id": item.id,
                "kind": item.kind,
                "label": item.label,
                "negative_type": item.negative_type,
                "tags": list(item.tags),
                "outcome": _outcome(item, selected),
                "notes": notes,
            }
        )
    return {"metrics": _metrics(records, pairs), "items": records}


def _metrics(records: list[dict[str, Any]], pairs: Counter) -> dict[str, Any]:
    answerable = [r for r in records if r["label"] == "answerable"]
    negatives = [r for r in records if r["label"] == "negative"]
    count = Counter(r["outcome"] for r in records)
    true_positive = pairs[(True, True)]
    precision_base = true_positive + pairs[(False, True)]
    recall_base = true_positive + pairs[(True, False)]

    def outcomes(rows: list[dict[str, Any]]) -> dict[str, int]:
        found = Counter(r["outcome"] for r in rows)
        return {name: found[name] for name in ("supported", "wrong_answer", "abstained")}

    return {
        "answerable": len(answerable),
        "negative": len(negatives),
        "false_accepts": count["false_accept"],
        "wrong_answers": count["wrong_answer"],
        "supported": count["supported"],
        "abstained_answerable": sum(1 for r in answerable if r["outcome"] == "abstained"),
        "false_accept_upper_95": upper_bound(count["false_accept"], len(negatives)),
        "wrong_answer_upper_95": upper_bound(count["wrong_answer"], len(answerable)),
        "supported_lower_95": lower_bound(count["supported"], len(answerable)),
        "sentence_precision": round(true_positive / precision_base, 4) if precision_base else None,
        "sentence_recall": round(true_positive / recall_base, 4) if recall_base else None,
        "sentence_pairs": {f"gold={g},accepted={a}": n for (g, a), n in sorted(pairs.items())},
        "answerable_by_kind": {
            kind: outcomes([r for r in answerable if r["kind"] == kind]) for kind in KINDS
        },
        "answerable_by_tag": {
            tag: outcomes([r for r in answerable if tag in r["tags"]]) for tag in TAGS
        },
        "false_accepts_by_kind": {
            kind: sum(1 for r in negatives if r["kind"] == kind and r["outcome"] != "abstained")
            for kind in KINDS
        },
        "false_accepts_by_type": {
            kind: sum(
                1 for r in negatives if r["negative_type"] == kind and r["outcome"] != "abstained"
            )
            for kind in NEGATIVE_TYPES
        },
    }


def bar(metrics: dict[str, Any]) -> dict[str, bool]:
    checks = {
        "false_accepts": metrics["false_accepts"] <= MAX_FALSE_ACCEPTS,
        "wrong_answers": metrics["wrong_answers"] <= MAX_WRONG_ANSWERS,
        "supported": metrics["supported"] >= MIN_SUPPORTED,
    }
    return {**checks, "clears": all(checks.values())}


def decide(candidates: dict[str, dict[str, Any]]) -> str | None:
    """Decision rule 1: deterministic over hybrid over learned, then the counts."""

    passing = [
        name
        for name, result in candidates.items()
        if result["status"] == "ok" and result["adoptable"] and result["bar"]["clears"]
    ]
    if not passing:
        return None
    return min(
        passing,
        key=lambda name: (
            PREFERENCE[CLASSES[name]],
            -candidates[name]["metrics"]["supported"],
            candidates[name]["metrics"]["wrong_answers"],
        ),
    )


def _versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    from learned import runtime_versions

    found = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        found["grounded-agent-lab"] = version("grounded-agent-lab")
    except PackageNotFoundError:
        found["grounded-agent-lab"] = "not installed"
    return {**found, **runtime_versions()}


def evaluate(split: str, names: tuple[str, ...]) -> dict[str, Any]:
    from learned import (
        NO_ANSWER_THRESHOLD,
        QNLI_MODEL,
        QNLI_REVISION,
        QNLI_THRESHOLD,
        SQUAD_MODEL,
        SQUAD_REVISION,
        CandidateBlocked,
        build_learned,
    )

    items = load_items(split)
    learned = build_learned() if any(name in names for name in ("C3a", "C3b", "C4", "C5")) else {}
    results: dict[str, dict[str, Any]] = {}
    for name in names:
        judge = DETERMINISTIC.get(name) or learned.get(name)
        entry: dict[str, Any] = {"class": CLASSES[name], "adoptable": name not in {"B0", "B1"}}
        if isinstance(judge, CandidateBlocked) or judge is None:
            entry.update(status="blocked", reason=str(judge) if judge else "unknown candidate")
            results[name] = entry
            continue
        try:
            outcome = run_candidate(judge, items)
        except CandidateBlocked as exc:
            entry.update(status="blocked", reason=str(exc))
        else:
            entry.update(status="ok", **outcome)
            entry["bar"] = bar(outcome["metrics"]) if split == "test" else None
        results[name] = entry
    return {
        "models": {
            QNLI_MODEL: QNLI_REVISION,
            SQUAD_MODEL: SQUAD_REVISION,
            "qnli_threshold": QNLI_THRESHOLD,
            "squad_no_answer_threshold": NO_ANSWER_THRESHOLD,
        },
        "candidates": results,
    }


def summary_markdown(receipt: dict[str, Any]) -> str:
    split = receipt["split"]
    lines = [
        f"# Open-question support check: {split} results",
        "",
        f"Generated by `run_eval.py` from `results/{split}-receipt.json`. Do not edit by hand.",
        "",
        f"Commit `{receipt['commit'][:12]}`, tree clean: {receipt['clean_tree']}, "
        f"data manifest intact: {not receipt['manifest_problems']}.",
        "",
        "| Candidate | Class | Status | False accepts | Wrong answers | Supported | "
        "Abstained (answerable) | FA upper 95% | Supported lower 95% | Sentence P / R | Bar |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, result in receipt["candidates"].items():
        if result["status"] != "ok":
            lines.append(f"| {name} | {result['class']} | blocked: {result['reason']} |"
                         " | | | | | | | |")
            continue
        m = result["metrics"]
        if result["bar"] is None:
            verdict = "n/a on dev"
        else:
            verdict = "clears" if result["bar"]["clears"] else "fails"
        if not result["adoptable"]:
            verdict += " (reference)"
        lines.append(
            f"| {name} | {result['class']} | ok | {m['false_accepts']}/{m['negative']} | "
            f"{m['wrong_answers']}/{m['answerable']} | {m['supported']}/{m['answerable']} | "
            f"{m['abstained_answerable']} | {m['false_accept_upper_95']:.3f} | "
            f"{m['supported_lower_95']:.3f} | {m['sentence_precision']} / "
            f"{m['sentence_recall']} | {verdict} |"
        )
    lines += ["", f"Decision: {receipt['decision'] or 'no candidate clears the bar'}.", ""]
    lines += ["## Supported answers by kind", "", "| Candidate | " + " | ".join(KINDS) + " |"]
    lines.append("|---|" + "---|" * len(KINDS))
    for name, result in receipt["candidates"].items():
        if result["status"] == "ok":
            by_kind = result["metrics"]["answerable_by_kind"]
            cells = [
                f"{by_kind[kind]['supported']} ok, {by_kind[kind]['wrong_answer']} wrong"
                for kind in KINDS
            ]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## False accepts by negative type", ""]
    lines.append("| Candidate | " + " | ".join(NEGATIVE_TYPES) + " |")
    lines.append("|---|" + "---|" * len(NEGATIVE_TYPES))
    for name, result in receipt["candidates"].items():
        if result["status"] == "ok":
            by_type = result["metrics"]["false_accepts_by_type"]
            lines.append(
                f"| {name} | " + " | ".join(str(by_type[kind]) for kind in NEGATIVE_TYPES) + " |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--final", action="store_true", help="the single test run")
    parser.add_argument("--rerun-reason", help="only after a harness crash; recorded")
    parser.add_argument("--candidates", default=",".join(ORDER))
    args = parser.parse_args(argv)

    names = tuple(name.strip() for name in args.candidates.split(",") if name.strip())
    receipt_path = RESULTS / f"{args.split}-receipt.json"
    problems = manifest_problems()
    clean = _git("status", "--porcelain", "--untracked-files=all") == ""
    if args.split == "test":
        if not args.final:
            parser.error("the test split runs only with --final")
        if names != ORDER:
            parser.error("the final run covers every candidate")
        if problems:
            parser.error(f"data manifest does not match: {problems}")
        if not clean:
            parser.error("the final run needs a clean git tree")
        if receipt_path.exists() and not args.rerun_reason:
            parser.error("a test receipt already exists; the test split runs once")

    started = datetime.now(UTC).isoformat(timespec="seconds")
    evaluation = evaluate(args.split, names)
    receipt = {
        "experiment": "open-question-support",
        "split": args.split,
        "final": bool(args.final),
        "rerun_reason": args.rerun_reason,
        "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "HEAD"),
        "clean_tree": clean,
        "protocol_commit": PROTOCOL_COMMIT,
        "data_commit": DATA_COMMIT,
        "manifest_problems": problems,
        "versions": _versions(),
        "bar": {
            "max_false_accepts": MAX_FALSE_ACCEPTS,
            "max_wrong_answers": MAX_WRONG_ANSWERS,
            "min_supported": MIN_SUPPORTED,
            "applies_to": "test",
        },
        **evaluation,
    }
    receipt["decision"] = decide(receipt["candidates"]) if args.split == "test" else None
    RESULTS.mkdir(exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    summary = summary_markdown(receipt)
    (RESULTS / f"{args.split}-summary.md").write_text(summary, encoding="utf-8")
    sys.stdout.write(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
