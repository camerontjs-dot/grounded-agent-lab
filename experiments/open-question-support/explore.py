"""Exploratory analysis after the decision. Nothing here can change it.

    python experiments/open-question-support/explore.py

Reads the committed dev and test receipts and writes results/exploratory.json:

1. C2 split by whether its question parser recognised the shape.
2. Threshold sweeps for the learned candidates, recomputed from the selected
   sentence's score in each receipt. The threshold is chosen on dev (the lowest
   one with no dev false accept) and then read off test. The protocol fixed 0.5
   and 0.0; these numbers are not candidates and were not preregistered.
3. The frozen candidates on the Harbor fixture's open questions, run once after
   the decision, as the protocol asks, to show what each would have answered.
   Selections pass through `redact_text`, as the drafter's output does.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from typing import Any

from candidates import DETERMINISTIC, parse_shape, select
from evalset import Note, load_items
from run_eval import ORDER, RESULTS

from grounded_agent.models import ResearchRequest
from grounded_agent.redact import redact_text
from grounded_agent.retrieve import retrieve
from grounded_agent.router import route_intent
from grounded_agent.support import requested_quantity, split_sentences

HARBOR_OPEN_QUESTIONS = (
    "What must Harbor do when combining retrieval results from different indexes?",
    "What is the next action for lantern-agent?",
    "How does the lantern-agent next action relate to Harbor trust labels?",
    "When a Harbor receipt omits snippet text, what remains unproven?",
    "What is the Harbor operator contact alias?",
)
_SCORE = re.compile(r"(?:qnli|span-minus-null) (-?\d+\.\d+)")


def _receipt(split: str) -> dict[str, Any]:
    return json.loads((RESULTS / f"{split}-receipt.json").read_text(encoding="utf-8"))


def c2_by_scope() -> dict[str, int]:
    items = {item.id: item for item in load_items("test")}
    counts: Counter[str] = Counter()
    for row in _receipt("test")["candidates"]["C2"]["items"]:
        scope = "in_inventory" if parse_shape(items[row["id"]].question) else "outside"
        counts[f"{row['label']}/{scope}/{row['outcome']}"] += 1
    return dict(sorted(counts.items()))


def _at_threshold(rows: list[dict[str, Any]], gold: dict[str, Any], cut: float) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        chosen = []
        for note in row["notes"]:
            match = _SCORE.search(note["detail"] or "")
            if note["selected"] and match and float(match.group(1)) >= cut:
                chosen.append((note["note"], note["selected"]))
        if not chosen:
            counts["abstained_" + row["label"]] += 1
        elif row["label"] == "negative":
            counts["false_accepts"] += 1
        elif all(choice in gold[row["id"]] for choice in chosen):
            counts["supported"] += 1
        else:
            counts["wrong_answers"] += 1
    return {key: counts[key] for key in ("false_accepts", "wrong_answers", "supported")}


def threshold_sweeps() -> dict[str, Any]:
    gold = {
        split: {item.id: item.gold for item in load_items(split)} for split in ("dev", "test")
    }
    receipts = {split: _receipt(split) for split in ("dev", "test")}
    sweeps: dict[str, Any] = {}
    for name in ("C3a", "C3b", "C4", "C5"):
        rows = {split: receipts[split]["candidates"][name]["items"] for split in receipts}
        scores = sorted(
            {
                float(match.group(1))
                for split in rows
                for row in rows[split]
                for note in row["notes"]
                if note["selected"] and (match := _SCORE.search(note["detail"] or ""))
            }
        )
        dev_clean = [
            cut
            for cut in scores
            if _at_threshold(rows["dev"], gold["dev"], cut)["false_accepts"] == 0
        ]
        chosen = min(dev_clean) if dev_clean else None
        sweeps[name] = {
            "rule": "lowest selected-sentence score with no dev false accept",
            "threshold": chosen,
            "dev": _at_threshold(rows["dev"], gold["dev"], chosen) if chosen else None,
            "test": _at_threshold(rows["test"], gold["test"], chosen) if chosen else None,
        }
    return sweeps


def harbor_consequences() -> dict[str, Any]:
    from learned import CandidateBlocked, build_learned

    learned = build_learned()
    judges = {name: DETERMINISTIC.get(name) or learned.get(name) for name in ORDER}
    report: dict[str, Any] = {}
    for question in HARBOR_OPEN_QUESTIONS:
        assert requested_quantity(question) is None
        request = ResearchRequest(request_id="explore-harbor", question=question)
        citable = retrieve(request, route_intent(request)).citable()
        notes = [
            Note(
                path=item.source_path,
                trust_profile=item.trust_profile,
                title=item.title,
                sentences=split_sentences(item.snippet),
            )
            for item in citable
        ]
        rows: dict[str, Any] = {"nominated": [note.path for note in notes]}
        for name, judge in judges.items():
            if isinstance(judge, CandidateBlocked) or judge is None:
                rows[name] = "blocked"
                continue
            chosen = [
                redact_text(choice.sentence)
                for note in notes
                if (choice := select(judge(question, note))) is not None
            ]
            rows[name] = chosen or "abstained"
        report[question] = rows
    return report


def main() -> int:
    payload = {
        "note": "exploratory; computed after the preregistered decision and cannot change it",
        "c2_test_by_scope": c2_by_scope(),
        "threshold_sweeps": threshold_sweeps(),
        "harbor_open_questions": harbor_consequences(),
    }
    path = RESULTS / "exploratory.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
