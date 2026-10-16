"""Load and check the open-question evaluation set. See PROTOCOL.md.

Sentences come from `grounded_agent.support.split_sentences`, the splitter the
drafter uses, so a gold label names exactly the text a candidate can select.

    python experiments/open-question-support/evalset.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from grounded_agent.support import requested_quantity, split_sentences

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SPLITS = ("dev", "test")
KINDS = ("attribute", "rule", "consequence", "relation")
NEGATIVE_TYPES = ("topic", "wrong_subject", "collision", "echo", "unanswerable")
TAGS = ("first", "later", "paraphrase", "pronoun", "distractor")
_IDENTIFIER = re.compile(r"\b[a-z0-9]+(?:-[a-z0-9]+)+\b")


@dataclass(frozen=True)
class Note:
    path: str
    trust_profile: str
    title: str
    sentences: tuple[str, ...]


@dataclass(frozen=True)
class Item:
    id: str
    kind: str
    label: str
    negative_type: str | None
    tags: tuple[str, ...]
    question: str
    notes: tuple[Note, ...]
    gold: frozenset[tuple[str, str]]


def _parse_note(path: Path, relative: str) -> Note:
    text = path.read_text(encoding="utf-8")
    _, header, body = text.split("---", 2)
    meta = dict(line.split(":", 1) for line in header.strip().splitlines())
    meta = {key.strip(): value.strip() for key, value in meta.items()}
    sentences = split_sentences(" ".join(body.split()))
    return Note(
        path=relative,
        trust_profile=meta["trust_profile"],
        title=meta["title"],
        sentences=sentences,
    )


def load_corpus(split: str) -> dict[str, Note]:
    root = DATA / split / "corpus"
    return {
        path.relative_to(root).as_posix(): _parse_note(path, path.relative_to(root).as_posix())
        for path in sorted(root.glob("*/*.md"))
    }


def load_items(split: str) -> tuple[Item, ...]:
    corpus = load_corpus(split)
    raw = json.loads((DATA / split / "items.json").read_text(encoding="utf-8"))
    return tuple(
        Item(
            id=row["id"],
            kind=row["kind"],
            label=row["label"],
            negative_type=row["negative_type"],
            tags=tuple(row["tags"]),
            question=row["question"],
            notes=tuple(corpus[path] for path in row["notes"]),
            gold=frozenset((entry["note"], entry["sentence"]) for entry in row["gold"]),
        )
        for row in raw
    )


def _item_problems(item: Item) -> list[str]:
    problems: list[str] = []
    paths = [note.path for note in item.notes]
    if not 1 <= len(paths) <= 3 or len(set(paths)) != len(paths):
        problems.append("needs one to three distinct notes")
    if item.kind not in KINDS:
        problems.append(f"unknown kind {item.kind!r}")
    if requested_quantity(item.question) is not None:
        problems.append("question asks for a quantity")
    by_path = {note.path: note for note in item.notes}
    for path, sentence in item.gold:
        if path not in by_path:
            problems.append(f"gold note {path} is not nominated")
        elif sentence not in by_path[path].sentences:
            problems.append(f"gold sentence not in {path}: {sentence!r}")
    if item.label == "negative":
        if item.gold or item.tags:
            problems.append("negative control carries gold or tags")
        if item.negative_type not in NEGATIVE_TYPES:
            problems.append(f"unknown negative type {item.negative_type!r}")
        return problems
    if item.label != "answerable" or item.negative_type is not None or not item.gold:
        problems.append("answerable item needs gold and no negative type")
        return problems
    unknown = set(item.tags) - set(TAGS)
    if unknown:
        problems.append(f"unknown tags {sorted(unknown)}")
    all_first = all(
        path in by_path and by_path[path].sentences.index(sentence) == 0
        for path, sentence in item.gold
        if path in by_path and sentence in by_path[path].sentences
    )
    if ("first" in item.tags) != all_first or ("later" in item.tags) == all_first:
        problems.append("first/later tag does not match gold positions")
    gold_paths = {path for path, _ in item.gold}
    has_distractor = any(path not in gold_paths for path in paths)
    if ("distractor" in item.tags) != has_distractor:
        problems.append("distractor tag does not match the nominated notes")
    return problems


def _quota_problems(split: str, items: tuple[Item, ...]) -> list[str]:
    scale = 1 if split == "test" else 2
    problems: list[str] = []
    answerable = [item for item in items if item.label == "answerable"]
    negatives = [item for item in items if item.label == "negative"]
    if len(answerable) != 40 // scale or len(negatives) != 40 // scale:
        problems.append(f"{split}: want {40 // scale} answerable and negative items")
    for kind in KINDS:
        for group, name in ((answerable, "answerable"), (negatives, "negative")):
            count = sum(1 for item in group if item.kind == kind)
            if count != 10 // scale:
                problems.append(f"{split}: {count} {name} {kind} items, want {10 // scale}")
    for negative_type in NEGATIVE_TYPES:
        typed = [item for item in negatives if item.negative_type == negative_type]
        if len(typed) != 8 // scale:
            problems.append(f"{split}: {len(typed)} {negative_type} controls, want {8 // scale}")
        if split == "test" and len({item.kind for item in typed}) < 3:
            problems.append(f"test: {negative_type} controls cover fewer than three kinds")
    if split == "test":
        tags = Counter(tag for item in answerable for tag in item.tags)
        if tags["first"] != 10 or tags["later"] != 30:
            problems.append(f"test: first={tags['first']} later={tags['later']}, want 10 and 30")
        for tag, minimum in (("paraphrase", 10), ("pronoun", 10), ("distractor", 15)):
            if tags[tag] < minimum:
                problems.append(f"test: {tags[tag]} {tag} items, want at least {minimum}")
    return problems


def check() -> list[str]:
    problems: list[str] = []
    identifiers: dict[str, set[str]] = {}
    for split in SPLITS:
        items = load_items(split)
        ids = [item.id for item in items]
        if len(set(ids)) != len(ids):
            problems.append(f"{split}: duplicate item ids")
        for item in items:
            problems.extend(f"{split} {item.id}: {problem}" for problem in _item_problems(item))
        problems.extend(_quota_problems(split, items))
        corpus = load_corpus(split)
        text = " ".join(
            " ".join((note.title, *note.sentences)).lower() for note in corpus.values()
        ) + " " + " ".join(item.question.lower() for item in items)
        identifiers[split] = set(_IDENTIFIER.findall(text))
    shared = identifiers["dev"] & identifiers["test"]
    if shared:
        problems.append(f"dev and test share identifiers: {sorted(shared)}")
    return problems


def summary(split: str) -> str:
    items = load_items(split)
    answerable = [item for item in items if item.label == "answerable"]
    negatives = [item for item in items if item.label == "negative"]
    tags = Counter(tag for item in answerable for tag in item.tags)
    types = Counter(item.negative_type for item in negatives)
    pairs = sum(len(note.sentences) for item in items for note in item.notes)
    return (
        f"{split}: {len(answerable)} answerable, {len(negatives)} negative, "
        f"{pairs} question-sentence pairs; tags {dict(sorted(tags.items()))}; "
        f"types {dict(sorted(types.items()))}"
    )


def main() -> int:
    problems = check()
    for split in SPLITS:
        print(summary(split))
    for problem in problems:
        print(f"PROBLEM {problem}")
    print("ok" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
