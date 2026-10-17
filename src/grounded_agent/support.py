"""Deterministic support gate between nominated evidence and an answer.

Retrieval nominates notes. A citation path, an overlap score, or a schema-valid
model draft is not warrant for a claim. This module rejects two kinds of
unsupported answer that a deterministic check can see:

1. Requested quantity. A question that asks "how many", "how much", or for the
   "number of" something is answerable only from a cited sentence that states
   a quantity. For "how many <noun>", the number has to sit in front of that
   noun ("six notes", "6 Harbor notes"), not anywhere in the sentence.
2. Claim specifics. A model claim may not introduce a number or a name that the
   notes it cites do not contain. Names here are acronyms, identifiers such as
   `lantern-agent`, e-mail addresses, URLs, letter-digit tokens, and
   capitalised words after the first word of a sentence.

That is the whole contract. It is not an entailment check. A paraphrase built
from ordinary words, a dropped negation, a swapped relation, or a count taken
from the wrong index can still pass. docs/limitations.md lists those gaps.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from grounded_agent.models import EvidenceItem

UNITS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}
TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
NUMBER_WORDS = {
    "zero": 0,
    **UNITS,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    **TENS,
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}
IRREGULAR_PLURALS = {
    "people": "person",
    "children": "child",
    "men": "man",
    "women": "woman",
    "indices": "index",
    "criteria": "criterion",
}
# Words that may open the counted noun phrase ("how many of the notes").
NOUN_PHRASE_LEAD = frozenset({"of", "the", "a", "an", "these", "those", "such"})
# Words that end the counted noun phrase ("how many notes ARE in ...").
NOUN_PHRASE_STOP = frozenset(
    {
        "are",
        "is",
        "was",
        "were",
        "be",
        "been",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "can",
        "could",
        "will",
        "would",
        "should",
        "must",
        "may",
        "might",
        "there",
        "in",
        "on",
        "at",
        "for",
        "from",
        "with",
        "by",
        "about",
        "to",
        "into",
        "than",
        "per",
        "that",
        "which",
        "who",
    }
)
NOUN_PHRASE_MAX = 4
# A number binds to a noun when the noun is among the next few tokens:
# "six notes", "six Harbor notes", "6 durable knowledge notes".
BINDING_WINDOW = 4

_COUNT_TRIGGER = re.compile(r"\b(?:how\s+many|number\s+of|count\s+of)\b", re.IGNORECASE)
_AMOUNT_TRIGGER = re.compile(r"\bhow\s+much\b", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-.,'’][A-Za-z0-9]+)*")
_DIGITS = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*(?!\w)")
_DIGIT_TOKEN = re.compile(r"\d+(?:[.,]\d+)*")
_ALPHA_RUN = re.compile(r"[a-z]+(?:-[a-z]+)*")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"https?://[^\s)\]>\"']+")
_IDENTIFIER = re.compile(r"\b[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+\b")
_ACRONYM = re.compile(r"\b[A-Z]{2,}\b")
_LETTER_DIGIT = re.compile(r"\b(?=[A-Za-z]*\d)(?=\d*[A-Za-z])[A-Za-z0-9]+\b")
_WORD = re.compile(r"[A-Za-z][A-Za-z'’]*")
_CAPITALISED = re.compile(r"[A-Z][a-z]+")
_POSSESSIVE = re.compile(r"['’]s$")
_NOT_ALNUM = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class QuantityRequest:
    """A question that needs a stated quantity. `noun` is None for "how much"."""

    noun: str | None


def singular(word: str) -> str:
    lowered = _POSSESSIVE.sub("", word.lower())
    if lowered in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[lowered]
    if lowered.endswith("ies") and len(lowered) > 4:
        return lowered[:-3] + "y"
    if lowered.endswith(("sses", "uses", "xes", "ches", "shes", "zes")):
        return lowered[:-2]
    if lowered.endswith("s") and not lowered.endswith("ss") and len(lowered) > 3:
        return lowered[:-1]
    return lowered


def _is_plural(word: str) -> bool:
    return word in IRREGULAR_PLURALS or (
        word.endswith("s") and not word.endswith("ss") and len(word) > 3
    )


def _counted_noun(rest: str) -> str | None:
    words = [_POSSESSIVE.sub("", word) for word in re.findall(r"[a-z][a-z'’-]*", rest.lower())]
    while words and words[0] in NOUN_PHRASE_LEAD:
        words.pop(0)
    phrase: list[str] = []
    for word in words[:NOUN_PHRASE_MAX]:
        if word in NOUN_PHRASE_STOP:
            break
        phrase.append(word)
    for word in phrase:
        if _is_plural(word):
            return singular(word)
    return singular(phrase[-1]) if phrase else None


def requested_quantity(question: str) -> QuantityRequest | None:
    """Return what quantity the question asks for, or None for other questions."""

    count = _COUNT_TRIGGER.search(question)
    if count is not None:
        return QuantityRequest(noun=_counted_noun(question[count.end() :]))
    if _AMOUNT_TRIGGER.search(question):
        return QuantityRequest(noun=None)
    return None


def split_sentences(text: str) -> tuple[str, ...]:
    parts = _SENTENCE_END.split(" ".join(text.split()))
    return tuple(part.strip() for part in parts if part.strip())


def _normalise_number(raw: str) -> str:
    value = raw.replace(",", "")
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value


def _number_value(token: str) -> str | None:
    lowered = token.lower()
    if _DIGIT_TOKEN.fullmatch(lowered):
        return _normalise_number(lowered)
    if lowered in NUMBER_WORDS:
        return str(NUMBER_WORDS[lowered])
    parts = lowered.split("-")
    if len(parts) == 2 and parts[0] in TENS and parts[1] in UNITS:
        return str(TENS[parts[0]] + UNITS[parts[1]])
    return None


def numbers(text: str) -> frozenset[str]:
    """Numbers written as digits or as number words, normalised to digits."""

    found = {_normalise_number(match.group(0)) for match in _DIGITS.finditer(text)}
    for run in _ALPHA_RUN.findall(text.lower()):
        compound = _number_value(run)
        if compound is not None:
            found.add(compound)
            continue
        for part in run.split("-"):
            if part in NUMBER_WORDS:
                found.add(str(NUMBER_WORDS[part]))
    return frozenset(found)


def quantities_for(sentence: str, noun: str | None) -> frozenset[str]:
    """Numbers in `sentence` attached to `noun` (any number when noun is None)."""

    tokens = _TOKEN.findall(sentence)
    bound: set[str] = set()
    for index, token in enumerate(tokens):
        value = _number_value(token)
        if value is None:
            continue
        if noun is None:
            bound.add(value)
            continue
        for follower in tokens[index + 1 : index + 1 + BINDING_WINDOW]:
            if _number_value(follower) is not None:
                break
            if singular(follower) == noun:
                bound.add(value)
                break
    return frozenset(bound)


def states_requested_quantity(sentence: str, request: QuantityRequest) -> bool:
    return bool(quantities_for(sentence, request.noun))


def _phrase(text: str) -> str:
    lowered = text.lower().replace("-", " ").replace("_", " ")
    return f" {' '.join(_NOT_ALNUM.sub(' ', lowered).split())} "


def _names(text: str) -> tuple[tuple[str, str], ...]:
    """Name-like tokens as (kind, normalised value). Numbers are handled apart."""

    found: list[tuple[str, str]] = []
    remainder = text
    for kind, pattern in (("url", _URL), ("email", _EMAIL)):
        for match in pattern.findall(remainder):
            found.append((kind, match.lower().rstrip(".,;:")))
        remainder = pattern.sub(" ", remainder)
    for pattern in (_IDENTIFIER, _ACRONYM, _LETTER_DIGIT):
        for match in pattern.findall(remainder):
            if _DIGIT_TOKEN.fullmatch(match):
                continue
            found.append(("name", _phrase(match).strip()))
    for sentence in split_sentences(remainder):
        for word in _WORD.findall(sentence)[1:]:
            base = _POSSESSIVE.sub("", word)
            if _CAPITALISED.fullmatch(base):
                found.append(("name", base.lower()))
    return tuple(dict.fromkeys(found))


def unsupported_specifics(claim: str, grounding: str) -> tuple[str, ...]:
    """Numbers and names in `claim` that `grounding` does not contain."""

    missing = [f"number {value}" for value in sorted(numbers(claim) - numbers(grounding))]
    raw = grounding.lower()
    phrase = _phrase(grounding)
    for kind, value in _names(claim):
        if kind in {"url", "email"}:
            present = value in raw
        else:
            present = f" {value} " in phrase
        if not present:
            missing.append(f"{kind} {value}")
    return tuple(missing)


def grounding_text(items: Iterable[EvidenceItem]) -> str:
    """Everything a citation carries: path, trust label, title, and snippet."""

    return "\n".join(
        f"{item.source_path}\n{item.trust_profile}\n{item.title}\n{item.snippet}" for item in items
    )


def _source_sentences(items: Iterable[EvidenceItem]) -> tuple[str, ...]:
    sentences: list[str] = []
    for item in items:
        sentences.append(item.title)
        sentences.extend(split_sentences(item.snippet))
    return tuple(sentences)


def claim_support_problem(
    claim: str,
    cited: tuple[EvidenceItem, ...],
    *,
    question: str | None = None,
) -> str | None:
    """Return why `claim` is not supported by `cited`, or None if no check fails."""

    if unsupported_specifics(claim, grounding_text(cited)):
        return "claim adds a number or name the cited notes do not contain"
    request = requested_quantity(question) if question else None
    if request is None:
        return None
    claimed = {
        value
        for sentence in split_sentences(claim)
        for value in quantities_for(sentence, request.noun)
    }
    if not claimed:
        return "claim does not state the requested quantity"
    stated = {
        value
        for sentence in _source_sentences(cited)
        for value in quantities_for(sentence, request.noun)
    }
    if not claimed <= stated:
        return "cited notes do not state that quantity"
    return None
