"""Deterministic candidates for the open-question support check. See PROTOCOL.md.

Each candidate looks at one nominated note and returns at most one sentence. B0
and B1 are references. C1 and C2 can be adopted. The learned candidates live in
learned.py, so this module needs no model runtime.

C1's rules and constants are fixed by the protocol. C2's question shapes are
fixed by the protocol; the surface patterns inside each shape were built on the
dev split only.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from evalset import Note

from grounded_agent.draft import _first_sentence
from grounded_agent.retrieve import STOPWORDS
from grounded_agent.support import _phrase, singular


@dataclass(frozen=True)
class Judgement:
    """One candidate's verdict on one sentence. Higher `rank` is preferred."""

    sentence: str
    accepted: bool
    rank: float
    detail: str


Judge = Callable[[str, Note], tuple[Judgement, ...]]


def select(judgements: tuple[Judgement, ...]) -> Judgement | None:
    """The accepted sentence with the highest rank, ties to the earlier sentence."""

    best: Judgement | None = None
    for judgement in judgements:
        if judgement.accepted and (best is None or judgement.rank > best.rank):
            best = judgement
    return best


def content_tokens(text: str) -> frozenset[str]:
    """Lowercase alphanumeric runs longer than two characters, minus stopwords, singular."""

    buffered = "".join(char if char.isalnum() else " " for char in text.lower())
    return frozenset(
        singular(token) for token in buffered.split() if len(token) > 2 and token not in STOPWORDS
    )


# B0: the current drafter.


def b0_first_sentence(question: str, note: Note) -> tuple[Judgement, ...]:
    first = _first_sentence(" ".join(note.sentences))
    return tuple(
        Judgement(sentence, sentence == first, 0.0, "first sentence")
        for sentence in note.sentences
    )


# B1: lexical overlap, the route ADR-010 turned down.

OVERLAP_FLOOR = 2


def b1_overlap(question: str, note: Note) -> tuple[Judgement, ...]:
    asked = content_tokens(question)
    judgements = []
    for sentence in note.sentences:
        shared = len(asked & content_tokens(sentence))
        judgements.append(
            Judgement(sentence, shared >= OVERLAP_FLOOR, float(shared), f"{shared} shared tokens")
        )
    return tuple(judgements)


# C1: entity binding at note level, then attribute coverage.

_IDENTIFIER = re.compile(r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b")
_WORD = re.compile(r"[A-Za-z][A-Za-z'’]*")
_CAPITALISED = re.compile(r"[A-Z][a-z]+")
_POSSESSIVE = re.compile(r"['’]s$")
ORGANISATION = "Harbor"
ATTRIBUTE_FLOOR = 2
NEW_TOKEN_FLOOR = 1


def entity_terms(question: str) -> tuple[str, ...]:
    """Hyphenated identifiers and capitalised non-initial words, except Harbor."""

    terms = [_phrase(match).strip() for match in _IDENTIFIER.findall(question)]
    for word in _WORD.findall(question)[1:]:
        base = _POSSESSIVE.sub("", word)
        if _CAPITALISED.fullmatch(base) and base != ORGANISATION:
            terms.append(base.lower())
    return tuple(dict.fromkeys(terms))


def note_is_about(note: Note, terms: tuple[str, ...]) -> bool:
    """Every entity term appears in the note's title or first sentence."""

    heading = _phrase(f"{note.title} {note.sentences[0] if note.sentences else ''}")
    return all(f" {term} " in heading for term in terms)


def c1_entity_attribute(question: str, note: Note) -> tuple[Judgement, ...]:
    terms = entity_terms(question)
    if not note_is_about(note, terms):
        return tuple(
            Judgement(sentence, False, 0.0, "note is not about the asked entity")
            for sentence in note.sentences
        )
    asked = content_tokens(question)
    attribute = asked - content_tokens(" ".join(terms))
    needed = min(ATTRIBUTE_FLOOR, len(attribute))
    judgements = []
    for sentence in note.sentences:
        tokens = content_tokens(sentence)
        covered = len(tokens & attribute)
        qualifies = covered >= needed and len(tokens - asked) >= NEW_TOKEN_FLOOR
        judgements.append(
            Judgement(sentence, qualifies, float(covered), f"{covered} attribute terms")
        )
    return tuple(judgements)


# C2: a closed inventory of question shapes, each with an answer pattern.

STATUS_WORDS = (
    "active",
    "paused",
    "blocked",
    "on hold",
    "deferred",
    "archived",
    "complete",
    "completed",
    "cancelled",
    "shipped",
    "live",
    "deprecated",
    "in review",
    "draft",
    "frozen",
)
PRONOUNS = frozenset({"it", "its", "they", "their", "them", "the project"})
COPULA = r"(?:is|are|was|were)"
# Deontic markers for rule answers, and the families R3 maps its word onto.
DEONTIC = re.compile(
    r"\b(?:must|should|shall|is required|are required|is forbidden|are forbidden|"
    r"do not|does not|never|cannot)\b",
    re.IGNORECASE,
)
DEONTIC_FAMILY = {
    "required": re.compile(r"\b(?:must|required|shall)\b", re.IGNORECASE),
    "forbidden": re.compile(
        r"\b(?:must not|forbidden|never|cannot|do not|does not|not allowed)\b", re.IGNORECASE
    ),
    "allowed": re.compile(r"\b(?:may|can|allowed)\b", re.IGNORECASE),
}
CONDITION_MARKERS = re.compile(r"\b(?:when|if|after)\b", re.IGNORECASE)
# A sentence whose subject is an embedded question points at another document.
EMBEDDED_QUESTION = re.compile(r"^(?:what|how|which|who)\b", re.IGNORECASE)
RELATION_VERBS = ("relate to", "relates to", "depend on", "depends on")
AUXILIARIES = frozenset({"is", "are", "was", "were", "has", "does", "do", "did"})


@dataclass(frozen=True)
class Shape:
    name: str
    subject: str = ""
    attribute: str = ""
    verb: str = ""
    extra: str = ""
    modal: str = ""
    condition: str = ""
    target: str = ""
    other: str = ""


def _clean(question: str) -> str:
    return " ".join(question.strip().rstrip("?").split())


def parse_shape(question: str) -> Shape | None:
    """Match the question against the protocol's inventory. None means out of scope."""

    q = _clean(question)
    lowered = q.lower()

    match = re.fullmatch(r"what (?:is|are) the (.+?) (?:for|of) (.+)", lowered)
    if match:
        attribute, subject = match.groups()
        if attribute == "status":
            return Shape("status", subject=subject)
        return Shape("attribute", subject=subject, attribute=attribute)
    match = re.fullmatch(r"what (?:is|are) ([a-z0-9-]+)['’]s (.+)", lowered)
    if match:
        return Shape("attribute", subject=match.group(1), attribute=match.group(2))
    match = re.fullmatch(r"who (?:is|are) the (.+?) (?:for|of) (.+)", lowered)
    if match:
        return Shape("attribute", subject=match.group(2), attribute=match.group(1))
    match = re.fullmatch(r"who ([a-z]+s) (.+)", lowered)
    if match and match.group(1) not in AUXILIARIES:
        return Shape("who", subject=match.group(2), verb=match.group(1))
    match = re.fullmatch(
        r"what (?:does|do) (.+?) ([a-z]+)((?: (?:when|if|after|before|with|for|to|on) .+)?)",
        lowered,
    )
    if match:
        return Shape(
            "does", subject=match.group(1), verb=match.group(2), extra=match.group(3).strip()
        )
    match = re.fullmatch(r"what (must|should) (.+?) do (when|before|after|if) (.+)", lowered)
    if match:
        return Shape(
            "do_when", modal=match.group(1), subject=match.group(2), condition=match.group(4)
        )
    match = re.fullmatch(r"what (must|should|may|can) (.+)", lowered)
    if match:
        return Shape("modal", modal=match.group(1), extra=match.group(2))
    match = re.fullmatch(r"what is (required|forbidden|allowed) (?:when|before|for) (.+)", lowered)
    if match:
        return Shape("deontic", modal=match.group(1), condition=match.group(2))
    match = re.fullmatch(r"what happens(?: to (.+?))? (?:when|if|after) (.+)", lowered)
    if match:
        return Shape("happens", target=match.group(1) or "", condition=match.group(2))
    match = re.fullmatch(r"(?:when|if) (.+?), what (.+)", lowered)
    if match:
        rest = re.sub(r"^happens(?: to)?\s*", "", match.group(2))
        return Shape("happens", target=rest, condition=match.group(1))
    match = re.fullmatch(r"what is the relationship between (.+?) and (.+)", lowered)
    if match:
        return Shape("relation", subject=match.group(1), other=match.group(2))
    match = re.fullmatch(r"how (?:does|do) (.+)", q, flags=re.IGNORECASE)
    if match:
        return _relation_shape(match.group(1))
    return None


def _relation_shape(rest: str) -> Shape | None:
    identifiers = _IDENTIFIER.findall(rest)
    if len(identifiers) == 2:
        return Shape("relation", subject=identifiers[0].lower(), other=identifiers[1].lower())
    lowered = rest.lower()
    for verb in RELATION_VERBS:
        left, found, right = lowered.partition(f" {verb} ")
        if found and left and right:
            return Shape("relation", subject=left, other=right)
    return None


def _note_subject(note: Note) -> str:
    """A project note's identifier, or a knowledge note's title."""

    first = note.sentences[0] if note.sentences else ""
    identifiers = _IDENTIFIER.findall(first)
    if note.trust_profile == "project_status" and identifiers:
        return identifiers[0].lower()
    return note.title.lower()


def _head(identifier: str) -> str:
    return identifier.split("-")[-1]


def _strip_determiner(text: str) -> str:
    return re.sub(r"^(?:the|a|an)\s+", "", text.strip().lower())


def _names(text: str, entity: str) -> bool:
    """`text` names `entity`: its identifier phrase, or every content token."""

    entity = _strip_determiner(entity)
    if entity.lower() == ORGANISATION.lower():
        return True
    if _IDENTIFIER.fullmatch(entity):
        return f" {_phrase(entity).strip()} " in _phrase(text)
    wanted = content_tokens(entity)
    return bool(wanted) and wanted <= content_tokens(text)


def _subject_matches(subject_text: str, entity: str, note: Note) -> bool:
    """The grammatical subject of a sentence refers to `entity`."""

    entity = _strip_determiner(entity)
    if entity == ORGANISATION.lower():
        return True
    cleaned = _strip_determiner(subject_text)
    note_subject = _note_subject(note)
    identifiers = [found.lower() for found in _IDENTIFIER.findall(subject_text)]
    if identifiers:
        return entity in identifiers
    if subject_text.strip().lower() in PRONOUNS or cleaned == "":
        return _same_entity(note_subject, entity)
    if _IDENTIFIER.fullmatch(note_subject) and cleaned == _head(note_subject):
        return _same_entity(note_subject, entity)
    if not _IDENTIFIER.fullmatch(entity):
        return content_tokens(entity) <= content_tokens(subject_text)
    return False


def _same_entity(a: str, b: str) -> bool:
    a, b = _strip_determiner(a), _strip_determiner(b)
    if _IDENTIFIER.fullmatch(a) or _IDENTIFIER.fullmatch(b):
        return _phrase(a) == _phrase(b)
    return bool(content_tokens(b)) and content_tokens(b) <= content_tokens(a)


def _adds_value(text: str, question: str) -> bool:
    return bool(content_tokens(text) - content_tokens(question))


def _phrase_pattern(words: str) -> str:
    parts = [re.escape(word) + r"(?:s|es)?" for word in words.split()]
    return r"\s+".join(parts)


def _attribute_answer(shape: Shape, sentence: str, note: Note, question: str) -> bool:
    attribute = _strip_determiner(shape.attribute)
    pattern = (
        rf"^(?P<lead>.*?)\b{_phrase_pattern(attribute)}"
        rf"(?:\s+(?:for|of)\s+(?P<obj>[^,]+?))?\s+{COPULA}\s+(?P<value>.+?)\.?$"
    )
    match = re.match(pattern, sentence, flags=re.IGNORECASE)
    if not match or not _adds_value(match.group("value"), question):
        return False
    entity = shape.subject
    if match.group("obj"):
        return _names(match.group("obj"), entity) or _subject_matches(
            match.group("obj"), entity, note
        )
    lead = match.group("lead").strip()
    possessive = re.fullmatch(r"(.+?)['’]s", lead)
    if possessive:
        return _subject_matches(possessive.group(1), entity, note)
    if lead.lower() in {"", "the", "its", "their"}:
        return _subject_matches("it" if lead else "", entity, note)
    return False


def _status_answer(shape: Shape, sentence: str, note: Note, question: str) -> bool:
    match = re.match(
        rf"^(?:project\s+)?(?P<subj>.+?)\s+{COPULA}\s+(?P<value>.+?)\.?$",
        sentence,
        flags=re.IGNORECASE,
    )
    if not match:
        return False
    value = f" {match.group('value').lower()} "
    if not any(f" {word} " in value for word in STATUS_WORDS):
        return False
    entity = shape.subject
    return _subject_matches(match.group("subj"), entity, note)


def _verb_forms(verb: str) -> tuple[str, str, str]:
    """Base, third person, and past participle of a regular verb."""

    if verb.endswith(("ches", "shes", "sses", "xes")):
        base = verb[:-2]
    elif verb.endswith("s") and not verb.endswith("ss"):
        base = verb[:-1]
    else:
        base = verb
    third = base + ("es" if base.endswith(("ch", "sh", "ss", "x")) else "s")
    past = base + ("d" if base.endswith("e") else "ed")
    return base, third, past


def _object_matches(obj: str, entity: str, note: Note) -> bool:
    if _names(obj, entity):
        return True
    words = _strip_determiner(obj).split()
    if not words:
        return False
    if obj.strip().lower() in PRONOUNS:
        return _same_entity(_note_subject(note), entity)
    note_subject = _note_subject(note)
    if _IDENTIFIER.fullmatch(note_subject) and words[0] == _head(note_subject):
        return _same_entity(note_subject, entity)
    return False


def _who_answer(shape: Shape, sentence: str, note: Note, question: str) -> bool:
    base, third, past = _verb_forms(shape.verb)
    entity = shape.subject
    active = re.match(
        rf"^(?P<agent>.+?)\s+(?:{base}|{third})\s+(?P<obj>.+?)\.?$",
        sentence,
        flags=re.IGNORECASE,
    )
    if active and _adds_value(active.group("agent"), question):
        if _object_matches(active.group("obj"), entity, note):
            return True
    passive = re.match(
        rf"^(?P<obj>.+?)\s+{COPULA}\s+{past}\s+by\s+(?P<agent>.+?)\.?$",
        sentence,
        flags=re.IGNORECASE,
    )
    if passive and _adds_value(passive.group("agent"), question):
        return _subject_matches(passive.group("obj"), entity, note)
    return False


def _does_answer(shape: Shape, sentence: str, note: Note, question: str) -> bool:
    words = sentence.rstrip(".").split()
    _, third, _ = _verb_forms(shape.verb)
    for cut in range(1, len(words)):
        subject = " ".join(words[:cut])
        if not _subject_matches(subject, shape.subject, note):
            continue
        verb = words[cut].lower()
        if shape.verb != "do" and verb != third:
            continue
        rest = " ".join(words[cut + 1 :])
        if not rest or not _adds_value(rest, question):
            continue
        if shape.extra and not content_tokens(shape.extra) <= content_tokens(sentence):
            continue
        return True
    return False


def _modal_answer(shape: Shape, sentence: str, note: Note, question: str) -> bool:
    match = re.match(
        rf"^(?P<subj>.+?)\s+{shape.modal}\s+(?P<verb>[a-z]+)(?P<obj>.*?)\.?$",
        sentence,
        flags=re.IGNORECASE,
    )
    if not match:
        return False
    verb = match.group("verb").lower()
    asked = shape.extra.split()
    if verb in asked:
        position = asked.index(verb)
    elif "do" in asked:
        position = asked.index("do")
    else:
        return False
    subject = " ".join(asked[:position])
    remainder = " ".join(asked[position + 1 :])
    if not subject or not _subject_matches(match.group("subj"), subject, note):
        return False
    obj = f"{verb if asked[position] == 'do' else ''} {match.group('obj')}"
    if remainder and not content_tokens(remainder) <= content_tokens(obj):
        return False
    return _adds_value(match.group("obj"), question)


def _covers(clause: str, wanted: str) -> bool:
    tokens = content_tokens(wanted)
    return bool(tokens) and tokens <= content_tokens(clause)


def _about(sentence: str, subject: str, note: Note) -> bool:
    if _strip_determiner(subject) == ORGANISATION.lower():
        return True
    return _names(sentence, subject) or _same_entity(_note_subject(note), subject)


def _do_when_answer(shape: Shape, sentence: str, note: Note, question: str) -> bool:
    if not DEONTIC.search(sentence) or not _covers(sentence, shape.condition):
        return False
    return _about(sentence, shape.subject, note) and _adds_value(sentence, question)


def _deontic_answer(shape: Shape, sentence: str, note: Note, question: str) -> bool:
    family = DEONTIC_FAMILY[shape.modal]
    if not family.search(sentence) or not _covers(sentence, shape.condition):
        return False
    return _adds_value(sentence, question)


def _happens_answer(shape: Shape, sentence: str, note: Note, question: str) -> bool:
    if EMBEDDED_QUESTION.match(sentence):
        return False
    for marker in CONDITION_MARKERS.finditer(sentence):
        clause = sentence[marker.end() :].split(",")[0]
        if not _covers(clause, shape.condition):
            continue
        target = re.sub(r"^(?:its|their|the|a|an)\s+", "", shape.target.strip())
        if target and not _covers(sentence, target):
            continue
        if _adds_value(sentence, question):
            return True
    return False


def _refers(sentence: str, entity: str, note: Note) -> bool:
    if _names(sentence, entity):
        return True
    subject_word = sentence.split()[0].lower() if sentence.split() else ""
    starts_with_pronoun = subject_word in {"it", "its", "they", "their"}
    note_subject = _note_subject(note)
    if note.trust_profile == "durable_knowledge":
        return _same_entity(note_subject, entity)
    return starts_with_pronoun and _same_entity(note_subject, entity)


def _relation_answer(shape: Shape, sentence: str, note: Note, question: str) -> bool:
    return _refers(sentence, shape.subject, note) and _refers(sentence, shape.other, note)


ANSWER_PATTERNS = {
    "attribute": _attribute_answer,
    "status": _status_answer,
    "who": _who_answer,
    "does": _does_answer,
    "modal": _modal_answer,
    "do_when": _do_when_answer,
    "deontic": _deontic_answer,
    "happens": _happens_answer,
    "relation": _relation_answer,
}


def c2_templates(question: str, note: Note) -> tuple[Judgement, ...]:
    shape = parse_shape(question)
    if shape is None:
        return tuple(
            Judgement(sentence, False, 0.0, "question outside the inventory")
            for sentence in note.sentences
        )
    check = ANSWER_PATTERNS[shape.name]
    total = len(note.sentences)
    return tuple(
        Judgement(
            sentence,
            check(shape, sentence, note, question),
            float(total - index),
            f"shape {shape.name}",
        )
        for index, sentence in enumerate(note.sentences)
    )


DETERMINISTIC: dict[str, Judge] = {
    "B0": b0_first_sentence,
    "B1": b1_overlap,
    "C1": c1_entity_attribute,
    "C2": c2_templates,
}
