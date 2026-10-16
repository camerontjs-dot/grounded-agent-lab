"""Learned candidates for the open-question support check. See PROTOCOL.md.

They need torch and transformers, which the package does not install. Models load
from the local Hugging Face cache at pinned revisions with `local_files_only=True`,
so a run never downloads anything. A missing runtime or model raises
`CandidateBlocked`, and the harness records the candidate as blocked rather than
skipping or scoring it.

Thresholds are the protocol's a priori values. None of them is tuned here.
"""

from __future__ import annotations

from typing import Any

from candidates import Judgement, entity_terms, note_is_about
from evalset import Note

QNLI_MODEL = "cross-encoder/qnli-electra-base"
QNLI_REVISION = "c7dea87c98b2269a935686c31336e97e837cbbeb"
QNLI_THRESHOLD = 0.5
SQUAD_MODEL = "deepset/roberta-base-squad2"
SQUAD_REVISION = "adc3b06f79f797d1c575d5479d6f5efe54a9e3b4"
NO_ANSWER_THRESHOLD = 0.0
MAX_ANSWER_TOKENS = 30
MAX_SEQUENCE = 384


class CandidateBlocked(Exception):
    """The runtime or the pinned model is not available locally."""


def _runtime() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise CandidateBlocked("torch and transformers are not installed") from exc
    return torch, transformers


class QnliScorer:
    """Sigmoid of the QNLI cross-encoder logit for (question, passage)."""

    def __init__(self) -> None:
        self.torch, transformers = _runtime()
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                QNLI_MODEL, revision=QNLI_REVISION, local_files_only=True
            )
            self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
                QNLI_MODEL, revision=QNLI_REVISION, local_files_only=True
            )
        except OSError as exc:
            raise CandidateBlocked(f"{QNLI_MODEL}@{QNLI_REVISION} is not cached") from exc
        if self.model.config.num_labels != 1:
            raise CandidateBlocked(f"{QNLI_MODEL} does not have a single output logit")
        self.model.eval()
        self._cache: dict[tuple[str, str], float] = {}

    def score(self, question: str, passage: str) -> float:
        key = (question, passage)
        if key not in self._cache:
            # One pair per forward pass: no padding, so scores do not depend on batching.
            features = self.tokenizer(question, passage, truncation=True, return_tensors="pt")
            with self.torch.no_grad():
                logit = self.model(**features).logits[0, 0]
            self._cache[key] = float(self.torch.sigmoid(logit))
        return self._cache[key]


class QnliCandidate:
    def __init__(self, scorer: QnliScorer, *, with_title: bool, bind_entity: bool) -> None:
        self.scorer = scorer
        self.with_title = with_title
        self.bind_entity = bind_entity

    def __call__(self, question: str, note: Note) -> tuple[Judgement, ...]:
        if self.bind_entity and not note_is_about(note, entity_terms(question)):
            return tuple(
                Judgement(sentence, False, 0.0, "note is not about the asked entity")
                for sentence in note.sentences
            )
        judgements = []
        for sentence in note.sentences:
            passage = f"{note.title}. {sentence}" if self.with_title else sentence
            score = self.scorer.score(question, passage)
            judgements.append(
                Judgement(sentence, score >= QNLI_THRESHOLD, score, f"qnli {score:.4f}")
            )
        return tuple(judgements)


class SquadCandidate:
    """SQuAD 2.0 reader over "<title>. <body>"; the span's sentence, or nothing."""

    def __init__(self) -> None:
        self.torch, transformers = _runtime()
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                SQUAD_MODEL, revision=SQUAD_REVISION, local_files_only=True
            )
            self.model = transformers.AutoModelForQuestionAnswering.from_pretrained(
                SQUAD_MODEL, revision=SQUAD_REVISION, local_files_only=True
            )
        except OSError as exc:
            raise CandidateBlocked(f"{SQUAD_MODEL}@{SQUAD_REVISION} is not cached") from exc
        if not self.tokenizer.is_fast:
            raise CandidateBlocked("the SQuAD reader needs a fast tokenizer for offsets")
        self.model.eval()

    def _read(self, question: str, context: str) -> tuple[int, float, float] | None:
        """Start character of the best span, its score, and the no-answer score."""

        features = self.tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=MAX_SEQUENCE,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = features.pop("offset_mapping")[0].tolist()
        sequence = features.sequence_ids(0)
        if sequence[-2] != 1 or offsets[-2][1] < len(context.rstrip()):
            raise CandidateBlocked("a note was truncated by the SQuAD reader")
        with self.torch.no_grad():
            output = self.model(**features)
        start = output.start_logits[0].tolist()
        end = output.end_logits[0].tolist()
        no_answer = start[0] + end[0]
        context_tokens = [index for index, part in enumerate(sequence) if part == 1]
        best: tuple[float, int] | None = None
        for i in context_tokens:
            for j in context_tokens:
                if j < i or j - i + 1 > MAX_ANSWER_TOKENS:
                    continue
                score = start[i] + end[j]
                if best is None or score > best[0]:
                    best = (score, i)
        if best is None:
            return None
        return offsets[best[1]][0], best[0], no_answer

    def __call__(self, question: str, note: Note) -> tuple[Judgement, ...]:
        prefix = f"{note.title}. "
        context = prefix + " ".join(note.sentences)
        read = self._read(question, context)
        chosen = None
        detail = "no span"
        margin = 0.0
        if read is not None:
            start_char, span_score, no_answer = read
            margin = span_score - no_answer
            detail = f"span-minus-null {margin:.4f}"
            if margin > NO_ANSWER_THRESHOLD:
                position = len(prefix)
                for sentence in note.sentences:
                    if position <= start_char < position + len(sentence):
                        chosen = sentence
                        break
                    position += len(sentence) + 1
                if chosen is None:
                    detail += ", span in title"
        return tuple(
            Judgement(sentence, sentence == chosen, margin, detail) for sentence in note.sentences
        )


def build_learned() -> dict[str, Any]:
    """Build every learned candidate, or record why it is blocked."""

    built: dict[str, Any] = {}
    try:
        scorer = QnliScorer()
    except CandidateBlocked as exc:
        for name in ("C3a", "C3b", "C4"):
            built[name] = exc
    else:
        built["C3a"] = QnliCandidate(scorer, with_title=False, bind_entity=False)
        built["C3b"] = QnliCandidate(scorer, with_title=True, bind_entity=False)
        built["C4"] = QnliCandidate(scorer, with_title=True, bind_entity=True)
    try:
        built["C5"] = SquadCandidate()
    except CandidateBlocked as exc:
        built["C5"] = exc
    return built


def runtime_versions() -> dict[str, str]:
    try:
        torch, transformers = _runtime()
    except CandidateBlocked:
        return {}
    return {"torch": torch.__version__, "transformers": transformers.__version__}
