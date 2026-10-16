# Open-question support check: protocol

Preregistered on 2026-10-16. I committed this file before writing any evaluation item or candidate code, and before running any model. Base: `fix/11-claim-support-gate` at `c6dce3b` (PR #22).

## Question

For an open question, one where `requested_quantity()` returns None, the extractive drafter returns the first sentence of every citable note. `tests/test_support.py::test_known_gap_open_question_first_sentence` records the result: "What is the next action for lantern-agent?" is answered with "Project lantern-agent is active."

Can a deterministic or locally run mechanism check that a selected sentence answers an open question, well enough to replace first-sentence extraction, without turning into a semantic evaluator that nobody has qualified?

This covers sentence selection for open questions only. Quantity questions and model-claim checks (the dropped-negation, ordinary-word and wrong-index gaps) are out of scope.

## When a sentence answers

A sentence answers a question when, read with only its note's title for context, it states what the question asks for, about the thing the question asks about. "It" and "its" refer to the note's subject. A sentence does not answer when it:

- mentions the topic without giving the value asked for;
- gives that kind of value for a different project, rule or actor, even if it names the asked one;
- uses the question's words in another sense;
- states the attribute or rule in general terms, with no value for the asked case.

An item's gold set is every sentence in its notes that answers. A negative control has an empty gold set. Titles are context, never answers.

## Evaluation set

Synthetic Harbor-style notes only, under `experiments/open-question-support/`. Nothing is copied from `fixtures/`, from real systems or from private notes. An item is a question plus the one to three notes nominated for it, in nomination order. Every nominated note counts as citable, because the check runs after retrieval.

Two splits, with no notes or entity names in common:

| Split | Answerable | Negative controls | Use |
|---|---|---|---|
| dev | 20 | 20 | building and debugging candidates |
| test | 40 | 40 | one final run, nothing changed after it |

The four question kinds come from the open questions the Harbor fixture already asks: attribute (the next action for a project), rule (what Harbor must do when combining results), consequence (what remains unproven when a receipt omits snippets) and relation (how one thing relates to another). Test has 10 answerable items and 10 negative controls of each kind.

Test answerable items: in exactly 10 items every gold sentence is the first sentence of its note, and in the other 30 the gold sentence comes later. At least 10 items word the answer differently from the question, at least 10 refer to the subject by pronoun or leave it implicit, and at least 15 include a distractor note.

Test negative controls, 8 of each type, each type spread over at least three kinds:

- topic: the notes are about the asked subject, but none states the value asked for;
- wrong subject: a sentence gives that kind of value for another subject;
- collision: a sentence reuses the question's key words in another sense;
- echo: a sentence names the attribute or rule in general terms, with no value for the asked case;
- unanswerable: the notes share words with the question but none addresses it.

Dev uses the same kinds and types at half the counts.

Order of work:

1. This protocol.
2. Both splits, written and labelled, then committed with SHA-256 hashes in `MANIFEST.sha256`. I review every label once before that commit. After it, a test label changes only for a mechanical error, such as a gold sentence that does not exist in its note, and the results list any such change.
3. Candidate code, developed against dev only.
4. Candidate code committed. Then one run on test, written to `results/test-receipt.json`. The harness refuses a test run from a dirty tree and records the commit. A rerun is allowed only if the harness crashes, and the receipt records that.

The Harbor fixtures (`fixtures/corpus/`, `fixtures/questions.json`, `fixtures/gold.json`) and the frozen PR #21 gates in `tests/test_audit_regressions.py` are not used to build or choose any candidate. I have read them, which is a known exposure. They run once, after the decision, to report consequences.

## Candidates

Each candidate looks at one nominated note at a time and selects at most one sentence from it. An item is answered if any note yields a sentence, and abstained otherwise.

| Id | Mechanism | Class | Can be adopted |
|---|---|---|---|
| B0 | first sentence of each note (the current drafter) | reference | no |
| B1 | sentence sharing the most content tokens with the question, at least 2 | reference | no |
| C1 | entity binding at note level, then attribute coverage | deterministic | yes |
| C2 | closed list of question templates with matching answer patterns | deterministic | yes |
| C3a | QNLI cross-encoder on the sentence alone | local model | yes |
| C3b | QNLI cross-encoder on note title plus sentence | local model | yes |
| C4 | C3b restricted to notes that pass C1's entity binding | hybrid | yes |
| C5 | SQuAD 2.0 extractive reader with a no-answer option, per note | local model | yes |

B1 is the "more overlap thresholds" route that ADR-010 already turned down. It runs to show how far overlap gets, and it cannot be adopted.

Fixed rules:

- **B1.** Content tokens are lowercase alphanumeric runs longer than two characters, minus `retrieve.STOPWORDS`, singularised with `support.singular`. A note yields the sentence with the most tokens shared with the question if that count is at least 2, the same floor as `MIN_CITABLE_TOKENS`. Ties go to the earlier sentence.
- **C1.** Entity terms are hyphenated identifiers and capitalised words after the first word of the question, except "Harbor", which every note in the corpus belongs to. Matching ignores case and treats hyphens as spaces, as `support.unsupported_specifics` does. A note is eligible only if every entity term appears in its title or its first sentence, so a note about one project cannot answer for another project it mentions. A question with no entity term leaves every note eligible. Attribute terms are the question's other content tokens. A sentence qualifies if it contains at least two attribute terms (all of them if there are fewer than two) and at least one content token that is not in the question. The note yields its qualifying sentence with the most attribute terms, ties to the earlier sentence. These constants are set here and are not tuned.
- **C2.** A closed inventory of question shapes. Attribute: `What is the <A> for|of <E>?`, `What is <E>'s <A>?`, `Who is the <A> for|of <E>?`, `Who <verb>s <E>?`, `What is the status of <E>?`, `What does <S> <verb>?`. Rule: `What must|should|may|can <S> <verb> ...?`, `What must|should <S> do when|before|after|if <C>?`, `What is required|forbidden|allowed when|before|for <C>?`. Consequence: `What happens when|if|after <C>?`, `When|If <C>, what ...?`. Relation: `How does <E1> <verb phrase> <E2>?`, `What is the relationship between <E1> and <E2>?`. Each shape has an answer pattern: the attribute phrase followed by a copula and a value, the same modal and verb followed by an object, a deontic sentence that covers the condition, a "when" or "if" sentence whose condition matches, or a sentence that names both entities. Status answers need a word from this list: active, paused, blocked, on hold, deferred, archived, complete, completed, cancelled, shipped, live, deprecated, in review, draft, frozen. "It", "its", "they", "their" and "the project" resolve to the note's subject, and a sentence that resolves to another subject is rejected. A question outside the inventory abstains. This list fixes the inventory. The surface patterns inside each shape are built on dev.
- **C3a, C3b.** `cross-encoder/qnli-electra-base` at revision `c7dea87c98b2269a935686c31336e97e837cbbeb`, on CPU, score = sigmoid of the logit. C3a scores `(question, sentence)`. C3b scores `(question, "<title>. <sentence>")`. A note yields its highest-scoring sentence if that score is at least 0.5, the model's own decision boundary.
- **C4.** C3b, run only on notes that pass C1's entity binding.
- **C5.** `deepset/roberta-base-squad2` at revision `adc3b06f79f797d1c575d5479d6f5efe54a9e3b4`, on CPU. The context is `"<title>. <body>"`. The best span of at most 30 tokens is compared with the no-answer score at threshold 0.0, the SQuAD 2.0 default. If the span wins and starts inside a body sentence, the note yields that sentence. A span inside the title yields nothing.

Learned candidates run offline from a pinned local copy. If a model or its runtime is missing, the candidate is reported as `blocked`, not skipped and not scored. No learned threshold is tuned for the decision. Thresholds tuned on dev may be reported afterwards, labelled exploratory.

I left out an LLM judge. ADR-010 turned one down as the gate, and a judge prompt adds a free parameter that a set this small cannot qualify. I also left out plain MNLI models: an open question gives no hypothesis to test, and QNLI is the NLI framing built for question and sentence pairs.

## Metric

Per candidate, on test:

- **false accepts:** negative controls that get any sentence;
- **wrong answers:** answerable items where any selected sentence is outside the gold set, even if a gold sentence is selected too. The drafter prints one line per note, so an extra unsupported line is a defect;
- **supported answers:** answerable items where at least one sentence is selected and every selected sentence is gold.

Reported but outside the bar: results by kind, by negative type and by tag; sentence-level precision and recall of each check over every sentence in the nominated notes; abstain rate.

## Pass bar

A candidate clears the bar only if all three hold on test:

| Property | Bar | Exact one-sided 95% bound at the bar |
|---|---|---|
| False accepts | 0 of 40 negative controls | upper bound 7.2% |
| Wrong answers | at most 2 of 40 answerable items | upper bound 14.9% |
| Supported answers | at least 26 of 40 answerable items | lower bound 50.8% |

Bounds are Clopper-Pearson. One false accept moves the upper bound to 11.3%, above the 10% I want from a refusal gate, so one is a fail. I allow two wrong answers because an answerable item that gets one extra line has still shown its reader the gold sentence. Those two still count against the candidate.

## Decision rules

1. If several candidates clear the bar, prefer deterministic over hybrid over learned, then more supported answers, then fewer wrong answers.
2. A deterministic winner goes into the default extractive path. An open question then gets only sentences that pass the check, and it abstains with `unsupported_claim` if none does. Only the `tests/test_support.py` xfails that then pass lose their markers. Tests that assert first-sentence text, such as `test_open_questions_still_get_first_sentence_nominations`, may be updated with the reason stated.
3. If wiring a winner into the default path would change an expected outcome in `tests/test_audit_regressions.py`, `fixtures/questions.json` or the demo checklist, I do not change that expectation here. That includes gates that are still xfail on this branch but pass once the other open repairs (#23 to #28) merge, which I check on a local merge. The check then ships disabled by default, and the PR lists which expectations would change and why, for the owner to decide.
4. A learned winner ships only as an opt-in behind an optional extra. It loads the pinned revision from local files and fails closed when it is enabled but missing. The default path and its xfails stay as they are.
5. If nothing clears the bar, the drafter is unchanged. The negative result goes into `docs/limitations.md` with a link to the receipt, and the four xfails stay.

## What a pass would not show

A pass on 80 synthetic items says the mechanism checked these 80 items. It does not measure accuracy on real notes, other writing styles or other question kinds, and it is not an entailment check. One author wrote the notes, the labels and the candidates. The order of work and the hash manifest make that order checkable. They do not remove the bias.
