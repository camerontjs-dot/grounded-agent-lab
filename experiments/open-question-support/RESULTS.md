# Open-question support check: results

No candidate cleared the preregistered bar, so the extractive drafter is unchanged and the four support-gate xfails in `tests/test_support.py` stay. The check that refused every negative control answered too few of the answerable questions. The checks that answered enough questions also answered negative controls.

[`PROTOCOL.md`](PROTOCOL.md) was committed at `069da32` before any item or candidate existed, and the evaluation set was sealed at `1b21a6e` before any candidate code existed. The candidates were frozen at `fc2fe3b`. The single test run is [`results/test-receipt.json`](results/test-receipt.json), from `ea533d3` on a clean tree with an intact data manifest. [`results/test-summary.md`](results/test-summary.md) is generated from it.

## Test result

The test split has 40 answerable items and 40 negative controls. The bar was 0 false accepts, at most 2 wrong answers and at least 26 supported answers.

| Candidate | Mechanism | False accepts | Wrong answers | Supported | Bar |
|---|---|---|---|---|---|
| B0 | first sentence (current drafter) | 40 | 36 | 4 | reference |
| B1 | most shared tokens, at least 2 | 33 | 16 | 24 | reference |
| C1 | note-level entity binding, attribute coverage | 13 | 6 | 24 | fails |
| C2 | closed question templates | **0** | 1 | 17 | fails on supported |
| C3a | QNLI cross-encoder, sentence only | 23 | 7 | 30 | fails |
| C3b | QNLI cross-encoder, title plus sentence | 23 | 14 | 22 | fails |
| C4 | C3b plus C1's entity binding | 13 | 8 | 24 | fails |
| C5 | SQuAD 2.0 reader with no-answer | 8 | 2 | **32** | fails on false accepts |

The current drafter supports 4 of 40 answerable items and answers all 40 negative controls. That is the open-question gap from #11, counted on this set.

## What failed

**Templates (C2) are safe and narrow.** C2 refused all 40 controls and made one wrong answer, but it abstained on 22 answerable items. Ten of those were question shapes outside its fixed inventory, such as "When is the osprey-gateway review due?" and "How do source revocation and quarantine relate?". Twelve were inside the inventory but worded differently from its answer pattern: "next step" for "next action", a rule stated without a modal ("Emergency fixes need sign-off..."), an outcome stated without a "when" clause ("Snapshots older than ninety days are deleted."), and relations that name a project as "the mirror" or "the gateway". One of the twelve, "Who signs off on osprey-gateway changes?", is the particle-verb case I took out of C2 before freezing (see below). The wrong answer was "What does skua-importer do?", answered with "Project skua-importer is active.": the pattern that lets "do" stand for any verb also accepted a copula. No dev item exercised that.

**The SQuAD 2.0 reader (C5) is broad and not safe.** It supported 32 of 40 and never accepted a wrong-subject or unanswerable control. It did accept three word collisions ("The mirror is checked against the primary every night." for "What does curlew-mirror check?"), three topical sentences and two echo controls, one of them through a pointer sentence ("Partner terms are kept in the partner agreement.").

**The QNLI cross-encoder (C3a, C3b, C4) did not separate answers from topical sentences.** On dev it scored "Project heron-backfill is paused." at 0.995 as an answer to "What is the next action for heron-backfill?". The same loaded model reproduces its model-card example (0.998 for the Berlin population sentence, 0.0004 for the unrelated New York sentence), so this is how the model behaves on these notes, not a loading error. Adding the note title made it worse. Entity binding (C4) cut C3b's false accepts from 23 to 13, but 5 of the 13 were still word collisions.

**Entity binding with attribute coverage (C1)** cut false accepts from 40 to 13 relative to the current drafter. Requiring every named entity in a note's title or first sentence also meant it abstained on all three test questions that relate two named projects.

## Exploratory, after the decision

None of this was preregistered, and none of it can change the decision. [`explore.py`](explore.py) computes it from the committed receipts into [`results/exploratory.json`](results/exploratory.json).

- **Thresholds chosen on dev.** Raising each learned candidate's threshold to the lowest value with no dev false accept gives, on test: C5 at a span margin of 6.98, 0 false accepts, 0 wrong and 18 supported; C4 at 0.9983, 0, 0 and 8; C3a at 0.9988, 0, 0 and 1. C3b has no such threshold on dev. Every mechanism here that avoids false accepts lands at or below 18 of 40.
- **C2 by scope.** On the 30 answerable test items whose shape C2 recognised, it supported 17, abstained on 12 and made the 1 wrong answer. It answered none of the 38 in-scope controls.
- **Harbor fixture, run once.** C2 and C5 both answer "What is the next action for lantern-agent?" with "The next action is implement the deterministic intent router." and skip the atlas-search sentence, which names lantern-agent but states atlas-search's own next action. Both abstain on "What must Harbor do when combining retrieval results from different indexes?", which `fixtures/questions.json` and the demo expect to be answered, and C2 also abstains on the contact-alias question. So under decision rule 3, even a passing C2 would have shipped disabled by default.

## How far this goes

The set is 80 synthetic items written by the person who wrote the candidates. I built C2 on dev, but I had also written the test items. Before freezing I removed three C2 features I had designed with test items in mind (particle verbs, "need" as a deontic marker, and ignoring extra words around an identifier); none of them fired on dev. The labels are strict: a sentence has to state the asked value for the asked subject, so "Emergency fixes need sign-off" does not answer "What changes are blocked during a freeze window?". The bar of 26 supported answers is a judgment I fixed in advance. The four question kinds carry equal weight, and C2 did worst on rule questions (3 of 10 supported) and relation questions (4 of 10).

So this shows that none of these eight mechanisms checks open questions well enough on this set. It does not show that no deterministic check could, and it says nothing about real notes.

## What would change the decision

The smaller change is a scoped template check: check only the shapes C2 recognises, and leave other open questions as labelled first-sentence nominations, as they are now. That variant was not preregistered. It would need its own evaluation with enough in-scope controls to bound the false-accept rate, plus a decision about the Harbor trust-label question, which it would stop answering. The other route, the SQuAD reader behind a collision and pointer guard, would need a fresh test split, because this one has been used.

## Reproduce

```bash
python -m pip install -e ".[dev]" "torch>=2.4,<3" "transformers>=4.44,<5"
python -c "from huggingface_hub import snapshot_download as get; get('cross-encoder/qnli-electra-base', revision='c7dea87c98b2269a935686c31336e97e837cbbeb', allow_patterns=['*.json', '*.txt', 'model.safetensors']); get('deepset/roberta-base-squad2', revision='adc3b06f79f797d1c575d5479d6f5efe54a9e3b4', allow_patterns=['*.json', '*.txt', 'model.safetensors'])"
python experiments/open-question-support/evalset.py
python experiments/open-question-support/run_eval.py --split dev
python experiments/open-question-support/explore.py
```

The learned candidates read `cross-encoder/qnli-electra-base` at `c7dea87` and `deepset/roberta-base-squad2` at `adc3b06` from the local Hugging Face cache and report `blocked` if either is missing. The receipt records torch 2.14.1, transformers 4.57.6 and Python 3.12.13 on macOS arm64. `run_eval.py --split test --final` refuses to run again while `results/test-receipt.json` exists.
