# Claim boundary

If a sentence is not in the left column with a matching proof, I should not say it in an interview.

## Local experiment versus conceptual

| Status | Meaning here |
|---|---|
| `built` | There is code, a test, and a command a stranger can run |
| `measured` | There is a fixture gold set and a generated receipt |
| `conceptual` | I mapped the idea. I did not run the product |

| Thing | Status | Proof | Do not infer |
|---|---|---|---|
| Extractive grounded loop | `built` | `grounded-agent demo`, pipeline tests | production RAG quality |
| Support gate (counts, numbers, names) | `built` | `tests/test_support.py`, audit gates | entailment, factual accuracy, checked answers to open questions |
| LangGraph wrap | `built` | graph tests match baseline hashes | LangGraph is always better |
| Read-only MCP tools | `built` | MCP and tool tests | GitHub/Slack/Drive MCP |
| Retrieval shootout | `measured` | `reports/retrieval-shootout.json` | vendor ranking, n>5 |
| Injection / redaction / review | `built` | security and checkpoint tests | Lakera, NeMo, Presidio |
| Ollama adapter | `built` | fail-closed tests and demo closed-port case | I serve a local model in CI |
| Hosted OpenAI-compatible adapter | `built` | fail-closed without credentials | I called OpenAI or Anthropic |
| Temporal / Airflow / n8n / Prefect | `conceptual` | [`workflow-portability.md`](workflow-portability.md) | I deployed those engines |
| Pinecone / Weaviate / CrewAI / LangSmith | `conceptual` | not in the runnable path | I used those products |

## Fixture limits

Harbor is synthetic. The gold set is five questions. `hashed_vector` is a sha256 bag-of-tokens index, not MiniLM and not Pinecone. Graph expansion is one-hop wiki links, not Microsoft GraphRAG.

A live MindGraph URL is fail-closed unless you set it. CI never requires it.

Every routed index is required, and outages are injected in-process, so a live index outage is not exercised.

## Support gate

The extractive drafter and model drafts go through one deterministic gate, `src/grounded_agent/support.py`. It refuses, with `abstain_reason=unsupported_claim`:

- a "how many", "how much", or "number of" question when no cited sentence states the quantity. For "how many <noun>" the number has to sit in front of that noun, so "two separate indexes" is not a count of notes;
- a model claim that adds a number, acronym, identifier, e-mail address, URL, or capitalised mid-sentence name that its cited notes do not contain.

It does not judge meaning. These gaps are recorded as strict xfails in `tests/test_support.py`:

- a dropped negation passes ("A labelled group is proof that a nominated note is true.");
- a claim built from ordinary words passes ("Harbor pays its chief executive a salary.");
- a count is tied to its noun, not to the rest of the question, so a count taken from the wrong index passes;
- open questions still get the first sentence of each citable note. "What is the next action for lantern-agent?" returns `answered` without the sentence that states the next action.

The gate can also refuse a supported claim. Number words are mapped one word at a time, so "two hundred" does not match "200". A model paraphrase that renames something the note names is refused.

## Resume phrases that are safe

- I built a grounded research agent that abstains when evidence is missing.
- I wrapped that loop in LangGraph and checked the fixture decisions match.
- I exported two read-only MCP tools and froze the FastMCP tool manager, so a write tool cannot be registered, listed, or called.
- I measured retrieval on a frozen Harbor gold set.
- I built a fail-closed Ollama adapter.

Phrases that are not safe without a new receipt: I used Pinecone. I used CrewAI. I ran Temporal. I red-teamed with Lakera. I evaluated with Ragas as the gate.
