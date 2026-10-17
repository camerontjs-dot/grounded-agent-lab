# Claim boundary

The status labels describe code and fixture checks. They do not establish truth, general semantic support, or production suitability.

## Local experiment versus conceptual

| Status | Meaning here |
|---|---|
| `built` | There is code, a test, and a command a stranger can run |
| `measured` | There is a fixture gold set and a generated receipt |
| `conceptual` | I mapped the idea. I did not run the product |

| Thing | Status | Proof | Do not infer |
|---|---|---|---|
| Extractive grounded loop | `built` | `grounded-agent demo`, pipeline tests | production RAG quality |
| Narrow deterministic support gate (requested quantities and some model-added specifics) | `built` | `tests/test_support.py`, audit gates | entailment, factual accuracy, or general missing-evidence abstention |
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

Each shootout method has its own gate, so the abstain column compares gates, not one threshold. The `hashed_vector` cutoff comes from a null built on the corpus without gold labels. On this corpus it is close to the scores it separates: `trust-labels.md` clears it by 0.005 on one question, and the project-scope cutoff rests on 51 null scores.

A live MindGraph URL is fail-closed unless you set it. CI never requires it.

Every routed index is required, and outages are injected in-process, so a live index outage is not exercised.

<a id="support-gate"></a>

## Support gate and known gaps

The extractive drafter and model drafts go through one deterministic gate, `src/grounded_agent/support.py`. Its checks are narrow; a citation or a returned `answered` status does not establish that the text supports the requested claim. The gate refuses, with `abstain_reason=unsupported_claim`:

- a "how many", "how much", or "number of" question when no cited sentence states the quantity. For "how many <noun>" the number has to sit in front of that noun, so "two separate indexes" is not a count of notes;
- a model claim that adds a number, acronym, identifier, e-mail address, URL, or capitalised mid-sentence name that its cited notes do not contain.

It does not judge entailment or generally abstain when evidence is missing. Four known gaps are recorded as strict xfails in `tests/test_support.py`:

- a dropped negation passes ("A labelled group is proof that a nominated note is true.");
- a claim built from ordinary words passes ("Harbor pays its chief executive a salary.");
- a count is tied to its noun, not to the rest of the question, so a count taken from the wrong index passes;
- open questions get the first sentence of each citable note without checking whether it answers the question. "What is the next action for lantern-agent?" returns `answered` without the sentence that states the next action.

The gate can also refuse a supported claim. Number words are mapped one word at a time, so "two hundred" does not match "200". A model paraphrase that renames something the note names is refused.

## Resume phrases that are safe

- I built a grounded research loop with deterministic checks for requested quantities and certain model-added specifics; its known gaps include open questions that return `answered` without the requested answer.
- I wrapped that loop in LangGraph and checked the fixture decisions match.
- I exported two read-only MCP tools and froze the FastMCP tool manager, so a write tool cannot be registered, listed, or called.
- I measured retrieval on a frozen Harbor gold set.
- I built a fail-closed Ollama adapter.

Phrases that are not safe without a new receipt: I used Pinecone. I used CrewAI. I ran Temporal. I red-teamed with Lakera. I evaluated with Ragas as the gate.
