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

## Resume phrases that are safe

- I built a grounded research agent that abstains when evidence is missing.
- I wrapped that loop in LangGraph and checked the fixture decisions match.
- I exported two read-only MCP tools and froze `add_tool`.
- I measured retrieval on a frozen Harbor gold set.
- I built a fail-closed Ollama adapter.

Phrases that are not safe without a new receipt: I used Pinecone. I used CrewAI. I ran Temporal. I red-teamed with Lakera. I evaluated with Ragas as the gate.
