# Proposed 0.8.0 migration

This is the migration boundary for the proposed next package version, `0.8.0`. It is an unreleased proposal on the successor worktree. No `0.8.0` tag or release exists, and this document does not qualify the successor for merge or release.

## Receipt schema: legacy V1 to V2

The old run receipt has no `receipt_schema` or `implementation_version`. Its hash covers redacted run metadata, but it does not bind cited file bytes or the corpus that was searched. The successor emits `grounded-agent-receipt-v2` receipts. Their canonical SHA-256 payload includes the schema, package version, request and routing metadata, citation paths, cited-source digests, corpus digests, trust profiles, abstention reason, tools and errors, provider, and redaction flag. It excludes the `receipt_hash` field itself.

For each citation, V2 carries the SHA-256 of the raw bytes read from that cited note, in citation order. It also carries a SHA-256 manifest digest for each corpus that retrieval searched. The builder refuses a citation when its evidence has no unique source digest. These digests bind the recorded inputs; they do not establish that an answer is true or supported.

The typed `Receipt` model accepts the V2 schema and checks digest syntax and alignment with citation paths. The captured legacy receipt was rejected because it lacks `implementation_version`; its citation list also cannot be paired with source digests that were never recorded. Do not fill missing fields and call the result V2. That would create a new record, not recover the old execution.

No V1 compatibility reader is proposed for 0.8.0. The first-party inventory found the package's own typed model, builder, CLI serialization, graph checkpoint use, and tests, but no separate MainFrame receipt parser or hash consumer. External consumers are `UNKNOWN`; the scoped search cannot establish that no external consumer exists. If an external archival reader becomes necessary, add an explicit legacy branch that labels those records unverifiable under V2 and preserves their original bytes and hash. It must never re-hash them as V2.

Keep existing V1 receipts and the dated retrieval report as historical artifacts. Do not rewrite old JSON, recompute its hashes under the new rules, or change its schema label. The prior frozen support study remains historical: its 0/8 result is not promoted or reinterpreted by this package migration.

## Implementation identity and version

The recommended next release is package version `0.8.0`, an unreleased pre-1.0 minor migration boundary. The successor source sets `0.8.0` in both `pyproject.toml` and `grounded_agent.__version__`; the version change remains proposed until an authorized release.

`implementation_version` records that package version. It does not identify one source tree: multiple unreleased commits can carry `0.8.0`, and the receipt does not contain a Git SHA. Any qualification or release record that makes a reproducibility claim must pair the package version with the exact source commit and tree. Do not infer exact code identity from the package version, branch name, or receipt hash.

## MCP dependency boundary

The successor declares `mcp>=1.21.1,<2`. The observed failing boundary is MCP 1.21.0 with Pydantic 2.14.1; MCP 1.21.1 with that Pydantic version passed the focused MCP controls on Python 3.11, 3.12, and 3.13. MCP 1.21.1's own metadata requires Pydantic `>=2.11.0,<3`, so the current direct `pydantic>=2.6,<3` declaration does not mean every lower Pydantic pairing is supported. The MCP minimum is one dependency boundary, not a complete floor for all transitive packages.

The successor CI is intended to cover the MCP minimum with Pydantic 2.14.1 and a fresh resolution at the declared ranges on Python 3.11–3.13. Treat those as release-gate requirements until results exist on the exact successor head. Retain the current Pydantic declaration unless a reproduced resolver failure requires a narrower direct constraint.

## Packaged Harbor fixtures and shootout output

The default Harbor corpus, questions, and gold labels now live inside the Python package at `grounded_agent/fixtures/`. Installed wheels use those packaged resources, so consumers should stop resolving the defaults from a repository-root `fixtures/` directory. The report records package-relative fixture labels.

`grounded-agent shootout` writes `retrieval-shootout.json` and `retrieval-shootout.md` under `./reports` relative to the caller's current directory by default. Pass `--out` when a different destination is needed. Scripts should not assume the command writes into the checkout's `reports/` directory.

The current report schema is `grounded-agent-shootout-v2`. Its three methods have distinct gates: lexical uses query-token recall plus a two-shared-token minimum; `hashed_vector` uses a cosine cutoff calibrated from a single-shared-token null; and `graph_expand` follows one-hop links from lexical seeds that pass the lexical gate. A method with no qualifying evidence abstains.

Query latency is the median over 25 runs with the index already built. Index build time and hashed-vector calibration time are reported separately. The historical V1 table used a shared threshold on different score scales, allowed graph expansion without an abstention gate, and included per-query index work in timing. Those values answer different measurement questions and are not directly comparable to V2. Keep the V1 files unchanged under `reports/history/`; use the V2 receipt and its per-question rows for new comparisons. The current gold set has five Harbor questions and does not establish vendor or production rankings.

## First-party evidence and remaining boundary

The migration details above follow the successor source and tests:

| Surface | Behavior |
| --- | --- |
| `src/grounded_agent/models.py` | Declares `grounded-agent-receipt-v2`, the `Receipt` fields, SHA-256 syntax, and cited-digest/path alignment. |
| `src/grounded_agent/receipt.py` | Builds the canonical V2 hash, binds each citation to a unique raw-source digest, and includes corpus digests and package version. |
| `src/grounded_agent/retrieve.py` and `src/grounded_agent/pipeline.py` | Hash note bytes and build a manifest digest for each corpus searched. |
| `tests/test_receipt_binding.py` and `tests/test_models.py` | Check source and corpus digest changes, version binding, canonical hash coverage, schema shape, and redacted serialization. |
| `src/grounded_agent/paths.py` and `tests/test_packaging.py` | Resolve the default corpus inside the package and require fixture files to be included in the wheel. |
| `src/grounded_agent/shootout.py` and `tests/test_shootout.py` | Emit and read the V2 report, recompute metrics from per-question rows, and keep the V1 report as historical evidence. |
| `src/grounded_agent/cli.py` | Serializes a run receipt for `ask --json`; it does not provide a separate legacy-receipt import command. |

The MainFrame search found project notes, issue references, and a merge runbook describing this repository and its commands. It found no separate first-party parser or verifier for Grounded Agent Lab receipts in MainFrame's `bin/`, outer `tests/`, or other project code. This is a bounded local search, not evidence about package downloads, forks, private integrations, or other external consumers. External use and migration requirements remain `UNKNOWN`.
