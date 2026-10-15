# Audit regression gates (2026-10-15)

**Scope:** synthetic Harbor fixtures, tests only. Baseline: `ddc65e347c30b1300df4cb21646bba1b14908942`.
No material production code, model/provider behaviour, shared contract, or main branch has been changed by this test pack.

## Commands

- `python -m pytest -q tests/test_audit_regressions.py`: existing positive controls pass; known defects are visible as **strict xfails**.
- `python -m pytest --runxfail -q tests/test_audit_regressions.py`: run the same gates without expected-failure handling to observe the unfixed defects.
- `ruff check . && python -m pytest -q`: the normal source-level suite, including the audit gates.
- `python scripts/smoke_wheel.py`: build a wheel, install into a new venv outside the checkout, and run the installed CLI demo. **This is currently an opt-in red gate; it is not yet wired into CI.**

## Frozen acceptance properties

| Case | Expected property | Work |
|---|---|---|
| Positive extractive literal count | Supplied exact count is answerable | #11 |
| Positive model literal | Verbatim cited statement remains answerable | #11 |
| Repeated source-identical receipt | Same inputs give same redacted digest | #14 |
| Unsupported count question | Baseline and graph both abstain | #11 |
| High-overlap near miss | Nomination without the requested count abstains | #11 |
| Unsupported model claim | Valid path does not authorize invented salary | #11 |
| Model numeric mutation | Seven-index claim is rejected against two-index source | #11 |
| Same-path source mutation | Cited source bytes change the receipt digest | #14 |
| Federated index unavailable | Required-source failure is not reported as complete | #19 |
| Timeout wall clock | Caller returns within budget plus bounded slack | #12 |
| MCP manager registration | No write capability appears in tool registry | #16 |
| Retrieval missing fact | Hashed-vector and graph ranking can reject an insufficient query | #15 |
| Built wheel used outside checkout | CLI demo functions with packaged corpus | #20 |

The tests do not exercise a real MCP stdio client (#13), since that transport is not implemented or selected by the current product contract. A separate transport smoke belongs with that feature. The five-question shootout is not a model benchmark, and these gates do not validate general semantic entailment.

## Strict xfail is an explicit known-defect record

Each observed failing property has `pytest.mark.xfail(strict=True)` with an issue/reason. It is neither skipped nor declared repaired. A fix makes the test XPASS(strict), which fails CI until that repair PR **removes its own marker** and preserves the test as an ordinary passing regression. Do not erase or relax unrelated failures to obtain a green check. On the exact base, use `--runxfail` to see the failures without suppression.

The installed-wheel smoke is intentionally independent and must fail until the wheel actually carries the required fixture data (or an explicit portable corpus path). After repair, add it as a required CI job; an editable install does not qualify.

## Delivery boundary

- Implement #11 first; retrieval nomination and claim warrant must remain separate. No general entailment-engine accuracy claim follows from a few deterministic cases.
- The existing `codex/day5-grounded-timeout` branch already contains a #12 candidate. Reconcile and verify instead of recreating it.
- Keep repairs in coherent issue-sized PRs. The tests-only draft can be merged separately once the expected-failure record is reviewed, or the relevant test can accompany its own fix PR.
- Preserve the result of the original defect-reproduction runs. Do not change fixture/gold expectations after seeing a failure without a separate documented decision.
- Tests use a fixed request ID for receipt identity comparisons, mutate only a temp-copied corpus, and use no hosted model, private note, network daemon, or real write tool.
- No release or production promotion is implied by these tests.
