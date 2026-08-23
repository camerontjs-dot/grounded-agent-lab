## Context & Purpose

<!-- Why does this change exist? -->

## Proposed Changes

-

## Verification & Test Receipts

- [ ] `ruff check .`
- [ ] `python -m pytest -q`
- [ ] `python -m compileall src`

## Security & Leak Prevention Checklist

- [ ] No hardcoded local machine paths (`/Users/*`, `/home/*`)
- [ ] No live API keys, tokens, or credentials in the diff
- [ ] `.env` and local caches remain ignored
