# Commit convention

This repository follows Conventional Commits 1.0.0.

```
<type>(<optional scope>): <short description in imperative mood>
```

Types: `feat`, `fix`, `test`, `docs`, `refactor`, `chore`, `perf`.

Rules:

1. One logical change per commit. Include the tests that lock that change.
2. Work on `feat/...` or `fix/...` branches. Open a pull request into `main`.
3. Do not squash a multi-commit PR into a single dump commit.
4. Never hardcode machine paths or secrets. The pre-commit hook rejects `/Users/*`, `/home/*`, and common API token shapes.
