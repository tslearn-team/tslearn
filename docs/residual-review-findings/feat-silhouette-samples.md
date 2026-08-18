# Residual Review Findings

Source run: ce-code-review run `20260819-004907-bb598598` on branch `feat/silhouette-samples` (tslearn-team/tslearn), plan `docs/plans/2026-08-18-001-feat-silhouette-samples-plan.md`.

## Residual Review Findings

- P3 — `tslearn/clustering/utils.py:317` — sample_size/random_state rejection guard is bypassable via metric_params — filed as [tslearn-team/tslearn#701](https://github.com/tslearn-team/tslearn/issues/701)
- P3 — `tslearn/clustering/utils.py:335` — Precomputed distance matrix passed without metric="precomputed" is silently re-interpreted as raw time-series data — filed as [tslearn-team/tslearn#702](https://github.com/tslearn-team/tslearn/issues/702)
