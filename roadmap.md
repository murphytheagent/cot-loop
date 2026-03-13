# Roadmap - CoT Loop Detection

Last updated: 2026-03-13 17:04 UTC

Scope:
- Build and validate a probe pipeline for CoT loop detection across prefill and completion feature views.
- Quantify generalization across natural eval splits and OOD tasks.

## Current Status
- Milestone 1 gate: complete.
- Milestone 2 gate: complete.
- Active milestone: Milestone 3 (metadata-aware prefill residual validation).
- Latest result: under the fixed Round 3 metadata-aware frame, simple boundary-summary replacements do not beat the all-layer last-token anchor, but `anchor + last16` gives a modest matched PR-AUC lift to `0.650` versus the Round 4 anchor at `0.642`.
- Active experiment: Round 8 is running as a fixed-label order-sensitive suffix-view diagnostic (`job 1240`) to test whether preserving late prompt-token order can beat the Round 6 anchor without changing labels.
- Active review surface: upstream PR #2 (`task/1772391564-ood-feature-ablation`) is still `OPEN` / `CLEAN` at head `50e0c9d`, but it now has `3` unresolved non-outdated review threads and no terminal local-review verdict inside the `300`-second maintenance bound.

## Milestone 1 - Pipeline and multi-view infrastructure
Status: done (2026-03-05 18:45 UTC)
Success criteria:
- Dataset builder supports prefill and completion feature views in a single pass.
- Training and eval support explicit feature-key selection, multi-view reuse, and imbalance-aware metrics.
- Slurm launchers exist for shared-dataset and k=5 three-view sweeps.

## Milestone 2 - Completion-vs-prefill baseline package
Status: done (2026-03-05 18:45 UTC)
Success criteria:
- Consolidated findings doc and PDF reflect the k=5 three-view results.
- PR #2 includes the multi-view, metrics, and RFM follow-up code paths.
- Completion-view and prefill-view performance are directly comparable on shared labels.

## Milestone 3 - Metadata-aware prefill residual validation
Status: in progress (set 2026-03-13 13:05 UTC)
Success criteria:
- Re-run prefill follow-ups under fixed labels / metadata-aware controls instead of raw composition drift.
- Determine whether any prompt-summary or augmentation beats the metadata-aware last-token anchor on matched evaluation.
- Record the best prefill candidate and its incremental lift versus both metadata-only and anchor-only baselines.

## Milestone 4 - Cross-dataset validation
Status: next (set 2026-03-13 13:05 UTC)
Success criteria:
- Replicate the current best prefill/completion findings on additional evaluation sets or model variants.
- Measure whether the preferred feature view survives varied token budgets and source mixes.
- Separate true robustness gains from prompt-length or source-composition effects.

## Milestone 5 - Deployment readiness
Status: future (set 2026-03-13 13:05 UTC)
Success criteria:
- Recommend a default feature view and probe configuration for routine use.
- Document performance tradeoffs, expected failure modes, and cost profile.
